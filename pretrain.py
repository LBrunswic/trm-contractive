from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import json
import time
import subprocess
import yaml
import shutil
import copy

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
try:
    from adam_atan2 import AdamATan2
except (ImportError, OSError):  # CUDA extension unavailable (e.g. no nvcc): exact pure-PyTorch port
    from models.adam_atan2_torch import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper
from models.regularizers import RegConfig, ContractionRegularizer


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

    # Benchmark infrastructure
    max_train_steps: Optional[int] = None  # hard cap on optimizer steps (smoke tests)
    log_every: int = 50                    # JSONL train-metric window
    abort_on_nan: bool = True
    resume: bool = True                    # resume from <checkpoint_path>/last_state.pt if present
    save_raw_final: bool = True            # also save the raw (non-EMA) weights at the end

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    if config.max_train_steps is not None:
        total_steps = min(total_steps, config.max_train_steps)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def save_full_state(config: PretrainConfig, train_state: TrainState, ema_helper: Optional[EMAHelper],
                    regularizer: Optional[ContractionRegularizer], completed_iters: int):
    """Everything needed to continue training exactly (except the in-flight ACT carry)."""
    if config.checkpoint_path is None:
        return
    state = dict(
        model=train_state.model.state_dict(),
        optimizers=[optim.state_dict() for optim in train_state.optimizers],
        ema=ema_helper.state_dict() if ema_helper is not None else None,
        regularizer=regularizer.state_dict() if regularizer is not None else None,
        step=train_state.step,
        completed_iters=completed_iters,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(),
    )
    path = os.path.join(config.checkpoint_path, "last_state.pt")
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_full_state(config: PretrainConfig, train_state: TrainState, ema_helper: Optional[EMAHelper],
                    regularizer: Optional[ContractionRegularizer]) -> int:
    """Returns the number of completed train iterations (0 if nothing to resume)."""
    path = os.path.join(config.checkpoint_path, "last_state.pt") if config.checkpoint_path is not None else None
    if not (config.resume and path is not None and os.path.exists(path)):
        return 0
    print(f"Resuming from {path}")
    state = torch.load(path, map_location="cuda", weights_only=False)
    train_state.model.load_state_dict(state["model"])
    for optim, optim_state in zip(train_state.optimizers, state["optimizers"]):
        optim.load_state_dict(optim_state)
    if ema_helper is not None and state["ema"] is not None:
        ema_helper.load_state_dict(state["ema"])
    if regularizer is not None and state["regularizer"] is not None:
        regularizer.load_state_dict(state["regularizer"])
    train_state.step = state["step"]
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    return state["completed_iters"]


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int,
                regularizer: Optional[ContractionRegularizer] = None):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Contraction regularizer on the anchors stashed at the previous step (backward now: peak memory = max, not sum)
    reg_metrics = {}
    reg_grads = None
    if regularizer is not None:
        reg_loss, reg_metrics = regularizer(train_state.step)
        if reg_loss is not None:
            reg_loss.backward()
            if train_state.step % config.log_every == 0:  # effective weighting: ||g_reg|| vs ||g_main||
                reg_grads = [p.grad.detach().clone() if p.grad is not None else None for p in train_state.model.parameters()]

    # Forward
    train_state.carry, loss, metrics, outputs, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
    if regularizer is not None and "reg_s_H" in outputs:
        regularizer.stash(outputs, train_state.carry.steps, train_state.step)
    del outputs

    total_loss = (1 / global_batch_size) * loss
    total_loss.backward()
    if reg_grads is not None:
        with torch.no_grad():
            g_reg2 = g_main2 = 0.0
            for p, g_reg in zip(train_state.model.parameters(), reg_grads):
                if g_reg is not None and p.grad is not None:
                    g_reg2 += g_reg.float().square().sum()
                    g_main2 += (p.grad - g_reg).float().square().sum()
            reg_metrics["reg/grad_norm_reg"] = float(torch.as_tensor(g_reg2).sqrt())
            reg_metrics["reg/grad_norm_main"] = float(torch.as_tensor(g_main2).sqrt())
        del reg_grads

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            reduced_metrics.update(reg_metrics)
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name),
        get_model_source_path("regularizers@ContractionRegularizer"),
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    if wandb.run is not None:
        try:
            wandb.run.log_code(config.checkpoint_path)
        except Exception as e:
            print(f"[wandb] log_code failed: {e}")


def _wandb_settings():
    # `_disable_stats` was renamed `x_disable_stats` in recent wandb versions
    for kwargs in ({"x_disable_stats": True}, {"_disable_stats": True}):
        try:
            return wandb.Settings(**kwargs)
        except Exception:
            pass
    return None


class JsonlLogger:
    def __init__(self, path: str):
        self.file = open(path, "a", buffering=1)

    def log(self, record: dict):
        self.file.write(json.dumps(_to_py(record)) + "\n")


def _to_py(v):
    if isinstance(v, dict):
        return {k: _to_py(x) for k, x in v.items()}
    if isinstance(v, (np.generic, np.ndarray)) or torch.is_tensor(v):
        return v.item()
    return v


def _flatten(d: dict, prefix: str = ""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}/"))
        else:
            out[f"{prefix}{k}"] = v
    return out


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    assert config.eval_interval is None or eval_loader is not None, "Eval data missing (upstream silently swallows this)"

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Contraction regularizer / monitor (only for models exporting regularizer anchors)
    regularizer = None
    inner = getattr(train_state.model, "_orig_mod", train_state.model).model.inner
    if hasattr(inner, "cycle"):
        reg_config = RegConfig.from_arch_extra(config.arch.__pydantic_extra__)  # type: ignore
        regularizer = ContractionRegularizer(reg_config, inner, batch_size=config.global_batch_size // WORLD_SIZE,
                                             seed=config.seed + RANK)
        print(f"Regularizer: {reg_config.model_dump()}  (k={regularizer.k}, rho={regularizer.rho:.4f}, mu={regularizer.mu:.4f})")

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    jsonl = None
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Resume
    start_iter = load_full_state(config, train_state, ema_helper, regularizer)
    train_loader.dataset._iters = start_iter  # type: ignore  # continue the data order (Philox(seed + iters))

    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step)
        os.makedirs(config.checkpoint_path, exist_ok=True)  # type: ignore
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(),
                   mode=os.environ.get("WANDB_MODE", "offline"), dir=config.checkpoint_path, settings=_wandb_settings())  # type: ignore
        num_params = sum(x.numel() for x in train_state.model.parameters())
        wandb.log({"num_params": num_params}, step=train_state.step)
        save_code_and_config(config)
        jsonl = JsonlLogger(os.path.join(config.checkpoint_path, "metrics.jsonl"))  # type: ignore
        jsonl.log({"split": "meta", "step": train_state.step, "resumed_from_iter": start_iter, "num_params": num_params,
                   "total_steps": train_state.total_steps, "total_iters": total_iters})

    # Training Loop
    batches_per_iter = int(train_epochs_per_iter * train_metadata.total_groups * train_metadata.mean_puzzle_examples) // config.global_batch_size
    t_start = time.perf_counter()
    train_seconds = eval_seconds = 0.0
    window, window_n, t_window, step_window = {}, 0, time.perf_counter(), train_state.step
    last_eval = None
    done = False
    for _iter_id in range(start_iter, total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        t_iter = time.perf_counter()
        n_batches_iter = 0
        for set_name, batch, global_batch_size in train_loader:
            n_batches_iter += 1
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE, regularizer=regularizer)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

                for k, v in metrics.items():
                    if v is not None and math.isfinite(float(v)):
                        window[k] = window.get(k, 0.0) + float(v)
                        window[k + "#n"] = window.get(k + "#n", 0) + 1
                window_n += 1
                if config.abort_on_nan and not math.isfinite(float(metrics["train/lm_loss"])):
                    raise FloatingPointError(f"Non-finite lm_loss at step {train_state.step}")
                if train_state.step % config.log_every == 0:
                    now = time.perf_counter()
                    record = {k: v / window[k + "#n"] for k, v in window.items() if not k.endswith("#n")}
                    record.update(split="train", step=train_state.step, wall_s=now - t_start,
                                  steps_per_s=(train_state.step - step_window) / (now - t_window),
                                  mem_alloc_peak_gb=torch.cuda.max_memory_allocated() / 2**30,
                                  mem_reserved_peak_gb=torch.cuda.max_memory_reserved() / 2**30)
                    jsonl.log(record)  # type: ignore
                    window, window_n, t_window, step_window = {}, 0, now, train_state.step
            if config.ema:
                ema_helper.update(train_state.model)

            if train_state.step >= train_state.total_steps:
                done = True
                break
        # A stop at total_steps before the loader is exhausted (max_train_steps) leaves the iteration incomplete
        iter_complete = (not done) or n_batches_iter >= batches_per_iter
        train_seconds += time.perf_counter() - t_iter
        is_last = done or (_iter_id == total_iters - 1)

        if _iter_id >= config.min_eval_interval or is_last:
            ############ Evaluation
            t_eval = time.perf_counter()
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)
            eval_dt = time.perf_counter() - t_eval
            eval_seconds += eval_dt

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                last_eval = _flatten(_to_py(metrics))
                jsonl.log({"split": "eval", "step": train_state.step, "iter": _iter_id + 1, "eval_s": eval_dt, **last_eval})  # type: ignore
                print({k: round(float(v), 4) for k, v in last_eval.items()})
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or is_last):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

        if RANK == 0:
            # A max_train_steps stop mid-iteration does not count the iteration as completed
            save_full_state(config, train_state, ema_helper, regularizer, completed_iters=_iter_id + (1 if iter_complete else 0))
            if is_last and config.save_raw_final:
                torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}_raw"))  # type: ignore
        if done:
            break

    # Summary
    if RANK == 0:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        summary = dict(
            run_name=config.run_name, final_step=train_state.step, total_steps=train_state.total_steps,
            wall_s=time.perf_counter() - t_start, train_s=train_seconds, eval_s=eval_seconds,
            peak_mem_alloc_gb=torch.cuda.max_memory_allocated() / 2**30,
            peak_mem_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
            num_params=sum(x.numel() for x in train_state.model.parameters()),
            final_eval=last_eval,
            final_ckpt=os.path.join(config.checkpoint_path, f"step_{train_state.step}"),  # type: ignore
            git_commit=commit, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        )
        with open(os.path.join(config.checkpoint_path, "summary.json"), "w") as f:  # type: ignore
            json.dump(_to_py(summary), f, indent=2)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
