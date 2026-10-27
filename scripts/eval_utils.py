"""Shared helpers for standalone evaluation / diagnostics of saved TRM checkpoints."""
import json
import os
import sys
from typing import Dict, Iterator, Optional, Tuple

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.functions import load_model_class  # noqa: E402
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # noqa: E402


def load_model(ckpt: str, data_path: str, config_path: Optional[str] = None, device: str = "cuda",
               overrides: Optional[dict] = None):
    """Build the ACT model from <ckpt dir>/all_config.yaml and load weights (EMA weights for `step_N`).

    Handles the `_orig_mod.` (torch.compile) and `model.` (ACTLossHead) key prefixes. Returns (model, config dict).
    """
    config_path = config_path or os.path.join(os.path.dirname(ckpt), "all_config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    arch = dict(cfg["arch"])
    name = arch.pop("name")
    arch.pop("loss", None)
    arch.update(overrides or {})

    state_dict = torch.load(ckpt, map_location=device, weights_only=True)
    state_dict = {k.removeprefix("_orig_mod.").removeprefix("model."): v for k, v in state_dict.items()}

    with open(os.path.join(data_path, "test", "dataset.json")) as f:
        meta = json.load(f)
    model_cfg = dict(
        **arch,
        batch_size=cfg["global_batch_size"],  # only sizes the train-mode puzzle-emb buffer
        vocab_size=meta["vocab_size"],
        seq_len=meta["seq_len"],
        num_puzzle_identifiers=state_dict["inner.puzzle_emb.weights"].shape[0] if "inner.puzzle_emb.weights" in state_dict else 1,
        causal=False,
    )
    with torch.device(device):
        model = load_model_class(name)(model_cfg)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cfg


def iter_test(data_path: str, batch_size: int, max_examples: Optional[int] = None, device: str = "cuda") -> Iterator[Dict[str, torch.Tensor]]:
    dataset = PuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_paths=[data_path], global_batch_size=batch_size,
                                                test_set_mode=True, epochs_per_iter=1, rank=0, num_replicas=1), split="test")
    seen = 0
    for _set_name, batch, effective in dataset:
        if max_examples is not None and seen >= max_examples:
            return
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if max_examples is not None and seen + effective > max_examples:
            # Mask rows beyond the requested count as padding
            keep = max_examples - seen
            batch["labels"][keep:] = -100
            effective = keep
        seen += effective
        yield batch


def init_carry(inner, batch_size: int, sigma: float = 0.0, generator: Optional[torch.Generator] = None,
               dtype: Optional[torch.dtype] = None, device: str = "cuda"):
    """Fresh carry (H_init / L_init broadcast), optionally with Gaussian noise of std `sigma`."""
    with torch.device(device):
        carry = inner.reset_carry(torch.ones(batch_size, dtype=torch.bool), inner.empty_carry(batch_size))
    z_H, z_L = carry.z_H, carry.z_L
    if sigma > 0:
        z_H = (z_H.float() + sigma * torch.randn(z_H.shape, device=device, generator=generator)).to(z_H.dtype)
        z_L = (z_L.float() + sigma * torch.randn(z_L.shape, device=device, generator=generator)).to(z_L.dtype)
    if dtype is not None:
        z_H, z_L = z_H.to(dtype), z_L.to(dtype)
    return type(carry)(z_H=z_H, z_L=z_L)


def correctness(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (preds, valid rows, per-row exact, per-row cell accuracy)."""
    preds = logits.argmax(-1)
    mask = labels != -100
    counts = mask.sum(-1)
    correct = mask & (preds == labels)
    valid = counts > 0
    exact = valid & (correct.sum(-1) == counts)
    cell = torch.where(valid, correct.sum(-1).float() / counts.clamp_min(1), 0.0)
    return preds, valid, exact, cell
