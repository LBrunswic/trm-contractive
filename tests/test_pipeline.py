"""Pipeline tests: test-subset builder, real subsets, Sudoku bench config recipe, pretrain helpers.

CPU only by design: run with
    CUDA_VISIBLE_DEVICES="" DISABLE_COMPILE=1 pytest tests/test_pipeline.py
Read-only use of data/; every file written goes to pytest's tmp_path.
"""
import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("DISABLE_COMPILE", "1")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from dataset.make_test_subset import make_subset  # noqa: E402
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # noqa: E402
from models.losses import IGNORE_LABEL_ID  # noqa: E402

FIELDS = ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices")
SUDOKU_META = dict(pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=11, seq_len=81,
                   num_puzzle_identifiers=1, mean_puzzle_examples=1.0, sets=["all"])


# =============================================================================================
# (1) make_subset on a synthetic Sudoku-shaped source
# =============================================================================================

N_SRC = 200


def _write_sudoku_source(root: Path, n: int = N_SRC, seed: int = 1234, *, broken_layout: bool = False) -> Path:
    """Sudoku layout exactly like build_sudoku_dataset's test split (uint8 tokens, 1 = blank, 2..10 = digits)."""
    rng = np.random.default_rng(seed)
    test = root / "test"
    test.mkdir(parents=True)
    labels = rng.integers(2, 11, size=(n, 81), dtype=np.uint8)
    inputs = np.where(rng.random((n, 81)) < 0.7, 1, labels).astype(np.uint8)
    assert len({r.tobytes() for r in inputs}) == n  # rows identifiable
    np.save(test / "all__inputs.npy", inputs)
    np.save(test / "all__labels.npy", labels)
    np.save(test / "all__puzzle_identifiers.npy", np.zeros(n, dtype=np.int32))
    idx = np.arange(n + 1, dtype=np.int32)
    if broken_layout:  # two examples in the first puzzle
        idx = np.concatenate([[0], np.arange(2, n + 1)]).astype(np.int32)
    np.save(test / "all__puzzle_indices.npy", idx)
    np.save(test / "all__group_indices.npy", np.arange(len(idx), dtype=np.int32))
    with open(test / "dataset.json", "w") as f:
        json.dump(dict(SUDOKU_META, total_groups=n, total_puzzles=n), f)
    with open(root / "identifiers.json", "w") as f:
        json.dump(["<blank>"], f)
    return root


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return _write_sudoku_source(tmp_path_factory.mktemp("src"))


def _load(d: Path, field: str):
    return np.load(d / "test" / f"all__{field}.npy")


def test_make_subset_contents(src, tmp_path):
    n, seed = 37, 5
    out = tmp_path / "sub"
    make_subset(str(src), str(out), n, seed)

    sel = np.load(out / "subset_indices.npy")
    assert sel.shape == (n,)
    assert np.all(np.diff(sel) > 0), "indices must be sorted and unique"
    assert sel.min() >= 0 and sel.max() < N_SRC
    # Exactly the first n entries of the seeded permutation (what makes subsets nested)
    np.testing.assert_array_equal(sel, np.sort(np.random.default_rng(seed).permutation(N_SRC)[:n]))

    for field in ("inputs", "labels", "puzzle_identifiers"):
        got, ref = _load(out, field), _load(src, field)
        np.testing.assert_array_equal(got, ref[sel])
    assert _load(out, "inputs").dtype == np.uint8 and _load(out, "labels").dtype == np.uint8
    for field in ("puzzle_identifiers", "puzzle_indices", "group_indices"):
        assert _load(out, field).dtype == np.int32, field
    np.testing.assert_array_equal(_load(out, "puzzle_indices"), np.arange(n + 1))
    np.testing.assert_array_equal(_load(out, "group_indices"), np.arange(n + 1))

    with open(out / "test" / "dataset.json") as f:
        meta = json.load(f)
    assert meta == dict(SUDOKU_META, total_groups=n, total_puzzles=n)
    with open(src / "test" / "dataset.json") as f:  # source untouched
        assert json.load(f)["total_groups"] == N_SRC

    assert json.loads((out / "identifiers.json").read_text()) == ["<blank>"]
    info = json.loads((out / "subset.json").read_text())
    assert info["n"] == n and info["seed"] == seed
    assert info["source"] == os.path.abspath(str(src))
    assert info["sha256_inputs"] == hashlib.sha256(_load(out, "inputs").tobytes()).hexdigest()
    assert not (out / "train").exists(), "only a test/ split is written"


def test_make_subset_deterministic(src, tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    make_subset(str(src), str(a), 50, seed=0)
    make_subset(str(src), str(b), 50, seed=0)
    make_subset(str(src), str(c), 50, seed=1)
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    assert {str(p) for p in files} >= {"subset_indices.npy", "subset.json", "identifiers.json", "test/dataset.json",
                                       *(f"test/all__{f}.npy" for f in FIELDS)}
    for rel in files:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), f"{rel} differs between identical runs"
    assert not np.array_equal(np.load(a / "subset_indices.npy"), np.load(c / "subset_indices.npy"))


def test_make_subset_nested(src, tmp_path):
    for seed in (0, 7):
        make_subset(str(src), str(tmp_path / f"s10_{seed}"), 10, seed)
        make_subset(str(src), str(tmp_path / f"s50_{seed}"), 50, seed)
        make_subset(str(src), str(tmp_path / f"s200_{seed}"), N_SRC, seed)
        i10, i50, i200 = (np.load(tmp_path / f"s{k}_{seed}" / "subset_indices.npy") for k in (10, 50, 200))
        assert np.isin(i10, i50).all() and np.isin(i50, i200).all()
        np.testing.assert_array_equal(i200, np.arange(N_SRC))  # n == total keeps everything
        # Row-level nesting: subset rows of the small subset appear in the larger one at the right positions
        pos = np.searchsorted(i50, i10)
        np.testing.assert_array_equal(_load(tmp_path / f"s10_{seed}", "inputs"),
                                      _load(tmp_path / f"s50_{seed}", "inputs")[pos])


def test_make_subset_rejects_bad_inputs(src, tmp_path):
    with pytest.raises(AssertionError):
        make_subset(str(src), str(tmp_path / "too_big"), N_SRC + 1, 0)
    bad = _write_sudoku_source(tmp_path / "bad_src", n=20, broken_layout=True)
    with pytest.raises(AssertionError, match="Sudoku layout"):
        make_subset(str(bad), str(tmp_path / "bad_out"), 5, 0)


def _iterate_test(path: Path, batch: int):
    ds = PuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_paths=[str(path)], global_batch_size=batch, test_set_mode=True,
                                           epochs_per_iter=1, rank=0, num_replicas=1), split="test")
    return ds, list(ds)


@pytest.mark.parametrize("n", [10, 16, 50, N_SRC])
def test_subset_through_puzzle_dataset(src, tmp_path, n):
    B = 8
    out = tmp_path / f"sub{n}"
    make_subset(str(src), str(out), n, seed=0)
    ds, batches = _iterate_test(out, B)
    assert ds.metadata.total_groups == n and ds.metadata.total_puzzles == n

    assert len(batches) == math.ceil(n / B)
    assert sum(eff for _, _, eff in batches) == n, "iterates exactly n examples"
    assert all(set_name == "all" for set_name, _, _ in batches)

    sel = np.load(out / "subset_indices.npy")
    ref_inputs, ref_labels = _load(src, "inputs")[sel], _load(src, "labels")[sel]
    for b, (_, batch, eff) in enumerate(batches):
        assert set(batch) == {"inputs", "labels", "puzzle_identifiers"}
        assert batch["inputs"].shape == (B, 81) and batch["labels"].shape == (B, 81)
        assert batch["puzzle_identifiers"].shape == (B,)
        assert all(v.dtype == torch.int32 for v in batch.values())
        rows = slice(b * B, b * B + eff)
        np.testing.assert_array_equal(batch["inputs"][:eff].numpy(), ref_inputs[rows])
        np.testing.assert_array_equal(batch["labels"][:eff].numpy(), ref_labels[rows])
        # Padding of the last batch
        assert (batch["labels"][eff:] == IGNORE_LABEL_ID).all()
        assert (batch["inputs"][eff:] == SUDOKU_META["pad_id"]).all()
        assert (batch["puzzle_identifiers"][eff:] == SUDOKU_META["blank_identifier_id"]).all()
        assert (batch["labels"][:eff] != IGNORE_LABEL_ID).all()

    last_eff = batches[-1][2]
    assert last_eff == (n % B or B)
    if n % B:
        assert (batches[-1][1]["labels"][last_eff:] == -100).all()


def _write_sudoku_train(root: Path, groups: int = 12, per_group: int = 3, seed: int = 99) -> Path:
    """Sudoku train layout: each group = original + augmentations, one example per puzzle."""
    rng = np.random.default_rng(seed)
    d = root / "train"
    d.mkdir(parents=True)
    n = groups * per_group
    labels = rng.integers(2, 11, size=(n, 81), dtype=np.uint8)
    np.save(d / "all__inputs.npy", np.where(rng.random((n, 81)) < 0.7, 1, labels).astype(np.uint8))
    np.save(d / "all__labels.npy", labels)
    np.save(d / "all__puzzle_identifiers.npy", np.zeros(n, dtype=np.int32))
    np.save(d / "all__puzzle_indices.npy", np.arange(n + 1, dtype=np.int32))
    np.save(d / "all__group_indices.npy", np.arange(0, n + 1, per_group, dtype=np.int32))
    with open(d / "dataset.json", "w") as f:  # upstream writes total_puzzles = #groups for Sudoku train
        json.dump(dict(SUDOKU_META, total_groups=groups, total_puzzles=groups), f)
    return root


def test_create_dataloader_paths_and_resume_order(src, tmp_path, bench):
    """create_dataloader: test split uses data_paths_test; resume via dataset._iters reproduces the data order
    (the DataLoader worker is persistent, so _iters set before the first iteration carries over)."""
    import pretrain
    train_root = _write_sudoku_train(tmp_path / "train_src")
    sub = tmp_path / "sub"
    make_subset(str(src), str(sub), 13, seed=0)
    cfg = pretrain.PretrainConfig(**dict(bench, data_paths=[str(train_root)], data_paths_test=[str(sub)]))
    kw = dict(rank=0, world_size=1, global_batch_size=8)

    eval_loader, eval_meta = pretrain.create_dataloader(cfg, "test", test_set_mode=True, epochs_per_iter=1, **kw)
    assert eval_meta.total_groups == 13
    assert sum(eff for _, _, eff in eval_loader) == 13

    def epoch_inputs(ds_iterable):
        return [b["inputs"].numpy().copy() for _, b, eff in ds_iterable]

    ref = PuzzleDataset(PuzzleDatasetConfig(seed=cfg.seed, dataset_paths=[str(train_root)], global_batch_size=8,
                                            test_set_mode=False, epochs_per_iter=2, rank=0, num_replicas=1), split="train")
    ref_epochs = [epoch_inputs(ref) for _ in range(4)]
    assert all(len(e) == 3 for e in ref_epochs)  # 12 groups x 2 epochs / 8, no partial batch
    assert not all(np.array_equal(a, b) for a, b in zip(ref_epochs[0], ref_epochs[1]))  # reshuffled each iter

    train_loader, train_meta = pretrain.create_dataloader(cfg, "train", test_set_mode=False, epochs_per_iter=2, **kw)
    assert train_meta.total_groups == 12
    train_loader.dataset._iters = 2  # what launch() does with start_iter = 2
    for k in (2, 3):
        got = epoch_inputs(train_loader)
        assert len(got) == len(ref_epochs[k])
        for a, b in zip(got, ref_epochs[k]):
            np.testing.assert_array_equal(a, b)


# =============================================================================================
# (2) Real subsets in data/ (read-only)
# =============================================================================================

DATA = REPO / "data"
FULL = DATA / "sudoku-extreme-1k-aug-1000-s0"
REAL_SUBSETS = {1000: DATA / "sudoku-extreme-test1k", 10000: DATA / "sudoku-extreme-test10k", 25000: DATA / "sudoku-extreme-test25k"}
needs_real_data = pytest.mark.skipif(not (FULL / "test").is_dir() or not all(p.is_dir() for p in REAL_SUBSETS.values()),
                                     reason="real Sudoku data not built")


@needs_real_data
def test_real_subsets_nested_and_match_full_split():
    with open(FULL / "test" / "dataset.json") as f:
        full_meta = json.load(f)
    total = full_meta["total_puzzles"]
    full_inputs = np.load(FULL / "test" / "all__inputs.npy", mmap_mode="r")
    full_labels = np.load(FULL / "test" / "all__labels.npy", mmap_mode="r")
    full_pid = np.load(FULL / "test" / "all__puzzle_identifiers.npy", mmap_mode="r")
    assert full_inputs.shape == (total, 81)

    idx = {}
    for n, d in REAL_SUBSETS.items():
        sel = np.load(d / "subset_indices.npy")
        idx[n] = sel
        assert sel.shape == (n,)
        assert np.all(np.diff(sel) > 0) and sel[0] >= 0 and sel[-1] < total

        info = json.loads((d / "subset.json").read_text())
        assert info["n"] == n
        assert os.path.basename(info["source"]) == FULL.name
        np.testing.assert_array_equal(sel, np.sort(np.random.default_rng(info["seed"]).permutation(total)[:n]))

        sub_inputs = np.load(d / "test" / "all__inputs.npy")
        np.testing.assert_array_equal(sub_inputs, full_inputs[sel])
        np.testing.assert_array_equal(np.load(d / "test" / "all__labels.npy"), full_labels[sel])
        np.testing.assert_array_equal(np.load(d / "test" / "all__puzzle_identifiers.npy"), full_pid[sel])
        for field in ("puzzle_indices", "group_indices"):
            np.testing.assert_array_equal(np.load(d / "test" / f"all__{field}.npy"), np.arange(n + 1))
        assert info["sha256_inputs"] == hashlib.sha256(np.ascontiguousarray(sub_inputs).tobytes()).hexdigest()

        with open(d / "test" / "dataset.json") as f:
            assert json.load(f) == dict(full_meta, total_groups=n, total_puzzles=n)

    assert np.isin(idx[1000], idx[10000]).all()
    assert np.isin(idx[10000], idx[25000]).all()


@needs_real_data
def test_real_test1k_through_puzzle_dataset():
    ds, batches = _iterate_test(REAL_SUBSETS[1000], 384)
    assert [eff for _, _, eff in batches] == [384, 384, 232]
    ref = np.load(REAL_SUBSETS[1000] / "test" / "all__labels.npy")
    got = np.concatenate([b["labels"][:eff].numpy() for _, b, eff in batches])
    np.testing.assert_array_equal(got, ref.astype(np.int32))  # Sudoku labels never hit ignore_label_id 0
    assert (batches[-1][1]["labels"][232:] == IGNORE_LABEL_ID).all()


# =============================================================================================
# (3) Config recipe (hydra compose of cfg_sudoku_bench)
# =============================================================================================

def _compose(config_name: str, overrides=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def _resolved(cfg):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=True)


@pytest.fixture(scope="module")
def bench_cfg():
    return _compose("cfg_sudoku_bench")


@pytest.fixture(scope="module")
def bench(bench_cfg):
    return _resolved(bench_cfg)


README_RECIPE = dict(lr=1e-4, puzzle_emb_lr=1e-4, weight_decay=1.0, puzzle_emb_weight_decay=1.0, ema=True, ema_rate=0.999,
                     lr_min_ratio=1.0, lr_warmup_steps=2000, beta1=0.9, beta2=0.95, evaluators=[])
README_ARCH = dict(mlp_t=True, pos_encodings="none", L_layers=2, H_cycles=3, L_cycles=6, hidden_size=512, expansion=4,
                   halt_max_steps=16, halt_exploration_prob=0.1, puzzle_emb_ndim=512, puzzle_emb_len=16,
                   forward_dtype="bfloat16", flow=False, reg_mode="none")


def test_bench_config_recipe_values(bench):
    assert "hydra" not in bench
    for k, v in README_RECIPE.items():
        assert bench[k] == v, k
        assert type(bench[k]) is type(v), f"{k}: {type(bench[k]).__name__}"  # e.g. YAML '1e-4' parsed as float
    for k, v in README_ARCH.items():
        assert bench["arch"][k] == v, f"arch.{k}"
        assert type(bench["arch"][k]) is type(v), f"arch.{k}: {type(bench['arch'][k]).__name__}"
    assert bench["arch"]["name"] == "recursive_reasoning.trm_contractive@TinyRecursiveReasoningModel_ACTV1"
    assert bench["arch"]["loss"] == dict(name="losses@ACTLossHead", loss_type="stablemax_cross_entropy")
    assert bench["epochs"] % bench["eval_interval"] == 0  # launch() asserts this


def test_bench_puzzle_emb_ndim_follows_hidden_size():
    assert _resolved(_compose("cfg_sudoku_bench", ["arch.hidden_size=64"]))["arch"]["puzzle_emb_ndim"] == 64


def test_bench_matches_readme_command(bench):
    """cfg_sudoku_bench == README 'pretrain_mlp_t_sudoku' command on cfg_pretrain + arch=trm, up to declared deltas."""
    readme = _resolved(_compose("cfg_pretrain", [
        "arch=trm", "data_paths=[data/sudoku-extreme-1k-aug-1000]", "evaluators=[]", "epochs=50000", "eval_interval=5000",
        "lr=1e-4", "puzzle_emb_lr=1e-4", "weight_decay=1.0", "puzzle_emb_weight_decay=1.0", "arch.mlp_t=True",
        "arch.pos_encodings=none", "arch.L_layers=2", "arch.H_cycles=3", "arch.L_cycles=6", "+run_name=x", "ema=True"]))
    # Benchmark-specific by design: data location/subset, schedule length, batch size, naming
    deltas = {"arch", "data_paths", "data_paths_test", "epochs", "eval_interval", "global_batch_size", "run_name"}
    for k, v in readme.items():
        if k not in deltas:
            assert bench[k] == v, k
    ra, ba = readme["arch"], bench["arch"]
    for k, v in ra.items():
        if k != "name":
            assert ba[k] == v, f"arch.{k}"
    extra = set(ba) - set(ra)
    assert all(k.startswith(("flow", "reg_")) for k in extra), extra


def test_pretrain_config_validates(bench_cfg, bench):
    import pretrain
    cfg = pretrain.PretrainConfig(**bench)
    assert cfg.arch.name.endswith("@TinyRecursiveReasoningModel_ACTV1")
    assert cfg.lr == 1e-4 and cfg.weight_decay == 1.0 and cfg.ema and cfg.evaluators == []
    assert cfg.arch.puzzle_emb_ndim == 512  # type: ignore[attr-defined]
    assert pretrain.create_evaluators(cfg, eval_metadata=None) == []
    # Exactly as launch() does: straight from the DictConfig, naming filled in, nothing written
    synced = pretrain.load_synced_config(bench_cfg, rank=0, world_size=1)
    assert synced.project_name == "sudoku-contractive" and synced.run_name
    assert synced.checkpoint_path == os.path.join("checkpoints", "sudoku-contractive", synced.run_name)
    assert synced.model_dump(exclude={"run_name", "checkpoint_path"}) == cfg.model_dump(exclude={"run_name", "checkpoint_path"})


@needs_real_data
def test_pretrain_config_dataset_dirs_exist(bench):
    import pretrain
    cfg = pretrain.PretrainConfig(**bench)
    for p in cfg.data_paths:
        assert (REPO / p / "train" / "dataset.json").is_file(), p
    for p in cfg.data_paths_test:
        assert (REPO / p / "test" / "dataset.json").is_file(), p


def test_arch_extras_parse_into_reg_and_model_configs(bench):
    import pretrain
    from models.regularizers import RegConfig
    from utils.functions import load_model_class
    from models import losses

    cfg = pretrain.PretrainConfig(**bench)
    extra = cfg.arch.__pydantic_extra__
    reg = RegConfig.from_arch_extra(extra)
    assert reg.reg_eta == 0.05 and reg.reg_mode == "none"
    yaml_reg = {k: v for k, v in extra.items() if k.startswith("reg_")}
    assert set(yaml_reg) <= set(RegConfig.model_fields), set(yaml_reg) - set(RegConfig.model_fields)  # typo guard
    for k, v in yaml_reg.items():
        assert getattr(reg, k) == v, k

    # The model config ignores unknown keys silently: make sure every non-reg arch key is a real field
    model_cls = load_model_class(cfg.arch.name)
    cfg_cls = sys.modules[model_cls.__module__].TinyRecursiveReasoningModel_ACTV1Config
    unknown = {k for k in extra if not k.startswith("reg_")} - set(cfg_cls.model_fields)
    assert not unknown, unknown
    mcfg = cfg_cls(**extra, batch_size=cfg.global_batch_size, vocab_size=11, seq_len=81, num_puzzle_identifiers=1, causal=False)
    assert mcfg.flow is False and mcfg.mlp_t is True and mcfg.puzzle_emb_ndim == 512
    assert load_model_class(cfg.arch.loss.name) is losses.ACTLossHead
    assert callable(getattr(losses, cfg.arch.loss.__pydantic_extra__["loss_type"]))


ARMS = {  # scripts/run_benchmark.sh arm_overrides (ETA=0.05)
    "A0": [],
    "A1": ["arch.reg_mode=cycle_lip", "arch.reg_eta=0.05"],
    "A2": ["arch.flow=True"],
    "A3": ["arch.flow=True", "arch.reg_mode=flow_mono", "arch.reg_eta=0.05"],
    "A4": ["arch.flow=True", "arch.reg_mode=cycle_lip", "arch.reg_eta=0.05"],
}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_benchmark_arm_overrides(arm):
    """The hydra overrides used by scripts/run_benchmark.sh and scripts/smoke.sh compose and validate."""
    import pretrain
    from models.regularizers import RegConfig
    B, epi, iters = 256, 1280, 7
    run = _compose("cfg_sudoku_bench", ["project_name=sudoku-contractive", f"run_name={arm}_s0", f"checkpoint_path=/nonexistent/{arm}",
                                        "seed=0", f"global_batch_size={B}", f"epochs={epi * iters}", f"eval_interval={epi}", *ARMS[arm]])
    cfg = pretrain.PretrainConfig(**_resolved(run))
    reg = RegConfig.from_arch_extra(cfg.arch.__pydantic_extra__)
    assert reg.reg_mode == {"A1": "cycle_lip", "A3": "flow_mono", "A4": "cycle_lip"}.get(arm, "none")
    assert cfg.arch.flow == (arm in ("A2", "A3", "A4"))  # type: ignore[attr-defined]
    assert cfg.epochs % cfg.eval_interval == 0

    # run_benchmark.sh target_step = epochs * 1000 / B must equal init_train_state's total_steps estimate
    if (FULL / "train" / "dataset.json").is_file():
        meta = json.loads((FULL / "train" / "dataset.json").read_text())
        total_steps = int(cfg.epochs * meta["total_groups"] * meta["mean_puzzle_examples"] / cfg.global_batch_size)
        assert total_steps == cfg.epochs * 1000 // B == 35000

    smoke = _compose("cfg_sudoku_bench", [f"run_name=t_{arm}", f"checkpoint_path=/nonexistent/smoke_{arm}", "global_batch_size=64",
                                          "epochs=1000", "eval_interval=1000", "max_train_steps=20", "log_every=20",
                                          "data_paths_test=[data/sudoku-extreme-test1k]", "resume=False", *ARMS[arm]])
    scfg = pretrain.PretrainConfig(**_resolved(smoke))
    assert scfg.max_train_steps == 20 and scfg.resume is False and scfg.data_paths_test == ["data/sudoku-extreme-test1k"]


def test_invalid_reg_mode_rejected():
    import pydantic
    import pretrain
    from models.regularizers import RegConfig
    cfg = pretrain.PretrainConfig(**_resolved(_compose("cfg_sudoku_bench", ["arch.reg_mode=bogus"])))
    with pytest.raises(pydantic.ValidationError):
        RegConfig.from_arch_extra(cfg.arch.__pydantic_extra__)


# =============================================================================================
# (4) pretrain helpers
# =============================================================================================

def test_to_py_nested_numpy_and_torch_scalars():
    from pretrain import _to_py
    d = {"all": {"accuracy": np.float32(0.5), "count": np.int64(3), "ok": np.bool_(True), "t": torch.tensor(2.5),
                 "arr0": np.array(1.25), "deep": {"x": np.float64(1e-3)}},
         "name": "A0", "none": None, "f": 1.0, "i": 2}
    out = _to_py(d)
    assert out == {"all": {"accuracy": 0.5, "count": 3, "ok": True, "t": 2.5, "arr0": 1.25, "deep": {"x": 1e-3}},
                   "name": "A0", "none": None, "f": 1.0, "i": 2}
    assert type(out["all"]["accuracy"]) is float and type(out["all"]["count"]) is int and type(out["all"]["ok"]) is bool
    assert type(out["all"]["t"]) is float and type(out["all"]["deep"]["x"]) is float
    json.dumps(out)  # serialisable
    assert d["all"]["accuracy"].dtype == np.float32  # input not mutated


def test_flatten_nested_dicts():
    from pretrain import _flatten, _to_py
    d = {"all": {"accuracy": 0.5, "exact_accuracy": 0.25}, "x": 1, "a": {"b": {"c": 2}, "d": 3}, "empty": {}}
    assert _flatten(d) == {"all/accuracy": 0.5, "all/exact_accuracy": 0.25, "x": 1, "a/b/c": 2, "a/d": 3}
    assert _flatten({"k": 1}, prefix="eval/") == {"eval/k": 1}
    # evaluate() returns {set_name: {metric: np.float32}}; the jsonl eval record uses _flatten(_to_py(.))
    ev = {"all": {"accuracy": np.float32(0.75), "exact_accuracy": np.float32(0.5), "steps": np.float32(16.0)}}
    flat = _flatten(_to_py(ev))
    assert flat == {"all/accuracy": 0.75, "all/exact_accuracy": 0.5, "all/steps": 16.0}
    assert all(type(v) is float for v in flat.values())


def test_jsonl_logger(tmp_path):
    from pretrain import JsonlLogger
    path = tmp_path / "metrics.jsonl"
    log = JsonlLogger(str(path))
    log.log({"split": "train", "step": 50, "train/lm_loss": np.float32(1.5), "nested": {"v": torch.tensor(3)}})
    log.log({"split": "eval", "step": 100, "all/exact_accuracy": np.float64(0.875)})
    log.file.close()
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert recs == [{"split": "train", "step": 50, "train/lm_loss": 1.5, "nested": {"v": 3}},
                    {"split": "eval", "step": 100, "all/exact_accuracy": 0.875}]


def test_lr_schedule_constant_after_warmup(bench):
    import pretrain
    cfg = pretrain.PretrainConfig(**bench)
    ts = SimpleNamespace(step=0, total_steps=35000)
    for step, expected in [(0, 0.0), (1, 1e-4 / 2000), (1000, 0.5e-4), (2000, 1e-4), (2001, 1e-4), (20000, 1e-4), (35000, 1e-4)]:
        ts.step = step
        assert pretrain.compute_lr(cfg.lr, cfg, ts) == pytest.approx(expected, rel=1e-12, abs=1e-15), step
    # min_ratio < 1 really is cosine (guards against the schedule being silently constant)
    assert pretrain.cosine_schedule_with_warmup_lr_lambda(10, base_lr=1.0, num_warmup_steps=0, num_training_steps=10,
                                                          min_ratio=0.0) == pytest.approx(0.0, abs=1e-12)


def test_train_batch_source_normalises_metrics():
    import pretrain
    src = inspect.getsource(pretrain.train_batch)
    assert 'v / (global_batch_size if k.endswith("loss") else count)' in src
    assert 'count = max(reduced_metrics["count"], 1)' in src
    assert "(1 / global_batch_size) * loss" in src


# --- direct invocation of train_batch on CPU: Tensor.cuda is patched to the identity, carry pre-initialised ---

class _RecordingOpt(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, dict(lr=0.0))
        self.calls = []

    def step(self, closure=None):
        self.calls.append((self.param_groups[0]["lr"], [p.grad.clone() for g in self.param_groups for p in g["params"]]))


class _StubModel(nn.Module):
    """Mimics ACTLossHead(model): returns (carry, loss, metrics, outputs, all_finish)."""

    def __init__(self, metrics, outputs, events):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3))
        self.metrics, self.outputs, self.events = metrics, outputs, events
        self.new_carry = SimpleNamespace(steps=torch.tensor([1, 2, 3, 4], dtype=torch.int32))

    def forward(self, carry, batch, return_keys):
        self.events.append(("forward", sorted(batch)))
        return self.new_carry, 2.0 * self.w.sum(), dict(self.metrics), dict(self.outputs), torch.tensor(False)


class _FakeRegularizer:
    """Records the call order; optionally returns a reg loss built on the model parameter."""

    def __init__(self, events, model=None, coef=None):
        self.events, self.model, self.coef = events, model, coef
        self.stashed = []

    def __call__(self, step):
        self.events.append(("reg", step))
        reg_loss = None if self.coef is None else self.coef * self.model.w.sum()
        return reg_loss, {"reg/lambda": 2.0, "reg/value": 0.125}

    def stash(self, outs, steps, *args):
        self.events.append(("stash",))
        self.stashed.append((outs, steps, args))


def _metrics(count):
    return {
        "count": torch.tensor(count),                        # int64, like valid_metrics.sum()
        "accuracy": torch.tensor(3.0),
        "exact_accuracy": torch.tensor(2),
        "q_halt_accuracy": torch.tensor(4),
        "steps": torch.tensor(12),
        "fpres": torch.tensor(1.0),
        "lm_loss": torch.tensor(8.0),
        "q_halt_loss": torch.tensor(2.0),
        "q_continue_loss": torch.tensor(4.0),
    }


ANCHORS = ("reg_s_H", "reg_s_L", "reg_phi_H", "reg_phi_L", "reg_x")


@pytest.fixture
def cpu_train_batch(monkeypatch, bench):
    import pretrain
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    cfg = pretrain.PretrainConfig(**bench)

    def run(count=4, outputs=None, make_regularizer=None, step=0, total_steps=100, gbs=8):
        events = []
        model = _StubModel(_metrics(count), outputs or {}, events)
        regularizer = None if make_regularizer is None else make_regularizer(events, model)
        opt = _RecordingOpt(model.parameters())
        ts = pretrain.TrainState(model=model, optimizers=[opt], optimizer_lrs=[cfg.lr],
                                 carry=SimpleNamespace(steps=None), step=step, total_steps=total_steps)
        batch = {k: torch.zeros(gbs, 81, dtype=torch.int32) for k in ("inputs", "labels")}
        batch["puzzle_identifiers"] = torch.zeros(gbs, dtype=torch.int32)
        out = pretrain.train_batch(cfg, ts, batch, gbs, rank=0, world_size=1, regularizer=regularizer)
        return SimpleNamespace(out=out, ts=ts, model=model, opt=opt, events=events, reg=regularizer)

    return run


def test_train_batch_metric_normalisation(cpu_train_batch):
    gbs = 8
    r = cpu_train_batch(count=4, gbs=gbs)
    assert r.ts.step == 1
    expected = {
        "train/count": 1.0, "train/accuracy": 0.75, "train/exact_accuracy": 0.5, "train/q_halt_accuracy": 1.0,
        "train/steps": 3.0, "train/fpres": 0.25,                                         # / count
        "train/lm_loss": 1.0, "train/q_halt_loss": 0.25, "train/q_continue_loss": 0.5,   # / global_batch_size
        "train/lr": 1e-4 / 2000,
    }
    assert set(r.out) == set(expected)
    for k, v in expected.items():
        assert float(r.out[k]) == pytest.approx(v, rel=1e-6), k
    # Backward on loss / global_batch_size, optimizer stepped once at the scheduled lr, grads cleared
    (lr, grads), = r.opt.calls
    assert lr == pytest.approx(1e-4 / 2000)
    torch.testing.assert_close(grads[0], torch.full((3,), 2.0 / gbs))
    assert r.model.w.grad is None or (r.model.w.grad == 0).all()
    assert r.ts.carry is r.model.new_carry


def test_train_batch_zero_count_avoids_nan(cpu_train_batch):
    out = cpu_train_batch(count=0).out
    assert all(math.isfinite(float(v)) for v in out.values())
    assert float(out["train/accuracy"]) == pytest.approx(3.0) and float(out["train/lm_loss"]) == pytest.approx(1.0)


def test_train_batch_past_total_steps_is_noop(cpu_train_batch):
    r = cpu_train_batch(step=100, total_steps=100, make_regularizer=lambda ev, m: _FakeRegularizer(ev, m))
    assert r.out is None and r.ts.step == 101 and r.opt.calls == [] and r.events == []


def test_train_batch_regularizer_without_anchors(cpu_train_batch):
    r = cpu_train_batch(make_regularizer=lambda ev, m: _FakeRegularizer(ev, m))
    assert [e[0] for e in r.events] == ["reg", "forward"] and r.reg.stashed == []
    assert r.out["reg/lambda"] == 2.0


def test_train_batch_with_regularizer(cpu_train_batch):
    gbs = 8
    anchors = {k: torch.zeros(gbs, 2) for k in ANCHORS}
    r = cpu_train_batch(outputs=anchors, gbs=gbs, make_regularizer=lambda ev, m: _FakeRegularizer(ev, m, coef=3.0))
    # Previous step's stash is regularized (and back-propagated) before this step's forward, then the new anchors stashed
    assert r.events == [("reg", 1), ("forward", ["inputs", "labels", "puzzle_identifiers"]), ("stash",)]
    (outs, steps, _), = r.reg.stashed
    assert set(anchors) <= set(outs) and steps is r.model.new_carry.steps
    # Regularizer metrics are merged unnormalised next to the normalised train metrics
    assert r.out["reg/lambda"] == 2.0 and r.out["reg/value"] == 0.125
    assert float(r.out["train/lm_loss"]) == pytest.approx(1.0)
    # Reg loss gradient accumulates into the same optimizer step, unscaled by global_batch_size
    (lr, grads), = r.opt.calls
    torch.testing.assert_close(grads[0], torch.full((3,), 3.0 + 2.0 / gbs))


# =============================================================================================
# (5) Training-loop smoke
# =============================================================================================

@pytest.mark.skip(reason="needs a GPU: pretrain.launch/create_model hard-code CUDA (torch.device('cuda'), .cuda()); the suite is CPU-only")
def test_training_loop_smoke():
    pass
