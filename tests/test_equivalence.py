"""Bitwise equivalence of models/recursive_reasoning/trm_contractive.py with upstream trm.py (eager, CPU).

trm_contractive is a fork of trm.py whose defaults (flow=False) must reproduce TRM exactly: same parameter
init under the same seed, same ACT rollout (train with exploration, eval), same gradients, same ACTLossHead
loss/metrics. With flow=True, h_L = h_H = 1 and a zero-initialized time embedding it must also be bitwise TRM
(loaded from an upstream state dict). The detached regularizer anchors (reg_s_*, reg_phi_*, reg_x) and the
"fpres" metric of models/losses.py are checked against independent recomputations.
"""
import contextlib
from typing import Dict, List

import pytest
import torch

from models.layers import CastedLinear
from models.losses import ACTLossHead, IGNORE_LABEL_ID
from models.recursive_reasoning import trm as trm_up
from models.recursive_reasoning import trm_contractive as trm_co

N_ACT_STEPS = 5
TEMB_KEYS = {"inner.temb_proj.weight", "inner.temb_proj.bias"}

BASE = dict(
    batch_size=4, seq_len=81, vocab_size=11, num_puzzle_identifiers=8,
    H_cycles=2, L_cycles=2, H_layers=0, L_layers=2,
    hidden_size=32, expansion=4, num_heads=2,
    halt_max_steps=3, halt_exploration_prob=0.5,
    no_ACT_continue=True,
)

CONFIGS = {
    # Sudoku recipe (MLP-T, no positional encodings, 16 puzzle-embedding positions)
    "mlpt_f32": dict(mlp_t=True, pos_encodings="none", puzzle_emb_ndim=32, puzzle_emb_len=16, forward_dtype="float32"),
    # Attention + RoPE, puzzle_emb_len = ceil(48 / 32) = 2 with zero padding, 3 H-cycles
    "rope_f32": dict(mlp_t=False, pos_encodings="rope", puzzle_emb_ndim=48, puzzle_emb_len=0, H_cycles=3,
                     forward_dtype="float32"),
    # Attention + learned positions, no puzzle embedding
    "learned_nopuzzle_f32": dict(mlp_t=False, pos_encodings="learned", puzzle_emb_ndim=0, puzzle_emb_len=0,
                                 batch_size=6, forward_dtype="float32"),
    # bf16 forward
    "mlpt_bf16": dict(mlp_t=True, pos_encodings="none", puzzle_emb_ndim=32, puzzle_emb_len=16, forward_dtype="bfloat16"),
}

FLOW_TRM = dict(flow=True, flow_h_L=1.0, flow_h_H=1.0, flow_time_embed=True)


def cfg_of(name: str, **overrides) -> dict:
    return {**BASE, **CONFIGS[name], **overrides}


# ---------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------

def build(module, cfg: dict, seed: int = 0):
    torch.manual_seed(seed)
    return module.TinyRecursiveReasoningModel_ACTV1(dict(cfg))


def perturbed_state_dict(model, seed: int = 1) -> Dict[str, torch.Tensor]:
    """Upstream init has zero puzzle embeddings and q_head = (0, -5) (never halts by itself). Randomize those so
    puzzle embeddings matter and q-halting fires on some rows."""
    gen = torch.Generator().manual_seed(seed)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    for k, v in sd.items():
        if k.endswith("puzzle_emb.weights"):
            sd[k] = (0.5 * torch.randn(v.shape, generator=gen)).to(v.dtype)
        elif k.endswith("q_head.weight"):
            sd[k] = (0.15 * torch.randn(v.shape, generator=gen)).to(v.dtype)
        elif k.endswith("q_head.bias"):
            sd[k] = (0.2 * torch.randn(v.shape, generator=gen)).to(v.dtype)
            if model.config.puzzle_emb_ndim == 0:
                # Without puzzle embeddings q reads token 0, whose state is nearly row-independent here; center
                # q_halt around 0 so that some rows halt early (by q) and others do not.
                sd[k][0] += 0.25
    return sd


def make_batches(cfg: dict, n: int, seed: int = 0) -> List[Dict[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    B, S, V = cfg["batch_size"], cfg["seq_len"], cfg["vocab_size"]
    batches = []
    for i in range(n):
        inputs = torch.randint(1, V, (B, S), generator=gen).to(torch.int32)
        labels = torch.randint(1, V, (B, S), generator=gen).to(torch.int32)
        labels[torch.rand(B, S, generator=gen) < 0.3] = IGNORE_LABEL_ID
        labels[i % B] = IGNORE_LABEL_ID  # one fully ignored row (loss_counts == 0)
        labels[(i + 1) % B] = inputs[(i + 1) % B]  # one row that is trivially predictable
        puzzle_identifiers = torch.randint(0, cfg["num_puzzle_identifiers"], (B,), generator=gen).to(torch.int32)
        batches.append(dict(inputs=inputs, labels=labels, puzzle_identifiers=puzzle_identifiers))
    return batches


def assert_equal(a: torch.Tensor, b: torch.Tensor, what: str):
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} != {b.dtype}"
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} != {tuple(b.shape)}"
    if not torch.equal(a, b):
        diff = (a.double() - b.double()).abs()
        raise AssertionError(f"{what}: not bitwise equal (max |diff| {diff.max().item():.3e}, "
                             f"{(diff > 0).sum().item()} / {diff.numel()} elements differ)")


def assert_state_dicts_equal(sd_a, sd_b, what=""):
    assert list(sd_a.keys()) == list(sd_b.keys()), f"{what}: keys differ {set(sd_a) ^ set(sd_b)}"
    for k in sd_a:
        assert_equal(sd_a[k], sd_b[k], f"{what} state_dict[{k}]")


def _zero_grads(model):
    for p in model.parameters():
        p.grad = None
    inner = model.inner
    if hasattr(inner, "puzzle_emb"):
        inner.puzzle_emb.local_weights.grad = None


def _probe_scalar(outputs) -> torch.Tensor:
    return (outputs["logits"].float().square().mean()
            + outputs["q_halt_logits"].sum() + 0.5 * outputs["q_continue_logits"].sum())


def rollout(model, batches, train: bool, backward: bool, reseed: bool = True):
    """Run len(batches) ACT steps with torch.manual_seed(100 + i) before each call (ACT exploration RNG).

    reseed=False keeps drawing from the global RNG stream as it was left by model construction."""
    model.train(train)
    carry = None
    records = []
    for i, batch in enumerate(batches):
        _zero_grads(model)
        ctx = contextlib.nullcontext() if train else torch.inference_mode()
        with ctx:
            if carry is None:
                carry = model.initial_carry(batch)
            if reseed:
                torch.manual_seed(100 + i)
            carry, outputs = model(carry=carry, batch=batch)
        rec = dict(
            outputs={k: v.detach().clone() for k, v in outputs.items()},
            z_H=carry.inner_carry.z_H.clone(), z_L=carry.inner_carry.z_L.clone(),
            halted=carry.halted.clone(), steps=carry.steps.clone(),
            current_data={k: v.clone() for k, v in carry.current_data.items()},
        )
        if backward:
            _probe_scalar(outputs).backward()
            grads = {n: (None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}
            if hasattr(model.inner, "puzzle_emb"):
                lw = model.inner.puzzle_emb.local_weights.grad
                grads["inner.puzzle_emb.local_weights"] = None if lw is None else lw.clone()
            rec["grads"] = grads
        records.append(rec)
    return records


def assert_rollouts_equal(rec_up, rec_co, what: str, grad_names=None):
    assert len(rec_up) == len(rec_co)
    for i, (a, b) in enumerate(zip(rec_up, rec_co)):
        w = f"{what} step {i}"
        for k in ("logits", "q_halt_logits", "q_continue_logits"):
            assert_equal(a["outputs"][k], b["outputs"][k], f"{w} {k}")
        for k in ("z_H", "z_L", "halted", "steps"):
            assert_equal(a[k], b[k], f"{w} {k}")
        assert a["current_data"].keys() == b["current_data"].keys()
        for k in a["current_data"]:
            assert_equal(a["current_data"][k], b["current_data"][k], f"{w} current_data[{k}]")
        if "grads" in a:
            names = grad_names if grad_names is not None else a["grads"].keys()
            for n in names:
                ga, gb = a["grads"][n], b["grads"][n]
                assert (ga is None) == (gb is None), f"{w} grad[{n}] presence differs"
                if ga is not None:
                    assert_equal(ga, gb, f"{w} grad[{n}]")


def _check_rollouts_are_nontrivial(records, cfg, train: bool):
    """Guard against vacuous equality: the rollout must exercise halting/reset and produce gradients."""
    halted = torch.stack([r["halted"] for r in records])
    assert halted.any() and not halted.all(), "rollout never halts or always halts: ACT path not exercised"
    if train:
        # q-halting with a random q_head plus exploration should produce at least one early halt
        steps = torch.stack([r["steps"] for r in records])
        assert ((steps < cfg["halt_max_steps"]) & halted).any(), "no early (q / exploration) halt in train mode"
        g = records[0]["grads"]
        assert g["inner.lm_head.weight"] is not None and g["inner.lm_head.weight"].abs().max() > 0
        assert g["inner.q_head.weight"] is not None and g["inner.q_head.weight"].abs().max() > 0
        if cfg["puzzle_emb_ndim"] > 0:
            assert g["inner.puzzle_emb.local_weights"].abs().max() > 0


# ---------------------------------------------------------------------------------------------------------
# Upstream vs trm_contractive (defaults)
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_name", list(CONFIGS))
def test_init_state_dict_identical(cfg_name):
    cfg = cfg_of(cfg_name)
    up, co = build(trm_up, cfg), build(trm_co, cfg)
    assert_state_dicts_equal(up.state_dict(), co.state_dict(), "init")
    assert [n for n, _ in up.named_parameters()] == [n for n, _ in co.named_parameters()]
    assert [n for n, _ in up.named_buffers()] == [n for n, _ in co.named_buffers()]
    for (n, b_up), (_, b_co) in zip(up.named_buffers(), co.named_buffers()):
        assert_equal(b_up, b_co, f"buffer {n}")  # includes non-persistent rotary caches / local weights
    assert co.inner.h_L == 1.0 and co.inner.h_H == 1.0 and co.inner.time_embeddings() is None


@pytest.mark.parametrize("cfg_name", list(CONFIGS))
def test_act_rollout_and_grads_bitwise_equal(cfg_name):
    cfg = cfg_of(cfg_name)
    up, co = build(trm_up, cfg), build(trm_co, cfg)
    sd = perturbed_state_dict(up)
    up.load_state_dict(sd)
    co.load_state_dict(sd)
    batches = make_batches(cfg, N_ACT_STEPS)

    rec_up = rollout(up, batches, train=True, backward=True)
    rec_co = rollout(co, batches, train=True, backward=True)
    _check_rollouts_are_nontrivial(rec_up, cfg, train=True)
    assert rec_up[0]["grads"].keys() == rec_co[0]["grads"].keys()
    assert_rollouts_equal(rec_up, rec_co, f"{cfg_name} train")

    rec_up = rollout(up, batches, train=False, backward=False)
    rec_co = rollout(co, batches, train=False, backward=False)
    _check_rollouts_are_nontrivial(rec_up, cfg, train=False)
    assert_rollouts_equal(rec_up, rec_co, f"{cfg_name} eval")
    # Eval always runs halt_max_steps, uniformly over the batch
    for i, r in enumerate(rec_co):
        last = i % cfg["halt_max_steps"] == cfg["halt_max_steps"] - 1
        assert bool(r["halted"].all()) == last and bool(r["halted"].any()) == last, i
        assert_equal(r["steps"], torch.full_like(r["steps"], i % cfg["halt_max_steps"] + 1), f"eval steps {i}")


@pytest.mark.parametrize("cfg_name", ["mlpt_f32", "rope_f32"])
def test_act_rollout_outputs_anchor_keys(cfg_name):
    """The ACT wrapper of trm_contractive passes the detached anchors through; reg_phi_* is the new carry."""
    cfg = cfg_of(cfg_name)
    co = build(trm_co, cfg)
    co.load_state_dict(perturbed_state_dict(co))
    for train in (True, False):
        for r in rollout(co, make_batches(cfg, 3), train=train, backward=False):
            out = r["outputs"]
            assert {"reg_s_H", "reg_s_L", "reg_phi_H", "reg_phi_L", "reg_x"} <= out.keys()
            assert_equal(out["reg_phi_H"], r["z_H"], "reg_phi_H vs carry")
            assert_equal(out["reg_phi_L"], r["z_L"], "reg_phi_L vs carry")


# ---------------------------------------------------------------------------------------------------------
# Euler-TRM flags
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("time_embed", [True, False], ids=["temb", "no_temb"])
@pytest.mark.parametrize("cfg_name", ["mlpt_f32", "rope_f32", "mlpt_bf16"])
def test_flow_h1_zero_temb_is_bitwise_trm(cfg_name, time_embed):
    cfg = cfg_of(cfg_name)
    up = build(trm_up, cfg)
    co = build(trm_co, {**cfg, **FLOW_TRM, "flow_time_embed": time_embed})

    # Extensions are created after every upstream parameter: the upstream RNG stream / init is unchanged
    co_sd = co.state_dict()
    assert set(co_sd) - set(up.state_dict()) == (TEMB_KEYS if time_embed else set())
    assert_state_dicts_equal(up.state_dict(), {k: v for k, v in co_sd.items() if k not in TEMB_KEYS}, "init")

    sd = perturbed_state_dict(up)
    up.load_state_dict(sd)
    result = co.load_state_dict(sd, strict=False)
    assert set(result.missing_keys) == (TEMB_KEYS if time_embed else set())
    assert result.unexpected_keys == []

    if time_embed:
        temb = co.inner.time_embeddings()
        n_sub = cfg["H_cycles"] * (cfg["L_cycles"] + 1)
        assert temb.shape == (n_sub, cfg["hidden_size"]) and temb.dtype == getattr(torch, cfg["forward_dtype"])
        assert torch.count_nonzero(co.inner.temb_proj.weight) == 0 and torch.count_nonzero(temb) == 0

    batches = make_batches(cfg, N_ACT_STEPS)
    up_names = [n for n, _ in up.named_parameters()] + (
        ["inner.puzzle_emb.local_weights"] if cfg["puzzle_emb_ndim"] > 0 else [])
    rec_up = rollout(up, batches, train=True, backward=True)
    rec_co = rollout(co, batches, train=True, backward=True)
    _check_rollouts_are_nontrivial(rec_up, cfg, train=True)
    assert_rollouts_equal(rec_up, rec_co, f"{cfg_name} flow train", grad_names=up_names)
    if time_embed:
        # temb_proj is in the graph (it gets a gradient), it just contributes exactly zero at init
        assert rec_co[0]["grads"]["inner.temb_proj.weight"] is not None
        assert rec_co[0]["grads"]["inner.temb_proj.bias"].abs().max() > 0

    rec_up = rollout(up, batches, train=False, backward=False)
    rec_co = rollout(co, batches, train=False, backward=False)
    assert_rollouts_equal(rec_up, rec_co, f"{cfg_name} flow eval")


@pytest.mark.parametrize("variant", ["h_half", "h1_nonzero_temb"])
def test_flow_variants_differ_from_trm(variant):
    cfg = cfg_of("mlpt_f32")
    up = build(trm_up, cfg)
    if variant == "h_half":
        co = build(trm_co, {**cfg, "flow": True, "flow_h_L": 0.5, "flow_h_H": 0.5, "flow_time_embed": True})
    else:
        co = build(trm_co, {**cfg, **FLOW_TRM})
    sd = perturbed_state_dict(up)
    up.load_state_dict(sd)
    co.load_state_dict(sd, strict=False)
    if variant == "h1_nonzero_temb":
        with torch.no_grad():
            co.inner.temb_proj.weight.normal_(std=0.3, generator=torch.Generator().manual_seed(5))
    batches = make_batches(cfg, 2)
    for train in (True, False):
        rec_up = rollout(up, batches, train=train, backward=False)
        rec_co = rollout(co, batches, train=train, backward=False)
        a, b = rec_up[0], rec_co[0]
        assert torch.isfinite(b["outputs"]["logits"]).all()
        assert not torch.equal(a["outputs"]["logits"], b["outputs"]["logits"])
        assert not torch.equal(a["z_H"], b["z_H"]) and not torch.equal(a["z_L"], b["z_L"])


def test_flow_odd_time_freqs_rejected():
    """An odd flow_time_freqs (sin/cos pairs) is rejected at construction instead of crashing at forward."""
    cfg = cfg_of("mlpt_f32", **FLOW_TRM, flow_time_freqs=63)
    with pytest.raises(AssertionError, match="even"):
        build(trm_co, cfg)


FLOW_VARIANTS = {
    "flow_h1_temb": FLOW_TRM,
    "flow_h_half_temb": dict(flow=True, flow_h_L=0.5, flow_h_H=0.5, flow_time_embed=True),
    "flow_no_temb": {**FLOW_TRM, "flow_time_embed": False},
}


@pytest.mark.parametrize("flow_variant", list(FLOW_VARIANTS))
@pytest.mark.parametrize("cfg_name", ["mlpt_f32", "rope_f32"])
def test_flow_model_construction_does_not_advance_global_rng(cfg_name, flow_variant):
    """temb_proj is created under a forked RNG: after building a flow model the global (CPU) RNG state is exactly
    the one left by building the non-flow model (and upstream TRM) with the same seed."""
    cfg = cfg_of(cfg_name)
    build(trm_co, cfg, seed=3)
    ref = torch.get_rng_state()
    build(trm_up, cfg, seed=3)
    assert torch.equal(torch.get_rng_state(), ref)
    co = build(trm_co, {**cfg, **FLOW_VARIANTS[flow_variant]}, seed=3)
    assert torch.equal(torch.get_rng_state(), ref), "building a flow model advanced the global RNG"
    assert co.inner.use_time_embed == FLOW_VARIANTS[flow_variant]["flow_time_embed"]
    # Non-vacuous: the temb_proj init does draw from the RNG, so creating it un-forked would move the stream
    build(trm_co, cfg, seed=3)
    CastedLinear(co.config.flow_time_freqs, cfg["hidden_size"], bias=True)
    assert not torch.equal(torch.get_rng_state(), ref)


@pytest.mark.parametrize("cfg_name", ["mlpt_f32", "rope_f32"])
def test_flow_trm_exploration_stream_seed_paired_without_reseeding(cfg_name):
    """Seed pairing in practice: with one seed at construction and no reseeding, the ACT exploration draws of the
    Euler-TRM arm (h = 1, zero time embedding) are those of TRM, so the train rollouts stay bitwise equal."""
    cfg = cfg_of(cfg_name)
    sd = perturbed_state_dict(build(trm_up, cfg))
    batches = make_batches(cfg, N_ACT_STEPS)

    def run(module, extra, extra_draw=False):
        model = build(module, {**cfg, **extra}, seed=11)
        if extra_draw:
            torch.rand(1)                       # what an un-forked temb_proj init would do to the stream
        model.load_state_dict(sd, strict=False)  # load_state_dict consumes no randomness
        return rollout(model, batches, train=True, backward=False, reseed=False)

    rec_up = run(trm_up, {})
    _check_rollouts_are_nontrivial(rec_up, cfg, train=False)
    assert_rollouts_equal(rec_up, run(trm_co, FLOW_TRM), f"{cfg_name} flow train (no reseed)")
    # Non-vacuous: shifting the stream by a single draw changes the exploration (halting) pattern
    shifted = run(trm_co, FLOW_TRM, extra_draw=True)
    assert any(not torch.equal(a["halted"], b["halted"]) or not torch.equal(a["steps"], b["steps"])
               for a, b in zip(rec_up, shifted)), "exploration does not depend on the RNG stream here"


# ---------------------------------------------------------------------------------------------------------
# Regularizer anchors
# ---------------------------------------------------------------------------------------------------------

def _euler(z: torch.Tensor, out: torch.Tensor, h: float) -> torch.Tensor:
    """z + h * (out - z), evaluated in fp32 and rounded once to the state dtype."""
    return (z.float() + h * (out.float() - z.float())).to(z.dtype)


def _upstream_recursion(inner, z_H, z_L, x, seq_info, n_cycles, temb=None, h_L=1.0, h_H=1.0, start=0):
    """Independent re-implementation (no inner.cycle): TRM's loop, written as explicit Euler steps."""
    L = inner.config.L_cycles
    for i in range(start, start + n_cycles):
        for j in range(L):
            ctx = z_H + x
            if temb is not None:
                ctx = ctx + temb[i * (L + 1) + j]
            out = inner.L_level(z_L, ctx, **seq_info)
            z_L = out if h_L == 1.0 else _euler(z_L, out, h_L)
        ctx = z_L
        if temb is not None:
            ctx = ctx + temb[i * (L + 1) + L]
        out = inner.L_level(z_H, ctx, **seq_info)
        z_H = out if h_H == 1.0 else _euler(z_H, out, h_H)
    return z_H, z_L


ANCHOR_VARIANTS = {
    "trm": {},
    "flow_h1_temb0": FLOW_TRM,
    "flow_h_half_temb": dict(flow=True, flow_h_L=0.5, flow_h_H=0.7, flow_time_embed=True),
}


@pytest.mark.parametrize("train", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("variant", list(ANCHOR_VARIANTS))
@pytest.mark.parametrize("cfg_name", ["mlpt_f32", "rope_f32", "mlpt_bf16"])
def test_anchors(cfg_name, variant, train):
    cfg = cfg_of(cfg_name, **ANCHOR_VARIANTS[variant])
    exact = variant != "flow_h_half_temb"
    co = build(trm_co, cfg)
    co.load_state_dict(perturbed_state_dict(co), strict=False)
    if co.inner.use_time_embed and not exact:
        with torch.no_grad():  # non-zero time embedding so the substep indexing is actually checked
            co.inner.temb_proj.weight.normal_(std=0.3, generator=torch.Generator().manual_seed(7))
            co.inner.temb_proj.bias.normal_(std=0.3, generator=torch.Generator().manual_seed(8))
    co.train(train)
    inner = co.inner
    batch = make_batches(cfg, 1)[0]
    B = cfg["batch_size"]

    with torch.no_grad():  # a non-trivial incoming carry: one inner step from the init state
        carry = inner.reset_carry(torch.ones(B, dtype=torch.bool), inner.empty_carry(B))
        carry, *_ = inner(carry, batch)

    new_carry, logits, (q_halt, q_cont), aux = inner(carry, batch)
    assert set(aux) == {"reg_s_H", "reg_s_L", "reg_phi_H", "reg_phi_L", "reg_x"}
    for k, v in aux.items():
        assert not v.requires_grad, f"{k} is not detached"
    assert logits.requires_grad  # the grad cycle is still differentiable

    # The model runs the first H_cycles - 1 cycles under no_grad and the last one with autograd enabled. The
    # recomputations below use the same autograd mode per cycle: on CPU, bf16 F.linear on a non-contiguous
    # input (the MLP-T token mixer transposes) dispatches to different kernels with and without grad mode and
    # differs by ~1 bf16 ulp, so a no_grad recomputation of the grad cycle is not bitwise comparable.
    H = cfg["H_cycles"]
    with torch.no_grad():
        x = inner.input_embeddings(batch)
        temb = inner.time_embeddings()
        seq_info = inner.seq_info()
        assert_equal(aux["reg_x"], x, "reg_x")

        # (a) via the public operator API
        z_H, z_L = carry.z_H, carry.z_L
        for i in range(H - 1):
            z_H, z_L = inner.cycle(z_H, z_L, x, temb, i, seq_info)
        assert_equal(aux["reg_s_H"], z_H, "reg_s_H vs cycle")
        assert_equal(aux["reg_s_L"], z_L, "reg_s_L vs cycle")
    with torch.enable_grad():
        temb_g = inner.time_embeddings()
        phi_H, phi_L = (t.detach() for t in inner.cycle(z_H, z_L, x, temb_g, H - 1, seq_info))
    assert_equal(aux["reg_phi_H"], phi_H, "reg_phi_H vs cycle")
    assert_equal(aux["reg_phi_L"], phi_L, "reg_phi_L vs cycle")
    assert_equal(aux["reg_phi_H"], new_carry.z_H, "reg_phi_H vs new carry")
    assert_equal(aux["reg_phi_L"], new_carry.z_L, "reg_phi_L vs new carry")
    assert not torch.equal(aux["reg_s_H"], aux["reg_phi_H"])

    # (b) independent explicit recursion (no cycle API)
    with torch.no_grad():
        s_H, s_L = _upstream_recursion(inner, carry.z_H, carry.z_L, x, seq_info, H - 1, temb, inner.h_L, inner.h_H)
    with torch.enable_grad():
        p_H, p_L = (t.detach() for t in _upstream_recursion(inner, s_H, s_L, x, seq_info, 1, temb_g,
                                                            inner.h_L, inner.h_H, start=H - 1))
    pairs = [(aux["reg_s_H"], s_H, "reg_s_H"), (aux["reg_s_L"], s_L, "reg_s_L"),
             (aux["reg_phi_H"], p_H, "reg_phi_H"), (aux["reg_phi_L"], p_L, "reg_phi_L")]
    for got, want, what in pairs:
        if exact:
            assert_equal(got, want, f"{what} vs explicit recursion")
        else:
            # torch.lerp vs z + h * (out - z): rounding differs (in bf16 an occasional 1-ulp difference in s
            # propagates through the next cycle), so compare in relative norm
            rel = 2e-2 if cfg["forward_dtype"] == "bfloat16" else 1e-5
            err = ((got.float() - want.float()).norm() / want.float().norm()).item()
            assert err <= rel, f"{what} vs explicit Euler recursion: relative error {err:.2e} > {rel:.0e}"

    # Output heads read the grad-cycle result
    with torch.enable_grad():
        expected_logits = inner.lm_head(aux["reg_phi_H"])[:, inner.puzzle_emb_len:]
    assert_equal(logits.detach(), expected_logits.detach(), "logits")


@pytest.mark.parametrize("L_cycles", [1, 3])
@pytest.mark.parametrize("cfg_name", ["mlpt_f32", "mlpt_bf16"])
def test_field_matches_cycle_definition(cfg_name, L_cycles):
    """field(z_H, z_L, x, tau_H, tau_L) = (L(z_H, z_L + tau_H) - z_H, L(z_L, z_H + x + tau_L) - z_L), in fp32.

    With h = 1, cycle i is z_L^{j+1} = z_L^j + v_L(z_H, z_L^j; tau_L = temb[substep_index(i, j)]) for the L-substeps
    j < L_cycles, then z_H' = z_H + v_H(z_H, z_L^{L_cycles}; tau_H = temb[substep_index(i, L_cycles)]) (v_H is a
    Jacobi field evaluated at the given z_L; the cycle is Gauss-Seidel and uses the new z_L), with the same
    (non-zero) time embedding at the same substep index, substep_index(i, j) = i * (L_cycles + 1) + j.
    """
    cfg = cfg_of(cfg_name, L_cycles=L_cycles, flow=True, flow_h_L=1.0, flow_h_H=1.0, flow_time_embed=True)
    co = build(trm_co, cfg)
    co.load_state_dict(perturbed_state_dict(co))
    with torch.no_grad():
        co.inner.temb_proj.weight.normal_(std=0.3, generator=torch.Generator().manual_seed(9))
    co.eval()
    inner = co.inner
    L, H = cfg["L_cycles"], cfg["H_cycles"]
    batch = make_batches(cfg, 1)[0]
    B = cfg["batch_size"]
    dtype = getattr(torch, cfg["forward_dtype"])
    tol = dict(rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        carry = inner.reset_carry(torch.ones(B, dtype=torch.bool), inner.empty_carry(B))
        x, si, temb = inner.input_embeddings(batch), inner.seq_info(), inner.time_embeddings()
        assert temb.shape[0] == H * (L + 1)
        for i in range(H):
            assert [inner.substep_index(i, j) for j in range(L + 1)] == list(range(i * (L + 1), (i + 1) * (L + 1)))
            tau_H = temb[i * (L + 1) + L]
            z_H1, z_L1 = inner.cycle(carry.z_H, carry.z_L, x, temb, i, si)
            z_L = carry.z_L
            for j in range(L):
                _, v_L = inner.field(carry.z_H, z_L, x, tau_H, temb[i * (L + 1) + j], si)
                assert v_L.dtype == torch.float32
                pred = z_L.float() + v_L                # fp32 prediction of this L-substep
                z_L = pred.to(dtype)                    # the cycle carries z_L in the forward dtype
            torch.testing.assert_close(z_L1.float(), pred, **tol, msg="z_L updates vs field")
            v_H, _ = inner.field(carry.z_H, z_L1, x, tau_H, temb[i * (L + 1)], si)
            assert v_H.dtype == torch.float32
            torch.testing.assert_close(z_H1.float(), carry.z_H.float() + v_H, **tol, msg="z_H update vs field")

            # Each block reads its own time: tau_L leaves v_H untouched and vice versa
            other = temb[(i * (L + 1) + L + 1) % temb.shape[0]]
            v_H_a, v_L_a = inner.field(carry.z_H, carry.z_L, x, tau_H, temb[i * (L + 1)], si)
            v_H_b, v_L_b = inner.field(carry.z_H, carry.z_L, x, tau_H, other, si)
            v_H_c, v_L_c = inner.field(carry.z_H, carry.z_L, x, other, temb[i * (L + 1)], si)
            assert torch.equal(v_H_a, v_H_b) and not torch.equal(v_L_a, v_L_b)
            assert torch.equal(v_L_a, v_L_c) and not torch.equal(v_H_a, v_H_c)

            # Per-row times (as the regularizer passes them): (B, 1, D) rows match per-row single-time calls
            j_rows = torch.arange(B) % L
            _, v_L_rows = inner.field(carry.z_H, carry.z_L, x, tau_H, temb[i * (L + 1) + j_rows].unsqueeze(1), si)
            for jj in range(L):
                _, v_L_j = inner.field(carry.z_H, carry.z_L, x, tau_H, temb[i * (L + 1) + jj], si)
                torch.testing.assert_close(v_L_rows[j_rows == jj], v_L_j[j_rows == jj], **tol)
            carry = type(carry)(z_H=z_H1, z_L=z_L1)


# ---------------------------------------------------------------------------------------------------------
# ACTLossHead (models/losses.py)
# ---------------------------------------------------------------------------------------------------------

def _expected_fpres(out, halted, labels):
    res2 = sum((out[f"reg_phi_{b}"].double() - out[f"reg_s_{b}"].double()).square().flatten(1).sum(-1) for b in "HL")
    ref2 = sum(out[f"reg_s_{b}"].double().square().flatten(1).sum(-1) for b in "HL")
    valid = halted & ((labels != IGNORE_LABEL_ID).sum(-1) > 0)
    return torch.sqrt(res2 / ref2)[valid].sum(), valid


ANCHOR_KEYS = ["reg_s_H", "reg_s_L", "reg_phi_H", "reg_phi_L", "reg_x"]


@pytest.mark.parametrize("flow", [False, True], ids=["default", "flow_h1_temb0"])
@pytest.mark.parametrize("cfg_name", ["mlpt_f32", "rope_f32", "mlpt_bf16"])
def test_act_loss_head_equivalence_and_fpres(cfg_name, flow, loss_type="stablemax_cross_entropy"):
    cfg = cfg_of(cfg_name)
    up = build(trm_up, cfg)
    co = build(trm_co, {**cfg, **(FLOW_TRM if flow else {})})
    sd = perturbed_state_dict(up)
    up.load_state_dict(sd)
    co.load_state_dict(sd, strict=not flow)
    head_up, head_co = ACTLossHead(up, loss_type), ACTLossHead(co, loss_type)
    batches = make_batches(cfg, N_ACT_STEPS)
    return_keys = ["logits", "q_halt_logits", "preds"]

    for train in (True, False):
        head_up.train(train)
        head_co.train(train)
        carry_up = carry_co = None
        saw_valid = False
        for i, batch in enumerate(batches):
            ctx = contextlib.nullcontext() if train else torch.inference_mode()
            _zero_grads(up)
            _zero_grads(co)
            with ctx:
                if carry_up is None:
                    carry_up, carry_co = head_up.initial_carry(batch), head_co.initial_carry(batch)
                prev_halted, prev_inner = carry_co.halted.clone(), carry_co.inner_carry
                rk = return_keys + ([] if train else ANCHOR_KEYS)  # anchors only auto-exported in training
                torch.manual_seed(100 + i)
                carry_up, loss_up, m_up, out_up, fin_up = head_up(return_keys=rk, carry=carry_up, batch=batch)
                torch.manual_seed(100 + i)
                carry_co, loss_co, m_co, out_co, fin_co = head_co(return_keys=rk, carry=carry_co, batch=batch)

            w = f"{'train' if train else 'eval'} step {i}"
            assert_equal(loss_up, loss_co, f"{w} loss")
            assert_equal(fin_up, fin_co, f"{w} all_finish")
            assert set(m_co) - set(m_up) == {"fpres"} and set(m_up) <= set(m_co), f"{w} metric keys"
            for k in m_up:
                assert_equal(m_up[k], m_co[k], f"{w} metric {k}")
            for k in return_keys:
                assert_equal(out_up[k], out_co[k], f"{w} output {k}")
            assert_equal(carry_up.halted, carry_co.halted, f"{w} halted")
            assert_equal(carry_up.inner_carry.z_H, carry_co.inner_carry.z_H, f"{w} z_H")

            # Anchor pass-through: training exports detached reg_* outputs, eval only what was asked for
            assert not any(k.startswith("reg_") for k in out_up)
            assert {k for k in out_co if k.startswith("reg_")} == set(ANCHOR_KEYS)
            for k in ANCHOR_KEYS:
                assert not out_co[k].requires_grad

            # fpres = sum over valid halted rows of sqrt(||phi - s||^2 / ||s||^2), (H, L) blocks jointly
            expected, valid = _expected_fpres(out_co, carry_co.halted, carry_co.current_data["labels"])
            saw_valid |= bool(valid.any())
            assert m_co["fpres"].dtype == torch.float32 and m_co["fpres"].ndim == 0
            torch.testing.assert_close(m_co["fpres"].double(), expected, rtol=1e-5, atol=1e-6, msg=f"{w} fpres")
            assert_equal(out_co["reg_phi_H"], carry_co.inner_carry.z_H, f"{w} reg_phi_H vs carry")
            assert_equal(out_co["reg_phi_L"], carry_co.inner_carry.z_L, f"{w} reg_phi_L vs carry")

            if train:
                loss_up.backward()
                loss_co.backward()
                for (n, p_up), (_, p_co) in zip(up.named_parameters(), co.named_parameters()):
                    assert (p_up.grad is None) == (p_co.grad is None), f"{w} grad presence {n}"
                    if p_up.grad is not None:
                        assert_equal(p_up.grad, p_co.grad, f"{w} loss grad {n}")
                if cfg["puzzle_emb_ndim"] > 0:
                    assert_equal(up.inner.puzzle_emb.local_weights.grad, co.inner.puzzle_emb.local_weights.grad,
                                 f"{w} loss grad puzzle_emb.local_weights")

            # reg_s_* is the state entering the grad cycle of this ACT step: recompute from the previous carry
            with torch.no_grad(), ctx:
                inner = co.inner
                reset = inner.reset_carry(prev_halted, prev_inner)
                x = inner.input_embeddings(carry_co.current_data)
                s_H, s_L = _upstream_recursion(inner, reset.z_H, reset.z_L, x, inner.seq_info(), cfg["H_cycles"] - 1)
            assert_equal(out_co["reg_s_H"], s_H, f"{w} reg_s_H vs recomputation")
            assert_equal(out_co["reg_s_L"], s_L, f"{w} reg_s_L vs recomputation")
            assert_equal(out_co["reg_x"], x, f"{w} reg_x vs recomputation")
        assert saw_valid, "no valid halted row: fpres never exercised"


@pytest.mark.xfail(strict=True, raises=TypeError, reason=(
    "BUG (inherited from upstream, still present in models/losses.py): ACTLossHead.forward calls "
    "self.loss_fn(..., valid_mask=mask) but softmax_cross_entropy(logits, labels, ignore_index) has no valid_mask "
    "parameter, so loss_type=softmax_cross_entropy raises TypeError on the first forward"))
@pytest.mark.parametrize("model_module", [trm_up, trm_co], ids=["trm", "trm_contractive"])
def test_act_loss_head_softmax_cross_entropy(model_module):
    cfg = cfg_of("mlpt_f32")
    head = ACTLossHead(build(model_module, cfg), "softmax_cross_entropy")
    batch = make_batches(cfg, 1)[0]
    carry = head.initial_carry(batch)
    torch.manual_seed(100)
    _, loss, metrics, _, _ = head(return_keys=[], carry=carry, batch=batch)
    assert torch.isfinite(loss)
