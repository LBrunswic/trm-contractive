"""D-gamma (dec_semi): semi-contraction onto the decoder fibres with leak tolerance gamma (CPU)."""
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("DISABLE_COMPILE", "1")

from models.regularizers import ContractionRegularizer, RegConfig, mapped_power_step, normalize_rows, rownorm2  # noqa: E402
from models.recursive_reasoning.trm_contractive import TinyRecursiveReasoningModel_ACTV1  # noqa: E402
from fixed_point_diagnostics import mapped_sigma_max  # noqa: E402


# ---------------------------------------------------------------- linear toy: state = (vis-carrying, other) tuple
N_H, N_L, VIS = 6, 5, 3  # z_H has 6 coords, the first 3 are "visible"


def make_ops(gamma):
    def P(v):
        h = torch.zeros_like(v[0], dtype=torch.float64)
        h[:, :VIS] = v[0][:, :VIS]
        return h, torch.zeros_like(v[1], dtype=torch.float64)

    def comp(v):
        return tuple(t.double() - p for t, p in zip(v, P(v)))

    W = lambda v: tuple(p + gamma * c for p, c in zip(P(v), comp(v)))
    Winv = lambda v: tuple(p + c / gamma for p, c in zip(P(v), comp(v)))
    return P, W, Winv


def flat_ops(gamma):
    n = N_H + N_L
    Pm = torch.zeros(n, n, dtype=torch.float64)
    Pm[:VIS, :VIS] = torch.eye(VIS, dtype=torch.float64)
    Wi = Pm + (torch.eye(n, dtype=torch.float64) - Pm) / gamma
    return Pm, Wi


def linear_phi(J):
    def phi(zh, zl):
        v = torch.cat([zh, zl], -1) @ J.T
        return v[:, :N_H], v[:, N_H:]
    return phi


@pytest.mark.parametrize("gamma", [1.0, 0.3, 0.1])
def test_mapped_power_step_converges_to_sigma_PJWinv(gamma):
    g = torch.Generator().manual_seed(0)
    n = N_H + N_L
    J = torch.randn(n, n, generator=g, dtype=torch.float64)
    J[:VIS, VIS:] *= 3.0                       # strong invisible->visible leak
    P, W, Winv = make_ops(gamma)
    Pm, Wi = flat_ops(gamma)
    target = torch.linalg.matrix_norm(Pm @ J @ Wi, 2).item()
    phi = linear_phi(J)
    s = (torch.randn(4, N_H, generator=g, dtype=torch.float64), torch.randn(4, N_L, generator=g, dtype=torch.float64))
    ref = phi(*s)
    d = normalize_rows((torch.randn(4, N_H, generator=g, dtype=torch.float64), torch.randn(4, N_L, generator=g, dtype=torch.float64)))
    for _ in range(200):
        d = mapped_power_step(phi, s, ref, tuple(0.1 * t for t in d), out_proj=P, in_map=Winv)
    # ratio along the converged direction, measured in the W metric
    delta = Winv(tuple(0.1 * t for t in d))
    out = phi(*(a + b for a, b in zip(s, delta)))
    num = rownorm2(*P(tuple(o - r for o, r in zip(out, ref)))).sqrt()
    den = rownorm2(*W(delta)).sqrt()
    assert torch.allclose(num / den, torch.full_like(num, target), rtol=1e-6)


def test_gamma_family_bounds_block_and_leak():
    """sigma(P J W^-1) >= max(||PJP||, ||PJ(I-P)|| / gamma): the constraint bounds both blocks."""
    g = torch.Generator().manual_seed(1)
    n = N_H + N_L
    J = torch.randn(n, n, generator=g, dtype=torch.float64)
    for gamma in (1.0, 0.3, 0.1):
        Pm, Wi = flat_ops(gamma)
        I = torch.eye(n, dtype=torch.float64)
        s_all = torch.linalg.matrix_norm(Pm @ J @ Wi, 2)
        A = torch.linalg.matrix_norm(Pm @ J @ Pm, 2)
        B = torch.linalg.matrix_norm(Pm @ J @ (I - Pm), 2)
        assert s_all >= A - 1e-12 and s_all >= B / gamma - 1e-12


def test_mapped_sigma_max_diagnostics_matches_matrix_norm():
    g = torch.Generator().manual_seed(2)
    n = N_H + N_L
    J = torch.randn(n, n, generator=g, dtype=torch.float64)
    Pm, Wi = flat_ops(0.3)
    f = lambda s: (s.flatten(1) @ J.T).view_as(s)
    out_proj = lambda v: (v.flatten(1) @ Pm.T).view_as(v)
    in_map = lambda v: (v.flatten(1) @ Wi.T).view_as(v)
    s = torch.randn(3, n, dtype=torch.float64, generator=g)
    sig = mapped_sigma_max(f, s, 300, g, out_proj, in_map)
    assert torch.allclose(sig, torch.full_like(sig, torch.linalg.matrix_norm(Pm @ J @ Wi, 2).item()), rtol=1e-6)


# ---------------------------------------------------------------- tiny real model
def tiny_model(dtype="float32"):
    cfg = dict(batch_size=8, seq_len=81, puzzle_emb_ndim=32, num_puzzle_identifiers=1, vocab_size=11, H_cycles=2, L_cycles=2,
               H_layers=0, L_layers=2, hidden_size=32, expansion=2, num_heads=2, pos_encodings="none", halt_max_steps=4,
               halt_exploration_prob=0.1, mlp_t=True, puzzle_emb_len=16, forward_dtype=dtype)
    torch.manual_seed(0)
    return TinyRecursiveReasoningModel_ACTV1(cfg)


def anchors(model):
    torch.manual_seed(1)
    batch = dict(inputs=torch.randint(1, 11, (8, 81)), labels=torch.randint(1, 11, (8, 81)),
                 puzzle_identifiers=torch.zeros(8, dtype=torch.int32))
    model.train()
    carry = model.initial_carry(batch)
    carry, out = model(carry, batch)
    return out, carry.steps


def make_reg(model, **kw):
    cfg = RegConfig(**{"reg_frac": 1.0, "reg_dual_warmup": 0, "reg_monitor_every": 0, **kw})
    return ContractionRegularizer(cfg, model.inner, batch_size=8, seed=0, device="cpu")


def test_semi_gamma1_equals_dec_out_fp32():
    model = tiny_model("float32")
    out, steps = anchors(model)
    vals = {}
    for mode, extra in (("dec_out", {}), ("dec_semi", {"reg_gamma": 1.0})):
        reg = make_reg(model, reg_mode=mode, reg_rho=1.0, **extra)
        reg.stash(out, steps, global_step=1)
        loss, m = reg(2)
        vals[mode] = (m["reg/value"], loss.item())
    assert vals["dec_out"][0] == pytest.approx(vals["dec_semi"][0], rel=1e-5)
    assert vals["dec_out"][1] == pytest.approx(vals["dec_semi"][1], rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_semi_end_to_end_grads_and_metrics(dtype):
    model = tiny_model(dtype)
    out, steps = anchors(model)
    reg = make_reg(model, reg_mode="dec_semi", reg_rho=1e-3, reg_gamma=0.3)  # tiny cap: always violated
    reg.stash(out, steps, global_step=1)
    loss, m = reg(2)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    grads = [p.grad for p in model.inner.L_level.parameters()]
    assert all(g_ is not None for g_ in grads) and sum(g_.abs().sum().item() for g_ in grads) > 0
    assert m["reg/viol"] == 1.0 and m["reg/value"] > 0
    # huge cap: no penalty
    reg2 = make_reg(model, reg_mode="dec_semi", reg_rho=1e6, reg_gamma=0.3)
    reg2.stash(out, steps, global_step=1)
    loss2, m2 = reg2(2)
    assert loss2.item() == 0.0 and m2["reg/viol"] == 0.0


def test_semi_ratio_decreases_with_gamma_for_fixed_operator():
    """Larger gamma inflates the denominator for invisible perturbations: the adversarial quotient is
    (weakly) larger for smaller gamma (stricter constraint)."""
    model = tiny_model("float32")
    out, steps = anchors(model)
    vals = []
    for gam in (1.0, 0.3, 0.1):
        reg = make_reg(model, reg_mode="dec_semi", reg_rho=1.0, reg_gamma=gam, reg_v1_power_iters=8)
        reg.stash(out, steps, global_step=1)
        _, m = reg(2)
        vals.append(m["reg/value"])
    assert vals[0] <= vals[1] * 1.05 and vals[1] <= vals[2] * 1.05


def test_monitor_reports_semi():
    model = tiny_model("float32")
    out, steps = anchors(model)
    reg = make_reg(model, reg_mode="none")
    reg.cfg.reg_monitor_every = 1
    reg.stash(out, steps, global_step=1)
    _, m = reg(2)
    assert math.isfinite(m["mon/dec_semi"]) and m["mon/dec_semi"] > 0
