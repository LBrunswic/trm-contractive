"""Tests for models/regularizers.py (contractivity regularizers of the TRM latent operator).

CPU only by design; run with
    CUDA_VISIBLE_DEVICES="" DISABLE_COMPILE=1 pytest tests/test_regularizers.py
Sections (1)-(8) drive the pure helpers with float64 toy operators whose spectra are known in closed
form (a toy state is a 2-tuple (z_H, z_L) of (B, n) tensors, operators act on their concatenation).
Section (9) runs ContractionRegularizer on a linear stub of the operator API (known spectra, full
warm-started loop), section (10) end to end on a tiny real trm_contractive model.
"""
import copy
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

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import models.regularizers as regmod  # noqa: E402
from models.regularizers import (  # noqa: E402
    ContractionRegularizer, RegConfig, f_gain_power_step, normalize_rows, osl_quotient, rowdot, rownorm2,
    secant_ratio, v1_power_step, v2_power_step,
)
from models.recursive_reasoning.trm_contractive import (  # noqa: E402
    TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1_Inner,
)
from models.losses import ACTLossHead  # noqa: E402

DT = torch.float64


@pytest.fixture(autouse=True, scope="module")
def _few_threads():
    """Tiny CPU ops: a handful of threads is fastest and avoids oversubscribing a shared machine."""
    n = torch.get_num_threads()
    torch.set_num_threads(min(4, n))
    yield
    torch.set_num_threads(n)


# =============================================================================================
# Toy-operator utilities
# =============================================================================================

def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(g, *shape, dtype=DT):
    return torch.randn(*shape, generator=g, dtype=dtype)


def _rand_orth(N: int, g) -> torch.Tensor:
    q, r = torch.linalg.qr(_randn(g, N, N))
    return q * torch.sign(torch.diagonal(r))


def _cat(t):
    return torch.cat([x.reshape(x.shape[0], -1) for x in t], -1)


def _split(v, like):
    """Inverse of _cat for a state shaped like `like` (tuple of tensors)."""
    out, o = [], 0
    for x in like:
        m = x[0].numel()
        out.append(v[:, o:o + m].reshape(x.shape))
        o += m
    return tuple(out)


def _linear_op(M):
    """Phi(z_H, z_L) = M @ concat(z_H, z_L), blocks reshaped back to the input shapes."""
    def op(*s):
        return _split(_cat(s) @ M.T.to(s[0].dtype), s)
    return op


def _state(g, B, shape, scale=1.0):
    return (scale * _randn(g, B, *shape), scale * _randn(g, B, *shape))


def _dir(g, B, shape):
    return normalize_rows(_state(g, B, shape))


def _scale(d, r):
    if not torch.is_tensor(r):
        return tuple(x * r for x in d)
    return tuple(x * r.view((-1,) + (1,) * (x.ndim - 1)).to(x.dtype) for x in d)


def _add(s, d):
    return tuple(x + y.to(x.dtype) for x, y in zip(s, d))


def _rq(M, d):
    """Rayleigh quotient d^T M d / d^T d in float64 for row-directions d."""
    v = _cat(d).to(DT)
    return ((v @ M.T) * v).sum(-1) / v.square().sum(-1)


def _jnorm2(J, d):
    """||J d||^2 / ||d||^2 in float64."""
    v = _cat(d).to(DT)
    return (v @ J.T).square().sum(-1) / v.square().sum(-1)


def _spectral_op(g, N, sv):
    U, V = _rand_orth(N, g), _rand_orth(N, g)
    return U @ torch.diag(sv.to(DT)) @ V.T


def _sym_with_spectrum(g, lam):
    Q = _rand_orth(lam.numel(), g)
    return Q @ torch.diag(lam.to(DT)) @ Q.T


def _skew(g, N, norm):
    K = _randn(g, N, N)
    K = (K - K.T) / 2
    return K / torch.linalg.matrix_norm(K, 2) * norm


# =============================================================================================
# (1) secant_ratio on a scaled orthogonal map
# =============================================================================================

@pytest.mark.parametrize("shape", [(16,), (5, 4)], ids=["2d_state", "3d_state"])
def test_secant_ratio_half_orthogonal_is_exactly_half(shape):
    g = _gen(0)
    B = 12
    N = 2 * math.prod(shape)
    phi = _linear_op(0.5 * _rand_orth(N, g))   # Q acts on the concatenated state

    # Arbitrary pairs with row scales spanning 4 decades, plus close pairs at the regularizer's radii
    row_scale = 10 ** (4 * torch.rand(B, generator=g, dtype=DT) - 2)
    a = _scale(_state(g, B, shape), row_scale)
    b = _scale(_state(g, B, shape), row_scale)
    near = _add(a, _scale(_dir(g, B, shape), 0.05 * rownorm2(*a).sqrt()))

    for u, v in [(a, b), (a, near), (near, a)]:
        ratio, den2 = secant_ratio(phi(*u), phi(*v), u, v)
        assert ratio.shape == (B,) and ratio.dtype == torch.float32 and den2.dtype == torch.float32
        torch.testing.assert_close(ratio, torch.full((B,), 0.5), rtol=2e-5, atol=0)
        torch.testing.assert_close(den2, _cat(u).sub(_cat(v)).square().sum(-1).float(), rtol=1e-5, atol=0)
        assert torch.equal(F.relu(ratio - (1 - 0.05)), torch.zeros(B))


# =============================================================================================
# (2) v1 power step: bimodal spectrum (catches a projected-gradient / normalised-leaf bug)
# =============================================================================================

def test_v1_power_step_bimodal_spectrum():
    g = _gen(1)
    B, n = 16, 16
    N = 2 * n
    sv2 = torch.cat([torch.zeros(N // 2), torch.full((N // 2,), 2.0)])[torch.randperm(N, generator=g)]
    J = _spectral_op(g, N, sv2.sqrt())
    phi = _linear_op(J)
    s = _state(g, B, (n,))
    phi_s = phi(*s)

    d0 = _dir(g, B, (n,))
    rq0 = _jnorm2(J, d0)
    assert 0.5 < rq0.mean() < 1.5            # random start: ~ mean(sigma^2) = 1

    d1 = v1_power_step(phi, s, phi_s, _scale(d0, 0.1))
    assert all(x.dtype == torch.float32 and not x.requires_grad for x in d1)
    torch.testing.assert_close(rownorm2(*d1), torch.ones(B), rtol=1e-6, atol=0)
    rq1 = _jnorm2(J, d1)
    # A true power step on J^T J kills the null half in one step (RQ = 2 exactly); a gradient
    # projected on the tangent of the sphere would give RQ ~ 1, a no-op step RQ ~ rq0.
    assert rq1.min() > 1.5, rq1
    torch.testing.assert_close(rq1, torch.full((B,), 2.0, dtype=DT), rtol=1e-4, atol=0)

    # Linear map: the step is invariant to the pair radius
    d1_big = v1_power_step(phi, s, phi_s, _scale(d0, 10.0))
    torch.testing.assert_close(_cat(d1_big), _cat(d1), rtol=0, atol=1e-5)

    d = d1
    for _ in range(29):
        d = v1_power_step(phi, s, phi_s, _scale(d, 0.1))
    torch.testing.assert_close(_jnorm2(J, d), torch.full((B,), 2.0, dtype=DT), rtol=1e-3, atol=0)


# =============================================================================================
# (3) v1 power iteration finds the spectral norm; random pairs only see the RMS singular value
# =============================================================================================

def test_v1_power_converges_to_spectral_norm_random_pairs_see_rms():
    g = _gen(2)
    B, n = 256, 32
    N = 2 * n
    sv = torch.cat([torch.tensor([3.0], dtype=DT), 0.05 + 0.45 * torch.rand(N - 1, generator=g, dtype=DT)])
    J = _spectral_op(g, N, sv)
    assert torch.linalg.matrix_norm(J, 2).item() == pytest.approx(3.0, rel=1e-12)
    rms = sv.square().mean().sqrt().item()

    phi = _linear_op(J)
    s = _state(g, B, (n,))
    phi_s = phi(*s)
    r = 0.1 * rownorm2(*s).sqrt()                 # radius relative to ||s||, as in the regularizer

    def ratio_of(d):
        s2 = _add(s, _scale(d, r))
        return secant_ratio(phi(*s2), phi_s, s2, s)[0]

    d = _dir(g, B, (n,))
    rand_ratio = ratio_of(d)
    assert rand_ratio.square().mean().sqrt().item() == pytest.approx(rms, rel=0.05)
    assert rand_ratio.mean().item() == pytest.approx(rms, rel=0.15)
    assert rand_ratio.max().item() < 0.75 * 3.0   # random pairs badly underestimate Lip

    for _ in range(30):
        d = v1_power_step(phi, s, phi_s, _scale(d, r))
    torch.testing.assert_close(ratio_of(d), torch.full((B,), 3.0), rtol=1e-3, atol=0)


# =============================================================================================
# (4) v2 shifted power step on linear fields: closed form, convergence, shift requirement
# =============================================================================================

def _jf_norm(A):
    """||J_F||_2 for F = V + I with V(s) = A s."""
    return torch.linalg.matrix_norm(A + torch.eye(A.shape[0], dtype=A.dtype), 2).item()


def _run_v2(A, g, B, n, steps, r=0.2, s_scale=0.2, shift=None):
    """Warm-started v2 power steps on V(s) = A s; returns (B, steps+1) quotients and exact RQs.

    shift: None (the c = 1 + 1.5 * gain_F(d) fallback of v2_power_step) or a float c used for every row.
    """
    field = _linear_op(A)
    s = _state(g, B, (n,), scale=s_scale)
    v_s = field(*s)
    d = _dir(g, B, (n,))
    S = (A + A.T) / 2
    c = None if shift is None else torch.full((B,), float(shift))
    qs, rqs = [], []
    for k in range(steps + 1):
        s2 = _add(s, _scale(d, r))
        qs.append(osl_quotient(field(*s2), v_s, s2, s)[0])
        rqs.append(_rq(S, d))
        if k < steps:
            d, _ = v2_power_step(field, s, v_s, _scale(d, r), shift=c)
    return torch.stack(qs, 1), torch.stack(rqs, 1)


def _first_within(q, target, tol):
    ok = ((q - target).abs() < tol).all(0)
    return int(ok.nonzero()[0]) if ok.any() else None


def test_v2_power_step_closed_form_shift_and_gain():
    """Linear V = A s: new = normalize(sym(A) d + c d) with c = shift (per row) or 1 + 1.5 gain_F, where
    gain_F = ||(A + I) d|| / ||d|| is the gain of F = V + I along the *input* d."""
    g = _gen(12)
    B, n = 6, 8
    N = 2 * n
    A = _randn(g, N, N)                                   # generic, non-symmetric
    S = (A + A.T) / 2
    JF = A + torch.eye(N, dtype=DT)
    field = _linear_op(A)
    s = _state(g, B, (n,))
    v_s = field(*s)
    r = 0.05 * (1 + torch.arange(B, dtype=DT)) * rownorm2(*s).sqrt().to(DT)   # different radius per row
    delta = _scale(_dir(g, B, (n,)), r)
    dv = _cat(delta).to(DT)
    gain_ref = (dv @ JF.T).norm(dim=-1) / dv.norm(dim=-1)

    def ref_dir(c):
        v = dv @ S.T + c.to(DT)[:, None] * dv
        return v / v.norm(dim=-1, keepdim=True)

    c = torch.linspace(2.0, 7.0, B)
    d1, gain = v2_power_step(field, s, v_s, delta, shift=c)
    assert all(t.dtype == torch.float32 and not t.requires_grad for t in d1)
    assert gain.shape == (B,) and gain.dtype == torch.float32 and not gain.requires_grad
    torch.testing.assert_close(rownorm2(*d1), torch.ones(B), rtol=1e-6, atol=0)
    torch.testing.assert_close(gain.to(DT), gain_ref, rtol=1e-5, atol=0)
    torch.testing.assert_close(_cat(d1).to(DT), ref_dir(c), rtol=0, atol=1e-5)

    # shift=None falls back to c = 1 + 1.5 * gain_F (same gain, different direction: the shift is really used)
    d2, gain2 = v2_power_step(field, s, v_s, delta)
    assert torch.equal(gain2, gain)
    torch.testing.assert_close(_cat(d2).to(DT), ref_dir(1.0 + 1.5 * gain_ref.float()), rtol=0, atol=1e-5)
    assert (_cat(d1) - _cat(d2)).norm(dim=-1).min() > 1e-2


@pytest.mark.parametrize("shift", ["fallback", "bound"])
def test_v2_power_step_indefinite_nonsymmetric_monotone_and_converges(shift):
    g = _gen(3)
    B, n = 8, 16
    N = 2 * n
    lam = torch.linspace(-3.0, 1.5, N, dtype=DT)
    A = _sym_with_spectrum(g, lam) + _skew(g, N, 1.0)       # indefinite, non-symmetric
    S = (A + A.T) / 2
    lam_max = torch.linalg.eigvalsh(S).max().item()
    assert lam_max == pytest.approx(1.5, abs=1e-12) and not torch.allclose(A, A.T)
    # "bound": c = 1 + ||J_F|| >= -lambda_min(sym J_V) makes the shifted operator PSD
    c = None if shift == "fallback" else 1.0 + _jf_norm(A)
    assert c is None or c >= -torch.linalg.eigvalsh(S).min().item() - 1e-9

    q, rq = _run_v2(A, g, B, n, steps=400, shift=c)
    # The secant quotient of a linear field is exactly the Rayleigh quotient of sym(A)
    torch.testing.assert_close(q.to(DT), rq, rtol=1e-4, atol=1e-5)
    # Monotone increase along warm-started steps (up to fp32 noise)
    worst_drop = (q[:, :-1] - q[:, 1:]).max().item()
    assert worst_drop < 1e-5, f"osl quotient decreased by {worst_drop:.3e}"
    assert (q[:, -1] - q[:, 0]).min() > 0.5

    n_1e2 = _first_within(q, lam_max, 1e-2)
    n_1e3 = _first_within(q, lam_max, 1e-3)
    print(f"\n[v2 indefinite, {shift}] steps to |q - lam_max| < 1e-2: {n_1e2}, < 1e-3: {n_1e3}")
    assert n_1e2 is not None and n_1e2 <= 300
    torch.testing.assert_close(q[:, -1], torch.full((B,), lam_max, dtype=torch.float32), rtol=0, atol=1e-2)


def _wide_negative_field(case, g, n):
    N = 2 * n
    if case == "sym_negdef_wide":
        return _sym_with_spectrum(g, torch.linspace(-6.0, 0.0, N, dtype=DT))
    # V = (Bm(zH+zL) - zH, Bm(zL+zH) - zL): sym(J_V) = {-1 (diff mode)} U {2 lam(Bm) - 1 (sum mode)}
    Bm = _sym_with_spectrum(g, torch.linspace(-2.5, 0.45, n, dtype=DT))
    I = torch.eye(n, dtype=DT)
    return torch.cat([torch.cat([Bm - I, Bm], 1), torch.cat([Bm, Bm - I], 1)], 0)


_FALLBACK_STALL = pytest.mark.xfail(strict=True, reason=(
    "Documented limitation (not a bug of the regularizer, which always passes an explicit shift): the shift=None "
    "fallback of v2_power_step uses c = 1 + 1.5 * gain_F along the *current* direction, which near lambda_max is "
    "~1 + 1.5 * 1 = 2.5 here, below -lambda_min(sym J_V) = 6, so sym(J_V) + c I is not PSD and its most negative "
    "eigenvalue dominates: the warm-started iteration stalls at a mixed fixed point (sym_negdef_wide: q ~ -0.194 "
    "instead of 0; trm_structured_wide: q ~ -0.358 instead of -0.1, and q is not monotone)."))


@pytest.mark.parametrize("shift", ["bound", pytest.param("fallback", marks=_FALLBACK_STALL)])
@pytest.mark.parametrize("case", ["sym_negdef_wide", "trm_structured_wide"])
def test_v2_power_step_wide_negative_spectrum_finds_lambda_max(case, shift):
    """lambda_min(sym J_V) = -6: with a guaranteed shift c = 1 + ||A + I||_2 (>= -lambda_min) the shifted
    operator is PSD and the iteration converges monotonically to lambda_max."""
    g = _gen(4)
    B, n = 8, 16
    A = _wide_negative_field(case, g, n)
    eig = torch.linalg.eigvalsh((A + A.T) / 2)
    lam_max, lam_min = eig.max().item(), eig.min().item()
    assert lam_min == pytest.approx(-6.0, abs=1e-9)
    c = 1.0 + _jf_norm(A) if shift == "bound" else None
    assert c is None or c >= -lam_min - 1e-9            # tight for sym_negdef_wide: sym(J_F) has eigenvalue -5
    q, _ = _run_v2(A, g, B, n, steps=1500, shift=c)
    print(f"\n[v2 {case}, {shift}] lam_max={lam_max:.3f} final q in [{q[:, -1].min():.3f}, {q[:, -1].max():.3f}]")
    worst_drop = (q[:, :-1] - q[:, 1:]).max().item()
    assert worst_drop < 1e-5, f"osl quotient decreased by {worst_drop:.3e}"
    torch.testing.assert_close(q[:, -1], torch.full((B,), lam_max, dtype=torch.float32), rtol=0, atol=1e-3)


# =============================================================================================
# (4b) f_gain power step: J_F^T J_F for F = V + I
# =============================================================================================

def test_f_gain_power_step_bimodal_spectrum_and_gain_along_input():
    g = _gen(13)
    B, n = 16, 16
    N = 2 * n
    sv2 = torch.cat([torch.zeros(N // 2), torch.full((N // 2,), 2.0)])[torch.randperm(N, generator=g)]
    JF = _spectral_op(g, N, sv2.sqrt())
    field = _linear_op(JF - torch.eye(N, dtype=DT))       # V = J_F - I, so F = V + I = J_F
    s = _state(g, B, (n,))
    v_s = field(*s)

    d0 = _dir(g, B, (n,))
    d1, gain = f_gain_power_step(field, s, v_s, _scale(d0, 0.1))
    assert all(x.dtype == torch.float32 and not x.requires_grad for x in d1)
    assert gain.shape == (B,) and gain.dtype == torch.float32 and not gain.requires_grad
    torch.testing.assert_close(rownorm2(*d1), torch.ones(B), rtol=1e-6, atol=0)
    # gain is the gain of F (not of V) along the *input* direction
    torch.testing.assert_close(gain.to(DT), _jnorm2(JF, d0).sqrt(), rtol=1e-4, atol=0)
    # A true power step on J_F^T J_F kills the null half in one step: ||J_F d1||^2 = 2 exactly
    torch.testing.assert_close(_jnorm2(JF, d1), torch.full((B,), 2.0, dtype=DT), rtol=1e-4, atol=0)
    # Linear map: invariant to the pair radius
    d1_big, gain_big = f_gain_power_step(field, s, v_s, _scale(d0, 10.0))
    torch.testing.assert_close(_cat(d1_big), _cat(d1), rtol=0, atol=1e-5)
    torch.testing.assert_close(gain_big, gain, rtol=1e-4, atol=0)


def test_f_gain_power_step_converges_to_norm_of_A_plus_I():
    g = _gen(11)
    B, n = 16, 16
    N = 2 * n
    sv = torch.cat([torch.tensor([2.5], dtype=DT), 0.1 + 1.5 * torch.rand(N - 1, generator=g, dtype=DT)])
    A = _spectral_op(g, N, sv) - torch.eye(N, dtype=DT)
    target = _jf_norm(A)
    assert target == pytest.approx(2.5, rel=1e-12)
    # The quantity is ||A + I||, not ||A|| (which differs here), i.e. the gain of the block maps F
    assert abs(torch.linalg.matrix_norm(A, 2).item() - target) > 0.2
    field = _linear_op(A)
    s = _state(g, B, (n,))
    v_s = field(*s)
    r = 0.1 * rownorm2(*s).sqrt()

    d = _dir(g, B, (n,))
    gains = []
    for _ in range(40):
        d, gain = f_gain_power_step(field, s, v_s, _scale(d, r))
        gains.append(gain)
    G = torch.stack(gains, 1)
    assert G[:, 0].max() < 0.8 * target                   # a random direction underestimates ||J_F||
    worst_drop = (G[:, :-1] - G[:, 1:]).max().item()
    assert worst_drop < 1e-5, f"F gain decreased by {worst_drop:.3e}"
    torch.testing.assert_close(G[:, -1], torch.full((B,), target, dtype=torch.float32), rtol=1e-3, atol=0)
    assert (G <= target * (1 + 1e-5)).all()               # a secant gain never exceeds the operator norm
    torch.testing.assert_close(_jnorm2(A + torch.eye(N, dtype=DT), d).sqrt(), torch.full((B,), target, dtype=DT),
                               rtol=1e-3, atol=0)


# =============================================================================================
# (5) TRM-structured linear toy, through the real Inner.field
# =============================================================================================

class _LinearLevel:
    """Stand-in for L_level: ReasoningModule(h, inj) = layers(h + inj), here a linear map Bm."""

    def __init__(self, Bm):
        self.Bm = Bm

    def __call__(self, hidden_states, input_injection, **kwargs):
        return (hidden_states + input_injection) @ self.Bm.T


def _trm_field(Bm, x=None, temb_H=None, temb_L=None):
    stub = SimpleNamespace(L_level=_LinearLevel(Bm))

    def field(z_H, z_L):
        xx = torch.zeros_like(z_H) if x is None else x
        return TinyRecursiveReasoningModel_ACTV1_Inner.field(stub, z_H, z_L, xx, temb_H, temb_L, {})
    return field


def test_trm_structured_field_modes():
    g = _gen(5)
    n = 12
    lam = torch.linspace(-0.5, 0.9, n, dtype=DT)
    Q = _rand_orth(n, g)
    Bm = Q @ torch.diag(lam) @ Q.T                    # symmetric, Bm' = Bm
    B = n
    x = _randn(g, B, n)
    t_H = _randn(g, B, n)
    t_L = _randn(g, B, n)
    field = _trm_field(Bm, x=x, temb_H=t_H, temb_L=t_L)
    s = _state(g, B, (n,))

    # Structure of the real field: V = (Bm(zH + zL + t_H) - zH, Bm(zL + zH + x + t_L) - zL), returned in fp32;
    # each block sees its own time embedding
    v = field(*s)
    assert all(t.dtype == torch.float32 for t in v)
    torch.testing.assert_close(v[0], ((s[0] + s[1] + t_H) @ Bm.T - s[0]).float(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(v[1], ((s[1] + s[0] + x + t_L) @ Bm.T - s[1]).float(), rtol=1e-6, atol=1e-6)
    v_noH = _trm_field(Bm, x=x, temb_H=None, temb_L=t_L)(*s)
    v_noL = _trm_field(Bm, x=x, temb_H=t_H, temb_L=None)(*s)
    assert torch.equal(v_noH[1], v[1]) and not torch.equal(v_noH[0], v[0])
    assert torch.equal(v_noL[0], v[0]) and not torch.equal(v_noL[1], v[1])
    torch.testing.assert_close(v_noL[1], ((s[1] + s[0] + x) @ Bm.T - s[1]).float(), rtol=1e-6, atol=1e-6)

    # Difference mode (d, -d): eigenvector of J_V with eigenvalue -1, for any d
    d = _randn(g, B, n)
    p = (0.3 * d, -0.3 * d)
    s2 = _add(s, p)
    v2 = field(*s2)
    torch.testing.assert_close(v2[0] - v[0], (-p[0]).float(), rtol=0, atol=1e-5)
    torch.testing.assert_close(v2[1] - v[1], (-p[1]).float(), rtol=0, atol=1e-5)
    q, _ = osl_quotient(v2, v, s2, s)
    torch.testing.assert_close(q, torch.full((B,), -1.0), rtol=0, atol=2e-5)

    # Sum mode (u_i, u_i) with Bm u_i = lam_i u_i: osl quotient 2 lam_i - 1 (one eigenvector per row)
    U = Q.T                                            # rows are eigenvectors
    p = (0.3 * U, 0.3 * U)
    s2 = _add(s, p)
    q, _ = osl_quotient(field(*s2), v, s2, s)
    torch.testing.assert_close(q, (2 * lam - 1).float(), rtol=0, atol=2e-5)

    # Benign spectrum: the v2 adversary (fallback shift) converges to max(-1, 2 lam_max(Bm) - 1) = 0.8,
    # where the F gain along the sum-mode eigenvector is 2 lam_max(Bm) = 1.8
    fz = _trm_field(Bm)
    Bk = 8
    s = _state(g, Bk, (n,), scale=0.2)
    v_s = fz(*s)
    dd = _dir(g, Bk, (n,))
    for k in range(400):
        dd, gain_F = v2_power_step(fz, s, v_s, _scale(dd, 0.2))
    s2 = _add(s, _scale(dd, 0.2))
    q, _ = osl_quotient(fz(*s2), v_s, s2, s)
    target = max(-1.0, 2 * lam.max().item() - 1)
    torch.testing.assert_close(q, torch.full((Bk,), target), rtol=0, atol=1e-2)
    torch.testing.assert_close(gain_F, torch.full((Bk,), 2 * lam.max().item()), rtol=0, atol=1e-2)


# =============================================================================================
# (6) osl_quotient of a damped rotation field
# =============================================================================================

def test_osl_quotient_skew_field_is_minus_one():
    g = _gen(6)
    B, n = 10, 8
    N = 2 * n
    K = _skew(g, N, 2.5)
    field = _linear_op(-torch.eye(N, dtype=DT) + K)
    a = _scale(_state(g, B, (n,)), 10 ** (2 * torch.rand(B, generator=g, dtype=DT) - 1))
    b = _add(a, _scale(_dir(g, B, (n,)), 0.1 * rownorm2(*a).sqrt()))
    q, den2 = osl_quotient(field(*a), field(*b), a, b)
    torch.testing.assert_close(q, torch.full((B,), -1.0), rtol=0, atol=1e-5)
    torch.testing.assert_close(den2, _cat(a).sub(_cat(b)).square().sum(-1).float(), rtol=1e-5, atol=0)

    for mu in (0.0, 0.5, 0.9, 0.99):
        assert torch.equal(F.relu(q + mu), torch.zeros(B)), mu
    for mu in (1.01, 1.5, 2.0):
        pen = F.relu(q + mu)
        assert (pen > 0).all(), mu
        torch.testing.assert_close(pen, torch.full((B,), mu - 1.0), rtol=0, atol=1e-5)

    # sym(J_V) = -I: every direction is optimal; the v2 step must stay well defined (no NaN, q = -1), with
    # and without an explicit shift; F = V + I = K, so gain_F = ||K d|| / ||d||
    s = _state(g, B, (n,))
    d0 = _dir(g, B, (n,))
    for shift in (None, torch.full((B,), 1.0 + 2.5)):
        d, gain = v2_power_step(field, s, field(*s), _scale(d0, 0.1), shift=shift)
        assert all(torch.isfinite(t).all() for t in d) and torch.isfinite(gain).all()
        torch.testing.assert_close(gain.to(DT), _jnorm2(K, d0).sqrt(), rtol=1e-4, atol=0)
        s2 = _add(s, _scale(d, 0.1))
        torch.testing.assert_close(osl_quotient(field(*s2), field(*s), s2, s)[0], torch.full((B,), -1.0),
                                   rtol=0, atol=1e-5)


# =============================================================================================
# (7) rownorm2 / rowdot / normalize_rows per-row semantics and dtypes
# =============================================================================================

def test_rownorm2_rowdot_normalize_rows_semantics():
    g = _gen(7)
    B, L, D = 5, 3, 4
    a32 = torch.randn(B, L, D, generator=g)
    b32 = torch.randn(B, L, D, generator=g)
    a16, b16 = a32.bfloat16(), b32.bfloat16()

    # Joint per-row norm over every block, accumulated in fp32 whatever the input dtype
    n2 = rownorm2(a16, b32)
    assert n2.shape == (B,) and n2.dtype == torch.float32
    ref = a16.float().flatten(1).square().sum(-1) + b32.flatten(1).square().sum(-1)
    torch.testing.assert_close(n2, ref, rtol=1e-6, atol=0)
    assert rownorm2(a16).dtype == torch.float32 and rownorm2(a16, b16).dtype == torch.float32
    assert rownorm2(a32.double()).dtype == torch.float32

    dot = rowdot((a16, b32), (b16, a32))
    assert dot.shape == (B,) and dot.dtype == torch.float32
    torch.testing.assert_close(dot, (a16.float() * b16.float() + b32 * a32).flatten(1).sum(-1), rtol=1e-5, atol=1e-5)

    # normalize_rows: every row of the *joint* state has unit norm, rows are independent, zero rows stay 0
    scales = torch.tensor([1e-3, 1.0, 1e3, 0.0, 7.0])
    x = (a16 * scales.view(-1, 1, 1).bfloat16(), b16 * scales.view(-1, 1, 1).bfloat16())
    y = normalize_rows(x)
    assert all(t.dtype == torch.float32 and t.shape == (B, L, D) for t in y)
    yn = rownorm2(*y)
    torch.testing.assert_close(yn[scales > 0], torch.ones(4), rtol=1e-5, atol=0)
    assert torch.equal(yn[3], torch.tensor(0.0)) and all(torch.isfinite(t).all() for t in y)
    # direction preserved; per-row: rescaling one row leaves every other row unchanged
    xf = _cat(tuple(t.float() for t in x))
    for i in (0, 1, 2, 4):
        torch.testing.assert_close(_cat(y)[i], xf[i] / xf[i].norm(), rtol=1e-5, atol=1e-6)
    x2 = (x[0].clone(), x[1].clone())
    x2[0][1] *= 50
    x2[1][1] *= 50
    y2 = normalize_rows(x2)
    for i in (0, 2, 3, 4):
        assert torch.equal(y2[0][i], y[0][i]) and torch.equal(y2[1][i], y[1][i]), i
    torch.testing.assert_close(_cat(y2)[1], _cat(y)[1], rtol=0, atol=1e-2)   # bf16 rescale rounding only

    # float64 inputs keep float64 directions (norm in fp32)
    y64 = normalize_rows((a32.double(), b32.double()))
    assert y64[0].dtype == torch.float64
    torch.testing.assert_close(rownorm2(*y64), torch.ones(B), rtol=1e-6, atol=0)


# =============================================================================================
# (8) Gradient efficacy of both penalties on parametrised toy maps
# =============================================================================================

def test_v1_penalty_drives_spectral_norm_below_target():
    g = _gen(8)
    B, n = 32, 8
    N = 2 * n
    W0 = _randn(g, N, N)
    W = torch.nn.Parameter(W0 / torch.linalg.matrix_norm(W0, 2) * 2.0)
    opt = torch.optim.Adam([W], lr=1e-2)

    def phi(*s):
        return _split(torch.tanh(_cat(s) @ W.T), s)

    target = 0.9
    d = _dir(g, B, (n,))
    first_loss = None
    for _ in range(300):
        s = _state(g, B, (n,), scale=0.05)            # fresh states near 0 (tanh' ~ 1)
        r = 0.1 * rownorm2(*s).sqrt()
        with torch.no_grad():
            phi_s = phi(*s)
        d = v1_power_step(phi, s, phi_s, _scale(d, r))   # warm-started adversary
        s2 = _add(s, _scale(d, r))
        ratio, _ = secant_ratio(phi(*s), phi(*s2), s, s2)
        loss = F.relu(torch.log(ratio) - math.log(target)).mean()   # the regularizer's (log) V1 penalty
        first_loss = loss.item() if first_loss is None else first_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first_loss > 0.5
    final = torch.linalg.matrix_norm(W.detach(), 2).item()
    assert final < target + 0.05, final


@pytest.mark.parametrize("adversary", ["fallback_shift", "regularizer_shift"])
def test_v2_penalty_drives_osl_below_minus_mu(adversary):
    g = _gen(9)
    B, n = 32, 8
    N = 2 * n
    I = torch.eye(N, dtype=DT)
    W0 = _randn(g, N, N)
    W = torch.nn.Parameter(W0 / torch.linalg.matrix_norm(W0, 2) * 2.0)
    opt = torch.optim.Adam([W], lr=1e-2)

    def field(*s):
        z = _cat(s)
        return _split(z @ W.T - z, s)

    def lam_max():
        Wd = W.detach()
        return torch.linalg.eigvalsh((Wd + Wd.T) / 2 - I).max().item()

    mu = 0.1
    assert lam_max() > 0.0          # starts violating
    d, dF = _dir(g, B, (n,)), _dir(g, B, (n,))
    gain = torch.zeros(B)
    for _ in range(300):
        s = _state(g, B, (n,))
        r = 0.1 * rownorm2(*s).sqrt()
        with torch.no_grad():
            v_s = field(*s)
        if adversary == "fallback_shift":
            d, _ = v2_power_step(field, s, v_s, _scale(d, r))
        else:
            # as ContractionRegularizer._v2: ||J_F|| power step, then a sym(J_V) step shifted by
            # c = 1 + 1.5 * running max of the F gains
            dF, gF = f_gain_power_step(field, s, v_s, _scale(dF, r))
            gain = torch.maximum(gain, gF)
            d, gV = v2_power_step(field, s, v_s, _scale(d, r), shift=1.0 + 1.5 * gain)
            gain = torch.maximum(gain, gV)
        s2 = _add(s, _scale(d, r))
        q, _ = osl_quotient(field(*s), field(*s2), s, s2)
        loss = F.relu(q + mu).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = lam_max()
    assert final < -mu + 0.05, final


# =============================================================================================
# (9) ContractionRegularizer on a linear stub of the operator API (known spectra)
# =============================================================================================

class _LinearInner:
    """Minimal operator API consumed by ContractionRegularizer (config, h_H, seq_info, time_embeddings,
    substep_index, cycle, field), with Phi(s) = M s and V(s) = A s on the concatenated (z_H, z_L)."""

    def __init__(self, M=None, A=None, H_cycles=2, L_cycles=2):
        self.config = SimpleNamespace(H_cycles=H_cycles, L_cycles=L_cycles)
        self.h_H = 1.0
        self._M, self._A = M, A

    def seq_info(self):
        return {}

    def time_embeddings(self):
        return None

    def substep_index(self, i, j):
        return i * (self.config.L_cycles + 1) + j

    def cycle(self, z_H, z_L, x, temb, i, seq_info):
        return _linear_op(self._M)(z_H, z_L)

    def field(self, z_H, z_L, x, temb_H, temb_L, seq_info):
        assert temb_H is None and temb_L is None
        return tuple(t.float() for t in _linear_op(self._A)(z_H, z_L))


def _stub_outs(g, B, n, scale=0.2):
    s = tuple(t.float() for t in _state(g, B, (n,), scale=scale))
    return dict(reg_s_H=s[0], reg_s_L=s[1], reg_phi_H=s[0].clone(), reg_phi_L=s[1].clone(),
                reg_x=torch.zeros_like(s[0]))


def _run_stub(reg, outs, B, n_calls):
    """Same puzzle in every row for n_calls consecutive steps (birth stays 1): warm starts carry over."""
    hist = []
    for t in range(1, n_calls + 1):
        reg.stash(outs, torch.full((B,), t, dtype=torch.int32), t)
        _, m = reg(step=t)
        hist.append(m)
    return hist


@pytest.mark.parametrize("case", ["indefinite", "sym_negdef_wide", "trm_structured_wide"])
def test_regularizer_v2_adversary_on_linear_stub_finds_lambda_max(case):
    """The full V2 loop (warm-started ||J_F|| step, reg_v2_power_iters shifted steps with c = 1 + 1.5 * running
    max gain, per-row memory) converges to lambda_max(sym J_V), including the wide negative spectra on which an
    unshifted / under-shifted iteration stalls; the logged F gain converges to ||J_F|| = ||A + I||."""
    g = _gen(14)
    B, n = 8, 16
    N = 2 * n
    if case == "indefinite":
        A = _sym_with_spectrum(g, torch.linspace(-3.0, 1.5, N, dtype=DT)) + _skew(g, N, 1.0)
    else:
        A = _wide_negative_field(case, g, n)
    lam_max = torch.linalg.eigvalsh((A + A.T) / 2).max().item()
    norm_F = _jf_norm(A)
    reg = ContractionRegularizer(RegConfig(reg_mode="flow_mono", reg_frac=1.0, reg_mu=0.0, reg_monitor_every=0),
                                 _LinearInner(A=A), batch_size=B, seed=0, device="cpu")
    assert reg.k == B and reg.cfg.reg_v2_power_iters == 3 and reg.cfg.reg_v2_shift_factor == 1.5
    hist = _run_stub(reg, _stub_outs(g, B, n), B, n_calls=150)
    q_hist = torch.tensor([m["reg/value"] for m in hist])
    print(f"\n[reg v2 stub {case}] lam_max={lam_max:.4f} q: first={q_hist[0]:.4f} last={q_hist[-1]:.4f}, "
          f"gain_F last={hist[-1]['reg/gain_F']:.4f} (||A+I||={norm_F:.4f})")
    last = hist[-1]
    assert last["reg/value"] == pytest.approx(lam_max, abs=1e-3)
    assert last["reg/value_max"] <= lam_max + 1e-4             # the secant quotient of a linear field <= lambda_max
    assert last["reg/gain_F"] == pytest.approx(norm_F, rel=1e-3)
    assert torch.all(reg.mem["gain_F"] <= norm_F * (1 + 1e-5))
    # once the gain estimate has converged the shift is guaranteed: 1 + 1.5 * gain >= 1 + ||J_F|| >= -lambda_min
    assert 1 + 1.5 * reg.mem["gain_F"].min().item() >= 1 + norm_F
    assert q_hist[-1] > q_hist[0]


def test_regularizer_v1_adversary_on_linear_stub_finds_spectral_norm():
    """One J^T J step per call (reg_v1_power_iters = 1): only the per-row warm start carries the iteration
    to ||Phi||; the log penalty is scale-free (Phi -> 10 Phi shifts it by exactly log 10)."""
    g = _gen(15)
    B, n = 8, 16
    N = 2 * n
    # small spectral gap (second singular value ~2.7): a single power step is far from converged
    sv = torch.cat([torch.tensor([3.0], dtype=DT), 0.5 + 2.2 * torch.rand(N - 1, generator=g, dtype=DT)])
    M = _spectral_op(g, N, sv)
    outs = _stub_outs(g, B, n)
    hists = []
    for scale in (1.0, 10.0):
        reg = ContractionRegularizer(RegConfig(reg_mode="cycle_lip", reg_frac=1.0, reg_monitor_every=0),
                                     _LinearInner(M=scale * M), batch_size=B, seed=0, device="cpu")
        assert reg.cfg.reg_v1_power_iters == 1
        hists.append(_run_stub(reg, outs, B, n_calls=80))
    hist, hist10 = hists
    print(f"\n[reg v1 stub] ratio: first={hist[0]['reg/value']:.4f} last={hist[-1]['reg/value']:.4f} (||M||=3)")
    assert hist[0]["reg/value"] < 0.95 * 3.0                   # a single cold step is far from Lip
    assert hist[-1]["reg/value"] == pytest.approx(3.0, rel=1e-3)
    assert hist[-1]["reg/value_max"] <= 3.0 * (1 + 1e-4)
    # log penalty: pen = mean(log ratio - log rho) when every row violates
    assert hist[-1]["reg/pen"] == pytest.approx(math.log(3.0 / reg.rho), rel=1e-3)
    assert "reg/gain_F" not in hist[-1]
    # Scale-free: same directions (linear map), ratio x 10, penalty + log 10 at every call
    for m1, m10 in zip(hist, hist10):
        assert m10["reg/value"] == pytest.approx(10 * m1["reg/value"], rel=1e-4)
        assert m10["reg/pen"] == pytest.approx(m1["reg/pen"] + math.log(10.0), rel=1e-4)


# =============================================================================================
# (10) ContractionRegularizer end to end on a tiny trm_contractive model (CPU)
# =============================================================================================

BATCH = 8
REG_KEYS = ("reg_s_H", "reg_s_L", "reg_phi_H", "reg_phi_L", "reg_x")
MEM_KEYS = {"dir", "dir_F", "gain_F", "birth"}
SD_KEYS = {"lam", "viol_ema", "n_calls", "gen", "mem"}
STATE_SHAPE = (BATCH, 2, 81 + 16, 64)


def _tiny_cfg(**over):
    cfg = dict(batch_size=BATCH, seq_len=81, puzzle_emb_ndim=64, num_puzzle_identifiers=4, vocab_size=11,
               H_cycles=2, L_cycles=2, H_layers=0, L_layers=2, hidden_size=64, expansion=4, num_heads=2,
               pos_encodings="none", halt_max_steps=4, halt_exploration_prob=0.1, forward_dtype="float32",
               mlp_t=True, puzzle_emb_len=16, no_ACT_continue=True,
               flow=False, flow_h_L=0.5, flow_h_H=0.5, flow_time_embed=True, flow_time_freqs=16)
    cfg.update(over)
    return cfg


def _batch(seed=0):
    g = _gen(seed)
    return dict(inputs=torch.randint(1, 11, (BATCH, 81), generator=g, dtype=torch.int32),
                labels=torch.randint(1, 11, (BATCH, 81), generator=g, dtype=torch.int32),
                puzzle_identifiers=torch.randint(0, 4, (BATCH,), generator=g, dtype=torch.int32))


def _build(seed=0, **over):
    torch.manual_seed(seed)
    head = ACTLossHead(TinyRecursiveReasoningModel_ACTV1(_tiny_cfg(**over)), loss_type="stablemax_cross_entropy")
    head.train()
    return head


def _forward(head, batch):
    carry = head.initial_carry(batch)
    carry, loss, metrics, outputs, _ = head(carry=carry, batch=batch, return_keys=[])
    return carry, loss, metrics, outputs


def _stash(reg, outputs, steps, global_step=1):
    reg.stash(outputs, steps, global_step)


def _setup(reg_kw, seed=0, **arch):
    head = _build(seed, **arch)
    carry, _, metrics, outputs = _forward(head, _batch(seed))
    inner = head.model.inner
    reg = ContractionRegularizer(RegConfig(**reg_kw), inner, batch_size=BATCH, seed=seed, device="cpu")
    head.zero_grad(set_to_none=True)
    return head, inner, reg, carry, metrics, outputs


def _randomize_temb(inner, seed=7):
    """Non-zero time embedding (zero at init) so that the substep indexing actually matters."""
    with torch.no_grad():
        inner.temb_proj.weight.normal_(std=0.3, generator=_gen(seed))
        inner.temb_proj.bias.normal_(std=0.3, generator=_gen(seed + 1))


def _grad_norm(params):
    return sum(0.0 if p.grad is None else p.grad.float().norm().item() for p in params)


def _mem_clone(reg):
    return {k: v.clone() for k, v in reg.mem.items()}


def test_regconfig_defaults():
    cfg = RegConfig()
    assert cfg.reg_mode == "none" and cfg.reg_v1_power_iters == 1 and cfg.reg_v2_power_iters == 3
    assert cfg.reg_v2_shift_factor == 1.5 and cfg.reg_dual_lr == 0.002 and cfg.reg_dual_warmup == 2000


def test_model_exports_detached_anchors_consistent_with_cycle():
    head = _build(0, flow=True)
    carry, _, metrics, outputs = _forward(head, _batch(0))
    inner = head.model.inner
    for k in REG_KEYS:
        assert k in outputs and not outputs[k].requires_grad
    # fpres is summed over halted rows only (like every ACT metric): none halted after step 1 here
    assert not carry.halted.any() and metrics["fpres"] == 0
    # halt_max_steps = 1: every row halts, fpres = sum_rows ||Phi(s) - s|| / ||s||
    head1 = _build(0, flow=True, halt_max_steps=1)
    carry1, _, metrics1, out1 = _forward(head1, _batch(0))
    assert carry1.halted.all()
    res = rownorm2(out1["reg_phi_H"] - out1["reg_s_H"], out1["reg_phi_L"] - out1["reg_s_L"])
    ref = (res / rownorm2(out1["reg_s_H"], out1["reg_s_L"])).sqrt().sum()
    assert torch.isfinite(metrics1["fpres"]) and metrics1["fpres"] > 0
    torch.testing.assert_close(metrics1["fpres"], ref, rtol=1e-5, atol=0)
    # Phi(s) recomputed from the anchors with the public operator API reproduces reg_phi
    with torch.no_grad():
        zH, zL = inner.cycle(outputs["reg_s_H"], outputs["reg_s_L"], outputs["reg_x"], inner.time_embeddings(),
                             inner.config.H_cycles - 1, inner.seq_info())
    torch.testing.assert_close(zH, outputs["reg_phi_H"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(zL, outputs["reg_phi_L"], rtol=1e-5, atol=1e-5)
    # eval mode: anchors are not passed through
    head.eval()
    _, _, _, out_eval = _forward(head, _batch(0))
    assert not any(k.startswith("reg_") for k in out_eval)


ARCHS = {
    "mlp_t": dict(),
    "attn_rope": dict(mlp_t=False, pos_encodings="rope"),
    "flow_temb": dict(flow=True),
    "mlp_t_bf16": dict(forward_dtype="bfloat16"),
}


@pytest.mark.parametrize("arch", list(ARCHS), ids=list(ARCHS))
def test_e2e_cycle_lip(arch):
    # reg_eta = 0.99 -> rho = 0.01: every row violates, so the penalty is active and must carry gradient
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="cycle_lip", reg_eta=0.99, reg_monitor_every=0),
                                                 **ARCHS[arch])
    assert reg.k == 2 and reg.rho == pytest.approx(0.01)
    _stash(reg, outputs, carry.steps)
    reg_loss, m = reg(step=1)
    assert reg.pending is None and reg.n_calls == 1
    assert reg_loss is not None and reg_loss.requires_grad and reg_loss.ndim == 0 and torch.isfinite(reg_loss)
    for k in ("reg/lambda", "reg/viol", "reg/value", "reg/value_max", "reg/pen", "reg/viol_ema", "reg/eps_rel",
              "reg/pair_rel_dist"):
        assert k in m and math.isfinite(m[k]), k
    assert "reg/gain_F" not in m
    assert m["reg/viol"] == 1.0 and m["reg/value"] > reg.rho
    # log penalty (every row violates): pen = mean(log ratio) - log rho <= log(mean ratio / rho) (Jensen)
    assert 0 < m["reg/pen"] <= math.log(m["reg/value"] / reg.rho) + 1e-5
    assert m["reg/pen"] >= math.log(m["reg/value_max"] / reg.rho) / reg.k - 1e-5
    assert reg_loss.item() == pytest.approx(m["reg/lambda"] * m["reg/pen"], rel=1e-5)
    assert reg.cfg.reg_eps_min <= m["reg/eps_rel"] <= reg.cfg.reg_eps_max
    # |s~ - s| = eps * ||s|| (exact up to rounding of the perturbed state in the forward dtype)
    tol = 2e-2 if arch.endswith("bf16") else 1e-4
    assert m["reg/pair_rel_dist"] == pytest.approx(m["reg/eps_rel"], rel=tol)

    reg_loss.backward()
    for name, p in inner.L_level.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    if inner.use_time_embed:
        assert _grad_norm(inner.temb_proj.parameters()) > 0
    # warm-start memory: the regularized rows hold unit directions and their birth, the others are untouched;
    # V1 never touches the V2 entries
    assert reg.mem is not None and set(reg.mem) == MEM_KEYS
    assert reg.mem["dir"].shape == STATE_SHAPE and reg.mem["dir_F"].shape == STATE_SHAPE
    assert reg.mem["gain_F"].shape == (BATCH,) and reg.mem["birth"].shape == (BATCH,)
    assert reg.mem["birth"].dtype == torch.long
    n2 = rownorm2(reg.mem["dir"])
    idx_mask = n2 > 0
    assert int(idx_mask.sum()) == reg.k
    torch.testing.assert_close(n2[idx_mask], torch.ones(reg.k), rtol=1e-5, atol=0)
    assert torch.equal(reg.mem["birth"], torch.where(idx_mask, 1, -1))   # birth = global_step - steps + 1 = 1
    assert torch.count_nonzero(reg.mem["dir_F"]) == 0 and torch.count_nonzero(reg.mem["gain_F"]) == 0


@pytest.mark.parametrize("arch", ["flow_temb", "flow_no_temb", "trm"])
def test_e2e_flow_mono(arch):
    a = dict(flow_temb=dict(flow=True), flow_no_temb=dict(flow=True, flow_time_embed=False), trm=dict())[arch]
    # reg_mu = 50 >> |osl|: every row violates
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="flow_mono", reg_mu=50.0, reg_monitor_every=0), **a)
    assert reg.mu == 50.0
    _stash(reg, outputs, carry.steps)
    reg_loss, m = reg(step=1)
    assert reg_loss is not None and reg_loss.requires_grad and torch.isfinite(reg_loss)
    for k in ("reg/lambda", "reg/viol", "reg/value", "reg/gain_F"):
        assert k in m and math.isfinite(m[k]), k
    assert m["reg/viol"] == 1.0
    assert m["reg/pen"] == pytest.approx(m["reg/value"] + 50.0, rel=1e-5)
    assert m["reg/gain_F"] > 0       # running max of secant F gains (a lower bound of ||J_F||)
    reg_loss.backward()
    for name, p in inner.L_level.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    if arch == "flow_temb":
        assert inner.temb_proj.weight.grad is not None and inner.temb_proj.weight.grad.abs().sum() > 0
        assert inner.temb_proj.bias.grad is not None and inner.temb_proj.bias.grad.abs().sum() > 0
    else:
        assert not hasattr(inner, "temb_proj")
    # warm-start memory: both directions (unit) and the running F gain are stored for the regularized rows
    assert set(reg.mem) == MEM_KEYS
    idx_mask = reg.mem["birth"] == 1
    assert int(idx_mask.sum()) == reg.k and torch.all(reg.mem["birth"][~idx_mask] == -1)
    for key in ("dir", "dir_F"):
        n2 = rownorm2(reg.mem[key])
        torch.testing.assert_close(n2[idx_mask], torch.ones(reg.k), rtol=1e-5, atol=0)
        assert torch.all(n2[~idx_mask] == 0)
    assert torch.all(reg.mem["gain_F"][idx_mask] > 0) and torch.all(reg.mem["gain_F"][~idx_mask] == 0)
    assert reg.mem["gain_F"][idx_mask].mean().item() == pytest.approx(m["reg/gain_F"], rel=1e-6)


def test_e2e_default_config_finite():
    for mode, arch in (("cycle_lip", dict()), ("flow_mono", dict(flow=True))):
        head, inner, reg, carry, _, outputs = _setup(dict(reg_mode=mode), **arch)
        assert reg.mu == pytest.approx(-math.log(0.95) / inner.h_H)
        _stash(reg, outputs, carry.steps)
        reg_loss, m = reg(step=1)
        assert reg_loss is not None and torch.isfinite(reg_loss) and reg_loss.item() >= 0
        reg_loss.backward()
        assert all(math.isfinite(v) for v in m.values())


def test_e2e_reg_mode_none_and_monitor():
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="none", reg_monitor_every=50), flow=True)
    assert reg(step=1) == (None, {})            # nothing stashed yet
    _stash(reg, outputs, carry.steps)
    reg_loss, m = reg(step=1)                   # not a monitor step
    assert reg_loss is None and m == {}
    assert reg.mem is None                      # inactive arm never builds warm-start state

    rng_before = torch.get_rng_state()
    _stash(reg, outputs, carry.steps)
    reg_loss, m = reg(step=50)                  # monitor step
    assert torch.equal(torch.get_rng_state(), rng_before), "regularizer consumed the global RNG"
    assert reg_loss is None
    expected = {"mon/v1_ratio", "mon/v1_ratio_p90", "mon/v1_frac_gt_rho", "mon/v1_frac_gt1", "mon/v2_q",
                "mon/v2_q_p90", "mon/v2_frac_gt_negmu", "mon/gain_F", "mon/fpres", "mon/x_over_z"}
    assert expected <= set(m)
    assert all(math.isfinite(m[k]) for k in expected)
    assert m["mon/v1_ratio"] > 0 and m["mon/fpres"] > 0 and 0 <= m["mon/v1_frac_gt1"] <= 1
    assert m["mon/gain_F"] > 0
    assert reg.mem is None                      # monitoring is cold-started, it keeps no warm-start state
    # monitoring is gradient-free: no .grad was populated
    assert all(p.grad is None for p in head.parameters())


@pytest.mark.parametrize("mode", ["cycle_lip", "flow_mono"])
def test_e2e_global_rng_untouched_by_active_regularizer(mode):
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode=mode, reg_monitor_every=1), flow=True)
    rng_before = torch.get_rng_state()
    _stash(reg, outputs, carry.steps)
    reg_loss, m = reg(step=1)
    reg_loss.backward()
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert "mon/v1_ratio" in m and "mon/v2_q" in m and "reg/value" in m


def test_e2e_v1_quantity_matches_manual_secant():
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="cycle_lip", reg_monitor_every=0))
    _stash(reg, outputs, carry.steps)
    p = reg.pending
    s, x = p["s"], p["x"]
    g = _gen(10)
    d = normalize_rows((torch.randn(s[0].shape, generator=g), torch.randn(s[1].shape, generator=g)))
    r = 0.1 * rownorm2(*s).sqrt()
    pen, ratio, den2, d_out = reg._v1(s, x, d, r, power_iters=0, with_grad=False)
    assert all(torch.equal(a, b) for a, b in zip(d_out, d))       # no power step: direction unchanged
    s2 = tuple(a + b * r.view(-1, 1, 1) for a, b in zip(s, d))
    with torch.no_grad():
        si, te, i = inner.seq_info(), inner.time_embeddings(), inner.config.H_cycles - 1
        ref, _ = secant_ratio(inner.cycle(*s2, x, te, i, si), inner.cycle(*s, x, te, i, si), s2, s)
    torch.testing.assert_close(ratio, ref, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(den2, r.square(), rtol=1e-4, atol=0)
    # V1 penalty is ReLU(log ratio - log rho) (val stays the ratio); same zero set as ReLU(ratio - rho)
    torch.testing.assert_close(pen, F.relu(torch.log(ref) - math.log(reg.rho)), rtol=1e-4, atol=1e-6)
    assert torch.equal(pen > 0, ratio > reg.rho)


def test_e2e_v2_quantity_matches_manual_osl_at_last_cycle_substeps():
    """V2 integrates z_L at a random L-substep j of the LAST H-cycle and z_H at that cycle's H-substep."""
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="flow_mono", reg_frac=1.0, reg_monitor_every=0),
                                                 flow=True, H_cycles=3)
    _randomize_temb(inner)
    L, i_last = inner.config.L_cycles, inner.config.H_cycles - 1
    assert reg.i_last == i_last == 2
    temb = inner.time_embeddings()
    D = temb.shape[-1]

    # _field_times: explicit indices i * (L + 1) + j (independent of substep_index)
    j = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    te_H, te_L = reg._field_times(temb, j)
    assert te_H.shape == (1, 1, D) and torch.equal(te_H[0, 0], temb[i_last * (L + 1) + L])
    assert te_L.shape == (BATCH, 1, D)
    for row, jj in enumerate(j.tolist()):
        assert torch.equal(te_L[row, 0], temb[i_last * (L + 1) + jj])
    assert reg._field_times(None, j) == (None, None)

    # _v2 with power_iters=0 evaluates exactly the osl quotient of inner.field at those times
    seen = []
    orig = reg._field_times

    def spy(temb_, j_):
        seen.append(j_.clone())
        return orig(temb_, j_)
    reg._field_times = spy

    _stash(reg, outputs, carry.steps)
    p = reg.pending
    s, x = p["s"], p["x"]
    g = _gen(16)
    d = normalize_rows((torch.randn(s[0].shape, generator=g), torch.randn(s[1].shape, generator=g)))
    dF = normalize_rows((torch.randn(s[0].shape, generator=g), torch.randn(s[1].shape, generator=g)))
    gain0 = torch.rand(BATCH, generator=g)
    r = 0.1 * rownorm2(*s).sqrt()
    pen, q, den2, d_out, dF_out, gain_out = reg._v2(s, x, d, dF, gain0, r, power_iters=0, with_grad=False)
    assert all(torch.equal(a, b) for a, b in zip(d_out, d)) and all(torch.equal(a, b) for a, b in zip(dF_out, dF))
    assert torch.equal(gain_out, gain0)
    assert len(seen) == 2 and torch.equal(seen[0], seen[1])          # same substep for power steps and penalty
    jr = seen[0]
    assert jr.shape == (BATCH,) and int(jr.min()) >= 0 and int(jr.max()) < L
    assert len(set(jr.tolist())) > 1, "all rows drew the same substep: per-row times not exercised"

    s2 = tuple(a + b * r.view(-1, 1, 1) for a, b in zip(s, d))
    si = inner.seq_info()
    with torch.no_grad():
        tH = temb[i_last * (L + 1) + L]
        tL = temb[i_last * (L + 1) + jr].unsqueeze(1)
        ref, ref_den2 = osl_quotient(inner.field(*s2, x, tH, tL, si), inner.field(*s, x, tH, tL, si), s2, s)
        # wrong times (first cycle, or a single substep for every row) give a different quotient
        q_first, _ = osl_quotient(inner.field(*s2, x, temb[L], temb[jr].unsqueeze(1), si),
                                  inner.field(*s, x, temb[L], temb[jr].unsqueeze(1), si), s2, s)
        t0 = temb[i_last * (L + 1)].view(1, 1, -1)
        q_j0, _ = osl_quotient(inner.field(*s2, x, tH, t0, si), inner.field(*s, x, tH, t0, si), s2, s)
    torch.testing.assert_close(q, ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(den2, ref_den2, rtol=1e-5, atol=0)
    torch.testing.assert_close(pen, F.relu(ref + reg.mu), rtol=1e-4, atol=1e-5)
    assert (q - q_first).abs().min() > 1e-4
    moved = jr != 0
    assert (q[moved] - q_j0[moved]).abs().min() > 1e-4
    torch.testing.assert_close(q[~moved], q_j0[~moved], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("mode", ["cycle_lip", "flow_mono"])
def test_e2e_power_iteration_counts_and_v2_shift(mode, monkeypatch):
    """V1: reg_v1_power_iters J^T J steps. V2: one ||J_F|| step, then reg_v2_power_iters sym(J_V) steps with
    shift = 1 + reg_v2_shift_factor * running max of the F gains (memory, ||J_F|| step, earlier V2 steps)."""
    calls = []
    real_v1, real_v2, real_f = regmod.v1_power_step, regmod.v2_power_step, regmod.f_gain_power_step

    def v1(*a, **kw):
        calls.append(("v1",))
        return real_v1(*a, **kw)

    def fg(*a, **kw):
        out = real_f(*a, **kw)
        calls.append(("f", out[1].clone()))
        return out

    def v2(field, s, v_s, delta, shift=None):
        assert shift is not None, "the regularizer must pass an explicit shift"
        out = real_v2(field, s, v_s, delta, shift=shift)
        calls.append(("v2", shift.clone(), out[1].clone()))
        return out

    monkeypatch.setattr(regmod, "v1_power_step", v1)
    monkeypatch.setattr(regmod, "f_gain_power_step", fg)
    monkeypatch.setattr(regmod, "v2_power_step", v2)

    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode=mode, reg_v1_power_iters=3, reg_v2_power_iters=4,
                                                      reg_v2_shift_factor=2.0, reg_frac=1.0, reg_monitor_every=0),
                                                 flow=True)
    for t in (1, 2):                            # same puzzles at t = 2 (birth 1): gain memory carries over
        calls.clear()
        if t == 2 and mode == "flow_mono":
            reg.mem["gain_F"].fill_(50.0)       # a remembered gain far above any fresh estimate must set the shift
        _stash(reg, outputs, carry.steps + (t - 1), t)
        idx = reg.pending["idx"].clone()
        gain_mem = reg.mem["gain_F"][idx].clone() if reg.mem is not None else torch.zeros(reg.k)
        _, m = reg(step=t)
        if mode == "cycle_lip":
            assert [c[0] for c in calls] == ["v1"] * 3
            continue
        assert [c[0] for c in calls] == ["f"] + ["v2"] * 4
        run = torch.maximum(gain_mem, calls[0][1])
        for c in calls[1:]:
            torch.testing.assert_close(c[1], 1.0 + 2.0 * run, rtol=1e-6, atol=0)
            run = torch.maximum(run, c[2])
        torch.testing.assert_close(reg.mem["gain_F"][idx], run, rtol=0, atol=0)
        assert m["reg/gain_F"] == pytest.approx(run.mean().item(), rel=1e-6)
        if t == 2:
            assert torch.all(gain_mem == 50.0) and torch.all(calls[0][1] < 50.0)
            assert all(torch.all(c[1] == 1.0 + 2.0 * 50.0) for c in calls[1:])

    # random pairs: no power step at all
    calls.clear()
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode=mode, reg_pair="rand", reg_monitor_every=0), flow=True)
    _stash(reg, outputs, carry.steps)
    reg(step=1)
    assert calls == []
    # the monitor runs reg_monitor_power_iters cold-started steps of both adversaries
    calls.clear()
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="none", reg_monitor_every=1,
                                                      reg_monitor_power_iters=2), flow=True)
    _stash(reg, outputs, carry.steps)
    reg(step=1)
    assert [c[0] for c in calls] == ["v1", "v1", "f", "v2", "v2"]


@pytest.mark.parametrize("mode", ["cycle_lip", "flow_mono"])
def test_e2e_adversary_beats_random_pairs(mode):
    """On the real model random pairs look (nearly) contractive while the power-iterated pairs expose expansion."""
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode=mode, reg_frac=1.0, reg_monitor_every=0), flow=True)
    _randomize_temb(inner)
    _stash(reg, outputs, carry.steps)
    p = reg.pending
    s, x = p["s"], p["x"]
    r, _ = reg._radius(s)
    d0 = reg._rand_dir(s)
    st = reg.gen.get_state()
    if mode == "cycle_lip":
        _, v_rand, _, _ = reg._v1(s, x, d0, r, 0, with_grad=False)
        _, v_adv, _, _ = reg._v1(s, x, d0, r, 10, with_grad=False)
    else:
        z = torch.zeros(reg.k)
        reg.gen.set_state(st)                   # same substeps j for both
        _, v_rand, *_ = reg._v2(s, x, d0, d0, z, r, 0, with_grad=False)
        reg.gen.set_state(st)
        _, v_adv, *_ = reg._v2(s, x, d0, d0, z, r, 10, with_grad=False)
    print(f"\n[{mode}] random {v_rand.mean():.3f}  adversarial {v_adv.mean():.3f}")
    assert torch.all(v_adv > v_rand + 0.5)


def _dual_setup(**kw):
    # EMA factor 1.0 freezes viol_ema, so the dual step is driven purely by the value we set
    return _setup(dict(reg_mode="cycle_lip", reg_viol_ema=1.0, reg_dual_lr=0.5, reg_target_viol=0.1,
                       reg_monitor_every=0, **kw))


def test_e2e_dual_ascent_direction_and_clipping():
    head, inner, reg, carry, _, outputs = _dual_setup(reg_dual_warmup=0)

    def step(global_step):
        _stash(reg, outputs, carry.steps, global_step)
        lam_used = reg.lam
        loss, m = reg(step=global_step)
        assert m["reg/lambda"] == lam_used
        assert loss.item() == pytest.approx(lam_used * m["reg/pen"], rel=1e-5, abs=1e-12)
        return lam_used, reg.lam

    reg.viol_ema = 0.6
    lams = [reg.lam]
    for t in range(1, 5):
        before, after = step(t)
        assert after == pytest.approx(before * math.exp(0.5 * (0.6 - 0.1)), rel=1e-12)
        lams.append(after)
    assert all(b > a for a, b in zip(lams, lams[1:]))

    reg.viol_ema = 0.0
    lams = [reg.lam]
    for t in range(5, 9):
        before, after = step(t)
        assert after == pytest.approx(before * math.exp(0.5 * (0.0 - 0.1)), rel=1e-12)
        lams.append(after)
    assert all(b < a for a, b in zip(lams, lams[1:]))

    # Clipping to [lambda_min, lambda_max] (viol_ema stays in [0, 1]; a large dual lr forces the bounds)
    reg.cfg = reg.cfg.model_copy(update=dict(reg_dual_lr=100.0))
    reg.viol_ema = 1.0
    step(9)
    assert reg.lam == reg.cfg.reg_lambda_max
    reg.viol_ema = 0.0
    step(10)
    assert reg.lam == reg.cfg.reg_lambda_min

    # reg_dual = False: lambda is frozen
    reg.cfg = reg.cfg.model_copy(update=dict(reg_dual=False))
    reg.viol_ema, lam = 0.9, reg.lam
    step(11)
    assert reg.lam == lam


@pytest.mark.parametrize("warmup", [5, None], ids=["warmup5", "default"])
def test_e2e_dual_ascent_frozen_during_warmup(warmup):
    head, inner, reg, carry, _, outputs = _dual_setup(**({} if warmup is None else dict(reg_dual_warmup=warmup)))
    W = reg.cfg.reg_dual_warmup
    assert W == (2000 if warmup is None else warmup)
    lam0 = reg.lam
    assert lam0 == reg.cfg.reg_lambda_init
    reg.viol_ema = 0.6
    for t in sorted({1, 2, W - 1, W}):          # lambda frozen for step <= warmup ...
        _stash(reg, outputs, carry.steps, t)
        _, m = reg(step=t)
        assert reg.lam == lam0 and m["reg/lambda"] == lam0, t
    _stash(reg, outputs, carry.steps, W + 1)    # ... and updated from the first step after it
    _, m = reg(step=W + 1)
    assert m["reg/lambda"] == lam0
    assert reg.lam == pytest.approx(lam0 * math.exp(0.5 * (0.6 - 0.1)), rel=1e-12)


def test_e2e_dual_ascent_with_real_violation_ema():
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="cycle_lip", reg_monitor_every=0, reg_dual_lr=0.5))
    beta, target = reg.cfg.reg_viol_ema, reg.cfg.reg_target_viol
    assert reg.viol_ema == target
    t0 = reg.cfg.reg_dual_warmup                # (default warmup) lambda moves from step t0 + 1 on

    def step(t):
        _stash(reg, outputs, carry.steps, t0 + t)
        ema0, lam0 = reg.viol_ema, reg.lam
        _, m = reg(step=t0 + t)
        assert reg.viol_ema == pytest.approx(beta * ema0 + (1 - beta) * m["reg/viol"], rel=1e-12)
        assert m["reg/viol_ema"] == reg.viol_ema
        assert reg.lam == pytest.approx(lam0 * math.exp(0.5 * (reg.viol_ema - target)), rel=1e-12)
        return m

    reg.rho = 1e-3                      # controlled input: every pair violates
    lams = [reg.lam]
    for t in range(1, 6):
        assert step(t)["reg/viol"] == 1.0
        lams.append(reg.lam)
    assert reg.viol_ema > target and all(b > a for a, b in zip(lams, lams[1:]))

    reg.rho = 1e6                       # controlled input: no pair violates
    for t in range(6, 200):
        assert step(t)["reg/viol"] == 0.0
        if reg.viol_ema < target:
            break
    lams = [reg.lam]
    for t in range(200, 205):
        step(t)
        lams.append(reg.lam)
    assert reg.viol_ema < target and all(b < a for a, b in zip(lams, lams[1:]))


def test_e2e_warm_start_reuses_direction_of_same_puzzle_only():
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="cycle_lip", reg_monitor_every=0, reg_frac=1.0))
    assert reg.k == BATCH
    seen = []
    orig = reg._v1

    def spy(s, x, d, r, power_iters, with_grad):
        seen.append(torch.stack(d, 1).clone())
        return orig(s, x, d, r, power_iters, with_grad)
    reg._v1 = spy

    steps1 = torch.ones(BATCH, dtype=torch.int32)
    _stash(reg, outputs, steps1, global_step=1)
    idx1 = reg.pending["idx"].clone()
    reg(step=1)
    stored = reg.mem["dir"].clone()
    assert torch.all(reg.mem["birth"] == 1)

    # same puzzles one ACT step later: birth = 2 - 2 + 1 = 1 -> warm start from the stored rows
    _stash(reg, outputs, steps1 + 1, global_step=2)
    idx2 = reg.pending["idx"].clone()
    reg(step=2)
    torch.testing.assert_close(seen[1], stored[idx2], rtol=1e-6, atol=1e-7)
    assert torch.all(reg.mem["birth"] == 1)

    # new puzzles in every row (steps restart at 1 at a later global step) -> cold start
    stored = reg.mem["dir"].clone()
    _stash(reg, outputs, steps1, global_step=3)
    idx3 = reg.pending["idx"].clone()
    reg(step=3)
    cos = (seen[2].flatten(1) * stored[idx3].flatten(1)).sum(-1)
    assert cos.abs().max() < 0.2
    assert torch.all(reg.mem["birth"] == 3)
    assert set(idx1.tolist()) == set(range(BATCH))


def test_e2e_warm_start_flow_mono_per_row():
    """V2 recalls dir, dir_F and the running F gain for rows still holding the same puzzle, and cold-starts
    (random unit directions, zero gain) the rows holding a new one, row by row."""
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="flow_mono", reg_monitor_every=0, reg_frac=1.0),
                                                 flow=True)
    seen = []
    orig = reg._v2

    def spy(s, x, d, dF, gain_F, r, power_iters, with_grad):
        seen.append(dict(d=torch.stack(d, 1).clone(), dF=torch.stack(dF, 1).clone(), gain=gain_F.clone()))
        return orig(s, x, d, dF, gain_F, r, power_iters, with_grad)
    reg._v2 = spy

    steps1 = torch.ones(BATCH, dtype=torch.int32)
    _stash(reg, outputs, steps1, global_step=1)
    reg(step=1)
    assert torch.equal(seen[0]["gain"], torch.zeros(BATCH))                  # cold start: zero gain
    for key in ("d", "dF"):
        torch.testing.assert_close(rownorm2(seen[0][key]), torch.ones(BATCH), rtol=1e-5, atol=0)
    stored = _mem_clone(reg)
    assert torch.all(stored["gain_F"] > 0)

    # global step 2: even rows continue their puzzle (steps 2 -> birth 1), odd rows hold a new one (birth 2)
    steps2 = torch.where(torch.arange(BATCH) % 2 == 0, 2, 1).to(torch.int32)
    _stash(reg, outputs, steps2, global_step=2)
    idx = reg.pending["idx"].clone()
    reg(step=2)
    same = (steps2[idx] == 2)
    assert same.any() and (~same).any()
    got = seen[1]
    torch.testing.assert_close(got["d"][same], stored["dir"][idx][same], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(got["dF"][same], stored["dir_F"][idx][same], rtol=1e-6, atol=1e-7)
    assert torch.equal(got["gain"][same], stored["gain_F"][idx][same])
    assert torch.all(got["gain"][~same] == 0)
    for key, mkey in (("d", "dir"), ("dF", "dir_F")):
        cos = (got[key][~same].flatten(1) * stored[mkey][idx][~same].flatten(1)).sum(-1)
        assert cos.abs().max() < 0.2
        torch.testing.assert_close(rownorm2(got[key]), torch.ones(BATCH), rtol=1e-5, atol=0)
    assert torch.equal(reg.mem["birth"][idx], torch.where(same, 1, 2))
    # the running gain only grows while the puzzle stays
    assert torch.all(reg.mem["gain_F"][idx][same] >= stored["gain_F"][idx][same])


@pytest.mark.parametrize("mode", ["cycle_lip", "flow_mono"])
def test_e2e_state_dict_roundtrip_reproduces_next_call(mode):
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode=mode, reg_monitor_every=0), flow=True)
    for t in (1, 2):
        _stash(reg, outputs, carry.steps + (t - 1), t)
        reg(step=t)
    sd = copy.deepcopy(reg.state_dict())
    assert set(sd) == SD_KEYS and set(sd["mem"]) == MEM_KEYS
    other = ContractionRegularizer(reg.cfg, inner, batch_size=BATCH, seed=123, device="cpu")
    other.load_state_dict(copy.deepcopy(sd))
    assert other.lam == reg.lam and other.viol_ema == reg.viol_ema and other.n_calls == reg.n_calls
    assert torch.equal(other.gen.get_state(), reg.gen.get_state())
    assert all(torch.equal(other.mem[k], reg.mem[k]) for k in MEM_KEYS)

    _stash(reg, outputs, carry.steps + 2, 3)
    _stash(other, outputs, carry.steps + 2, 3)
    l1, m1 = reg(step=3)
    l2, m2 = other(step=3)
    assert torch.equal(l1, l2) and m1 == m2
    assert all(torch.equal(reg.mem[k], other.mem[k]) for k in MEM_KEYS)


def test_e2e_state_dict_without_mem_roundtrips():
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="none", reg_monitor_every=0))
    sd = reg.state_dict()
    assert set(sd) == SD_KEYS and sd["mem"] is None
    other = ContractionRegularizer(RegConfig(reg_mode="cycle_lip", reg_monitor_every=0), inner, batch_size=BATCH,
                                   seed=5, device="cpu")
    _stash(other, outputs, carry.steps, 1)
    other(step=1)
    assert other.mem is not None
    other.load_state_dict(sd)
    assert other.mem is None


@pytest.mark.parametrize("old_format", ["no_mem", "dir_only", "dir_and_dir_birth"])
def test_e2e_load_state_dict_old_format_restarts_warm_starts_cold(old_format):
    """Checkpoints written before the `mem` dict (no entry, or the old dir / dir_birth tensors) load without
    error: lambda, EMA, call count and RNG are restored and the warm starts restart cold."""
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="cycle_lip", reg_monitor_every=0))
    _stash(reg, outputs, carry.steps, 1)
    reg(step=1)
    reg.lam, reg.viol_ema = 1.7, 0.3            # distinguishable from a fresh regularizer
    sd = copy.deepcopy(reg.state_dict())
    mem = sd.pop("mem")
    old = dict(sd)
    if old_format != "no_mem":
        old["dir"] = mem["dir"]
    if old_format == "dir_and_dir_birth":
        old["dir_birth"] = mem["birth"]

    other = ContractionRegularizer(reg.cfg, inner, batch_size=BATCH, seed=0, device="cpu")
    _stash(other, outputs, carry.steps, 1)      # give it warm-start state of its own, which must be dropped
    other(step=1)
    assert other.mem is not None
    other.load_state_dict(copy.deepcopy(old))
    assert other.mem is None
    assert other.lam == 1.7 and other.viol_ema == 0.3 and other.n_calls == reg.n_calls
    assert torch.equal(other.gen.get_state(), reg.gen.get_state())

    # Same next call as a regularizer restored with mem=None (cold), although the same puzzles still occupy the rows
    cold = ContractionRegularizer(reg.cfg, inner, batch_size=BATCH, seed=99, device="cpu")
    cold.load_state_dict(dict(copy.deepcopy(sd), mem=None))
    _stash(other, outputs, carry.steps + 1, 2)
    _stash(cold, outputs, carry.steps + 1, 2)
    l1, m1 = other(step=2)
    l2, m2 = cold(step=2)
    assert l1 is not None and torch.isfinite(l1)
    assert torch.equal(l1, l2) and m1 == m2
    assert all(torch.equal(other.mem[k], cold.mem[k]) for k in MEM_KEYS)


def test_e2e_state_dict_is_a_snapshot():
    head, inner, reg, carry, _, outputs = _setup(dict(reg_mode="flow_mono", reg_monitor_every=0, reg_frac=1.0),
                                                 flow=True)
    _stash(reg, outputs, carry.steps, 1)
    reg(step=1)
    sd = reg.state_dict()
    frozen = copy.deepcopy(sd)
    _stash(reg, outputs, carry.steps + 1, 2)
    reg(step=2)
    assert not torch.equal(reg.mem["dir"], frozen["mem"]["dir"])         # the live state did move on
    assert torch.equal(sd["gen"], frozen["gen"])
    assert all(torch.equal(sd["mem"][k], frozen["mem"][k]) for k in MEM_KEYS)

    # load_state_dict copies too: mutating the dict afterwards (or running the loaded regularizer) is independent
    other = ContractionRegularizer(reg.cfg, inner, batch_size=BATCH, seed=0, device="cpu")
    other.load_state_dict(sd)
    for v in sd["mem"].values():
        v.zero_()
    sd["gen"].zero_()
    assert all(torch.equal(other.mem[k], frozen["mem"][k]) for k in MEM_KEYS)
    assert torch.equal(other.gen.get_state(), frozen["gen"])
    sd2 = other.state_dict()
    _stash(other, outputs, carry.steps + 1, 2)
    other(step=2)
    assert all(torch.equal(sd2["mem"][k], frozen["mem"][k]) for k in MEM_KEYS)


def test_regconfig_from_arch_extra_and_rand_pairs():
    cfg = RegConfig.from_arch_extra(dict(reg_mode="flow_mono", reg_eta=0.1, reg_pair="rand", hidden_size=64,
                                         flow=True, name="x"))
    assert cfg.reg_mode == "flow_mono" and cfg.reg_eta == 0.1 and cfg.reg_pair == "rand"
    head, inner, reg, carry, _, outputs = _setup(cfg.model_dump() | dict(reg_monitor_every=0), flow=True)
    assert reg.mu == pytest.approx(-math.log(0.9) / 0.5)
    _stash(reg, outputs, carry.steps)
    loss, m = reg(step=1)
    assert loss is not None and torch.isfinite(loss)
    assert reg.mem is None          # random pairs keep no warm-start state
