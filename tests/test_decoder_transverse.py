"""Tests for the decoder-transverse regularizers D1 `dec_block` / D2 `dec_out` (models/regularizers.py) and the
matching Jacobian diagnostics (scripts/fixed_point_diagnostics.py: decoder_projector, projected_sigma_max).

CPU only by design; run with
    CUDA_VISIBLE_DEVICES="" DISABLE_COMPILE=1 pytest tests/test_decoder_transverse.py

P = projection of the z_H answer tokens onto the centered decoder row space (z_L and puzzle positions -> 0).
D1: sigma_max(P J P) (perturbations inside range(P)), D2: sigma_max(P J) (full-space perturbations).
Sections:
  (1) _decoder_basis           (2) _proj
  (3) proj_power_step on linear toy operators with a known block structure (visible / invisible), then the full
      warm-started ContractionRegularizer loop on a linear stub of the operator API
  (4) end to end on a tiny real trm_contractive model (vocab 11, hidden 32, mlp_t, puzzle_emb_len 16)
  (5) RegConfig               (6) fixed_point_diagnostics.decoder_projector / projected_sigma_max
  (7) basis construction on numerically hard / degenerate inputs (regressions for fixed bugs)
"""
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("DISABLE_COMPILE", "1")

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pydantic  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from models.regularizers import (  # noqa: E402
    ContractionRegularizer, RegConfig, normalize_rows, proj_power_step, rowdot, rownorm2, v1_power_step,
)
from models.recursive_reasoning.trm_contractive import TinyRecursiveReasoningModel_ACTV1  # noqa: E402
from models.losses import ACTLossHead  # noqa: E402
from fixed_point_diagnostics import decoder_projector, projected_sigma_max  # noqa: E402

DT = torch.float64


@pytest.fixture(autouse=True, scope="module")
def _few_threads():
    n = torch.get_num_threads()
    torch.set_num_threads(min(4, n))
    yield
    torch.set_num_threads(n)


@pytest.fixture
def float64_default():
    """projected_sigma_max draws its start vector with the default dtype (as in tests/test_diagnostics.py)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


# =============================================================================================
# Utilities
# =============================================================================================

def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(g, *shape, dtype=DT):
    return torch.randn(*shape, generator=g, dtype=dtype)


def _rand_orth(n, g):
    q, r = torch.linalg.qr(_randn(g, n, n))
    return q * torch.sign(torch.diagonal(r))


def _with_sv(g, m, n, sv):
    """m x n matrix with the given leading singular values (the rest zero)."""
    sv = torch.as_tensor(sv, dtype=DT)
    k = sv.numel()
    return _rand_orth(m, g)[:, :k] @ torch.diag(sv) @ _rand_orth(n, g)[:, :k].T


def _init_like_lm_head(g, V, D, dtype=torch.float32):
    """CastedLinear init: truncated normal, std 1 / sqrt(D)."""
    return (torch.randn(V, D, generator=g, dtype=DT).clamp(-2, 2) / math.sqrt(D)).to(dtype)


def _fake(W=None, P0=0):
    """Minimal `self` for the unbound ContractionRegularizer._decoder_basis / _proj."""
    return SimpleNamespace(inner=SimpleNamespace(lm_head=SimpleNamespace(weight=W), puzzle_emb_len=P0))


def _basis(W):
    return ContractionRegularizer._decoder_basis(_fake(W))


def _reg_proj(Q, P0):
    fake = _fake(None, P0)
    return lambda v: ContractionRegularizer._proj(fake, v, Q)


def _ref_basis(W):
    """Reference orthonormal basis of the centered row space, float64 SVD (independent of the QR path)."""
    Wc = W.double() - W.double().mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Wc, full_matrices=False)
    r = int((S > 1e-9 * S[0]).sum())
    return Vh[:r].T


def _ref_proj(Q, P0):
    """Reference projector on a state tuple (float64)."""
    Q = Q.double()

    def P(v):
        h = torch.zeros(v[0].shape, dtype=DT)
        h[:, P0:] = (v[0][:, P0:].double() @ Q) @ Q.T
        return h, torch.zeros(v[1].shape, dtype=DT)
    return P


def _cat(t):
    return torch.cat([x.reshape(x.shape[0], -1) for x in t], -1)


def _split(v, like):
    out, o = [], 0
    for x in like:
        m = x[0].numel()
        out.append(v[:, o:o + m].reshape(x.shape))
        o += m
    return tuple(out)


def _linear_op(J):
    """Phi(z_H, z_L) = J @ concat(z_H, z_L) (flattened per row), blocks reshaped back."""
    def op(*s):
        return _split(_cat(s) @ J.T.to(s[0].dtype), s)
    return op


def _scale(d, r):
    if not torch.is_tensor(r):
        return tuple(x * r for x in d)
    return tuple(x * r.view((-1,) + (1,) * (x.ndim - 1)).to(x.dtype) for x in d)


def _visible_basis(Q, P0, Ltok):
    """Orthonormal (N, (Ltok - P0) r) basis of range(P) in the flattened (z_H, z_L) coordinates."""
    D, r = Q.shape
    cols = []
    for t in range(P0, Ltok):
        for k in range(r):
            v = torch.zeros(2, Ltok, D, dtype=DT)
            v[0, t] = Q[:, k].double()
            cols.append(v.flatten())
    return torch.stack(cols, 1)


def _complement(Bv, g):
    N, nv = Bv.shape
    q, _ = torch.linalg.qr(torch.cat([Bv, _randn(g, N, N - nv)], 1))
    Bi = q[:, nv:]
    assert (Bv.T @ Bi).abs().max() < 1e-12
    return Bi


def _sigma(M):
    return torch.linalg.matrix_norm(M, 2).item()


# ---------------------------------------------------------------------------------------------
# Linear toys with a known visible / invisible block structure
# ---------------------------------------------------------------------------------------------
TOY = dict(B=6, P0=2, Ltok=5, D=8, V=4)      # r = V - 1 = 3, visible dim = (5 - 2) * 3 = 9, N = 2 * 5 * 8 = 80


def _toy(case, seed=0):
    """J = [Bv Bi] [[A_vv, A_vi], [A_iv, A_ii]] [Bv Bi]^T.

    D1 = sigma_max(P J P) = ||A_vv||, D2 = sigma_max(P J) = ||[A_vv, A_vi]||; A_iv and A_ii are invisible to both.
      invisible_expands : ||A_ii|| = 10 and ||A_iv|| = 8 (big), small leak ||A_vi|| = 0.3   -> D1 = 1.5, D2 ~ 1.5
      leak              : ||A_vi|| = 6 (invisible -> visible), A_ii mild                    -> D1 = 1.5, D2 >= 6
    """
    g = _gen(100 + seed)
    W = _init_like_lm_head(g, TOY["V"], TOY["D"])
    Q = _basis(W)
    assert Q.shape == (TOY["D"], TOY["V"] - 1)
    Qd, _ = torch.linalg.qr(Q.double())             # same span, orthonormal to float64 precision
    Bv = _visible_basis(Qd, TOY["P0"], TOY["Ltok"])
    Bi = _complement(Bv, g)
    nv, ni = Bv.shape[1], Bi.shape[1]
    A_vv = _with_sv(g, nv, nv, [1.5] + torch.linspace(0.1, 0.8, nv - 1).tolist())
    if case == "invisible_expands":
        A_vi = _with_sv(g, nv, ni, [0.3] + [0.2] * (nv - 1))
        A_iv = _with_sv(g, ni, nv, [8.0] + [1.0] * (nv - 1))
        A_ii = _with_sv(g, ni, ni, [10.0] + torch.linspace(0.1, 1.0, ni - 1).tolist())
    elif case == "leak":
        A_vi = _with_sv(g, nv, ni, [6.0] + torch.linspace(0.1, 1.0, nv - 1).tolist())
        A_iv = _with_sv(g, ni, nv, [0.5] * nv)
        A_ii = _with_sv(g, ni, ni, torch.linspace(0.1, 0.9, ni).tolist())
    else:
        raise ValueError(case)
    Bfull = torch.cat([Bv, Bi], 1)
    A = torch.cat([torch.cat([A_vv, A_vi], 1), torch.cat([A_iv, A_ii], 1)], 0)
    J = Bfull @ A @ Bfull.T
    Pm = Bv @ Bv.T
    return SimpleNamespace(W=W, Q=Q, J=J, Pm=Pm, Bv=Bv, Bi=Bi,
                           d1=_sigma(Pm @ J @ Pm), d2=_sigma(Pm @ J), full=_sigma(J),
                           A_vv=A_vv, A_vi=A_vi)


def _toy_state(g, B=TOY["B"], scale=0.3):
    shape = (B, TOY["Ltok"], TOY["D"])
    return (scale * torch.randn(*shape, generator=g), scale * torch.randn(*shape, generator=g))


# =============================================================================================
# (1) _decoder_basis
# =============================================================================================

@pytest.mark.parametrize("D", [16, 32, 64, 512])
def test_decoder_basis_orthonormal_rank_span_and_centering(D):
    V = 11
    g = _gen(D)
    W = _init_like_lm_head(g, V, D)
    Q = _basis(W)
    assert Q.dtype == torch.float32 and Q.shape == (D, V - 1)
    # orthonormal columns
    torch.testing.assert_close(Q.T @ Q, torch.eye(V - 1), rtol=0, atol=2e-6)

    Wd = W.double()
    Wc = Wd - Wd.mean(0, keepdim=True)
    Qd = Q.double()
    assert torch.linalg.matrix_rank(Wd).item() == V and torch.linalg.matrix_rank(Wc).item() == V - 1
    # span(Q) == centered row space (same orthogonal projector as an independent float64 SVD)
    Qr = _ref_basis(W)
    assert Qr.shape == (D, V - 1)
    torch.testing.assert_close(Qd @ Qd.T, Qr @ Qr.T, rtol=0, atol=2e-6)
    # every centered row lies in span(Q)
    assert (Wc - Wc @ Qd @ Qd.T).norm() / Wc.norm() < 2e-6

    # for any z: Wc z == Wc (Q Q^T z), i.e. projecting does not change the centered logits ...
    z = _randn(g, 64, D)
    zp = (z @ Qd) @ Qd.T
    torch.testing.assert_close(zp @ Wc.T, z @ Wc.T, rtol=0, atol=1e-5)
    # ... so the raw logits change by a per-row constant only: softmax and argmax unchanged
    dl = z @ Wd.T - zp @ Wd.T
    torch.testing.assert_close(dl - dl.mean(-1, keepdim=True), torch.zeros_like(dl), rtol=0, atol=1e-5)
    assert dl.std(-1).max() < 1e-5 and dl.abs().max() > 1e-2           # a real shift, invisible to softmax
    torch.testing.assert_close(torch.softmax(zp @ Wd.T, -1), torch.softmax(z @ Wd.T, -1), rtol=0, atol=1e-6)
    assert torch.equal((zp @ Wd.T).argmax(-1), (z @ Wd.T).argmax(-1))

    # the "all-ones in logit space" direction z1 (W z1 = 1: shifts every logit by 1) is in the null space of P
    z1 = torch.linalg.pinv(Wd) @ torch.ones(V, dtype=DT)
    torch.testing.assert_close(Wd @ z1, torch.ones(V, dtype=DT), rtol=0, atol=1e-8)
    assert (Qd.T @ z1).norm() / z1.norm() < 1e-5
    # whereas the uncentered row space contains it (centering is what removes it)
    Qu, _ = torch.linalg.qr(Wd.T)
    assert (Qu.T @ z1).norm() / z1.norm() > 0.999


def test_decoder_basis_invariant_to_a_common_row_shift_and_wide_vocab():
    g = _gen(1)
    V, D = 11, 64
    W = _init_like_lm_head(g, V, D)
    Q = _basis(W).double()
    # adding the same vector to every row of W (moderate size) does not change the projector
    m = torch.randn(1, D, generator=g)
    W2 = W + 0.5 * W.norm(dim=1).mean() * m / m.norm()
    Q2 = _basis(W2).double()
    assert Q2.shape == Q.shape
    torch.testing.assert_close(Q2 @ Q2.T, Q @ Q.T, rtol=0, atol=2e-6)
    # it is computed from the detached fp32 weight (a bf16 / grad-requiring parameter works and is not tracked)
    Wp = torch.nn.Parameter(W.clone())
    Qp = _basis(Wp)
    assert not Qp.requires_grad and Qp.dtype == torch.float32
    Qb = _basis(W.to(torch.bfloat16)).double()
    assert Qb.shape == Q.shape
    # hidden < vocab - 1: the centered rows span the whole hidden space, P is the identity on answer tokens
    Ws = _init_like_lm_head(g, V, 6)
    Qs = _basis(Ws)
    assert Qs.shape == (6, 6)
    torch.testing.assert_close(Qs @ Qs.T, torch.eye(6), rtol=0, atol=2e-6)


# =============================================================================================
# (2) _proj
# =============================================================================================

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64], ids=["fp32", "bf16", "fp64"])
def test_proj_structure_idempotent_self_adjoint(dtype):
    g = _gen(2)
    V, D, P0, L, B = 11, 32, 16, 16 + 81, 4
    W = _init_like_lm_head(g, V, D)
    Q = _basis(W)
    P = _reg_proj(Q, P0)
    Pref = _ref_proj(_ref_basis(W), P0)

    def rand_state():
        return tuple(torch.randn(B, L, D, generator=g).to(dtype) for _ in range(2))

    u, v = rand_state(), rand_state()
    pu = P(u)
    assert all(t.dtype == torch.float32 and t.shape == (B, L, D) for t in pu)
    assert torch.count_nonzero(pu[1]) == 0                      # z_L component is zero
    assert torch.count_nonzero(pu[0][:, :P0]) == 0              # puzzle-embedding positions are zero
    assert rownorm2(pu[0][:, P0:]).min() > 0
    # matches an independent float64 SVD projector
    torch.testing.assert_close(pu[0].double(), Pref(u)[0], rtol=0, atol=2e-5)
    # idempotent and non-expansive
    ppu = P(pu)
    torch.testing.assert_close(ppu[0], pu[0], rtol=0, atol=2e-6)
    assert torch.count_nonzero(ppu[1]) == 0
    assert torch.all(rownorm2(*pu) <= rownorm2(*u) * (1 + 1e-6))
    # rank: (L - P0) * (V - 1) per row, and the expected energy fraction for an isotropic state
    frac = (rownorm2(*pu) / rownorm2(*u)).mean().item()
    assert frac == pytest.approx((L - P0) * (V - 1) / (2 * L * D), rel=0.15)
    # self-adjoint: <P u, v> == <u, P v>
    a, b = rowdot(P(u), v), rowdot(u, P(v))
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    # linear
    w = tuple((2.0 * x.float() - 3.0 * y.float()) for x, y in zip(u, v))
    lin = tuple(2.0 * x - 3.0 * y for x, y in zip(P(u), P(v)))
    for x, y in zip(P(w), lin):
        torch.testing.assert_close(x, y, rtol=0, atol=1e-4)
    # centered logits of the answer tokens are unchanged by P
    Wc = W.double() - W.double().mean(0, keepdim=True)
    torch.testing.assert_close(pu[0][:, P0:].double() @ Wc.T, u[0][:, P0:].float().double() @ Wc.T,
                               rtol=0, atol=2e-4)


def test_proj_is_differentiable_with_adjoint_gradient():
    g = _gen(3)
    W = _init_like_lm_head(g, 11, 32)
    Q = _basis(W)
    P = _reg_proj(Q, 16)
    v = tuple(torch.randn(3, 20, 32, generator=g).requires_grad_(True) for _ in range(2))
    w = tuple(torch.randn(3, 20, 32, generator=g) for _ in range(2))
    out = P(v)
    assert out[0].requires_grad
    rowdot(out, w).sum().backward()
    # d<P v, w>/dv = P^T w = P w (z_L gets a zero gradient: it is not used by P)
    pw = P(w)
    torch.testing.assert_close(v[0].grad, pw[0], rtol=0, atol=1e-6)
    assert v[1].grad is None or torch.count_nonzero(v[1].grad) == 0


# =============================================================================================
# (3) proj_power_step on linear toys with a known visible / invisible structure
# =============================================================================================

@pytest.mark.parametrize("case", ["invisible_expands", "leak"])
def test_toy_construction_reference_values(case):
    t = _toy(case)
    # Pm (built from Q by hand) is the projector applied by _proj
    g = _gen(4)
    s = _toy_state(g)
    ps = _reg_proj(t.Q, TOY["P0"])(s)
    torch.testing.assert_close(_cat(ps).double(), _cat(s).double() @ t.Pm, rtol=0, atol=1e-6)
    assert t.d1 == pytest.approx(1.5, rel=1e-9)
    if case == "invisible_expands":
        assert t.full >= 10.0 and t.d2 < 1.6
    else:
        assert t.d2 >= 6.0 and t.full >= 6.0


@pytest.mark.parametrize("mode", ["D1", "D2"])
@pytest.mark.parametrize("case", ["invisible_expands", "leak"])
def test_proj_power_step_closed_form(case, mode):
    """Linear Phi: one step maps d to normalize(P_in J^T P_out J d) (exact power step, any radius)."""
    t = _toy(case)
    g = _gen(5)
    P = _reg_proj(t.Q, TOY["P0"])
    in_proj = P if mode == "D1" else None
    phi = _linear_op(t.J)
    s = _toy_state(g)
    phi_s = phi(*s)
    d0 = normalize_rows(_toy_state(g))
    if mode == "D1":
        d0 = normalize_rows(P(d0))
    r = 0.1 * rownorm2(*s).sqrt()
    d1 = proj_power_step(phi, s, phi_s, _scale(d0, r), out_proj=P, in_proj=in_proj)
    assert all(x.dtype == torch.float32 and not x.requires_grad for x in d1)
    torch.testing.assert_close(rownorm2(*d1), torch.ones(TOY["B"]), rtol=1e-6, atol=0)
    v = _cat(d0).double()
    Pin = t.Pm if mode == "D1" else torch.eye(t.J.shape[0], dtype=DT)
    ref = v @ (Pin @ t.J.T @ t.Pm @ t.J).T
    ref = ref / ref.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(_cat(d1).double(), ref, rtol=0, atol=2e-5)
    # linear map: invariant to the pair radius
    d1b = proj_power_step(phi, s, phi_s, _scale(d0, 10.0), out_proj=P, in_proj=in_proj)
    torch.testing.assert_close(_cat(d1b), _cat(d1), rtol=0, atol=2e-5)
    if mode == "D1":
        # stays in range(P): z_L and puzzle positions exactly zero, invariant under P
        assert torch.count_nonzero(d1[1]) == 0 and torch.count_nonzero(d1[0][:, :TOY["P0"]]) == 0
        torch.testing.assert_close(_cat(P(d1)), _cat(d1), rtol=0, atol=1e-6)


@pytest.mark.parametrize("case", ["invisible_expands", "leak"])
def test_proj_power_iteration_converges_to_projected_norms(case):
    """Warm-started steps: D1 -> sigma_max(P J P), D2 -> sigma_max(P J); neither sees the invisible block A_ii
    (sigma 10) nor the visible -> invisible block A_iv; only D2 sees the invisible -> visible leak A_vi."""
    t = _toy(case)
    g = _gen(6)
    P = _reg_proj(t.Q, TOY["P0"])
    phi = _linear_op(t.J)
    s = _toy_state(g)
    phi_s = phi(*s)
    r = 0.1 * rownorm2(*s).sqrt()
    B = TOY["B"]

    def gain(d):
        v = _cat(d).double()
        return (v @ (t.Pm @ t.J).T).norm(dim=-1) / v.norm(dim=-1)

    out = {}
    for mode in ("D1", "D2"):
        d = normalize_rows(_toy_state(g))
        if mode == "D1":
            d = normalize_rows(P(d))
        hist = [gain(d)]
        for _ in range(60):
            d = proj_power_step(phi, s, phi_s, _scale(d, r), out_proj=P, in_proj=P if mode == "D1" else None)
            hist.append(gain(d))
        hist = torch.stack(hist, 1)
        # monotone non-decreasing (power iteration on a PSD operator), up to fp32 noise
        assert (hist[:, :-1] - hist[:, 1:]).max() < 1e-5
        out[mode] = (d, hist[:, -1])

    torch.testing.assert_close(out["D1"][1], torch.full((B,), t.d1, dtype=DT), rtol=1e-4, atol=0)
    torch.testing.assert_close(out["D2"][1], torch.full((B,), t.d2, dtype=DT), rtol=1e-4, atol=0)
    dD1, dD2 = _cat(out["D1"][0]).double(), _cat(out["D2"][0]).double()
    # D1 iterate stays in range(P)
    assert ((dD1 - dD1 @ t.Pm).norm(dim=-1) < 1e-5).all()
    # the plain V1 adversary on the same operator finds the invisible expansion instead
    d = normalize_rows(_toy_state(g))
    for _ in range(60):
        d = v1_power_step(phi, s, phi_s, _scale(d, r))
    v1 = (_cat(d).double() @ t.J.T).norm(dim=-1)
    torch.testing.assert_close(v1, torch.full((B,), t.full, dtype=DT), rtol=1e-3, atol=0)
    if case == "invisible_expands":
        assert t.full >= 10 and max(t.d1, t.d2) < 1.6                    # neither D1 nor D2 picks up sigma = 10
        # the D2 maximizer is (almost) visible: the leak is small
        assert ((dD2 @ t.Pm).norm(dim=-1) > 0.9).all()
    else:
        assert t.d2 > 3.9 * t.d1                                          # D2 picks up the leak, D1 does not
        # the D2 maximizer lives mostly in the invisible subspace (it is the leaking direction)
        assert ((dD2 - dD2 @ t.Pm).norm(dim=-1) > 0.9).all()


# ---------------------------------------------------------------------------------------------
# Full ContractionRegularizer loop on a linear stub of the operator API
# ---------------------------------------------------------------------------------------------

class _LinearDecInner:
    """Operator API consumed by ContractionRegularizer, Phi(s) = J s, V(s) = (J - I) s, with an lm_head."""

    def __init__(self, J, W, P0, H_cycles=2, L_cycles=2):
        self.config = SimpleNamespace(H_cycles=H_cycles, L_cycles=L_cycles)
        self.h_H = 1.0
        self._J = J
        self.lm_head = SimpleNamespace(weight=W)
        self.puzzle_emb_len = P0

    def seq_info(self):
        return {}

    def time_embeddings(self):
        return None

    def substep_index(self, i, j):
        return i * (self.config.L_cycles + 1) + j

    def cycle(self, z_H, z_L, x, temb, i, seq_info):
        return _linear_op(self._J)(z_H, z_L)

    def field(self, z_H, z_L, x, temb_H, temb_L, seq_info):
        out = _linear_op(self._J)(z_H, z_L)
        return tuple(o.float() - z.float() for o, z in zip(out, (z_H, z_L)))


def _stub_outs(g, B):
    s = _toy_state(g, B)
    return dict(reg_s_H=s[0], reg_s_L=s[1], reg_phi_H=s[0].clone(), reg_phi_L=s[1].clone(),
                reg_x=torch.zeros_like(s[0]))


def _run_stub(reg, outs, B, n_calls):
    hist = []
    for t in range(1, n_calls + 1):
        reg.stash(outs, torch.full((B,), t, dtype=torch.int32), t)     # same puzzle in every row (birth 1)
        _, m = reg(step=t)
        hist.append(m)
    return hist


@pytest.mark.parametrize("case", ["invisible_expands", "leak"])
@pytest.mark.parametrize("mode", ["dec_block", "dec_out"])
def test_regularizer_dec_adversary_on_linear_stub(mode, case):
    t = _toy(case)
    target = t.d1 if mode == "dec_block" else t.d2
    B = TOY["B"]
    g = _gen(7)
    reg = ContractionRegularizer(RegConfig(reg_mode=mode, reg_frac=1.0, reg_monitor_every=0),
                                 _LinearDecInner(t.J, t.W, TOY["P0"]), batch_size=B, seed=0, device="cpu")
    assert reg.k == B and reg.cfg.reg_v1_power_iters == 1 and reg.rho == pytest.approx(0.95)
    hist = _run_stub(reg, _stub_outs(g, B), B, n_calls=60)
    first, last = hist[0], hist[-1]
    print(f"\n[{mode} stub {case}] target={target:.4f} value: first={first['reg/value']:.4f} "
          f"last={last['reg/value']:.4f} (full ||J||={t.full:.3f})")
    assert last["reg/value"] == pytest.approx(target, rel=1e-4)
    assert last["reg/value_max"] <= target * (1 + 1e-4)         # a linear secant quotient never exceeds the norm
    assert all(m["reg/value_max"] <= target * (1 + 1e-4) for m in hist)
    assert last["reg/viol"] == 1.0
    assert last["reg/pen"] == pytest.approx(math.log(target / reg.rho), rel=1e-3)
    assert first["reg/value"] < last["reg/value"] + 1e-6
    if case == "invisible_expands":
        assert last["reg/value_max"] < 0.2 * t.full                 # the invisible expansion is left free
    # warm-start memory: unit directions, in range(P) for D1
    dirs = reg.mem["dir"]
    torch.testing.assert_close(rownorm2(dirs), torch.ones(B), rtol=1e-5, atol=0)
    flat = dirs.flatten(1).double()
    off = (flat - flat @ t.Pm).norm(dim=-1)
    if mode == "dec_block":
        assert (off < 1e-5).all()
    elif case == "leak":
        assert (off > 0.5).all()
    assert torch.all(reg.mem["birth"] == 1)


def test_regularizer_dec_monitor_on_linear_stub_is_bounded_by_norms():
    t = _toy("leak")
    B = TOY["B"]
    reg = ContractionRegularizer(RegConfig(reg_mode="none", reg_frac=1.0, reg_monitor_every=1),
                                 _LinearDecInner(t.J, t.W, TOY["P0"]), batch_size=B, seed=0, device="cpu")
    reg.stash(_stub_outs(_gen(8), B), torch.ones(B, dtype=torch.int32), 1)
    _, m = reg(step=1)
    assert 0 < m["mon/dec_block"] <= t.d1 * (1 + 1e-4)
    assert 0 < m["mon/dec_out"] <= t.d2 * (1 + 1e-4)
    # 3 cold power steps already separate them: the leak is visible to D2 only
    assert m["mon/dec_out"] > 2 * m["mon/dec_block"]
    assert reg.mem is None


# =============================================================================================
# (4) End to end on a tiny real trm_contractive model (CPU)
# =============================================================================================

BATCH, SEQ, P0, VOCAB, HID = 8, 81, 16, 11, 32
L_TOK = SEQ + P0


def _tiny_cfg(**over):
    cfg = dict(batch_size=BATCH, seq_len=SEQ, puzzle_emb_ndim=HID, num_puzzle_identifiers=4, vocab_size=VOCAB,
               H_cycles=2, L_cycles=2, H_layers=0, L_layers=2, hidden_size=HID, expansion=2, num_heads=2,
               pos_encodings="none", halt_max_steps=4, halt_exploration_prob=0.1, forward_dtype="float32",
               mlp_t=True, puzzle_emb_len=P0, no_ACT_continue=True,
               flow=False, flow_h_L=0.5, flow_h_H=0.5, flow_time_embed=True, flow_time_freqs=16)
    cfg.update(over)
    return cfg


ARCHS = {
    "mlp_t": dict(),
    "mlp_t_bf16": dict(forward_dtype="bfloat16"),
    "flow_temb_bf16": dict(forward_dtype="bfloat16", flow=True),
}


def _batch(seed=0):
    g = _gen(seed)
    return dict(inputs=torch.randint(1, VOCAB, (BATCH, SEQ), generator=g, dtype=torch.int32),
                labels=torch.randint(1, VOCAB, (BATCH, SEQ), generator=g, dtype=torch.int32),
                puzzle_identifiers=torch.randint(0, 4, (BATCH,), generator=g, dtype=torch.int32))


def _build(seed=0, **over):
    torch.manual_seed(seed)
    head = ACTLossHead(TinyRecursiveReasoningModel_ACTV1(_tiny_cfg(**over)), loss_type="stablemax_cross_entropy")
    head.train()
    return head


def _setup(reg_kw, seed=0, batch_size=BATCH, **arch):
    head = _build(seed, **arch)
    carry = head.initial_carry(_batch(seed))
    carry, _, _, outputs, _ = head(carry=carry, batch=_batch(seed), return_keys=[])
    inner = head.model.inner
    if inner.use_time_embed:   # zero at init: give it a value so the temb path matters
        with torch.no_grad():
            inner.temb_proj.weight.normal_(std=0.3, generator=_gen(seed + 7))
            inner.temb_proj.bias.normal_(std=0.3, generator=_gen(seed + 8))
    reg = ContractionRegularizer(RegConfig(**reg_kw), inner, batch_size=batch_size, seed=seed, device="cpu")
    head.zero_grad(set_to_none=True)
    return head, inner, reg, carry, outputs


def _spy_dec(reg):
    calls = []
    orig = reg._dec

    def wrapped(s, x, d, power_iters, with_grad, block=False, kind=None):
        out = orig(s, x, d, power_iters, with_grad=with_grad, block=block, kind=kind)
        kind = kind or ("block" if block else "out")
        calls.append(dict(s=s, x=x, block=kind == "block", kind=kind, power_iters=power_iters, pen=out[0].detach(),
                          ratio=out[1].detach(), den2=out[2].detach(), d=out[3], eps=out[4]))
        return out
    reg._dec = wrapped
    return calls


def _spy_phi(reg):
    log = []
    orig = reg._phi

    def wrapped(z_H, z_L, x, temb):
        out = orig(z_H, z_L, x, temb)
        log.append(dict(n=z_H.shape[0], z_H=z_H.detach().clone(), z_L=z_L.detach().clone(),
                        dtypes=(z_H.dtype, z_L.dtype, x.dtype) + (() if temb is None else (temb.dtype,)),
                        out_dtypes=(out[0].dtype, out[1].dtype), grad=torch.is_grad_enabled()))
        return out
    reg._phi = wrapped
    return log


def _inner_basis(inner):
    return _ref_basis(inner.lm_head.weight.detach())


def _manual_ratio(inner, s, x, d, eps, block):
    """Independent recomputation of the D1 / D2 secant quotient for given directions and relative radii."""
    temb = inner.time_embeddings()
    dt = torch.float32 if block else s[0].dtype
    s_ = tuple(t.to(dt) for t in s)
    x_ = x.to(dt)
    temb = None if temb is None else temb.to(dt)
    P = _ref_proj(_inner_basis(inner), inner.puzzle_emb_len)
    base = rownorm2(*P(s_)) if block else rownorm2(*s_)
    r = eps * base.float().sqrt()
    s2 = tuple((a.float() + dd * r.view(-1, 1, 1)).to(dt) for a, dd in zip(s_, d))
    n = s[0].shape[0]
    # grad mode as in the penalty pass: CPU bf16 kernels round differently with and without autograd (~4e-4 here)
    with torch.enable_grad():
        oH, oL = inner.cycle(torch.cat([s_[0], s2[0]]), torch.cat([s_[1], s2[1]]), torch.cat([x_, x_]), temb,
                             inner.config.H_cycles - 1, inner.seq_info())
    oH, oL = oH.detach(), oL.detach()
    num = P((oH[:n].float() - oH[n:].float(), oL[:n].float() - oL[n:].float()))
    diff = tuple(a.float() - b.float() for a, b in zip(s_, s2))
    den = P(diff) if block else diff
    return rownorm2(*num).sqrt() / rownorm2(*den).sqrt()


@pytest.mark.parametrize("arch", list(ARCHS), ids=list(ARCHS))
@pytest.mark.parametrize("mode", ["dec_block", "dec_out"])
def test_e2e_dec_penalty_metrics_and_grads(mode, arch):
    # reg_rho = 1e-3: every row violates, so the penalty is active and must carry gradient
    head, inner, reg, carry, outputs = _setup(dict(reg_mode=mode, reg_rho=1e-3, reg_eta=0.05, reg_monitor_every=0),
                                              **ARCHS[arch])
    assert reg.rho == 1e-3 and reg.k == 2
    calls = _spy_dec(reg)
    reg.stash(outputs, carry.steps, 1)
    reg_loss, m = reg(step=1)
    assert len(calls) == 1 and calls[0]["block"] == (mode == "dec_block") and calls[0]["power_iters"] == 1
    c = calls[0]
    assert reg_loss is not None and reg_loss.requires_grad and reg_loss.ndim == 0 and torch.isfinite(reg_loss)
    for k in ("reg/lambda", "reg/viol", "reg/value", "reg/value_max", "reg/pen", "reg/viol_ema", "reg/eps_rel",
              "reg/pair_rel_dist"):
        assert k in m and math.isfinite(m[k]), k
    assert "reg/gain_F" not in m
    # reg/value is the D1 / D2 quotient returned by _dec, the penalty is ReLU(log ratio - log rho)
    assert c["ratio"].shape == (reg.k,) and torch.all(c["ratio"] > 0)
    assert m["reg/value"] == pytest.approx(c["ratio"].mean().item(), rel=1e-6)
    assert m["reg/value_max"] == pytest.approx(c["ratio"].max().item(), rel=1e-6)
    torch.testing.assert_close(c["pen"], torch.log(c["ratio"]) - math.log(1e-3), rtol=1e-5, atol=1e-5)
    assert m["reg/viol"] == 1.0
    assert m["reg/pen"] == pytest.approx(c["pen"].mean().item(), rel=1e-5)
    assert reg_loss.item() == pytest.approx(m["reg/lambda"] * m["reg/pen"], rel=1e-5)
    # eps_rel reported is the one used by _dec; pair_rel_dist is ||diff|| / ||s|| (D1: diff = eps ||P s|| d)
    assert m["reg/eps_rel"] == pytest.approx(c["eps"].mean().item(), rel=1e-6)
    assert torch.all((c["eps"] >= reg.cfg.reg_eps_min * (1 - 1e-6)) & (c["eps"] <= reg.cfg.reg_eps_max * (1 + 1e-6)))
    s = c["s"]
    P = _ref_proj(_inner_basis(inner), P0)
    if mode == "dec_block":
        expect = c["eps"] * (rownorm2(*P(s)) / rownorm2(*s)).sqrt().float()
        torch.testing.assert_close(c["den2"].sqrt(), c["eps"] * rownorm2(*P(s)).sqrt().float(), rtol=1e-4, atol=0)
    else:
        expect = c["eps"]
    tol = 2e-2 if (arch.endswith("bf16") and mode == "dec_out") else 1e-4
    assert m["reg/pair_rel_dist"] == pytest.approx(expect.mean().item(), rel=tol)

    reg_loss.backward()
    for name, p in inner.L_level.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    if inner.use_time_embed:
        assert inner.temb_proj.weight.grad is not None and inner.temb_proj.weight.grad.abs().sum() > 0
    # the projector is a constant (built from the detached lm_head): no gradient to the decoder or embeddings
    assert inner.lm_head.weight.grad is None
    assert inner.embed_tokens.embedding_weight.grad is None

    # warm-start memory: unit directions for the regularized rows; D1 directions live in range(P)
    n2 = rownorm2(reg.mem["dir"])
    sel = reg.mem["birth"] == 1
    assert int(sel.sum()) == reg.k and torch.all(n2[~sel] == 0)
    torch.testing.assert_close(n2[sel], torch.ones(reg.k), rtol=1e-5, atol=0)
    dirs = reg.mem["dir"][sel]
    dirs_t = (dirs[:, 0], dirs[:, 1])
    if mode == "dec_block":
        assert torch.count_nonzero(dirs[:, 1]) == 0 and torch.count_nonzero(dirs[:, 0, :P0]) == 0
        torch.testing.assert_close(P(dirs_t)[0].float(), dirs[:, 0], rtol=0, atol=1e-5)
    else:
        assert rownorm2(dirs[:, 1]).min() > 0.05                       # full-space direction (z_L too)
    assert torch.count_nonzero(reg.mem["dir_F"]) == 0 and torch.count_nonzero(reg.mem["gain_F"]) == 0


@pytest.mark.parametrize("arch", ["mlp_t", "mlp_t_bf16"])
@pytest.mark.parametrize("mode", ["dec_block", "dec_out"])
def test_e2e_dec_ratio_matches_manual_secant(mode, arch):
    head, inner, reg, carry, outputs = _setup(dict(reg_mode=mode, reg_monitor_every=0, reg_v1_power_iters=2),
                                              **ARCHS[arch])
    calls = _spy_dec(reg)
    reg.stash(outputs, carry.steps, 1)
    _, m = reg(step=1)
    c = calls[0]
    ratio = _manual_ratio(inner, c["s"], c["x"], c["d"], c["eps"], c["block"])
    print(f"\n[{mode} {arch}] ratio reg={c['ratio'].tolist()} manual={ratio.tolist()}")
    torch.testing.assert_close(ratio.float(), c["ratio"].float(), rtol=1e-4, atol=0)
    assert m["reg/value"] == pytest.approx(ratio.mean().item(), rel=1e-4)


@pytest.mark.parametrize("mode", ["dec_block", "dec_out"])
def test_e2e_dec_pair_geometry(mode):
    """D1 pairs differ only inside range(P) at radius eps ||P s||; D2 pairs are full-space at radius eps ||s||."""
    head, inner, reg, carry, outputs = _setup(dict(reg_mode=mode, reg_monitor_every=0))
    calls = _spy_dec(reg)
    log = _spy_phi(reg)
    reg.stash(outputs, carry.steps, 1)
    reg(step=1)
    c = calls[0]
    k = reg.k
    pen_pass = [e for e in log if e["n"] == 2 * k]
    assert len(pen_pass) == 1 and pen_pass[0]["grad"]
    zH, zL = pen_pass[0]["z_H"], pen_pass[0]["z_L"]
    s, s2 = (zH[:k], zL[:k]), (zH[k:], zL[k:])
    torch.testing.assert_close(s[0], c["s"][0].float(), rtol=0, atol=0)
    diff = tuple(b.double() - a.double() for a, b in zip(s, s2))
    P = _ref_proj(_inner_basis(inner), P0)
    if mode == "dec_block":
        assert torch.count_nonzero(diff[1]) == 0 and torch.count_nonzero(diff[0][:, :P0]) == 0
        torch.testing.assert_close(P(diff)[0], diff[0], rtol=0, atol=1e-6)
        torch.testing.assert_close(rownorm2(*diff).sqrt().double(),
                                   (c["eps"].double() * rownorm2(*P(s)).sqrt().double()), rtol=1e-4, atol=0)
    else:
        torch.testing.assert_close(rownorm2(*diff).sqrt().double(),
                                   (c["eps"].double() * rownorm2(*s).sqrt().double()), rtol=1e-4, atol=0)
        assert torch.all(rownorm2(diff[1]) > 0)
        # the D2 perturbation is not confined to range(P)
        assert torch.all(rownorm2(*P(diff)) < 0.9 * rownorm2(*diff))


@pytest.mark.parametrize("arch", ["mlp_t_bf16", "flow_temb_bf16"])
@pytest.mark.parametrize("mode,fp32,expect", [("dec_block", True, torch.float32),
                                              ("dec_block", False, torch.bfloat16),
                                              ("dec_out", True, torch.bfloat16)],
                         ids=["D1_fp32", "D1_bf16", "D2_flag_ignored"])
def test_e2e_dec_operator_dtype(mode, fp32, expect, arch):
    head, inner, reg, carry, outputs = _setup(dict(reg_mode=mode, reg_dec_fp32=fp32, reg_rho=1e-3,
                                                   reg_monitor_every=0, reg_v1_power_iters=2), **ARCHS[arch])
    assert outputs["reg_s_H"].dtype == torch.bfloat16
    log = _spy_phi(reg)
    seen = []
    hook = inner.L_level.register_forward_pre_hook(
        lambda mod, args, kwargs: seen.append((args[0] if args else kwargs["hidden_states"]).dtype), with_kwargs=True)
    try:
        reg.stash(outputs, carry.steps, 1)
        reg_loss, m = reg(step=1)
    finally:
        hook.remove()
    # 1 reference + reg_v1_power_iters power steps + 1 penalty pass
    assert len(log) == 1 + 2 + 1
    n_args = 4 if inner.use_time_embed else 3
    for e in log:
        assert e["dtypes"] == (expect,) * n_args, e["dtypes"]
        assert e["out_dtypes"] == (expect, expect)
    assert seen and all(dt == expect for dt in seen)
    assert torch.isfinite(reg_loss) and m["reg/viol"] == 1.0
    reg_loss.backward()
    for name, p in inner.L_level.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


@pytest.mark.parametrize("mode", ["dec_block", "dec_out"])
def test_e2e_dec_huge_rho_zero_penalty(mode):
    head, inner, reg, carry, outputs = _setup(dict(reg_mode=mode, reg_rho=1e6, reg_monitor_every=0))
    assert reg.rho == 1e6
    reg.stash(outputs, carry.steps, 1)
    reg_loss, m = reg(step=1)
    assert m["reg/pen"] == 0.0 and m["reg/viol"] == 0.0 and reg_loss.item() == 0.0
    assert m["reg/value"] > 0
    reg_loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0, name


@pytest.mark.parametrize("mode", ["dec_block", "dec_out", "cycle_lip", "flow_mono", "none"])
def test_reg_rho_overrides_eta(mode):
    t = _toy("leak")
    inner = _LinearDecInner(t.J, t.W, TOY["P0"])
    r1 = ContractionRegularizer(RegConfig(reg_mode=mode, reg_eta=0.3), inner, batch_size=4, seed=0, device="cpu")
    assert r1.rho == pytest.approx(0.7)
    r2 = ContractionRegularizer(RegConfig(reg_mode=mode, reg_eta=0.3, reg_rho=2.5), inner, batch_size=4, seed=0,
                                device="cpu")
    assert r2.rho == 2.5
    # mu (V2) is still derived from eta, reg_rho does not touch it
    assert r2.mu == pytest.approx(r1.mu) and r1.mu == pytest.approx(-math.log(0.7))
    # a falsy explicit value is not confused with "unset" (the override tests `is not None`)
    r3 = ContractionRegularizer(RegConfig(reg_mode=mode, reg_eta=0.3, reg_rho=0.0), inner, batch_size=4,
                                seed=0, device="cpu")
    assert r3.rho == 0.0


def test_e2e_cycle_lip_uses_reg_rho():
    head, inner, reg, carry, outputs = _setup(dict(reg_mode="cycle_lip", reg_rho=1e6, reg_monitor_every=0))
    reg.stash(outputs, carry.steps, 1)
    reg_loss, m = reg(step=1)
    assert reg.rho == 1e6 and m["reg/pen"] == 0.0 and m["reg/viol"] == 0.0
    head2, inner2, reg2, carry2, outputs2 = _setup(dict(reg_mode="cycle_lip", reg_rho=1e-3, reg_monitor_every=0))
    reg2.stash(outputs2, carry2.steps, 1)
    _, m2 = reg2(step=1)
    assert m2["reg/viol"] == 1.0
    assert m2["reg/pen"] <= math.log(m2["reg/value"] / 1e-3) + 1e-5


@pytest.mark.parametrize("arch", ["mlp_t", "mlp_t_bf16"])
@pytest.mark.parametrize("mode", ["none", "dec_block"])
def test_e2e_monitor_dec_metrics(mode, arch):
    head, inner, reg, carry, outputs = _setup(dict(reg_mode=mode, reg_monitor_every=1), **ARCHS[arch])
    calls = _spy_dec(reg)
    reg.stash(outputs, carry.steps, 1)
    reg_loss, m = reg(step=1)
    for k in ("mon/dec_block", "mon/dec_out", "mon/v1_ratio", "mon/v2_q", "mon/fpres"):
        assert k in m and math.isfinite(m[k]), k
    assert m["mon/dec_block"] > 0 and m["mon/dec_out"] > 0
    mon = calls[-3:]                                  # monitor: D1, D2, then D-gamma, cold-started, without gradient
    assert [c["kind"] for c in mon] == ["block", "out", "semi"]
    assert all(c["power_iters"] == reg.cfg.reg_monitor_power_iters and not c["ratio"].requires_grad for c in mon)
    assert m["mon/dec_block"] == pytest.approx(mon[0]["ratio"].mean().item(), rel=1e-6)
    assert m["mon/dec_out"] == pytest.approx(mon[1]["ratio"].mean().item(), rel=1e-6)
    if mode == "none":
        assert reg_loss is None and reg.mem is None and len(calls) == 3
        assert all(p.grad is None for p in head.parameters())
    else:
        mem_before = {k: v.clone() for k, v in reg.mem.items()}
        reg.stash(outputs, carry.steps, 1)
        # monitoring alone (no active regularizer call) keeps no warm-start state
        reg.monitor(reg.pending["s"], reg.pending["phi_s"], reg.pending["x"])
        for k, v in reg.mem.items():
            assert torch.equal(v, mem_before[k]), k


def test_e2e_monitor_dec_uses_at_most_16_rows():
    B = 24
    head, inner, reg, carry, outputs = _setup(dict(reg_mode="none", reg_frac=1.0, reg_monitor_every=1),
                                              batch_size=B)
    big = {k: torch.cat([v] * 3) for k, v in outputs.items() if k.startswith("reg_")}
    calls = _spy_dec(reg)
    reg.stash(big, torch.cat([carry.steps] * 3), 1)
    assert reg.k == B
    _, m = reg(step=1)
    assert [c["s"][0].shape[0] for c in calls] == [16, 16, 16]
    assert math.isfinite(m["mon/dec_block"]) and math.isfinite(m["mon/dec_out"])


@pytest.mark.parametrize("mode", ["dec_block", "dec_out"])
def test_e2e_rand_pairs_dec(mode):
    head, inner, reg, carry, outputs = _setup(dict(reg_mode=mode, reg_pair="rand", reg_monitor_every=0))
    log = _spy_phi(reg)
    reg.stash(outputs, carry.steps, 1)
    reg_loss, m = reg(step=1)
    assert len(log) == 1                                             # no reference / power-step forwards
    assert torch.isfinite(reg_loss) and m["reg/value"] > 0
    assert reg.mem is None                                           # random pairs keep no warm start


@pytest.mark.parametrize("mode", ["dec_block", "dec_out"])
def test_e2e_warm_started_adversary_reaches_projected_jacobian_norm(mode):
    """Tiny radius, same puzzle in every row: the warm-started secant adversary converges to the Jacobian
    quantity reported by the diagnostics (sigma_max(P J P) for D1, sigma_max(P J) for D2) from below."""
    head, inner, reg, carry, outputs = _setup(dict(reg_mode=mode, reg_frac=1.0, reg_eps_min=1e-3, reg_eps_max=1e-3,
                                                   reg_monitor_every=0))
    calls = _spy_dec(reg)
    for t in range(1, 41):
        reg.stash(outputs, torch.full((BATCH,), t, dtype=torch.int32), t)
        reg(step=t)
    # row order of the last call
    last = calls[-1]
    s_last = last["s"][0]
    s_all = outputs["reg_s_H"]
    idx = torch.cdist(s_last.flatten(1), s_all.flatten(1)).argmin(1)
    assert torch.equal(s_all[idx], s_last)

    x32 = outputs["reg_x"].float()
    temb = inner.time_embeddings()
    si = inner.seq_info()

    def phi(s):
        zH, zL = inner.cycle(s[:, 0], s[:, 1], x32, temb, inner.config.H_cycles - 1, si)
        return torch.stack([zH, zL], 1)

    S = torch.stack([outputs["reg_s_H"], outputs["reg_s_L"]], 1).float()
    Pd = decoder_projector(inner)
    sig = projected_sigma_max(phi, S, 150, _gen(0), Pd, Pd if mode == "dec_block" else None).detach()
    ratio = last["ratio"]
    print(f"\n[{mode}] warm-started ratio={ratio.tolist()}\n          jacobian sigma={sig[idx].tolist()}")
    rel = (ratio.double() - sig[idx].double()) / sig[idx].double()
    assert rel.max() < 5e-3                    # a secant quotient at tiny radius cannot exceed the norm (much)
    assert rel.min() > -2e-2                   # and the warm-started adversary has (nearly) found it
    first = calls[0]["ratio"]
    assert first.mean() < ratio.mean() + 1e-6


# =============================================================================================
# (5) RegConfig
# =============================================================================================

def test_regconfig_new_modes_and_fields():
    for mode in ("none", "cycle_lip", "flow_mono", "dec_block", "dec_out"):
        assert RegConfig(reg_mode=mode).reg_mode == mode
    with pytest.raises(pydantic.ValidationError):
        RegConfig(reg_mode="dec_blok")
    cfg = RegConfig()
    assert cfg.reg_rho is None and cfg.reg_dec_fp32 is True and cfg.reg_eta == 0.05
    assert RegConfig(reg_rho=2).reg_rho == 2.0 and isinstance(RegConfig(reg_rho=2).reg_rho, float)
    assert RegConfig(reg_dec_fp32=False).reg_dec_fp32 is False


def test_regconfig_from_arch_extra_picks_reg_rho_and_dec_fp32():
    extra = dict(reg_mode="dec_block", reg_rho=0.95, reg_frac=0.125, reg_dec_fp32=False, hidden_size=512,
                 loss=dict(name="x"), flow=True)
    cfg = RegConfig.from_arch_extra(extra)
    assert cfg.reg_mode == "dec_block" and cfg.reg_rho == 0.95 and cfg.reg_frac == 0.125
    assert cfg.reg_dec_fp32 is False
    cfg2 = RegConfig.from_arch_extra(dict(reg_mode="dec_out", reg_rho=1.0))
    assert cfg2.reg_mode == "dec_out" and cfg2.reg_rho == 1.0 and cfg2.reg_dec_fp32 is True
    assert RegConfig.from_arch_extra(dict(reg_mode="cycle_lip")).reg_rho is None
    # what run_benchmark.sh passes (+arch.reg_rho=...) ends up in the regularizer
    t = _toy("leak")
    reg = ContractionRegularizer(cfg2, _LinearDecInner(t.J, t.W, TOY["P0"]), batch_size=4, seed=0, device="cpu")
    assert reg.rho == 1.0


# =============================================================================================
# (6) fixed_point_diagnostics.decoder_projector / projected_sigma_max
# =============================================================================================

def _diag_linear(J):
    return lambda s: (s.flatten(1) @ J.T.to(s.dtype)).view_as(s)


def _diag_proj(Q, P0):
    Q = Q.double()

    def P(v):
        out = torch.zeros_like(v)
        out[:, 0, P0:] = (v[:, 0, P0:] @ Q.to(v.dtype)) @ Q.T.to(v.dtype)
        return out
    return P


@pytest.mark.parametrize("case", ["invisible_expands", "leak"])
def test_projected_sigma_max_matches_matrix_norms_float64(case, float64_default):
    t = _toy(case)
    g = _gen(9)
    f = _diag_linear(t.J)
    s = torch.randn(5, 2, TOY["Ltok"], TOY["D"], generator=g)
    assert s.dtype == torch.float64
    P = _diag_proj(t.Q, TOY["P0"])
    # the diagnostics state layout (b, 2, L, D) flattens like the toy's (z_H, z_L) tuple
    v = torch.randn(3, 2, TOY["Ltok"], TOY["D"], generator=g)
    torch.testing.assert_close(P(v).flatten(1), v.flatten(1) @ t.Pm, rtol=0, atol=1e-6)

    blk = projected_sigma_max(f, s, 200, g, P, P)
    out = projected_sigma_max(f, s, 200, g, P)
    ref_blk = _sigma(t.Pm @ t.J @ t.Pm)
    ref_out = _sigma(t.Pm @ t.J)
    assert blk.shape == (5,) and out.shape == (5,)
    torch.testing.assert_close(blk, torch.full_like(blk, ref_blk), rtol=1e-6, atol=0)
    torch.testing.assert_close(out, torch.full_like(out, ref_out), rtol=1e-6, atol=0)
    # out_proj = in_proj = identity recovers the plain spectral norm
    full = projected_sigma_max(f, s, 300, g, lambda w: w, lambda w: w)
    torch.testing.assert_close(full, torch.full_like(full, t.full), rtol=1e-4, atol=0)
    if case == "invisible_expands":
        assert blk.max() < 1.6 and out.max() < 1.6 and t.full >= 10
    else:
        assert out.min() > 3.9 * blk.max()


def test_decoder_projector_matches_reference_on_stub_and_sigma_fp32():
    t = _toy("leak")
    stub = SimpleNamespace(lm_head=SimpleNamespace(weight=t.W), puzzle_emb_len=TOY["P0"])
    Pd = decoder_projector(stub)
    g = _gen(10)
    v = torch.randn(4, 2, TOY["Ltok"], TOY["D"], generator=g)
    torch.testing.assert_close(Pd(v).flatten(1).double(), v.flatten(1).double() @ t.Pm, rtol=0, atol=2e-6)
    f = _diag_linear(t.J.float())
    s = torch.randn(4, 2, TOY["Ltok"], TOY["D"], generator=g)
    blk = projected_sigma_max(f, s, 200, g, Pd, Pd)
    out = projected_sigma_max(f, s, 200, g, Pd)
    torch.testing.assert_close(blk.double(), torch.full((4,), t.d1, dtype=DT), rtol=1e-4, atol=0)
    torch.testing.assert_close(out.double(), torch.full((4,), t.d2, dtype=DT), rtol=1e-4, atol=0)


@pytest.mark.parametrize("arch", ["mlp_t", "mlp_t_bf16"])
def test_decoder_projector_on_tiny_model_is_a_projection(arch):
    head, inner, reg, carry, outputs = _setup(dict(reg_mode="none"), **ARCHS[arch])
    Pd = decoder_projector(inner)
    g = _gen(11)
    u = torch.randn(3, 2, L_TOK, HID, generator=g)
    w = torch.randn(3, 2, L_TOK, HID, generator=g)
    pu = Pd(u)
    assert pu.shape == u.shape and pu.dtype == torch.float32
    assert torch.count_nonzero(pu[:, 1]) == 0 and torch.count_nonzero(pu[:, 0, :P0]) == 0
    torch.testing.assert_close(Pd(pu), pu, rtol=0, atol=2e-6)                         # idempotent
    a = (pu * w).flatten(1).sum(-1)
    b = (u * Pd(w)).flatten(1).sum(-1)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)                            # self-adjoint
    # rank (L - P0) (V - 1) per row: trace of the per-token block is V - 1
    eye = torch.eye(HID).view(HID, 1, 1, HID).expand(HID, 2, L_TOK, HID).clone()
    tr = torch.einsum("bd,bd->", Pd(eye)[:, 0, P0], torch.eye(HID))
    assert tr.item() == pytest.approx(VOCAB - 1, abs=1e-4)
    # same projector as the regularizer's
    Q = reg._decoder_basis()
    pr = reg._proj((u[:, 0], u[:, 1]), Q)
    torch.testing.assert_close(torch.stack(pr, 1), pu, rtol=0, atol=2e-6)


# =============================================================================================
# (7) Regression tests for the fixed basis-construction bugs (fp64 SVD with a standard rank tolerance)
# =============================================================================================

def _diag_basis_projector(W, D):
    """Per-token D x D projector applied by fixed_point_diagnostics.decoder_projector."""
    Pd = decoder_projector(SimpleNamespace(lm_head=SimpleNamespace(weight=W), puzzle_emb_len=0))
    eye = torch.eye(D).view(D, 1, 1, D).expand(D, 2, 1, D).clone()
    return Pd(eye)[:, 0, 0].double()


def _impl_projector(impl, W):
    if impl == "regularizer":
        Q = _basis(W).double()
        return Q @ Q.T
    return _diag_basis_projector(W, W.shape[1])


@pytest.mark.parametrize("impl", ["regularizer", "diagnostics"])
def test_decoder_basis_rank_with_common_row_component(impl):
    V, D = 11, 512
    g = _gen(12)
    W0 = _init_like_lm_head(g, V, D)
    m = torch.randn(1, D, generator=g)
    W = W0 + 3.0 * W0.norm(dim=1).mean() * m / m.norm()
    Pm = _impl_projector(impl, W)
    rank = round(torch.trace(Pm).item())
    z1 = torch.linalg.pinv(W.double()) @ torch.ones(V, dtype=DT)
    leak = ((Pm @ z1).norm() / z1.norm()).item()
    print(f"\n[{impl}] rank={rank} (expected {V - 1}), |P z1|/|z1| = {leak:.3e}")
    assert rank == V - 1
    assert leak < 1e-4


@pytest.mark.parametrize("impl", ["regularizer", "diagnostics"])
def test_decoder_basis_spans_row_space_with_duplicate_rows(impl):
    V, D = 11, 64
    g = _gen(13)
    W = _init_like_lm_head(g, V, D)
    W[3] = W[1]
    Wc = W.double() - W.double().mean(0, keepdim=True)
    assert torch.linalg.matrix_rank(Wc).item() == V - 2
    Pm = _impl_projector(impl, W)
    err = ((Wc - Wc @ Pm).norm() / Wc.norm()).item()
    print(f"\n[{impl}] rank={torch.trace(Pm).item():.3f}, relative span error {err:.3e}")
    assert err < 1e-5
