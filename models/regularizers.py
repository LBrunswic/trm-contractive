"""Contractivity regularizers for the TRM recursive latent operator.

State s = (z_H, z_L); norms are the joint Euclidean norm over the whole state, in fp32.

V1 `cycle_lip`  : Phi = one full H-cycle. Constraint  ||Phi(s) - Phi(s~)|| / ||s - s~|| <= rho = 1 - eta.
                  Penalty ReLU(log ratio - log rho) (same zero set as ReLU(ratio - rho), but scale-free, so
                  a far-from-contractive init does not swamp the task loss).
D1 `dec_block`  : visible-block contraction: with P = projection of the z_H answer tokens onto the (centered)
                  decoder row space, ||P(Phi(s+d) - Phi(s))|| <= rho ||d|| for d in range(P). Bounds only the
                  visible->visible block A = PJP; invisible->visible leakage B = PJ(I-P) is unconstrained, so this
                  is NOT contraction onto the decoder fibres (see dec_semi).
Dg `dec_semi`   : semi-contraction onto the decoder fibres with leak tolerance gamma: full-space pairs and
                  ||P(Phi(s) - Phi(s~))|| <= rho * sqrt(||P ds||^2 + gamma^2 ||(I-P) ds||^2), i.e. ||PJP|| <= rho and
                  ||PJ(I-P)|| <= rho*gamma; then limsup ||P(s_n - s'_n)|| <= rho*gamma*sup||(I-P) ds|| / (1 - rho).
                  gamma = 1 is D2, gamma -> 0 strict semi-contraction (fibres map into fibres), gamma -> inf is D1.
D2 `dec_out`    : the decoded part of the next state is rho-Lipschitz in the full state:
                  ||P(Phi(s) - Phi(s~))|| <= rho ||s - s~|| (invisible -> invisible expansion stays free).
V2 `flow_mono`  : V(s) = joint Euler-TRM vector field. Constraint  <V(s)-V(s~), s-s~> / ||s-s~||^2 <= -mu.
                  Penalty ReLU(q + mu). mu defaults to -ln(1 - eta) / h_H (the continuous-time rate whose
                  time-h_H flow contracts z_H by 1 - eta); it is a hyperparameter, not an exact match of V1
                  (an explicit Euler step is guaranteed to contract if h ||J_V||^2 < 2 mu, a sufficient condition; compare arms with the
                  common yardsticks logged by `monitor` and the diagnostics script).

Pairs are adversarial: s~ = s + delta, |delta| = eps * ||s|| with eps log-uniform, delta from power iteration
(V1: J^T J at scale eps; V2: sym(J_V) shifted by c = 1 + 1.5 * ||J_F|| estimate so the shifted operator is PSD),
warm-started per row for as long as the same puzzle occupies that row of the ACT carry.
The strength lambda follows dual ascent toward a target violation rate (frozen during LR warmup).
The regularizer runs outside the main compiled graph on detached anchors exported by the model,
with its own RNG, so the main path (incl. ACT exploration) consumes the same randomness in every arm.
"""
from typing import Dict, Literal, Optional, Tuple
import math
import os

import pydantic
import torch
import torch.nn.functional as F


class RegConfig(pydantic.BaseModel):
    reg_mode: Literal["none", "cycle_lip", "flow_mono", "dec_block", "dec_out", "dec_semi"] = "none"
    reg_eta: float = 0.05
    reg_rho: Optional[float] = None         # explicit Lipschitz cap (overrides 1 - eta), e.g. 2.0 for a looser V1
    reg_dec_fp32: bool = True               # dec_block / dec_semi: run the regularizer path in fp32 (small visible perturbations)
    reg_gamma: float = 0.3                  # dec_semi: leak tolerance (1 = dec_out, -> 0 = strict semi-contraction)
    reg_mu: Optional[float] = None          # None -> -ln(1 - eta) / h_H
    reg_pair: Literal["adv", "rand"] = "adv"
    reg_eps_min: float = 0.05               # pair radius, relative to ||s||, log-uniform in [min, max]
    reg_eps_max: float = 0.2
    reg_frac: float = 0.25                  # fraction of batch rows used per step
    reg_v1_power_iters: int = 1             # J^T J steps per call (each = 1 fwd+bwd of Phi)
    reg_v2_power_iters: int = 3             # shifted sym(J_V) steps per call (each = 1 fwd+bwd of V)
    reg_v2_shift_factor: float = 1.5        # c = 1 + factor * ||J_F|| estimate

    # Dual ascent on lambda to hold the violation rate at a target
    reg_dual: bool = True
    reg_lambda_init: float = 1.0
    reg_lambda_min: float = 1e-2
    reg_lambda_max: float = 1e2
    reg_dual_lr: float = 0.002
    reg_dual_warmup: int = 2000             # lambda frozen at lambda_init for the first steps (LR warmup)
    reg_target_viol: float = 0.10
    reg_viol_ema: float = 0.98

    # Monitoring (all arms): both contraction quantities with a cold-started adversary
    reg_monitor_every: int = 50
    reg_monitor_power_iters: int = 3

    @staticmethod
    def from_arch_extra(extra: dict) -> "RegConfig":
        return RegConfig(**{k: v for k, v in extra.items() if k.startswith("reg_")})


# ---------------------------------------------------------------------------------------------
# Pure helpers (state = tuple of tensors with matching leading batch dim)
# ---------------------------------------------------------------------------------------------

def rownorm2(*ts: torch.Tensor) -> torch.Tensor:
    out = None
    for t in ts:
        v = t.float().flatten(1).square().sum(-1)
        out = v if out is None else out + v
    return out


def rowdot(a: Tuple[torch.Tensor, ...], b: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    return sum((x.float() * y.float()).flatten(1).sum(-1) for x, y in zip(a, b))


def _bshape(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return v.view((-1,) + (1,) * (like.ndim - 1))


def normalize_rows(d: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    n = rownorm2(*d).sqrt().clamp_min(1e-30)
    return tuple(x / _bshape(n, x) for x in d)


def secant_ratio(fa, fb, a, b):
    """||F(a)-F(b)|| / ||a-b||; denominators from the (rounded) inputs actually fed to F."""
    num2 = rownorm2(*(x.float() - y.float() for x, y in zip(fa, fb)))
    den2 = rownorm2(*(x.float() - y.float() for x, y in zip(a, b)))
    return torch.sqrt(num2 + 1e-12) / torch.sqrt(den2 + 1e-12), den2


def osl_quotient(va, vb, a, b):
    """<V(a)-V(b), a-b> / ||a-b||^2 (one-sided Lipschitz secant)."""
    d = tuple(x.float() - y.float() for x, y in zip(a, b))
    inner = rowdot(tuple(p.float() - q.float() for p, q in zip(va, vb)), d)
    den2 = rownorm2(*d)
    return inner / (den2 + 1e-12), den2


def v1_power_step(phi, s, phi_s, delta):
    """One ALR power step for the secant Lipschitz quotient of `phi` at s.

    delta is used as a raw autograd leaf (no normalization inside the graph), so the gradient
    of ||phi(s + delta) - phi(s)||^2 is 2 J^T J delta (linear case): a true power-iteration step.
    Returns the new unit direction (fp32, detached).
    """
    with torch.enable_grad():
        delta = tuple(d.detach().float().requires_grad_(True) for d in delta)
        sp = tuple((x.float() + d).to(x.dtype) for x, d in zip(s, delta))
        out = phi(*sp)
        obj = rownorm2(*(o.float() - r.float() for o, r in zip(out, phi_s))).sum()
        grads = torch.autograd.grad(obj, delta)
    return normalize_rows(tuple(g.detach() for g in grads))


def proj_power_step(phi, s, phi_ref, delta, out_proj, in_proj=None):
    """Power step on (P_out J P_in)^T (P_out J P_in): gradient of ||P_out(phi(s + d) - phi(s))||^2 w.r.t. the raw
    leaf d, then projected by P_in (None = full space). Projections map a state tuple to a state tuple."""
    with torch.enable_grad():
        delta = tuple(d.detach().float().requires_grad_(True) for d in delta)
        sp = tuple((x.float() + d).to(x.dtype) for x, d in zip(s, delta))
        out = phi(*sp)
        obj = rownorm2(*out_proj(tuple(o.float() - r.float() for o, r in zip(out, phi_ref)))).sum()
        grads = torch.autograd.grad(obj, delta)
    grads = tuple(g.detach() for g in grads)
    return normalize_rows(in_proj(grads) if in_proj is not None else grads)


def mapped_power_step(phi, s, phi_ref, w, out_proj, in_map):
    """Power step on (P_out J M)^T (P_out J M) with delta = M(w): gradient of ||P_out(phi(s + M w) - phi(s))||^2
    w.r.t. the raw leaf w (M symmetric: identity, a projection, or a metric inverse), normalized."""
    with torch.enable_grad():
        w = tuple(t.detach().float().requires_grad_(True) for t in w)
        delta = in_map(w)
        sp = tuple((x.float() + d).to(x.dtype) for x, d in zip(s, delta))
        out = phi(*sp)
        obj = rownorm2(*out_proj(tuple(o.float() - r.float() for o, r in zip(out, phi_ref)))).sum()
        # in_map may ignore parts of w (e.g. a projection zeroes z_L): those components get a zero gradient
        grads = torch.autograd.grad(obj, w, allow_unused=True)
    return normalize_rows(tuple(torch.zeros_like(t) if g is None else g.detach() for g, t in zip(grads, w)))


def f_gain_power_step(field, s, v_s, delta):
    """Power step on J_F^T J_F for F = V + I (the block maps): returns (new unit direction, gain ||dF||/||d||)."""
    with torch.enable_grad():
        delta = tuple(d.detach().float().requires_grad_(True) for d in delta)
        sp = tuple((x.float() + d).to(x.dtype) for x, d in zip(s, delta))
        v_p = field(*sp)
        dF = tuple(p - r.float() + d for p, r, d in zip(v_p, v_s, delta))
        obj = rownorm2(*dF).sum()
        grads = torch.autograd.grad(obj, delta)
    with torch.no_grad():
        gain = rownorm2(*dF).sqrt() / rownorm2(*delta).sqrt().clamp_min(1e-30)
    return normalize_rows(tuple(g.detach() for g in grads)), gain


def v2_power_step(field, s, v_s, delta, shift: Optional[torch.Tensor] = None):
    """One shifted power step toward the most expanding direction of <V(s+d) - V(s), d>.

    grad_d <V(s+d) - V(s), d> = (V(s+d) - V(s)) + J_V(s+d)^T d  (= 2 sym(J_V) d in the linear case), so
    d1 = normalize(grad / 2 + c d) is a power step on sym(J_V) + c I. Power iteration converges to the
    eigenvalue of largest *magnitude*, so c must make the shifted operator PSD: c >= -lambda_min(sym J_V).
    Since sym(J_V) = sym(J_F) - I, c = 1 + ||J_F|| suffices (per row). If `shift` is None, c falls back to
    1 + 1.5 * (gain of F along d), which is not guaranteed.
    Returns (new unit direction, gain of F along d).
    """
    with torch.enable_grad():
        delta = tuple(d.detach().float().requires_grad_(True) for d in delta)
        sp = tuple((x.float() + d).to(x.dtype) for x, d in zip(s, delta))
        v_p = field(*sp)
        dv = tuple(p - r.float() for p, r in zip(v_p, v_s))
        g = rowdot(dv, delta).sum()
        grads = torch.autograd.grad(g, delta)

    with torch.no_grad():
        dn = rownorm2(*delta).sqrt().clamp_min(1e-30)
        gain_F = rownorm2(*(x + d for x, d in zip(dv, delta))).sqrt() / dn
        c = shift if shift is not None else 1.0 + 1.5 * gain_F
        new = tuple(0.5 * gr + _bshape(c, d) * d.detach() for d, gr in zip(delta, grads))
    return normalize_rows(new), gain_F


# ---------------------------------------------------------------------------------------------
# Regularizer driving the model operator
# ---------------------------------------------------------------------------------------------

class ContractionRegularizer:
    BUCKETS = ((1, 2), (3, 5), (6, 10), (11, 10_000))

    def __init__(self, cfg: RegConfig, inner: torch.nn.Module, batch_size: int, seed: int, device="cuda"):
        self.cfg = cfg
        self.inner = inner
        self.B = batch_size
        self.k = max(1, math.ceil(cfg.reg_frac * batch_size))
        self.device = device

        self.h = float(getattr(inner, "h_H", 1.0))
        self.mu = cfg.reg_mu if cfg.reg_mu is not None else -math.log(1.0 - cfg.reg_eta) / self.h
        self.rho = cfg.reg_rho if cfg.reg_rho is not None else 1.0 - cfg.reg_eta

        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed + 1_000_003)

        self.lam = cfg.reg_lambda_init
        self.viol_ema = cfg.reg_target_viol
        # Per-row warm-start memory, valid while the same puzzle occupies the row (same "birth" step)
        self.mem: Optional[Dict[str, torch.Tensor]] = None
        self.pending: Optional[dict] = None
        self.n_calls = 0

        L = inner.config.L_cycles
        self.i_last = inner.config.H_cycles - 1
        seq_info = inner.seq_info()
        i_last = self.i_last

        def phi(z_H, z_L, x, temb):
            return inner.cycle(z_H, z_L, x, temb, i_last, seq_info)

        def field(z_H, z_L, x, temb_H, temb_L):
            return inner.field(z_H, z_L, x, temb_H, temb_L, seq_info)

        if "DISABLE_COMPILE" not in os.environ:
            # (rows, dtype, grad-mode) variants of the regularizer/monitor exceed dynamo's default limit of 8
            torch._dynamo.config.recompile_limit = max(getattr(torch._dynamo.config, "recompile_limit", 8), 32)
            phi = torch.compile(phi, dynamic=False)
            field = torch.compile(field, dynamic=False)
        self._phi, self._field = phi, field
        self._L = L

    # ------------------------------------------------------------------ state
    def state_dict(self):
        return dict(lam=self.lam, viol_ema=self.viol_ema, n_calls=self.n_calls, gen=self.gen.get_state().clone(),
                    mem=None if self.mem is None else {k: v.detach().cpu().clone() for k, v in self.mem.items()})

    def load_state_dict(self, sd):
        self.lam, self.viol_ema, self.n_calls = sd["lam"], sd["viol_ema"], sd["n_calls"]
        self.gen.set_state(sd["gen"].cpu())
        mem = sd.get("mem")  # older checkpoints (no birth bookkeeping) restart warm starts cold
        self.mem = None if mem is None else {k: v.to(self.device).clone() for k, v in mem.items()}

    # ------------------------------------------------------------------ sampling
    def _rand(self, *shape):
        return torch.randn(shape, device=self.device, generator=self.gen)

    def _rand_dir(self, s):
        return normalize_rows((self._rand(*s[0].shape), self._rand(*s[1].shape)))

    def _radius(self, s):
        u = torch.rand((self.k,), device=self.device, generator=self.gen)
        lo, hi = math.log(self.cfg.reg_eps_min), math.log(self.cfg.reg_eps_max)
        eps_rel = torch.exp(lo + (hi - lo) * u)
        return eps_rel * rownorm2(*s).sqrt(), eps_rel

    def _ensure_mem(self, s):
        if self.mem is None:
            shape = (self.B, 2) + tuple(s[0].shape[1:])
            self.mem = dict(dir=torch.zeros(shape, device=self.device), dir_F=torch.zeros(shape, device=self.device),
                            gain_F=torch.zeros(self.B, device=self.device),
                            birth=torch.full((self.B,), -1, device=self.device, dtype=torch.long))

    def _recall(self, idx, birth, s):
        """Warm-start state for the selected rows (cold start for rows holding a new puzzle)."""
        self._ensure_mem(s)
        fresh = self.mem["birth"][idx] != birth
        f = _bshape(fresh, s[0])
        out = {}
        for key in ("dir", "dir_F"):
            stored = self.mem[key][idx]
            rand = self._rand_dir(s)
            out[key] = normalize_rows((torch.where(f, rand[0], stored[:, 0]), torch.where(f, rand[1], stored[:, 1])))
        out["gain_F"] = torch.where(fresh, torch.zeros_like(self.mem["gain_F"][idx]), self.mem["gain_F"][idx])
        return out

    def _store(self, idx, birth, **values):
        self.mem["birth"][idx] = birth
        for key, v in values.items():
            self.mem[key][idx] = torch.stack(v, 1).detach() if isinstance(v, tuple) else v.detach()

    @staticmethod
    def _scale(d, r):
        return tuple(x * _bshape(r, x) for x in d)

    @staticmethod
    def _perturb(s, delta):
        return tuple((x.float() + d).to(x.dtype) for x, d in zip(s, delta))

    def _temb(self, grad: bool):
        with torch.set_grad_enabled(grad):
            return self.inner.time_embeddings()

    # ------------------------------------------------------------------ quantities
    def _v1(self, s, x, d, r, power_iters: int, with_grad: bool):
        temb_ng = self._temb(False)
        phi_ng = lambda zh, zl: self._phi(zh, zl, x, temb_ng)
        if power_iters > 0:
            # Reference Phi(s) with the *current* parameters (stashed anchors are one optimizer step old)
            with torch.no_grad():
                phi_ref = phi_ng(*s)
        for _ in range(power_iters):
            d = v1_power_step(phi_ng, s, phi_ref, self._scale(d, r))
        s2 = self._perturb(s, self._scale(d, r))

        # Penalty pass: Phi(s) and Phi(s~) recomputed together (same kernels -> consistent bf16 rounding)
        with torch.set_grad_enabled(with_grad):
            temb = self._temb(with_grad)
            out_H, out_L = self._phi(torch.cat([s[0], s2[0]]), torch.cat([s[1], s2[1]]), torch.cat([x, x]), temb)
            fa, fb = (out_H[:self.k], out_L[:self.k]), (out_H[self.k:], out_L[self.k:])
            ratio, den2 = secant_ratio(fa, fb, s, s2)
            pen = F.relu(torch.log(ratio) - math.log(self.rho))
        return pen, ratio, den2, d

    def _field_times(self, temb, j):
        """Times at which each block is integrated in the last H-cycle (matching the anchor):
        z_L at L-substep j (per row), z_H at the H-substep."""
        if temb is None:
            return None, None
        te_L = temb[self.inner.substep_index(self.i_last, 0) + j].unsqueeze(1)     # (k, 1, D)
        te_H = temb[self.inner.substep_index(self.i_last, self._L)].view(1, 1, -1)
        return te_H, te_L

    def _v2(self, s, x, d, dF, gain_F, r, power_iters: int, with_grad: bool):
        j = torch.randint(0, self._L, (self.k,), device=self.device, generator=self.gen)
        temb_ng = self._temb(False)
        te = self._field_times(temb_ng, j)
        field_ng = lambda zh, zl: self._field(zh, zl, x, te[0], te[1])
        if power_iters > 0:
            with torch.no_grad():
                v_s = field_ng(*s)
            # ||J_F|| estimate (warm-started power step) -> PSD shift for the sym(J_V) iteration
            dF, g = f_gain_power_step(field_ng, s, v_s, self._scale(dF, r))
            gain_F = torch.maximum(gain_F, g)
            for _ in range(power_iters):
                shift = 1.0 + self.cfg.reg_v2_shift_factor * gain_F
                d, g = v2_power_step(field_ng, s, v_s, self._scale(d, r), shift=shift)
                gain_F = torch.maximum(gain_F, g)
        s2 = self._perturb(s, self._scale(d, r))

        with torch.set_grad_enabled(with_grad):
            temb = self._temb(with_grad)
            te_H, te_L = self._field_times(temb, j)
            te_L = None if te_L is None else torch.cat([te_L, te_L])
            v_H, v_L = self._field(torch.cat([s[0], s2[0]]), torch.cat([s[1], s2[1]]), torch.cat([x, x]), te_H, te_L)
            va, vb = (v_H[:self.k], v_L[:self.k]), (v_H[self.k:], v_L[self.k:])
            q, den2 = osl_quotient(va, vb, s, s2)
            pen = F.relu(q + self.mu)
        return pen, q, den2, d, dF, gain_F

    def _decoder_basis(self) -> torch.Tensor:
        """Orthonormal basis (D, r) of the centered decoder row space (argmax-invariant directions removed)."""
        W = self.inner.lm_head.weight.detach().double()
        Wc = W - W.mean(0, keepdim=True)
        # fp64 SVD with a standard rank tolerance: right singular vectors span the centered row space
        _, S, Vh = torch.linalg.svd(Wc, full_matrices=False)
        tol = max(Wc.shape) * torch.finfo(torch.float64).eps * S[0]
        return Vh[S > tol].T.float()

    def _proj(self, v, Q):
        """Project a state tuple onto the decoder-visible subspace: z_H answer tokens on span(Q); z_L -> 0."""
        P0 = self.inner.puzzle_emb_len
        vis = torch.zeros(v[0].shape, device=v[0].device, dtype=torch.float32)
        vis[:, P0:] = (v[0][:, P0:].float() @ Q) @ Q.T
        return vis, torch.zeros(v[1].shape, device=v[1].device, dtype=torch.float32)

    def _dec(self, s, x, d, power_iters: int, with_grad: bool, block: bool = False, kind: Optional[str] = None):
        """Decoder-transverse quotients. kind: 'block' (D1), 'out' (D2) or 'semi' (D-gamma).

        Pairs s~ = s + M(r w): M = P (block), I (out), W_gamma^{-1} (semi); denominators ||Metric(s - s~)|| with
        Metric = P, I, W_gamma; numerators ||P(Phi(s) - Phi(s~))||. Radius r = eps * ||Metric(s)||.
        """
        kind = kind or ("block" if block else "out")
        n = s[0].shape[0]
        fp32 = kind in ("block", "semi") and self.cfg.reg_dec_fp32
        dt = torch.float32 if fp32 else s[0].dtype
        s_, x_ = tuple(t.to(dt) for t in s), x.to(dt)
        Q = self._decoder_basis()
        P = lambda v: self._proj(v, Q)
        comp = lambda v: tuple(t.float() - p for t, p in zip(v, P(v)))           # (I - P) v
        g = self.cfg.reg_gamma
        if kind == "block":
            in_map, metric = P, P
        elif kind == "out":
            in_map = metric = (lambda v: tuple(t.float() for t in v))
        else:
            metric = lambda v: tuple(p + g * c for p, c in zip(P(v), comp(v)))     # W_gamma
            in_map = lambda v: tuple(p + c / g for p, c in zip(P(v), comp(v)))     # W_gamma^{-1}

        u = torch.rand((n,), device=self.device, generator=self.gen)
        lo, hi = math.log(self.cfg.reg_eps_min), math.log(self.cfg.reg_eps_max)
        eps_rel = torch.exp(lo + (hi - lo) * u)
        r = eps_rel * rownorm2(*metric(s_)).sqrt()
        d = normalize_rows(P(d)) if kind == "block" else normalize_rows(tuple(t.float() for t in d))

        temb_ng = self._temb(False)
        temb_ng = None if temb_ng is None else temb_ng.to(dt)
        phi_ng = lambda zh, zl: self._phi(zh, zl, x_, temb_ng)
        if power_iters > 0:
            with torch.no_grad():
                phi_ref = phi_ng(*s_)
        for _ in range(power_iters):
            d = mapped_power_step(phi_ng, s_, phi_ref, self._scale(d, r), out_proj=P, in_map=in_map)
            if kind == "block":
                d = normalize_rows(P(d))
        s2 = self._perturb(s_, in_map(self._scale(d, r)))

        with torch.set_grad_enabled(with_grad):
            temb = self._temb(with_grad)
            temb = None if temb is None else temb.to(dt)
            out_H, out_L = self._phi(torch.cat([s_[0], s2[0]]), torch.cat([s_[1], s2[1]]), torch.cat([x_, x_]), temb)
            fa, fb = (out_H[:n], out_L[:n]), (out_H[n:], out_L[n:])
            num2 = rownorm2(*P(tuple(a.float() - b.float() for a, b in zip(fa, fb))))
            den2 = rownorm2(*metric(tuple(a.float() - b.float() for a, b in zip(s_, s2))))
            ratio = torch.sqrt(num2 + 1e-12) / torch.sqrt(den2 + 1e-12)
            pen = F.relu(torch.log(ratio) - math.log(self.rho))
        return pen, ratio, den2, d, eps_rel

    # ------------------------------------------------------------------ main entry
    @torch.no_grad()
    def stash(self, outs: Dict[str, torch.Tensor], steps: torch.Tensor, global_step: int):
        """Keep k detached anchor rows of this step; they are regularized at the next step.

        Running the regularizer (and its backward) before the next main forward keeps peak memory at
        max(main, regularizer) instead of their sum. The states are one optimizer step stale, which is
        immaterial for a penalty that only needs samples of visited states.
        """
        idx = torch.randperm(self.B, device=self.device, generator=self.gen)[:self.k]
        self.pending = dict(
            idx=idx,
            s=(outs["reg_s_H"][idx].clone(), outs["reg_s_L"][idx].clone()),
            phi_s=(outs["reg_phi_H"][idx].clone(), outs["reg_phi_L"][idx].clone()),
            x=outs["reg_x"][idx].clone(),
            act_steps=steps[idx].clone(),
            birth=global_step - steps[idx].long() + 1,   # identifies the puzzle occupying the row
        )

    def __call__(self, step: int):
        """Regularize the stashed rows. Returns (reg_loss or None, metrics); reg_loss is on the per-row mean scale."""
        metrics = {}
        reg_loss = None
        pending = self.pending
        if pending is None:
            return reg_loss, metrics
        self.pending = None
        self.n_calls += 1

        idx, s, phi_s, x, act_steps, birth = (pending[k] for k in ("idx", "s", "phi_s", "x", "act_steps", "birth"))
        active = self.cfg.reg_mode != "none"

        if active:
            r, eps_rel = self._radius(s)
            adv = self.cfg.reg_pair == "adv"
            mem = self._recall(idx, birth, s) if adv else dict(dir=self._rand_dir(s), dir_F=self._rand_dir(s), gain_F=torch.zeros(self.k, device=self.device))
            if self.cfg.reg_mode == "cycle_lip":
                pen, val, den2, d = self._v1(s, x, mem["dir"], r, self.cfg.reg_v1_power_iters if adv else 0, with_grad=True)
                if adv:
                    self._store(idx, birth, dir=d)
            elif self.cfg.reg_mode in ("dec_block", "dec_out", "dec_semi"):
                kind = {"dec_block": "block", "dec_out": "out", "dec_semi": "semi"}[self.cfg.reg_mode]
                pen, val, den2, d, eps_rel = self._dec(s, x, mem["dir"], self.cfg.reg_v1_power_iters if adv else 0,
                                                       with_grad=True, kind=kind)
                if adv:
                    self._store(idx, birth, dir=d)
            else:
                pen, val, den2, d, dF, gain_F = self._v2(s, x, mem["dir"], mem["dir_F"], mem["gain_F"], r,
                                                        self.cfg.reg_v2_power_iters if adv else 0, with_grad=True)
                if adv:
                    self._store(idx, birth, dir=d, dir_F=dF, gain_F=gain_F)
                metrics["reg/gain_F"] = gain_F.mean().item()

            reg_loss = self.lam * pen.mean()

            with torch.no_grad():
                viol = (pen > 0).float()
                viol_rate = viol.mean().item()
                self.viol_ema = self.cfg.reg_viol_ema * self.viol_ema + (1 - self.cfg.reg_viol_ema) * viol_rate
                metrics.update({
                    "reg/lambda": self.lam,
                    "reg/pen": pen.mean().item(),
                    "reg/value": val.mean().item(),
                    "reg/value_max": val.max().item(),
                    "reg/viol": viol_rate,
                    "reg/viol_ema": self.viol_ema,
                    "reg/eps_rel": eps_rel.mean().item(),
                    "reg/pair_rel_dist": (den2 / rownorm2(*s).clamp_min(1e-12)).sqrt().mean().item(),
                })
                for lo, hi in self.BUCKETS:
                    m = ((act_steps >= lo) & (act_steps <= hi)).float()
                    n = m.sum().item()
                    if n > 0:
                        metrics[f"reg/viol_act{lo}"] = (viol * m).sum().item() / n

            if self.cfg.reg_dual and step > self.cfg.reg_dual_warmup:
                self.lam = float(min(self.cfg.reg_lambda_max, max(self.cfg.reg_lambda_min,
                                     self.lam * math.exp(self.cfg.reg_dual_lr * (self.viol_ema - self.cfg.reg_target_viol)))))

        if self.cfg.reg_monitor_every > 0 and (step % self.cfg.reg_monitor_every == 0):
            metrics.update(self.monitor(s, phi_s, x))
        return reg_loss, metrics

    @torch.no_grad()
    def monitor(self, s, phi_s, x):
        """Contraction census on training states, identical procedure for every arm (cold start)."""
        it = self.cfg.reg_monitor_power_iters
        r, _ = self._radius(s)
        _, ratio, _, _ = self._v1(s, x, self._rand_dir(s), r, it, with_grad=False)
        zeros = torch.zeros(self.k, device=self.device)
        _, q, _, _, _, gain_F = self._v2(s, x, self._rand_dir(s), self._rand_dir(s), zeros, r, it, with_grad=False)

        m = min(16, s[0].shape[0])
        s_m, x_m = (s[0][:m], s[1][:m]), x[:m]
        _, dec_block, *_ = self._dec(s_m, x_m, self._rand_dir(s_m), it, with_grad=False, block=True)
        _, dec_out, *_ = self._dec(s_m, x_m, self._rand_dir(s_m), it, with_grad=False, block=False)
        _, dec_semi, *_ = self._dec(s_m, x_m, self._rand_dir(s_m), it, with_grad=False, kind="semi")

        fpres = (rownorm2(*(a.float() - b.float() for a, b in zip(phi_s, s))) / rownorm2(*s)).sqrt()
        z = s[0].float() + s[1].float()
        return {
            "mon/v1_ratio": ratio.mean().item(),
            "mon/v1_ratio_p90": torch.quantile(ratio, 0.9).item(),
            "mon/v1_frac_gt_rho": (ratio > self.rho).float().mean().item(),
            "mon/v1_frac_gt1": (ratio > 1).float().mean().item(),
            "mon/v2_q": q.mean().item(),
            "mon/v2_q_p90": torch.quantile(q, 0.9).item(),
            "mon/v2_frac_gt_negmu": (q > -self.mu).float().mean().item(),
            "mon/gain_F": gain_F.mean().item(),
            "mon/dec_block": dec_block.mean().item(),
            "mon/dec_out": dec_out.mean().item(),
            "mon/dec_semi": dec_semi.mean().item(),
            "mon/fpres": fpres.mean().item(),
            "mon/x_over_z": (rownorm2(x) / rownorm2(z)).sqrt().mean().item(),
        }
