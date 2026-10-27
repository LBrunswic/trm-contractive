"""Tests for models/adam_atan2_torch.py, the pure-PyTorch port of the adam-atan2 0.0.3 CUDA optimizer.

The reference below is derived from adam-atan2 0.0.3 (https://github.com/imoneoi/adam_atan2), Copyright (c) its
authors, licensed under the Apache License, Version 2.0 (see LICENSES/Apache-2.0.txt). Modified: the kernel in
csrc/adam_atan2.cu is re-expressed per element in PyTorch and evaluated in the kernel's opmath (fp64 for fp64
params, fp32 otherwise):

    param      *= 1 - lr * wd
    exp_avg     = lerp(exp_avg,    grad,        1 - beta1)     lerp(v0, v1, t) = fma(t, v1, fma(-t, v0, v0))
    exp_avg_sq  = lerp(exp_avg_sq, grad * grad, 1 - beta2)
    denom       = sqrt(exp_avg_sq) / sqrt(1 - beta2^step)
    param      -= lr / (1 - beta1^step) * atan2(exp_avg, denom)

Host scalars (step_size, wd_alpha, 1 - beta, sqrt(bias_correction2)) are computed in double and cast to opmath.
Tolerances follow the upstream reference test: fp64 1e-15, fp32 1e-6, bf16 5e-3 (absolute), 100 steps,
weight_decay 0.1, betas (0.9, 0.95), init_std 0.02, grad_std 1e-3.
"""
import copy
import io
import math

import pytest
import torch

from models.adam_atan2_torch import AdamATan2

DEVICE = torch.device("cpu")
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
INIT_STD = 0.02
GRAD_STD = 1e-3
STEPS = 100
ATOL = {torch.float64: 1e-15, torch.float32: 1e-6, torch.bfloat16: 5e-3}


def _bf16_supported() -> bool:
    try:
        a = torch.randn(8, dtype=torch.bfloat16)
        torch._foreach_lerp_([a.float()], [a.float()], 0.5)
        torch.atan2(a.float(), a.float().abs() + 1)
        return True
    except Exception:  # pragma: no cover - CPU builds without bf16
        return False


DTYPES = [torch.float64, torch.float32] + ([torch.bfloat16] if _bf16_supported() else [])


# ---------------------------------------------------------------------------------------------------------
# Reference transcription of the CUDA kernel
# ---------------------------------------------------------------------------------------------------------

def _opmath(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fma(a, b, c) = a * b + c with a single rounding.

    fp32: the product of two fp32 values is exact in fp64, so computing in fp64 and rounding once to fp32
    reproduces a hardware fma up to (extremely rare) double-rounding ties. fp64: plain a * b + c.
    """
    if a.dtype == torch.float32:
        return (a.double() * b.double() + c.double()).float()
    return a * b + c


def _kernel_lerp(v0: torch.Tensor, v1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return _fma(t, v1, _fma(-t, v0, v0))


def reference_step(param, grad, exp_avg, exp_avg_sq, *, step: int, lr: float, beta1: float, beta2: float,
                   weight_decay: float) -> None:
    """One kernel step, in place on (param, exp_avg, exp_avg_sq) stored in the parameter dtype."""
    opmath = _opmath(param.dtype)

    def scalar(v: float) -> torch.Tensor:  # double host scalar -> opmath kernel argument
        return torch.tensor(v, dtype=opmath)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    step_size = scalar(lr / bias_correction1)
    wd_alpha = scalar(1 - lr * weight_decay)
    mbeta1 = scalar(1 - beta1)
    mbeta2 = scalar(1 - beta2)
    bias_correction2_sqrt = scalar(math.sqrt(bias_correction2))

    p = param.to(opmath)
    g = grad.to(opmath)
    m = exp_avg.to(opmath)
    v = exp_avg_sq.to(opmath)

    p = p * wd_alpha
    m = _kernel_lerp(m, g, mbeta1)
    v = _kernel_lerp(v, g * g, mbeta2)
    denom = torch.sqrt(v) / bias_correction2_sqrt
    p = p - step_size * torch.atan2(m, denom)

    param.copy_(p)
    exp_avg.copy_(m)
    exp_avg_sq.copy_(v)


class ReferenceState:
    def __init__(self, init: torch.Tensor):
        self.param = init.clone()
        self.exp_avg = torch.zeros_like(init)
        self.exp_avg_sq = torch.zeros_like(init)
        self.step = 0

    def update(self, grad, *, lr, beta1=BETAS[0], beta2=BETAS[1], weight_decay=WEIGHT_DECAY):
        self.step += 1
        reference_step(self.param, grad, self.exp_avg, self.exp_avg_sq, step=self.step, lr=lr,
                       beta1=beta1, beta2=beta2, weight_decay=weight_decay)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Integer view of a float tensor, for bit-exact comparisons (distinguishes -0.0 / +0.0, NaN payloads)."""
    int_dtype = {torch.float64: torch.int64, torch.float32: torch.int32,
                 torch.bfloat16: torch.int16, torch.float16: torch.int16}[t.dtype]
    return t.contiguous().view(int_dtype)


def assert_bitwise_equal(a: torch.Tensor, b: torch.Tensor, msg: str = ""):
    assert a.dtype == b.dtype and a.shape == b.shape, msg
    assert torch.equal(_bits(a), _bits(b)), f"{msg}: max |diff| = {(a.double() - b.double()).abs().max().item()}"


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, what: str):
    diff = (actual.double() - expected.double()).abs().max().item()
    assert diff <= atol, f"{what}: max |diff| {diff:.3e} > atol {atol:.1e}"


def _make_grads(shape, dtype, n, seed):
    gen = torch.Generator().manual_seed(seed)
    return [torch.empty(shape, dtype=torch.float64).normal_(std=GRAD_STD, generator=gen).to(dtype) for _ in range(n)]


def _make_init(shape, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.empty(shape, dtype=torch.float64).normal_(std=INIT_STD, generator=gen).to(dtype)


# ---------------------------------------------------------------------------------------------------------
# (1) Port vs kernel reference, 100 steps
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).replace("torch.", ""))
@pytest.mark.parametrize("lr", [1e-3, 1e-4, 5e-4])
@pytest.mark.parametrize("shape", [(1,), (4096,), (64, 512)], ids=lambda s: "x".join(map(str, s)))
def test_matches_kernel_reference(shape, lr, dtype):
    init = _make_init(shape, dtype, seed=0)
    grads = _make_grads(shape, dtype, STEPS, seed=1)

    p = torch.nn.Parameter(init.clone())
    opt = AdamATan2([p], lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY)
    ref = ReferenceState(init)
    atol = ATOL[dtype]

    for i, g in enumerate(grads):
        p.grad = g.clone()
        opt.step()
        ref.update(g, lr=lr)

        state = opt.state[p]
        assert float(state["step"]) == i + 1
        assert state["exp_avg"].dtype == dtype and state["exp_avg_sq"].dtype == dtype
        _assert_close(p.detach(), ref.param, atol, f"param @ step {i + 1}")
        _assert_close(state["exp_avg"], ref.exp_avg, atol, f"exp_avg @ step {i + 1}")
        _assert_close(state["exp_avg_sq"], ref.exp_avg_sq, atol, f"exp_avg_sq @ step {i + 1}")

    # The moments are tiny (|m| ~ 1e-3, v ~ 1e-6), so the absolute tolerances above are loose for them;
    # check them relative to their own scale as well in fp32/fp64 (only lerp rounding differs: PyTorch's
    # formula vs the kernel's fma form, i.e. a few ulps of the operands; elementwise relative error is
    # meaningless where exp_avg cancels to ~0).
    if dtype != torch.bfloat16:
        rel = 1e-13 if dtype == torch.float64 else 1e-6
        state = opt.state[p]
        for key, expected in (("exp_avg", ref.exp_avg), ("exp_avg_sq", ref.exp_avg_sq)):
            _assert_close(state[key], expected, rel * expected.abs().max().item(), f"{key} (scale-relative)")

    # The update must actually have moved the parameters (guards against a no-op optimizer passing)
    assert (p.detach().double() - init.double()).abs().max().item() > lr


def test_mixed_group_buckets_match_reference():
    """One param group with several shapes/dtypes and a param whose grad is None every other step.

    The port buckets by (step, device, dtype) for foreach kernels; each tensor must still follow its own
    step counter and dtype exactly as an independent per-tensor reference.
    """
    lr = 5e-4
    specs = [((7,), torch.float32), ((3, 5), torch.float64), ((16,), torch.float32), ((4, 4), torch.float64)]
    if torch.bfloat16 in DTYPES:
        specs.append(((33,), torch.bfloat16))
    params, refs, grads = [], [], []
    for k, (shape, dtype) in enumerate(specs):
        init = _make_init(shape, dtype, seed=10 + k)
        params.append(torch.nn.Parameter(init.clone()))
        refs.append(ReferenceState(init))
        grads.append(_make_grads(shape, dtype, 30, seed=100 + k))
    opt = AdamATan2(params, lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY)

    skip_every_other = 2  # params[2] (fp32, same dtype as params[0]) only gets a grad on even steps
    for i in range(30):
        for k, p in enumerate(params):
            p.grad = None if (k == skip_every_other and i % 2 == 1) else grads[k][i].clone()
        opt.step()
        for k, p in enumerate(params):
            if p.grad is not None:
                refs[k].update(grads[k][i], lr=lr)

    for k, (p, ref) in enumerate(zip(params, refs)):
        atol = ATOL[p.dtype]
        assert float(opt.state[p]["step"]) == ref.step, k
        _assert_close(p.detach(), ref.param, atol, f"param {k}")
        _assert_close(opt.state[p]["exp_avg"], ref.exp_avg, atol, f"exp_avg {k}")
        _assert_close(opt.state[p]["exp_avg_sq"], ref.exp_avg_sq, atol, f"exp_avg_sq {k}")
    assert refs[skip_every_other].step == 15 and refs[0].step == 30


# ---------------------------------------------------------------------------------------------------------
# (2) lr = 0
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).replace("torch.", ""))
def test_lr_zero_leaves_params_bit_identical_but_updates_moments(dtype):
    shape = (64, 33)
    init = _make_init(shape, dtype, seed=3)
    # Exact zero and tiny magnitudes must survive too. (-0.0 is left out on purpose: the kernel itself computes
    # -0.0 - (+0.0 * atan2(m, d)) and may legitimately return +0.0, so its sign is not preserved there either.)
    init.view(-1)[:3] = torch.tensor([0.0, 1e-30, -1e-30], dtype=torch.float64).to(dtype)
    grads = _make_grads(shape, dtype, 5, seed=4)
    grads[0].view(-1)[:2] = 0  # atan2(0, 0) path

    p = torch.nn.Parameter(init.clone())
    opt = AdamATan2([p], lr=0.0, betas=BETAS, weight_decay=WEIGHT_DECAY)
    ref = ReferenceState(init)
    for g in grads:
        p.grad = g.clone()
        opt.step()
        ref.update(g, lr=0.0)
        assert_bitwise_equal(p.detach(), init, "lr=0 changed the parameters")

    state = opt.state[p]
    assert float(state["step"]) == len(grads)
    assert state["exp_avg"].abs().max() > 0 and state["exp_avg_sq"].abs().max() > 0
    atol = ATOL[dtype]
    _assert_close(state["exp_avg"], ref.exp_avg, atol, "exp_avg")
    _assert_close(state["exp_avg_sq"], ref.exp_avg_sq, atol, "exp_avg_sq")
    assert torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()


def test_lr_zero_then_schedule_counts_steps():
    """pretrain.py builds the optimizer with lr=0 and sets group['lr'] every step: bias correction must use
    the total step count, including the lr=0 steps."""
    init = _make_init((128,), torch.float32, seed=5)
    grads = _make_grads((128,), torch.float32, 6, seed=6)
    p = torch.nn.Parameter(init.clone())
    opt = AdamATan2([p], lr=0, betas=BETAS, weight_decay=WEIGHT_DECAY)
    ref = ReferenceState(init)
    for i, g in enumerate(grads):
        lr = 0.0 if i < 3 else 1e-3 * i
        for group in opt.param_groups:
            group["lr"] = lr
        p.grad = g.clone()
        opt.step()
        ref.update(g, lr=lr)
    _assert_close(p.detach(), ref.param, ATOL[torch.float32], "param")
    assert float(opt.state[p]["step"]) == 6


# ---------------------------------------------------------------------------------------------------------
# (3) grad None is skipped
# ---------------------------------------------------------------------------------------------------------

def test_params_without_grad_are_skipped():
    a = torch.nn.Parameter(_make_init((10,), torch.float32, seed=7))
    b = torch.nn.Parameter(_make_init((10,), torch.float32, seed=8))
    b_init = b.detach().clone()
    opt = AdamATan2([a, b], lr=1e-3, betas=BETAS, weight_decay=WEIGHT_DECAY)

    for i in range(3):
        a.grad = _make_grads((10,), torch.float32, 1, seed=20 + i)[0]
        b.grad = None
        opt.step()

    # b: untouched (not even weight decay), no state created
    assert_bitwise_equal(b.detach(), b_init, "param without grad was modified")
    assert b not in opt.state or len(opt.state[b]) == 0
    assert float(opt.state[a]["step"]) == 3

    # Once b gets a grad, its own step counter starts at 1 and it matches a fresh reference
    gb = _make_grads((10,), torch.float32, 1, seed=30)[0]
    ref = ReferenceState(b_init)
    a.grad = None
    b.grad = gb.clone()
    a_before = a.detach().clone()
    opt.step()
    ref.update(gb, lr=1e-3)
    assert float(opt.state[b]["step"]) == 1
    assert float(opt.state[a]["step"]) == 3
    assert_bitwise_equal(a.detach(), a_before, "param without grad was modified")
    _assert_close(b.detach(), ref.param, ATOL[torch.float32], "param b")


def test_all_grads_none_is_a_noop_and_closure_is_returned():
    p = torch.nn.Parameter(_make_init((5,), torch.float32, seed=9))
    before = p.detach().clone()
    opt = AdamATan2([p], lr=1e-3)
    assert opt.step() is None
    assert_bitwise_equal(p.detach(), before)
    assert len(opt.state[p]) == 0

    def closure():
        opt.zero_grad()
        loss = (p * 2).sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert torch.is_tensor(loss) and loss.item() == pytest.approx(2 * before.sum().item(), rel=1e-6)
    assert float(opt.state[p]["step"]) == 1


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.zeros(4, 3))
    p.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.ones(1, 3), (4, 3), check_invariants=True)
    opt = AdamATan2([p], lr=1e-3)
    with pytest.raises(RuntimeError, match="sparse"):
        opt.step()


# ---------------------------------------------------------------------------------------------------------
# (4) state_dict round trip
# ---------------------------------------------------------------------------------------------------------

def _roundtrip_params():
    specs = [((32, 8), torch.float32), ((17,), torch.float32), ((9,), torch.float64)]
    if torch.bfloat16 in DTYPES:
        specs.append(((12, 3), torch.bfloat16))
    inits = [_make_init(shape, dtype, seed=40 + k) for k, (shape, dtype) in enumerate(specs)]
    grads = [_make_grads(shape, dtype, 20, seed=60 + k) for k, (shape, dtype) in enumerate(specs)]
    return inits, grads


def _run(opt, params, grads, steps, lr_fn):
    for i in steps:
        for group in opt.param_groups:
            group["lr"] = lr_fn(i)
        for k, p in enumerate(params):
            p.grad = grads[k][i].clone()
        opt.step()


@pytest.mark.parametrize("serialize", ["torch_save", "deepcopy"])
def test_state_dict_roundtrip_continues_identically(serialize):
    inits, grads = _roundtrip_params()
    lr_fn = lambda i: 1e-3 * (1 + 0.1 * i)  # noqa: E731 - varies per step like the scheduler in pretrain.py

    # Uninterrupted: 20 steps
    params_a = [torch.nn.Parameter(x.clone()) for x in inits]
    opt_a = AdamATan2(params_a, lr=0, betas=BETAS, weight_decay=WEIGHT_DECAY)
    _run(opt_a, params_a, grads, range(20), lr_fn)

    # Interrupted: 10 steps, save, load into a fresh optimizer on cloned params, 10 more
    params_b = [torch.nn.Parameter(x.clone()) for x in inits]
    opt_b = AdamATan2(params_b, lr=0, betas=BETAS, weight_decay=WEIGHT_DECAY)
    _run(opt_b, params_b, grads, range(10), lr_fn)
    if serialize == "torch_save":
        buf = io.BytesIO()
        torch.save({"params": [p.detach() for p in params_b], "optim": opt_b.state_dict()}, buf)
        buf.seek(0)
        saved = torch.load(buf, map_location="cpu", weights_only=True)
    else:
        saved = copy.deepcopy({"params": [p.detach() for p in params_b], "optim": opt_b.state_dict()})
    del opt_b, params_b

    params_c = [torch.nn.Parameter(x.clone()) for x in saved["params"]]
    opt_c = AdamATan2(params_c, lr=0, betas=BETAS, weight_decay=WEIGHT_DECAY)  # as constructed in pretrain.py
    opt_c.load_state_dict(saved["optim"])
    for p in params_c:
        step = opt_c.state[p]["step"]
        assert step.device.type == "cpu" and step.dtype == torch.float32 and float(step) == 10
    _run(opt_c, params_c, grads, range(10, 20), lr_fn)

    for k, (pa, pc) in enumerate(zip(params_a, params_c)):
        assert_bitwise_equal(pa.detach(), pc.detach(), f"param {k}")
        for key in ("exp_avg", "exp_avg_sq"):
            assert_bitwise_equal(opt_a.state[pa][key], opt_c.state[pc][key], f"{key} {k}")
        assert float(opt_a.state[pa]["step"]) == float(opt_c.state[pc]["step"]) == 20


# ---------------------------------------------------------------------------------------------------------
# (5) constructor validation
# ---------------------------------------------------------------------------------------------------------

def test_constructor_accepts_lr_zero():
    p = torch.nn.Parameter(torch.zeros(3))
    opt = AdamATan2([p], lr=0)
    assert opt.param_groups[0]["lr"] == 0
    opt = AdamATan2([p], lr=0.0, betas=(0.0, 0.0), weight_decay=0.0)
    assert opt.param_groups[0]["betas"] == (0.0, 0.0)


@pytest.mark.parametrize("kwargs", [
    dict(lr=-1e-3),
    dict(lr=-1e-12),
    dict(lr=1e-3, betas=(1.0, 0.95)),
    dict(lr=1e-3, betas=(0.9, 1.0)),
    dict(lr=1e-3, betas=(-0.1, 0.95)),
    dict(lr=1e-3, weight_decay=-0.1),
])
def test_constructor_rejects_invalid(kwargs):
    p = torch.nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError):
        AdamATan2([p], **kwargs)
