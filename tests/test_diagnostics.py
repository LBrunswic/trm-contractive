"""Estimators used by scripts/fixed_point_diagnostics.py, checked on linear maps with known spectra (CPU)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from fixed_point_diagnostics import sigma_max, spectral_radius, osl_lambda_max, fp32_rate, auroc  # noqa: E402

N = 24


@pytest.fixture(autouse=True)
def float64_default():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)  # state = (b, 2, 3, 4) -> 24 dims per row


def linear(A):
    return lambda s: (s.flatten(1) @ A.T).view_as(s)


def rand_orth(n, g):
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    return q


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(0)


def test_sigma_max_matches_spectral_norm(gen):
    U, V = rand_orth(N, gen), rand_orth(N, gen)
    sv = torch.linspace(0.1, 3.0, N)
    A = U @ torch.diag(sv) @ V.T
    s = torch.randn(5, 2, 3, 4, generator=gen)
    sig, u = sigma_max(linear(A), s, iters=200, gen=gen)
    assert torch.allclose(sig, torch.full_like(sig, 3.0), rtol=1e-3)


def test_spectral_radius_of_non_normal_map_is_below_spectral_norm(gen):
    # Upper-triangular, eigenvalues 0.5 on the diagonal, large nilpotent part: ||A|| >> 1 > rho(A) = 0.5
    A = 0.5 * torch.eye(N) + 2.0 * torch.triu(torch.ones(N, N), diagonal=1) / N
    s = torch.randn(3, 2, 3, 4, generator=gen)
    sig, _ = sigma_max(linear(A), s, iters=200, gen=gen)
    rho = spectral_radius(linear(A), s, iters=400, gen=gen, tail=50)
    assert sig.min() > 1.5
    # Jordan-like block converges slowly; the estimate approaches 0.5 from above
    assert torch.all(rho < 0.75) and torch.all(rho > 0.45)


def test_spectral_radius_complex_pair_geometric_mean(gen):
    # Rotation-scaling blocks: eigenvalues 0.8 * exp(+-i theta) (dominant) and 0.3
    blocks = []
    for k in range(N // 2):
        r = 0.8 if k == 0 else 0.3
        th = 0.7
        blocks.append(r * torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]))
    A = torch.block_diag(*blocks)
    Q = rand_orth(N, gen)
    A = Q @ A @ Q.T
    s = torch.randn(2, 2, 3, 4, generator=gen)
    rho = spectral_radius(linear(A), s, iters=200, gen=gen, tail=50)
    assert torch.allclose(rho, torch.full_like(rho, 0.8), rtol=2e-2)


def test_osl_lambda_max_matches_sym_part(gen):
    S = torch.randn(N, N, generator=gen)
    sym = torch.diag(torch.linspace(-3.0, 1.5, N))
    Q = rand_orth(N, gen)
    A = Q @ sym @ Q.T + 0.5 * (S - S.T)            # sym(A) has spectrum [-3, 1.5]
    s = torch.randn(4, 2, 3, 4, generator=gen)
    lam, rq = osl_lambda_max(linear(A), s, iters=400, gen=gen)
    assert torch.allclose(lam, torch.full_like(lam, 1.5), atol=1e-2)
    # random Rayleigh quotients concentrate near the trace / n
    assert abs(rq.mean().item() - (sym.trace() / N).item()) < 0.6


def test_fp32_rate_on_affine_contraction(gen):
    s_star = torch.randn(3, 2, 3, 4, generator=gen)
    f = lambda s: s_star + 0.3 * (s - s_star)
    s0 = s_star + torch.randn(3, 2, 3, 4, generator=gen)
    rate, res_last = fp32_rate(f, s0, steps=30, floor=1e-9)
    # residuals are relative to the moving ||s||, so the measured rate is ~0.3 up to a few percent
    assert torch.allclose(rate, torch.full_like(rate, 0.3), rtol=3e-2)
    assert torch.all(res_last < 1e-9)


def test_auroc_basic():
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])
    pos = np.array([False, False, False, True, True, True])
    assert auroc(score, pos) == 1.0
    assert auroc(-score, pos) == 0.0
    assert auroc(np.ones(6), pos) == 0.5
    assert np.isnan(auroc(score, np.zeros(6, dtype=bool)))
