"""Fixed-point / contraction diagnostics of a trained TRM, using the model's own operator.

Yardsticks (identical for every arm):
  Psi  = ACT-step map (all H-cycles of one inner forward) -- autonomous for every arm, primary.
  Phi  = last H-cycle map (the operator penalized by V1), evaluated at the state entering it.
  V    = joint field (F(z_H, z_L + tau) - z_H, F(z_L, z_H + x + tau) - z_L) (the field penalized by V2).

Reported:
  * rollout to K ACT steps (bf16, as trained): exact accuracy per step, residuals ||Psi(s_k)-s_k||/||s_k||,
    ||Phi(a_k)-a_k||/||a_k|| split by solved/unsolved at step 16, AUROC of the residual for errors;
  * at s_16 in fp32: sigma_max(J_Psi), sigma_max(J_Phi) by power iteration (torch.func jvp/vjp),
    secant ratios at several radii (random and top-singular directions), lambda_max(sym J_V) by shifted
    power iteration and random Rayleigh quotients;
  * init robustness: Gaussian noise on H_init/L_init -> accuracy, answer agreement, fixed-point distance.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from eval_utils import load_model, iter_test, init_carry, correctness


def rownorm(t: torch.Tensor) -> torch.Tensor:
    return t.float().flatten(1).norm(dim=-1)


def stats(v: torch.Tensor, rho=None) -> dict:
    v = v.double().cpu()
    out = dict(mean=v.mean().item(), median=v.median().item(), p10=torch.quantile(v, 0.1).item(),
               p90=torch.quantile(v, 0.9).item(), max=v.max().item(), min=v.min().item(), n=v.numel())
    if rho is not None:
        out.update(frac_gt1=(v > 1).double().mean().item(), frac_gt_rho=(v > rho).double().mean().item())
    return out


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(score_pos > score_neg) via ranks (ties averaged)."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1.0
    # average ties
    allv = np.concatenate([pos, neg])
    for val in np.unique(allv):
        m = allv == val
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout(inner, batch, K, sigma=0.0, gen=None, keep_state_at=16):
    x = inner.input_embeddings(batch)
    temb = inner.time_embeddings()
    si = inner.seq_info()
    H = inner.config.H_cycles
    c = init_carry(inner, x.shape[0], sigma=sigma, generator=gen)
    zH, zL = c.z_H, c.z_L
    rec = dict(exact=[], res_psi=[], res_phi=[], preds={}, state={})
    for k in range(K):
        SH, SL = zH, zL
        for i in range(H):
            aH, aL = zH, zL
            zH, zL = inner.cycle(zH, zL, x, temb, i, si)
        num_psi = torch.sqrt(rownorm(zH.float() - SH.float()) ** 2 + rownorm(zL.float() - SL.float()) ** 2)
        den_psi = torch.sqrt(rownorm(SH) ** 2 + rownorm(SL) ** 2)
        num_phi = torch.sqrt(rownorm(zH.float() - aH.float()) ** 2 + rownorm(zL.float() - aL.float()) ** 2)
        den_phi = torch.sqrt(rownorm(aH) ** 2 + rownorm(aL) ** 2)
        logits = inner.lm_head(zH)[:, inner.puzzle_emb_len:]
        preds, valid, exact, _ = correctness(logits, batch["labels"])
        rec["exact"].append(exact)
        rec["res_psi"].append(num_psi / den_psi)
        rec["res_phi"].append(num_phi / den_phi)
        if (k + 1) in (keep_state_at, K):
            rec["preds"][k + 1] = preds
            rec["state"][k + 1] = (zH.clone(), zL.clone())
    rec["valid"] = valid
    rec["x"] = x
    return rec


# ------------------------------------------------------------------------------------------- jacobians
def make_maps(inner, x32, temb32):
    si = inner.seq_info()
    H = inner.config.H_cycles

    def psi(s):
        zH, zL = s[:, 0], s[:, 1]
        for i in range(H):
            zH, zL = inner.cycle(zH, zL, x32, temb32, i, si)
        return torch.stack([zH, zL], 1)

    def phi(s):
        zH, zL = inner.cycle(s[:, 0], s[:, 1], x32, temb32, H - 1, si)
        return torch.stack([zH, zL], 1)

    def field(te_H, te_L):
        def f(s):
            vH, vL = inner.field(s[:, 0], s[:, 1], x32, te_H, te_L, si)
            return torch.stack([vH, vL], 1)
        return f

    def to_last_cycle(s):  # state entering the last cycle of the next ACT step
        zH, zL = s[:, 0], s[:, 1]
        for i in range(H - 1):
            zH, zL = inner.cycle(zH, zL, x32, temb32, i, si)
        return torch.stack([zH, zL], 1)

    return psi, phi, field, to_last_cycle


def _rn(u):
    return u.flatten(1).norm(dim=-1).view(-1, *([1] * (u.ndim - 1))).clamp_min(1e-30)


def _dot(a, b):
    return (a * b).flatten(1).sum(-1)


def sigma_max(f, s, iters, gen):
    """Per-row top singular value of J_f(s) by power iteration on J^T J (rows are independent)."""
    _, vjp_fn = torch.func.vjp(f, s)
    u = torch.randn(s.shape, device=s.device, generator=gen, dtype=s.dtype)
    u = u / _rn(u)
    for _ in range(iters):
        _, Ju = torch.func.jvp(f, (s,), (u,))
        (JtJu,) = vjp_fn(Ju)
        u = JtJu / _rn(JtJu)
    _, Ju = torch.func.jvp(f, (s,), (u,))
    return _rn(Ju).flatten(), u.detach()


def decoder_projector(inner):
    """P: state (b, 2, L, D) -> z_H answer tokens projected on the centered decoder row space (z_L -> 0)."""
    W = inner.lm_head.weight.detach().double()
    Wc = W - W.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Wc, full_matrices=False)    # fp64, standard rank tolerance
    Q = Vh[S > max(Wc.shape) * torch.finfo(torch.float64).eps * S[0]].T.float()
    P0 = inner.puzzle_emb_len

    def P(v):
        out = torch.zeros_like(v)
        out[:, 0, P0:] = (v[:, 0, P0:] @ Q) @ Q.T
        return out
    return P


def projected_sigma_max(f, s, iters, gen, out_proj, in_proj=None):
    """Top singular value of P_out J P_in (per row) by power iteration with jvp/vjp."""
    _, vjp_fn = torch.func.vjp(f, s)
    ip = in_proj if in_proj is not None else (lambda v: v)
    u = ip(torch.randn(s.shape, device=s.device, generator=gen, dtype=s.dtype))
    u = u / _rn(u)
    for _ in range(iters):
        w = ip(vjp_fn(out_proj(torch.func.jvp(f, (s,), (u,))[1]))[0])
        u = w / _rn(w)
    return _rn(out_proj(torch.func.jvp(f, (s,), (u,))[1])).flatten()


def mapped_sigma_max(f, s, iters, gen, out_proj, in_map):
    """Top singular value of P_out J M (M symmetric, e.g. W_gamma^{-1}) per row: power iteration on M J^T P J M."""
    _, vjp_fn = torch.func.vjp(f, s)
    u = torch.randn(s.shape, device=s.device, generator=gen, dtype=s.dtype)
    u = u / _rn(u)
    for _ in range(iters):
        w = in_map(vjp_fn(out_proj(torch.func.jvp(f, (s,), (in_map(u),))[1]))[0])
        u = w / _rn(w)
    return _rn(out_proj(torch.func.jvp(f, (s,), (in_map(u),))[1])).flatten()


def spectral_radius(f, s, iters, gen, tail=10):
    """Per-row |dominant eigenvalue| of J_f(s) by plain power iteration on J (geometric-mean growth over the
    last `tail` iterations, which also averages complex-pair oscillation)."""
    u = torch.randn(s.shape, device=s.device, generator=gen, dtype=s.dtype)
    u = u / _rn(u)
    logs = []
    for _ in range(iters):
        _, Ju = torch.func.jvp(f, (s,), (u,))
        n = _rn(Ju)
        logs.append(n.flatten().log())
        u = Ju / n
    return torch.stack(logs[-tail:], 1).mean(1).exp()


def fp32_rate(f, s, steps=30, floor=1e-5):
    """Empirical linear convergence rate of the fp32 iteration s <- f(s) started at s (below the bf16 floor):
    geometric mean of r_{k+1}/r_k over the steps whose residual is still above `floor` (fp32 precision)."""
    res = []
    with torch.no_grad():
        for _ in range(steps):
            s_new = f(s)
            res.append((_rn(s_new - s) / _rn(s)).flatten())
            s = s_new
    res = torch.stack(res, 1).clamp_min(1e-30)                     # (b, steps)
    above = res > floor
    last = (above.long().cumprod(1).sum(1) - 1).clamp_min(1)       # last index of the initial run above the floor
    r_last = res.gather(1, last.view(-1, 1)).squeeze(1)
    rate = (r_last / res[:, 0]).pow(1.0 / last.float())
    return rate, res[:, -1]


def osl_lambda_max(f, s, iters, gen, n_rand=4):
    """lambda_max(sym J_f(s)) per row via power iteration on sym(J) + c I, plus random Rayleigh quotients."""
    _, vjp_fn = torch.func.vjp(f, s)
    jv = lambda u: torch.func.jvp(f, (s,), (u,))[1]
    A = lambda u: 0.5 * (jv(u) + vjp_fn(u)[0])
    # spectral radius estimate of sym(J) -> shift
    u = torch.randn(s.shape, device=s.device, generator=gen, dtype=s.dtype)
    u = u / _rn(u)
    for _ in range(10):
        w = A(u)
        c = _rn(w)
        u = w / c
    shift = 1.1 * c
    u = torch.randn(s.shape, device=s.device, generator=gen, dtype=s.dtype)
    u = u / _rn(u)
    for _ in range(iters):
        w = A(u) + shift * u
        u = w / _rn(w)
    lam = _dot(u, A(u))
    rq = []
    for _ in range(n_rand):
        r = torch.randn(s.shape, device=s.device, generator=gen)
        r = r / _rn(r)
        rq.append(_dot(r, jv(r)))
    return lam, torch.stack(rq, 1).mean(1)


def secant(f, s, fs, direction, eps):
    d = direction / _rn(direction) * (eps * _rn(s))
    return (_rn(f(s + d) - fs) / _rn(d)).flatten()


# ------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="test subset dir (e.g. data/sudoku-extreme-test1k)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--n_jac", type=int, default=256)
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--jac_batch", type=int, default=16)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--power_iters", type=int, default=20)
    p.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.1, 0.5])
    p.add_argument("--noise", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    p.add_argument("--noise_seeds", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    t0 = time.perf_counter()
    model, cfg = load_model(a.ckpt, a.data)
    inner = model.inner
    for param in model.parameters():  # only state Jacobians are needed; avoids recording a parameter graph
        param.requires_grad_(False)
    arch = cfg["arch"]
    eta = float(arch.get("reg_eta", 0.05))
    h = float(getattr(inner, "h_H", 1.0))
    rho = float(arch["reg_rho"]) if arch.get("reg_rho") is not None else 1.0 - eta
    mu = -math.log(1.0 - eta) / h if eta < 1 else float("nan")
    gen = torch.Generator(device="cuda").manual_seed(0)
    K = a.K
    out = dict(ckpt=os.path.abspath(a.ckpt), n=0, K=K, eta=eta, rho=rho, reg_rho=arch.get("reg_rho"), mu=mu, h=h,
               flow=bool(arch.get("flow", False)), reg_mode=arch.get("reg_mode", "none"))

    # ---------------- rollout + init robustness
    ex_steps, res_psi, res_phi, valid_all = [], [], [], []
    rob = {s: {sd: dict(exact16=[], exactK=[], agree16=[], agreeK=[], dist16=[], distK=[]) for sd in range(a.noise_seeds)} for s in a.noise}
    jac_inputs = []
    x_over_z = []
    for batch in iter_test(a.data, a.batch, max_examples=a.n):
        rec = rollout(inner, batch, K)
        v = rec["valid"]
        valid_all.append(v)
        ex_steps.append(torch.stack(rec["exact"], 1))
        res_psi.append(torch.stack(rec["res_psi"], 1))
        res_phi.append(torch.stack(rec["res_phi"], 1))
        zH16, zL16 = rec["state"][16]
        x_over_z.append(rownorm(rec["x"]) / rownorm(zH16.float() + zL16.float()))
        if sum(t[0].shape[0] for t in jac_inputs) < a.n_jac:
            keep = v.nonzero().flatten()
            jac_inputs.append((zH16[keep], zL16[keep], {k: t[keep] for k, t in batch.items()}))

        for sigma in a.noise:
            for sd in range(a.noise_seeds):
                g = torch.Generator(device="cuda").manual_seed(1000 * sd + int(100 * sigma))
                rn = rollout(inner, batch, K, sigma=sigma, gen=g)
                r = rob[sigma][sd]
                for kk, tag in ((16, "16"), (K, "K")):
                    r[f"exact{tag}"].append(rn["exact"][kk - 1][v])
                    r[f"agree{tag}"].append((rn["preds"][kk] == rec["preds"][kk]).all(-1)[v])
                    a0, a1 = rec["state"][kk], rn["state"][kk]
                    dist = torch.sqrt(rownorm(a1[0].float() - a0[0].float()) ** 2 + rownorm(a1[1].float() - a0[1].float()) ** 2)
                    dist = dist / torch.sqrt(rownorm(a0[0]) ** 2 + rownorm(a0[1]) ** 2)
                    r[f"dist{tag}"].append(dist[v])

    valid = torch.cat(valid_all)
    ex = torch.cat(ex_steps)[valid].float()
    rp = torch.cat(res_psi)[valid]
    rf = torch.cat(res_phi)[valid]
    solved16 = ex[:, 15] > 0
    out["n"] = int(valid.sum())
    out["rollout"] = dict(
        exact_per_step=ex.mean(0).tolist(),
        res_psi_mean=rp.mean(0).tolist(), res_psi_median=rp.median(0).values.tolist(),
        res_psi_mean_solved=rp[solved16].mean(0).tolist() if solved16.any() else None,
        res_psi_mean_unsolved=rp[~solved16].mean(0).tolist() if (~solved16).any() else None,
        res_phi_mean=rf.mean(0).tolist(),
        res_phi_mean_solved=rf[solved16].mean(0).tolist() if solved16.any() else None,
        res_phi_mean_unsolved=rf[~solved16].mean(0).tolist() if (~solved16).any() else None,
    )
    out["auroc_res_psi16_for_error"] = auroc(rp[:, 15].cpu().numpy(), (~solved16).cpu().numpy())
    out["auroc_res_psi64_for_error"] = auroc(rp[:, K - 1].cpu().numpy(), (ex[:, K - 1] == 0).cpu().numpy())
    out["x_over_z"] = stats(torch.cat(x_over_z)[valid])
    out["init_robustness"] = {
        str(sigma): {k: float(torch.cat([torch.cat(rob[sigma][sd][k]).float() for sd in range(a.noise_seeds)]).mean())
                     for k in ("exact16", "exactK", "agree16", "agreeK", "dist16", "distK")}
        for sigma in a.noise
    }

    # ---------------- Jacobian-based contraction measures at s_16 (fp32)
    Pdec = decoder_projector(inner)
    Pinv = lambda v: v - Pdec(v)                        # decoder-invisible complement
    gen_dec = torch.Generator(device="cuda").manual_seed(1)  # separate stream: keeps the other metrics' draws unchanged
    acc = dict(psi_sigma=[], phi_sigma=[], psi_rho=[], phi_rho=[], psi_fp32_rate=[], psi_fp32_res30=[],
               phi_dec_block_sigma=[], phi_dec_out_sigma=[], phi_dec_leak_sigma=[],
               **{'phi_dec_semi_g0.3_sigma': [], 'phi_dec_semi_g0.1_sigma': []}, **{f"secant_{m}_{kind}_{e}": [] for m in ("psi", "phi") for kind in ("rand", "top") for e in a.eps})
    # Field times: z_L at L-substep j of the last H-cycle, z_H at its H-substep (None for TRM)
    L, i_last = inner.config.L_cycles, inner.config.H_cycles - 1
    t_list = [0, L - 1] if getattr(inner, "use_time_embed", False) else [None]
    osl_acc = {str(t): dict(lam=[], rq=[], vnorm=[]) for t in t_list}

    rows = 0
    for zH16, zL16, b in jac_inputs:
        for start in range(0, zH16.shape[0], a.jac_batch):
            if rows >= a.n_jac:
                break
            sl = slice(start, min(start + a.jac_batch, zH16.shape[0], start + a.n_jac - rows))
            sub = {k: t[sl] for k, t in b.items()}
            with torch.no_grad():
                x32 = inner.input_embeddings(sub).float()
                temb = inner.time_embeddings()
                temb32 = None if temb is None else temb.float()
            s = torch.stack([zH16[sl], zL16[sl]], 1).float()
            psi, phi, field, to_last = make_maps(inner, x32, temb32)
            with torch.no_grad():
                a_last = to_last(s)
                psi_s, phi_a = psi(s), phi(a_last)

            sig_psi, u_psi = sigma_max(psi, s, a.power_iters, gen)
            sig_phi, u_phi = sigma_max(phi, a_last, a.power_iters, gen)
            acc["psi_sigma"].append(sig_psi.detach())
            acc["phi_sigma"].append(sig_phi.detach())
            acc["phi_dec_block_sigma"].append(projected_sigma_max(phi, a_last, a.power_iters + 10, gen_dec, Pdec, Pdec).detach())
            acc["phi_dec_out_sigma"].append(projected_sigma_max(phi, a_last, a.power_iters + 10, gen_dec, Pdec).detach())
            acc["phi_dec_leak_sigma"].append(projected_sigma_max(phi, a_last, a.power_iters + 10, gen_dec, Pdec, Pinv).detach())
            for gam in (0.3, 0.1):
                Winv = lambda v, gam=gam: Pdec(v) + Pinv(v) / gam
                acc[f"phi_dec_semi_g{gam}_sigma"].append(mapped_sigma_max(phi, a_last, a.power_iters + 10, gen_dec, Pdec, Winv).detach())
            acc["psi_rho"].append(spectral_radius(psi, s, a.power_iters + 10, gen).detach())
            acc["phi_rho"].append(spectral_radius(phi, a_last, a.power_iters + 10, gen).detach())
            rate, res30 = fp32_rate(psi, s)
            acc["psi_fp32_rate"].append(rate)
            acc["psi_fp32_res30"].append(res30)
            with torch.no_grad():
                for e in a.eps:
                    r_dir = torch.randn(s.shape, device=s.device, generator=gen)
                    acc[f"secant_psi_rand_{e}"].append(secant(psi, s, psi_s, r_dir, e))
                    acc[f"secant_psi_top_{e}"].append(secant(psi, s, psi_s, u_psi, e))
                    acc[f"secant_phi_rand_{e}"].append(secant(phi, a_last, phi_a, r_dir, e))
                    acc[f"secant_phi_top_{e}"].append(secant(phi, a_last, phi_a, u_phi, e))

            for t in t_list:
                if t is None:
                    f = field(None, None)
                else:
                    f = field(temb32[inner.substep_index(i_last, L)].view(1, 1, -1), temb32[inner.substep_index(i_last, t)].view(1, 1, -1))
                lam, rq = osl_lambda_max(f, s, a.power_iters + 10, gen)
                osl_acc[str(t)]["lam"].append(lam.detach())
                osl_acc[str(t)]["rq"].append(rq.detach())
                with torch.no_grad():
                    osl_acc[str(t)]["vnorm"].append((_rn(f(s)) / _rn(s)).flatten())
            rows += sl.stop - sl.start
            del psi, phi, field
            torch.cuda.empty_cache()

    out["jac"] = {k: stats(torch.cat(v), rho=rho) for k, v in acc.items() if len(v)}
    out["osl"] = {t: dict(lam_max=stats(torch.cat(d["lam"])), lam_frac_gt0=(torch.cat(d["lam"]) > 0).float().mean().item(),
                          lam_frac_gt_negmu=(torch.cat(d["lam"]) > -mu).float().mean().item(),
                          rayleigh_rand=stats(torch.cat(d["rq"])), v_over_s=stats(torch.cat(d["vnorm"])))
                  for t, d in osl_acc.items() if len(d["lam"])}
    out["n_jac"] = rows
    out["time_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    brief = dict(n=out["n"], exact16=out["rollout"]["exact_per_step"][15], exactK=out["rollout"]["exact_per_step"][-1],
                 res_psi16=out["rollout"]["res_psi_mean"][15], psi_sigma=out["jac"]["psi_sigma"]["median"],
                 phi_sigma=out["jac"]["phi_sigma"]["median"], psi_rho=out["jac"]["psi_rho"]["median"],
                 phi_rho=out["jac"]["phi_rho"]["median"], psi_fp32_rate=out["jac"]["psi_fp32_rate"]["median"],
                 phi_dec_block=out["jac"]["phi_dec_block_sigma"]["median"], phi_dec_out=out["jac"]["phi_dec_out_sigma"]["median"],
                 osl={t: v["lam_max"]["median"] for t, v in out["osl"].items()}, time_s=out["time_s"])
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
