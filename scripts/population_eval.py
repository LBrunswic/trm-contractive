"""Population evaluation: residual-based stopping time + multi-init majority voting with confidence.

For each puzzle, M trajectories are run in one batch: member 0 starts from the canonical init (H_init/L_init),
members 1..M-1 from perturbed inits (Gaussian noise of std `sigma` on H_init/L_init, or fully random
per-token states with --init rand). Every trajectory runs K ACT steps (the ACT-step map Psi, as in eval).

Stopping rules (per trajectory):
  fixed@K       : answer after K ACT steps (K in {16, 32, 64})
  resid<tau     : first step whose fixed-point residual ||Psi(s_k) - s_k|| / ||s_k|| < tau (else K_max)
  qhalt@16      : first step with q_halt > 0 (else 16) -- TRM's learned halting, for reference
Aggregations (per puzzle, over the M answers at their stopping steps):
  single        : member 0 only (canonical init) -- the standard TRM protocol
  member_mean   : mean single-trajectory accuracy over all members
  vote          : majority vote over exact boards; confidence = vote share, margin = (top1 - top2) / M
  cell_vote     : per-cell majority (may yield an inconsistent board)
  min_resid     : the member with the smallest residual at its stopping step
Reported: accuracy, mean steps (compute = M x steps), AUROC of confidence for correctness, selective accuracy
at fixed coverage, ECE, and uniqueness (distinct endpoints, unanimity, latent dispersion of the final states).
"""
import argparse
import json
import math
import os
import time
from collections import Counter

import numpy as np
import torch

from eval_utils import load_model, iter_test, correctness


def auroc(score, positive):
    score, positive = np.asarray(score, dtype=np.float64), np.asarray(positive, dtype=bool)
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="stable")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):          # average ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def selective(conf, correct, coverages=(0.5, 0.8, 0.9, 1.0)):
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    order = np.argsort(-conf, kind="stable")
    out = {}
    for c in coverages:
        k = max(1, int(round(c * len(conf))))
        out[f"acc@cov{c}"] = float(correct[order[:k]].mean())
    return out


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def make_init(inner, rows, M, mode, sigma, gen, device="cuda"):
    """(rows*M) initial states; member 0 of every puzzle is the canonical init."""
    B = rows * M
    with torch.device(device):
        carry = inner.reset_carry(torch.ones(B, dtype=torch.bool), inner.empty_carry(B))
    z_H, z_L = carry.z_H.float(), carry.z_L.float()
    noisy = (torch.arange(B, device=device) % M != 0).view(-1, 1, 1)
    if mode == "noise":
        nH = torch.randn(z_H.shape, device=device, generator=gen) * sigma
        nL = torch.randn(z_L.shape, device=device, generator=gen) * sigma
        z_H, z_L = torch.where(noisy, z_H + nH, z_H), torch.where(noisy, z_L + nL, z_L)
    elif mode == "rand":
        rH = torch.randn(z_H.shape, device=device, generator=gen)
        rL = torch.randn(z_L.shape, device=device, generator=gen)
        z_H, z_L = torch.where(noisy, rH, z_H), torch.where(noisy, rL, z_L)
    else:
        raise ValueError(mode)
    dt = inner.forward_dtype
    return type(carry)(z_H=z_H.to(dt), z_L=z_L.to(dt))


@torch.inference_mode()
def rollout_population(inner, step_fn, batch, M, K, mode, sigma, gen):
    """Returns per-step predictions (K, B, 81) uint8, residuals (K, B), q_halt (K, B), exact (K, B), final states."""
    P = batch["inputs"].shape[0]
    rep = {k: v.repeat_interleave(M, dim=0) for k, v in batch.items()}
    carry = make_init(inner, P, M, mode, sigma, gen)
    preds, resid, qh, exact = [], [], [], []
    for _ in range(K):
        prev_H, prev_L = carry.z_H, carry.z_L
        carry, logits, (q_halt, _), _aux = step_fn(carry, rep)
        carry = type(carry)(z_H=carry.z_H.clone(), z_L=carry.z_L.clone())
        num = (carry.z_H.float() - prev_H.float()).flatten(1).square().sum(-1) + (carry.z_L.float() - prev_L.float()).flatten(1).square().sum(-1)
        den = prev_H.float().flatten(1).square().sum(-1) + prev_L.float().flatten(1).square().sum(-1)
        p, _valid, ex, _ = correctness(logits, rep["labels"])
        preds.append(p.to(torch.uint8))
        resid.append(torch.sqrt(num / den))
        qh.append(q_halt.float())
        exact.append(ex)
    final = torch.cat([carry.z_H.float().flatten(1), carry.z_L.float().flatten(1)], 1)
    return torch.stack(preds), torch.stack(resid), torch.stack(qh), torch.stack(exact), final


def stop_steps(resid, qh, rule, K):
    """Per-trajectory stopping index (0-based) for a rule."""
    Kmax = resid.shape[0]
    if rule.startswith("fixed@"):
        k = int(rule.split("@")[1])
        return torch.full((resid.shape[1],), min(k, Kmax) - 1, dtype=torch.long, device=resid.device)
    if rule.startswith("resid<"):
        tau = float(rule.split("<")[1])
        hit = resid < tau
        first = torch.where(hit.any(0), hit.float().argmax(0), torch.full_like(hit[0], Kmax - 1, dtype=torch.long))
        return first.long()
    if rule == "qhalt@16":
        cap = min(16, Kmax)
        hit = qh[:cap] > 0
        return torch.where(hit.any(0), hit.float().argmax(0), torch.full_like(hit[0], cap - 1, dtype=torch.long)).long()
    raise ValueError(rule)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--puzzles_per_batch", type=int, default=16)
    p.add_argument("--init", nargs="+", default=["noise:0.5", "rand"])
    p.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    t0 = time.perf_counter()
    model, cfg = load_model(a.ckpt, a.data)
    inner = model.inner
    step_fn = torch.compile(inner, dynamic=False) if a.compile else inner
    rules = [f"fixed@{k}" for k in (16, 32, a.K) if k <= a.K] + [f"resid<{t:g}" for t in a.taus] + ["qhalt@16"]
    rules = list(dict.fromkeys(rules))
    out = dict(ckpt=os.path.abspath(a.ckpt), n=0, M=a.M, K=a.K, rules=rules, modes={})

    for mode_spec in a.init:
        mode, sigma = (mode_spec.split(":") + ["0"])[:2]
        sigma = float(sigma)
        gen = torch.Generator(device="cuda").manual_seed(a.seed)
        per_rule = {r: dict(single=[], member_mean=[], vote=[], cell_vote=[], min_resid=[], vote_share=[], vote_margin=[],
                            n_distinct=[], steps=[], resid_single=[], single_steps=[]) for r in rules}
        dispersion, n_seen = [], 0
        for batch in iter_test(a.data, a.puzzles_per_batch, max_examples=a.n):
            valid = (batch["labels"] != -100).any(-1)
            P, M = batch["inputs"].shape[0], a.M
            preds, resid, qh, exact, final = rollout_population(inner, step_fn, batch, M, a.K, mode, sigma, gen)
            labels = batch["labels"]
            # latent dispersion of the final states inside each puzzle's population
            f = final.view(P, M, -1)
            mean = f.mean(1, keepdim=True)
            disp = ((f - mean).norm(dim=-1).mean(1) / mean.squeeze(1).norm(dim=-1))
            for r in rules:
                ks = stop_steps(resid, qh, r, a.K)                                          # (P*M,)
                idx = torch.arange(P * M, device=ks.device)
                ans = preds[ks, idx].view(P, M, -1)                                          # (P, M, 81)
                ex = exact[ks, idx].view(P, M)
                rs = resid[ks, idx].view(P, M)
                steps = (ks + 1).float().view(P, M)
                cell = torch.mode(ans.long(), dim=1).values                                  # (P, 81)
                cell_ok = (cell == labels.long()).all(-1)
                best = rs.argmin(1)
                for i in range(P):
                    if not valid[i]:
                        continue
                    boards = Counter(bytes(row) for row in ans[i].cpu().numpy())
                    top = boards.most_common(2)
                    winner = np.frombuffer(top[0][0], dtype=np.uint8)
                    pr = per_rule[r]
                    pr["single"].append(bool(ex[i, 0]))
                    pr["member_mean"].append(float(ex[i].float().mean()))
                    pr["vote"].append(bool((torch.from_numpy(winner.copy()).to(labels.device).long() == labels[i].long()).all()))
                    pr["vote_share"].append(top[0][1] / M)
                    pr["vote_margin"].append((top[0][1] - (top[1][1] if len(top) > 1 else 0)) / M)
                    pr["n_distinct"].append(len(boards))
                    pr["cell_vote"].append(bool(cell_ok[i]))
                    pr["min_resid"].append(bool(ex[i, best[i]]))
                    pr["steps"].append(float(steps[i].mean()))
                    pr["single_steps"].append(float(steps[i, 0]))
                    pr["resid_single"].append(float(rs[i, 0]))
            dispersion.extend(disp[valid].tolist())
            n_seen += int(valid.sum())

        res = {}
        for r, pr in per_rule.items():
            single, vote = np.array(pr["single"]), np.array(pr["vote"])
            share, margin = np.array(pr["vote_share"]), np.array(pr["vote_margin"])
            res[r] = dict(
                acc_single=float(single.mean()), acc_member_mean=float(np.mean(pr["member_mean"])),
                acc_vote=float(vote.mean()), acc_cell_vote=float(np.mean(pr["cell_vote"])), acc_min_resid=float(np.mean(pr["min_resid"])),
                mean_steps=float(np.mean(pr["steps"])), mean_steps_single=float(np.mean(pr["single_steps"])),
                unanimous=float(np.mean(np.array(pr["n_distinct"]) == 1)), mean_distinct=float(np.mean(pr["n_distinct"])),
                auroc_share=auroc(share, vote), auroc_margin=auroc(margin, vote),
                auroc_neg_resid_single=auroc(-np.array(pr["resid_single"]), single),
                selective_share=selective(share, vote), selective_neg_resid_single=selective(-np.array(pr["resid_single"]), single),
                ece_share=ece(share, vote),
            )
        out["modes"][mode_spec] = dict(rules=res, dispersion_mean=float(np.mean(dispersion)), dispersion_median=float(np.median(dispersion)))
        out["n"] = n_seen

    out["time_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for mode_spec, m in out["modes"].items():
        print(f"== {mode_spec}  n={out['n']}  M={a.M}  dispersion={m['dispersion_mean']:.4f}")
        for r, v in m["rules"].items():
            print(f"  {r:12s} single={v['acc_single']:.4f} vote={v['acc_vote']:.4f} cell={v['acc_cell_vote']:.4f} "
                  f"minres={v['acc_min_resid']:.4f} mean={v['acc_member_mean']:.4f} steps={v['mean_steps']:.1f} "
                  f"unan={v['unanimous']:.3f} distinct={v['mean_distinct']:.2f} auroc(share)={v['auroc_share']:.3f} "
                  f"auroc(-res)={v['auroc_neg_resid_single']:.3f} ece={v['ece_share']:.3f}")
    print(f"time {out['time_s']:.0f}s")


if __name__ == "__main__":
    main()
