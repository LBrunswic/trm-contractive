"""Standalone evaluation of a saved TRM checkpoint.

Matches the in-training eval semantics (fresh carry, the same data at every ACT step, no halting),
but runs up to K ACT steps in one pass and records accuracy at every step (test-time extrapolation),
plus Q-head halting accuracy (ACT-style inference capped at 16 and at K steps).

  python scripts/evaluate.py --ckpt checkpoints/.../step_N --data data/sudoku-extreme-1k-aug-1000-s0 --act_steps 16 --out eval_full.json
"""
import argparse
import json
import os
import time

import torch

from eval_utils import load_model, iter_test, init_carry, correctness


@torch.inference_mode()
def run_eval(model, data_path: str, batch_size: int, K: int, train_K: int = 16, compile_inner: bool = False,
             max_examples=None, sigma: float = 0.0, seed: int = 0, device: str = "cuda"):
    inner = model.inner
    step_fn = torch.compile(inner, dynamic=False) if compile_inner else inner
    f64 = lambda: torch.zeros(K, dtype=torch.float64, device=device)
    exact, cell, changed, halted = f64(), f64(), f64(), f64()
    caps = sorted({min(train_K, K), K})
    halt_exact = {c: torch.zeros((), dtype=torch.float64, device=device) for c in caps}
    halt_steps = {c: torch.zeros((), dtype=torch.float64, device=device) for c in caps}
    n = torch.zeros((), dtype=torch.float64, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)

    for batch in iter_test(data_path, batch_size, max_examples=max_examples, device=device):
        B = batch["inputs"].shape[0]
        carry = init_carry(inner, B, sigma=sigma, generator=gen, device=device)
        first_halt = torch.zeros(B, dtype=torch.long, device=device)      # 0 = not yet halted
        exact_at_halt = torch.zeros(B, dtype=torch.bool, device=device)
        prev = None
        valid = None
        for k in range(K):
            carry, logits, (q_halt, _q_cont), _aux = step_fn(carry, batch)
            carry = type(carry)(z_H=carry.z_H.clone(), z_L=carry.z_L.clone())  # detach from compiled-graph buffers
            preds, valid, ex, ce = correctness(logits, batch["labels"])
            exact[k] += ex.sum()
            cell[k] += ce.sum()
            if prev is not None:
                changed[k] += (valid & (preds != prev).any(-1)).sum()
            newly = (q_halt > 0) & (first_halt == 0)
            first_halt = torch.where(newly, k + 1, first_halt)
            exact_at_halt = torch.where(newly, ex, exact_at_halt)
            halted[k] += (valid & (first_halt > 0)).sum()
            if (k + 1) in halt_exact:
                # ACT inference capped at k+1 steps: answer at the first Q-halt, else at the cap
                halt_exact[k + 1] += (valid & torch.where(first_halt > 0, exact_at_halt, ex)).sum()
                halt_steps[k + 1] += torch.where(valid, torch.where(first_halt > 0, first_halt, k + 1), 0).sum()
            prev = preds
        n += valid.sum()

    N = n.item()
    per = lambda t: (t / N).tolist()
    return dict(
        n=int(N), act_steps=K,
        exact_accuracy=(exact[train_K - 1] / N).item() if K >= train_K else None,
        accuracy=(cell[train_K - 1] / N).item() if K >= train_K else None,
        exact_accuracy_last=(exact[-1] / N).item(),
        best_step=int(torch.argmax(exact).item()) + 1,
        per_step=dict(exact_accuracy=per(exact), accuracy=per(cell), pred_change_frac=per(changed), q_halted_frac=per(halted)),
        q_halt={f"exact_accuracy@cap{c}": (halt_exact[c] / N).item() for c in caps} |
               {f"mean_steps@cap{c}": (halt_steps[c] / N).item() for c in caps},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--act_steps", type=int, default=16)
    p.add_argument("--train_act_steps", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--sigma", type=float, default=0.0, help="init-noise std (robustness)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    t0 = time.perf_counter()
    model, _cfg = load_model(a.ckpt, a.data)
    res = run_eval(model, a.data, a.batch_size, a.act_steps, a.train_act_steps, compile_inner=a.compile,
                   max_examples=a.max_examples, sigma=a.sigma)
    res.update(ckpt=os.path.abspath(a.ckpt), data=os.path.abspath(a.data), time_s=time.perf_counter() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_step"}, indent=1))


if __name__ == "__main__":
    main()
