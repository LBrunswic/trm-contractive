"""Aggregate benchmark runs into reports/<project>/benchmark.md + figures.

Reads <root>/<ARM>_s<seed>/{summary.json, metrics.jsonl, eval_100k*.json, eval_full*.json, eval_ext64*.json, diag*.json,
pop_eval*.json}; <root> defaults to checkpoints/<project>. Only these JSON/JSONL files are read (no weights), so the
published runs regenerate the published report:

  python scripts/summarize.py --root results/sudoku-contractive/runs --out reports/sudoku-contractive
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_LABEL = {
    "A0": "A0 TRM", "A0@T": "A0 TRM @T (equal steps)", "A1": "A1 TRM + V1 cycle-Lip",
    "A2": "A2 Euler-TRM", "A3": "A3 Euler-TRM + V2 monotone", "A4": "A4 Euler-TRM + V1 cycle-Lip",
    "A5": "A5 TRM + V1 non-expansive (ρ=1)", "A6": "A6 TRM + V1 Lip≤2", "A7": "A7 TRM + V1 Lip≤4",
    "A8": "A8 TRM + Dγ semi-contraction ρ=.95 γ=.3", "A9": "A9 TRM + D2 decoder-output ρ=1",
    "A10": "A10 Euler-TRM + Dγ ρ=.95 γ=.3",
}
# Reference categorical palette (fixed order, light mode); regularized arms dashed as secondary encoding
COLOR = {"A0": "#2a78d6", "A0@T": "#2a78d6", "A1": "#eb6834", "A2": "#1baf7a", "A3": "#eda100", "A4": "#e87ba4", "A5": "#4a3aa7", "A6": "#008300", "A7": "#e34948", "A8": "#2a78d6", "A9": "#7a7974", "A10": "#1baf7a"}
DASH = {"A0": "-", "A0@T": ":", "A1": "--", "A2": "-", "A3": "--", "A4": "--", "A5": "-.", "A6": "-.", "A7": "-.", "A8": ":", "A9": ":", "A10": "-."}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PAIRS = [("A1", "A0@T", "V1 penalty (TRM)"), ("A1", "A0", "V1 vs compute-matched TRM"),
         ("A3", "A2", "V2 penalty (Euler-TRM)"), ("A2", "A0@T", "Euler parameterization"),
         ("A4", "A2", "V1 penalty (Euler-TRM)"), ("A5", "A0@T", "V1 non-expansive (TRM)"),
         ("A5", "A1", "margin: rho=1 vs rho=0.95"), ("A6", "A0@T", "V1 Lip<=2"), ("A7", "A0@T", "V1 Lip<=4"),
         ("A8", "A0@T", "D-gamma semi-contraction (TRM)"), ("A9", "A0@T", "D2 decoder-output Lipschitz"),
         ("A10", "A2", "D-gamma on Euler-TRM"), ("A10", "A8", "Euler vs TRM under D-gamma")]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_runs(root):
    runs = defaultdict(dict)  # arm -> seed -> record
    for d in sorted(glob.glob(os.path.join(root, "*_s*"))):
        m = re.match(r"(A\d+)_s(\d+)$", os.path.basename(d))
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        summary = load_json(os.path.join(d, "summary.json"))
        metrics = []
        if os.path.exists(os.path.join(d, "metrics.jsonl")):
            with open(os.path.join(d, "metrics.jsonl")) as f:
                metrics = [json.loads(line) for line in f if line.strip()]
        rec = dict(dir=d, summary=summary, metrics=metrics,
                   eval_main=load_json(os.path.join(d, "eval_100k.json")),
                   eval_full=load_json(os.path.join(d, "eval_full.json")),
                   eval_ext=load_json(os.path.join(d, "eval_ext64.json")),
                   diag=load_json(os.path.join(d, "diag.json")),
                   diag_raw=load_json(os.path.join(d, "diag_raw.json")),
                   pop=load_json(os.path.join(d, "pop_eval.json")))
        runs[arm][seed] = rec
        # A0's checkpoint at the common budget T is a separate (equal-steps) arm
        for f in glob.glob(os.path.join(d, "eval_100k_step*.json")):
            st = re.search(r"_step(\d+)\.json", f).group(1)
            runs["A0@T" if arm == "A0" else f"{arm}@{st}"][seed] = dict(
                dir=d, summary=None, metrics=[m for m in metrics if m.get("step", 0) <= int(st)],
                eval_main=load_json(f), eval_full=load_json(os.path.join(d, f"eval_full_step{st}.json")),
                eval_ext=load_json(os.path.join(d, f"eval_ext64_step{st}.json")),
                diag=load_json(os.path.join(d, f"diag_step{st}.json")), diag_raw=None,
                pop=load_json(os.path.join(d, f"pop_eval_step{st}.json")), step=int(st))
    return runs


def get(d, *path, default=None):
    for p in path:
        if d is None:
            return default
        d = d.get(p) if isinstance(d, dict) else None
    return default if d is None else d


def last_train(metrics, key, window=10):
    vals = [m[key] for m in metrics if m.get("split") == "train" and key in m and m[key] is not None]
    return float(np.mean(vals[-window:])) if vals else None


def osl_median(diag):
    osl = get(diag, "osl") or {}
    if not osl:
        return None
    key = "None" if "None" in osl else sorted(osl, key=lambda k: int(k))[0]
    return osl[key]["lam_max"]["median"]


def row_values(rec):
    s, em, ef, ee, dg = rec["summary"], rec["eval_main"], rec["eval_full"], rec["eval_ext"], rec["diag"]
    m = rec["metrics"]
    rob = get(dg, "init_robustness") or {}
    return {
        "exact@16 (100k)": get(em, "exact_accuracy"),
        "cell acc@16 (100k)": get(em, "accuracy"),
        "Q-halt exact@16 (100k)": get(em, "q_halt", "exact_accuracy@cap16"),
        "exact@16 (full 423k)": get(ef, "exact_accuracy"),
        "exact@16 (25k)": (get(ee, "per_step", "exact_accuracy") or [None] * 16)[15],
        "exact@64 (25k)": get(ee, "exact_accuracy_last"),
        "best step (25k)": get(ee, "best_step"),
        "resid Psi@16": (get(dg, "rollout", "res_psi_mean") or [None] * 16)[15],
        "sigma_max Psi (med)": get(dg, "jac", "psi_sigma", "median"),
        "sigma_max Phi (med)": get(dg, "jac", "phi_sigma", "median"),
        "sigma P.J.P (D1, med)": get(dg, "jac", "phi_dec_block_sigma", "median"),
        "sigma P.J (D2, med)": get(dg, "jac", "phi_dec_out_sigma", "median"),
        "sigma P.J.(I-P) (leak, med)": get(dg, "jac", "phi_dec_leak_sigma", "median"),
        "sigma P.J.W^-1 γ=.3 (med)": get(dg, "jac", "phi_dec_semi_g0.3_sigma", "median"),
        "rho(J_Psi) (med)": get(dg, "jac", "psi_rho", "median"),
        "fp32 rate Psi (med)": get(dg, "jac", "psi_fp32_rate", "median"),
        "frac sigma Phi > rho": get(dg, "jac", "phi_sigma", "frac_gt_rho"),
        "lam_max sym J_V (med)": osl_median(dg),
        "init agree@16 (sd .5)": get(rob, "0.5", "agree16"),
        "init agree@64 (sd .5)": get(rob, "0.5", "agreeK"),
        "x/z": get(dg, "x_over_z", "mean"),
        "lambda (end)": last_train(m, "reg/lambda"),
        "viol (end)": last_train(m, "reg/viol"),
        "train steps": get(s, "final_step", default=rec.get("step")),
        "train h": (get(s, "train_s") or float("nan")) / 3600 if s else None,
        "steps/s": (get(s, "final_step") / get(s, "train_s")) if s and get(s, "train_s") else None,
        "peak GB": get(s, "peak_mem_reserved_gb"),
    }


def fmt(vals, pct=False):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "–"
    k = 100.0 if pct else 1.0
    if len(vals) == 1:
        return f"{k * vals[0]:.2f}" if abs(k * vals[0]) < 1e4 else f"{k * vals[0]:.3g}"
    mu, sd = np.mean(vals) * k, np.std(vals, ddof=1) * k
    return f"{mu:.2f} ± {sd:.2f}" if abs(mu) < 1e4 else f"{mu:.3g} ± {sd:.2g}"


PCT = {"exact@16 (100k)", "cell acc@16 (100k)", "Q-halt exact@16 (100k)", "exact@16 (full 423k)", "exact@16 (25k)", "exact@64 (25k)",
       "frac sigma Phi > rho", "init agree@16 (sd .5)", "init agree@64 (sd .5)", "viol (end)"}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def band_plot(ax, arm, xs_list, ys_list):
    if not xs_list:
        return
    n = min(len(x) for x in xs_list)
    xs = np.array(xs_list[0][:n], dtype=float)
    ys = np.array([y[:n] for y in ys_list], dtype=float)
    mu = np.nanmean(ys, 0)
    ax.plot(xs, mu, DASH[arm], color=COLOR[arm], linewidth=2, label=ARM_LABEL.get(arm, arm))
    if len(ys) > 1:
        ax.fill_between(xs, np.nanmin(ys, 0), np.nanmax(ys, 0), color=COLOR[arm], alpha=0.15, linewidth=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="checkpoints/sudoku-contractive")
    p.add_argument("--out", default="reports/sudoku-contractive")
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)
    runs = load_runs(a.root)
    arms = [arm for arm in ("A0@T", "A0", "A1", "A5", "A6", "A7", "A8", "A9", "A2", "A3", "A4", "A10") if arm in runs]

    lines = ["# Sudoku-Extreme: contractive recursive operator benchmark", ""]
    lines += ["Mean ± sample std over seeds (individual seeds in the per-seed table). Accuracies in %.", ""]

    # ---- main table
    cols = list(row_values(next(iter(next(iter(runs.values())).values()))).keys()) if runs else []
    table = {arm: {seed: row_values(rec) for seed, rec in sorted(runs[arm].items())} for arm in arms}
    lines.append("| metric | " + " | ".join(f"{ARM_LABEL.get(arm, arm)} (n={len(table[arm])})" for arm in arms) + " |")
    lines.append("|---|" + "---|" * len(arms))
    for c in cols:
        lines.append(f"| {c} | " + " | ".join(fmt([v[c] for v in table[arm].values()], pct=c in PCT) for arm in arms) + " |")
    lines.append("")

    # ---- paired differences
    lines += ["## Paired differences (same seed)", "", "| contrast | metric | per-seed Δ (pp) | mean Δ ± sd | t (n-1 dof) |", "|---|---|---|---|---|"]
    for x, y, name in PAIRS:
        if x not in table or y not in table:
            continue
        for metric in ("exact@16 (100k)", "exact@64 (25k)"):
            seeds = sorted(set(table[x]) & set(table[y]))
            d = [100 * (table[x][s][metric] - table[y][s][metric]) for s in seeds
                 if table[x][s][metric] is not None and table[y][s][metric] is not None]
            if not d:
                continue
            mu = float(np.mean(d))
            sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
            t = mu / (sd / math.sqrt(len(d))) if len(d) > 1 and sd > 0 else float("nan")
            lines.append(f"| {name}: {x} − {y} | {metric} | " + ", ".join(f"{v:+.2f}" for v in d) + f" | {mu:+.2f} ± {sd:.2f} | {t:.2f} |")
    lines.append("")

    # ---- population evaluation: stopping time, multi-init voting, confidence, uniqueness
    pop_cols = [("acc_single", "single", True), ("acc_vote", "vote", True), ("acc_min_resid", "min-resid", True),
                ("acc_member_mean", "member mean", True), ("unanimous", "unanimous", True), ("mean_distinct", "distinct/16", False),
                ("auroc_share", "AUROC(vote share)", False), ("auroc_neg_resid_single", "AUROC(−resid)", False), ("ece_share", "ECE(share)", False)]
    for mode in ("noise:0.5", "rand"):
        for rule in ("fixed@64", "resid<0.02"):
            have = [arm for arm in arms if any(get(r.get("pop"), "modes", mode, "rules", rule) for r in runs[arm].values())]
            if not have:
                continue
            lines += [f"## Population eval — init `{mode}`, stop `{rule}` (test1k, M=16; member 0 = canonical init)", "",
                      "| metric | " + " | ".join(ARM_LABEL.get(arm, arm) for arm in have) + " |", "|---|" + "---|" * len(have)]
            for key, name, pct in pop_cols:
                vals = {arm: [get(r.get("pop"), "modes", mode, "rules", rule, key) for r in runs[arm].values()] for arm in have}
                lines.append(f"| {name} | " + " | ".join(fmt(vals[arm], pct) for arm in have) + " |")
            for c in (0.5, 0.8, 0.9):
                vals = {arm: [get(r.get("pop"), "modes", mode, "rules", rule, "selective_share", f"acc@cov{c}") for r in runs[arm].values()] for arm in have}
                lines.append(f"| vote acc @ {int(100 * c)}% coverage | " + " | ".join(fmt(vals[arm], True) for arm in have) + " |")
            vals = {arm: [get(r.get("pop"), "modes", mode, "rules", rule, "mean_steps") for r in runs[arm].values()] for arm in have}
            lines.append("| mean ACT steps | " + " | ".join(fmt(vals[arm]) for arm in have) + " |")
            vals = {arm: [get(r.get("pop"), "modes", mode, "dispersion_mean") for r in runs[arm].values()] for arm in have}
            lines += ["| latent dispersion of endpoints | " + " | ".join(fmt(vals[arm]) for arm in have) + " |", ""]

    # ---- per-seed table
    lines += ["## Per-seed results", "", "| run | exact@16 (100k) | exact@64 (25k) | σmax Φ | λmax sym J_V | train h |", "|---|---|---|---|---|---|"]
    for arm in arms:
        for seed, v in table[arm].items():
            f = lambda k, pct=False: fmt([v[k]], pct)
            lines.append(f"| {arm}_s{seed} | {f('exact@16 (100k)', True)} | {f('exact@64 (25k)', True)} | "
                         f"{f('sigma_max Phi (med)')} | {f('lam_max sym J_V (med)')} | {f('train h')} |")
    lines.append("")

    # ---- figures
    figs = []
    plot_arms = [arm for arm in arms if arm != "A0@T"]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    for arm in plot_arms:
        xs, ys = [], []
        for rec in runs[arm].values():
            ev = [m for m in rec["metrics"] if m.get("split") == "eval"]
            if ev:
                xs.append([m["step"] for m in ev]); ys.append([100 * m["all/exact_accuracy"] for m in ev])
        band_plot(ax, arm, xs, ys)
    style(ax, "In-training exact accuracy (test10k, EMA, 16 ACT steps)", "optimizer step", "exact accuracy (%)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    figs.append(("eval_curves.png", "In-training eval")); fig.savefig(os.path.join(a.out, "figs", figs[-1][0]), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    for arm in arms:
        ys = [100 * np.array(rec["eval_ext"]["per_step"]["exact_accuracy"]) for rec in runs[arm].values() if rec["eval_ext"]]
        band_plot(ax, arm, [list(range(1, len(y) + 1)) for y in ys], ys)
    ax.axvline(16, color=INK2, linewidth=0.8, linestyle=":")
    style(ax, "Test-time extrapolation: exact accuracy vs ACT steps (test25k)", "ACT steps at inference (trained with 16)", "exact accuracy (%)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    figs.append(("act_steps.png", "Accuracy vs ACT steps")); fig.savefig(os.path.join(a.out, "figs", figs[-1][0]), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    for arm in arms:
        ys = [rec["diag"]["rollout"]["res_psi_mean"] for rec in runs[arm].values() if rec["diag"]]
        band_plot(ax, arm, [list(range(1, len(y) + 1)) for y in ys], ys)
    ax.set_yscale("log")
    style(ax, "Fixed-point residual ‖Ψ(s_k) − s_k‖ / ‖s_k‖ (test1k)", "ACT step k", "relative residual (log)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    figs.append(("residual.png", "Fixed-point residual")); fig.savefig(os.path.join(a.out, "figs", figs[-1][0]), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, (("psi_sigma", "σ_max(J_Ψ) at s_16"), ("phi_sigma", "σ_max(J_Φ) at s_16"), ("osl", "λ_max(sym J_V) at s_16"))):
        for i, arm in enumerate(arms):
            for j, rec in enumerate(runs[arm].values()):
                dg = rec["diag"]
                if not dg:
                    continue
                if key == "osl":
                    osl = dg["osl"]; k0 = "None" if "None" in osl else sorted(osl, key=int)[0]; st = osl[k0]["lam_max"]
                else:
                    st = dg["jac"][key]
                x = i + 0.12 * (j - 1)
                ax.errorbar([x], [st["median"]], yerr=[[st["median"] - st["p10"]], [st["p90"] - st["median"]]],
                            fmt="o", color=COLOR[arm], markersize=6, capsize=3, linewidth=1.5)
        if key != "osl":
            ax.set_yscale("log")
        ax.axhline(0.0 if key == "osl" else 1.0, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(arms)), arms, fontsize=8)
        style(ax, title, "", "median, p10–p90")
    figs.append(("contraction.png", "Contraction measures")); fig.savefig(os.path.join(a.out, "figs", figs[-1][0]), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    for ax, key, title in ((axes[0], "mon/v1_ratio", "Monitor: V1 secant ratio of Φ (adv, train states)"),
                           (axes[1], "mon/v2_q", "Monitor: V2 one-sided quotient of V (adv)")):
        for arm in plot_arms:
            xs, ys = [], []
            for rec in runs[arm].values():
                tr = [m for m in rec["metrics"] if m.get("split") == "train" and key in m]
                if tr:
                    xs.append([m["step"] for m in tr]); ys.append([m[key] for m in tr])
            band_plot(ax, arm, xs, ys)
        style(ax, title, "optimizer step", key.split("/")[1])
        if key == "mon/v1_ratio":
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    figs.append(("train_monitor.png", "Training-time contraction monitor")); fig.savefig(os.path.join(a.out, "figs", figs[-1][0]), dpi=150, bbox_inches="tight"); plt.close(fig)

    # stopping-time curves: accuracy vs mean ACT steps for residual thresholds (single trajectory and majority vote)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    for ax, key, title in ((axes[0], "acc_single", "Residual stopping: single trajectory"), (axes[1], "acc_vote", "Residual stopping: 16-init majority vote")):
        for arm in arms:
            pts = []
            for rec in runs[arm].values():
                rules = get(rec.get("pop"), "modes", "noise:0.5", "rules") or {}
                rr = sorted([(v["mean_steps_single" if key == "acc_single" else "mean_steps"], 100 * v[key]) for r, v in rules.items() if r.startswith("resid<")])
                if rr:
                    pts.append(rr)
            if pts:
                n = min(len(p_) for p_ in pts)
                xs = np.mean([[p_[i][0] for i in range(n)] for p_ in pts], 0); ys = np.mean([[p_[i][1] for i in range(n)] for p_ in pts], 0)
                ax.plot(xs, ys, DASH[arm], marker="o", markersize=4, color=COLOR[arm], linewidth=2, label=ARM_LABEL.get(arm, arm))
        style(ax, title, "mean ACT steps used", "exact accuracy (%)")
        ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    figs.append(("stopping_time.png", "Residual-based stopping time")); fig.savefig(os.path.join(a.out, "figs", figs[-1][0]), dpi=150, bbox_inches="tight"); plt.close(fig)

    lines += ["## Figures", ""] + [f"### {t}\n\n![{t}](figs/{f})\n" for f, t in figs]
    with open(os.path.join(a.out, "benchmark.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
