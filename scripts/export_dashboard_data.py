"""Export the five datasets behind the interactive report from the published per-run files.

Reads only results/sudoku-contractive/runs/<ARM>_s<seed>/{eval_*.json, pop_eval*.json, diag*.json, metrics.jsonl}
and writes results/sudoku-contractive/report/data/{arms,seeds,curve,stopping,train}.json, the data embedded in
results/sudoku-contractive/report/TRM_contraction_benchmark.html.

  python scripts/export_dashboard_data.py [--root results/sudoku-contractive/runs] [--out results/sudoku-contractive/report/data]

Conventions: per-arm rows use seed 0; "A0@T" is A0 at its 35k-step checkpoint (files with suffix _step35000), "A0" is
A0 at 45k steps. Accuracies are percentages. Population columns use 16 starts x 64 steps: `rand_*` from fully random
initial states, `noise_*` from the canonical state plus Gaussian noise of std 0.5. Contraction columns are medians
over the diagnostic states (sigma_max(J_Phi), rho(J_Psi), lambda_max(sym J_V), decoder-projected sigma_max).
A missing file gives null.
"""
import argparse
import json
import os

ARMS = [  # arm, label, model, constraint, cap, train_steps
    ("A0@T", "Vanilla TRM", "TRM", "none", None, 35000),
    ("A0", "Vanilla TRM, 45k steps (approximately compute-matched)", "TRM", "none", None, 45000),
    ("A1", "V1 Lipschitz cap 0.95", "TRM", "V1 Lipschitz", 0.95, 35000),
    ("A5", "V1 Lipschitz cap 1", "TRM", "V1 Lipschitz", 1.0, 35000),
    ("A6", "V1 Lipschitz cap 2", "TRM", "V1 Lipschitz", 2.0, 35000),
    ("A7", "V1 Lipschitz cap 4", "TRM", "V1 Lipschitz", 4.0, 35000),
    ("A9", "D2 decoded output 1-Lipschitz", "TRM", "D2 decoder-aware", 1.0, 35000),
    ("A8", "Dγ semi-contraction (γ 0.3)", "TRM", "Dγ decoder-aware", 0.95, 35000),
    ("A2", "Euler-TRM", "Euler-TRM", "none", None, 35000),
    ("A4", "Euler-TRM + V1 cap 0.95", "Euler-TRM", "V1 Lipschitz", 0.95, 35000),
    ("A3", "Euler-TRM + V2 monotone field", "Euler-TRM", "V2 monotone", None, 35000),
    ("A10", "Euler-TRM + Dγ (γ 0.3)", "Euler-TRM", "Dγ decoder-aware", 0.95, 35000),
]
SEED_ARMS = ["A0@T", "A0", "A1", "A2", "A3"]
TAUS = [0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]


def run_dir(arm, seed=0):
    """(run directory name, file suffix) of an arm."""
    return ("A0_s%d" % seed, "_step35000") if arm == "A0@T" else ("%s_s%d" % (arm, seed), "")


def load(root, arm, name, seed=0):
    d, suf = run_dir(arm, seed)
    path = os.path.join(root, d, name % suf)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pct(x, nd):
    return None if x is None else round(100 * x, nd)


def rnd(x, nd):
    return None if x is None else round(x, nd)


def med(d, key):
    v = (d or {}).get(key)
    return v["median"] if isinstance(v, dict) else v


def pop_rule(pop, mode, rule):
    return pop["modes"][mode]["rules"].get(rule) if pop else None


def arms_rows(root):
    rows = []
    for arm, label, model, constraint, cap, steps in ARMS:
        full, e100, e64 = (load(root, arm, n) for n in ("eval_full%s.json", "eval_100k%s.json", "eval_ext64%s.json"))
        diag, pop = load(root, arm, "diag%s.json"), load(root, arm, "pop_eval%s.json")
        jac = (diag or {}).get("jac", {})
        osl = None if diag is None else (diag["osl"].get("0") or diag["osl"].get("None"))
        r, n = pop_rule(pop, "rand", "fixed@64"), pop_rule(pop, "noise:0.5", "fixed@64")
        g = lambda rule, k, nd=1: pct(rule[k], nd) if rule else None
        rows.append(dict(
            arm=arm, label=label, model=model, constraint=constraint, cap=cap, train_steps=steps,
            exact_full=pct(full and full["exact_accuracy"], 2), exact_100k=pct(e100 and e100["exact_accuracy"], 2),
            cell_100k=pct(e100 and e100["accuracy"], 2), exact_64=pct(e64 and e64["exact_accuracy_last"], 2),
            sigma_phi=rnd(med(jac, "phi_sigma"), 2), rho_psi=rnd(med(jac, "psi_rho"), 3),
            lam_max=rnd(med(osl, "lam_max"), 2),
            dec_block=rnd(med(jac, "phi_dec_block_sigma"), 2), dec_out=rnd(med(jac, "phi_dec_out_sigma"), 2),
            resid16=rnd(diag["rollout"]["res_psi_mean"][15], 4) if diag else None,  # mean ||Psi(s)-s||/||s|| at ACT step 16
            rand_single=g(r, "acc_single"), rand_member_mean=g(r, "acc_member_mean"), rand_vote=g(r, "acc_vote"),
            rand_min_resid=g(r, "acc_min_resid"), rand_unanimous=g(r, "unanimous"),
            rand_dispersion=rnd(pop and pop["modes"]["rand"]["dispersion_mean"], 3),
            noise_vote=g(n, "acc_vote"), noise_auroc_share=rnd(n and n["auroc_share"], 3),
            noise_sel80=pct(n and n["selective_share"]["acc@cov0.8"], 1),
        ))
    return rows


def seeds_rows(root):
    rows = []
    for arm in SEED_ARMS:
        for seed in (0, 1, 2):
            e100, e64 = load(root, arm, "eval_100k%s.json", seed), load(root, arm, "eval_ext64%s.json", seed)
            r = pop_rule(load(root, arm, "pop_eval%s.json", seed), "rand", "fixed@64")
            rows.append(dict(arm=arm, seed=seed, exact_100k=pct(e100 and e100["exact_accuracy"], 2),
                             exact_64=pct(e64 and e64["exact_accuracy_last"], 2),
                             rand_unanimous=pct(r and r["unanimous"], 1), rand_min_resid=pct(r and r["acc_min_resid"], 1)))
    return rows


def curve_rows(root):
    rows = []
    for arm, *_ in ARMS:
        e64 = load(root, arm, "eval_ext64%s.json")
        if e64:
            rows += [dict(arm=arm, step=k + 1, exact=pct(x, 2)) for k, x in enumerate(e64["per_step"]["exact_accuracy"])]
    return rows


def stopping_rows(root):
    rows = []
    for arm, *_ in ARMS:
        pop = load(root, arm, "pop_eval%s.json")
        for tau in TAUS:
            r = pop_rule(pop, "noise:0.5", "resid<%s" % tau)
            if r:
                rows.append(dict(arm=arm, tau=tau, steps_single=rnd(r["mean_steps_single"], 2), acc_single=pct(r["acc_single"], 1),
                                 steps_vote=rnd(r["mean_steps"], 2), acc_vote=pct(r["acc_vote"], 1)))
    return rows


def train_rows(root):
    rows = []
    for arm, *_ in ARMS:
        if arm == "A0@T":
            continue
        d, _ = run_dir(arm)
        path = os.path.join(root, d, "metrics.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                m = json.loads(line)
                if m.get("split") == "eval" and "all/exact_accuracy" in m:
                    rows.append(dict(arm=arm, step=m["step"], exact=pct(m["all/exact_accuracy"], 2)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="results/sudoku-contractive/runs")
    ap.add_argument("--out", default="results/sudoku-contractive/report/data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in (("arms", arms_rows), ("seeds", seeds_rows), ("curve", curve_rows), ("stopping", stopping_rows), ("train", train_rows)):
        rows = fn(a.root)
        with open(os.path.join(a.out, name + ".json"), "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=0)
            f.write("\n")
        print("%-8s %4d rows -> %s" % (name, len(rows), os.path.join(a.out, name + ".json")))


if __name__ == "__main__":
    main()
