# Benchmark plan: baselines and variations for a CVPR-level study of monotone latent operators in recursive reasoners

*Plan v2, 27 October 2026. This is a plan: none of the new experiments below have been run. Measured numbers come
from the runs published in this repository (`results/sudoku-contractive/`); checkpoint and optimizer-state sizes and
the FLOP counts were measured locally on files that are not published (marked where used). External numbers were
checked against the primary sources; anything that could not be checked is marked "unverified" or left out.*

**How to read this document**

- **Arms and IDs.** Existing arms keep their repository names A0–A10 (see `README.md`, *Arms*). V1 is the two-sided
  Lipschitz cap (`reg_mode=cycle_lip`), V2 the monotone vector-field penalty (`reg_mode=flow_mono`). New IDs:
  **B** trained external baseline, **E** test-time baseline on the same weights, **X** internal variation,
  **C** test-time protocol, **M** metric, **D** diagnostic, **T** task, **EV-\*** evaluation package (§2.5;
  EV-Full is exact@16 on the full test set).
  **B★** is the random-init (RI) baseline chosen on validation among B1, B2 and B3 (§7.5). Every row that uses B★
  means B3 when B3 is the validation choice; the B★ row's own runs (seeds 3–5) are then not needed [n12].
- **Compute unit.** G-h = wall-clock hours on one RTX 5080 (16 GB). Ranges "lo–hi" span the two throughput regimes
  measured on the 5080s (training up to 1.65× apart, evaluation up to about 2.0× apart; the hi end of evaluation
  ranges uses the slow-regime median, §2.1). Wall-clock estimates use the midpoint. The RTX 4090 Laptop GPU (16 GB)
  is counted at 0.35 G-h per wall-hour (checked in §2.1).
- **Budgets.** R = 35k steps at B = 256 (8.96M samples, the published budget). 2R = 70k steps. L = 195k steps at
  B = 256 = 49.9M samples, the paper's 50k epochs; this equals the paper recipe (about 65,104 steps at B = 768) in
  samples and is the paper budget used in the tables. One L run costs 5.57× an R run. (P denotes the number of
  power steps in §2.4, and the decoder projector inside W_γ in X16.)
- **Compute columns.** Every runnable row carries: *Train G-h/seed* (one training run at the stated budget, including
  regulariser overhead and in-training evaluation), *Seeds* (new runs), *Train G-h total*, *Eval G-h* (the evaluation
  protocol that row needs, summed over its checkpoints), *Peak mem* (GB; **bold** when it exceeds 16 GB), *Eng. days*
  (person-days of implementation). Test-time and metric rows give the cost per checkpoint and the number of
  checkpoints. Rows that only quote published numbers cost 0 G-h.
- **Tiers.** Tier 0 = minimum viable submission (its compute fits the CVPR 2027 window; its engineering needs two
  people, §8.3), Tier 1 = strong, Tier 2 = full.
- **Notes.** "[nX]" points to the estimation notes in §10.

---

## Summary

| Tier | Scope | Train G-h | Eval G-h | Total G-h (mid) | Eng. days | Wall-clock at 60% 5080 availability [n17] |
|---|---|---|---|---|---|---|
| 0 | Random-init baselines and the {RI} × {V2} factorial at 6 seeds, matched-compute test-time baselines, held-out inits, mechanism minimum (dose-response, random pairs, trace hinge, JR), visual Sudoku | 141–245 | 176–332 | 317–577 (447) | 14–20 | ≈12.3 days of compute; fits 28 Oct–11 Nov with ≈2.7 days of buffer. Engineering needs two people (≈14–20 person-days in days 1–11); one person cannot do it without the cuts of §8.6 |
| 1 | Full-budget runs (1 seed), remaining ablations, restoration (T5), RAFT fine-tune (T6), ungrounded visual Sudoku | 211–442 | 103–195 | 314–637 (476) | 15–22 | ≈13 days of compute; engineering-bound (≈3–4.5 weeks for one person) |
| 2 | Full-budget runs at 3 seeds, Maze-Hard (gated), ARC, nice-to-have rows | 347–654 | 119–227 | 467–881 (674) | 13–22 | ≈18.5 days of compute |
| **All** | | **699–1,340** | **398–755** | **1,097–2,095 (1,596)** | **42–64** | **≈44 days of compute** |

Evaluation is the largest single cost of Tier 0 (57%): population, attack and held-out-init protocols on about 90
checkpoints. §8 gives the per-table totals, the schedule, the critical path and the cut order.

---

## 1. Framing

### 1.1 The claim the benchmark must support

- **Mechanism.** A soft one-sided Lipschitz penalty on the update field V = F − I of a weight-tied recursive
  operator, enforced with margin μ on secant pairs found by an adversarial search and held at a target violation rate
  by dual ascent, makes the decoded answer independent of the initial latent, including initial distributions never
  seen in training, and makes the label-free signals (fixed-point residual, vote share) reliable enough for selection
  and selective prediction.
- **Matched compute.** This must hold against baselines that get the same training FLOPs, the same tuning budget and
  the same number of test-time ACT steps. The canonical-accuracy cost of the penalty is reported, not hidden.

The paper is therefore a mechanism study under matched conditions, not a state-of-the-art paper. The decisive
comparisons are B1–B3 against X1 (V2 on top of random-init training), evaluated on the held-out init families of C5.

### 1.2 What prior work already covers, and what may be claimed

The literature search behind this table is written up in [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) (72 verified
papers: 26 synthesised in detail, the other 46 listed).

| Prior work | What it already shows | What this study may claim | Deciding rows | Compute of deciding rows (G-h, Tier 0) |
|---|---|---|---|---|
| Equilibrium Reasoners (EqR, 2605.21488); Path Independent Equilibrium Models (PI, 2211.09961); xRRMs: Explorative Recurrent Reasoning Models (NeurIPS 2026 workshop, abstract only) | Random restarts, majority vote and min-residual (Top1Conv) selection on TRM; random-init (RI) and noise-injection (NI) training make the model path-independent (Δ_PI 3.58 → 0.10 on Sudoku-Lite with RI); PI's adversarial init attack. xRRMs trains TRM/HRM from several random initial latents plus learned-variance noise with a best-of-K loss, i.e. training from random starts that keeps trajectories diverse | Only whether a field-level penalty adds invariance beyond RI, in particular on init families outside every arm's training distribution and under attack. The selection rules are EqR's; training from random initial latents is prior art (EqR, PI, xRRMs) | B1, B2, B★, B3, X1, C5, C6 (H1, H5) | 102–181 |
| Dynamical Systems Theory Behind a Hierarchical Reasoning Model (CMM, 2603.22871) | TRM read as an Euler step of an ODE with a squared Routh–Hurwitz trace hinge on z_H | V2 implies the trace condition on the joint field (tr J_V / d ≤ λ_max(sym J_V) ≤ −μ), not the converse. In all 18 runs with diagnostics the random-direction Rayleigh quotient of sym J_V (≈ tr J_V / d over the joint (z_H, z_L) state) has medians of −0.65 to −0.93 and is negative at every measured step-16 state. The z_H-block trace that CMM penalises was not measured (added to D1). B4a/B4b test whether the worst-direction condition matters | B4a, B4b | 7–14 |
| Fixed-Point Reasoners (FPRM, 2606.18206), Spherical FPRM (2610.25646), Geometric Fixed-Point Models (2609.33540) | Residual halting; local convergence assuming Re λ(J_F) < 1 at the fixed point; architectural stability | Residual halting is theirs and is used only as baseline E2. V2 targets a sufficient condition for their hypothesis, but only on sampled pairs; D1/D2 report where it actually holds | E2, D1, D2 | inside EV packages (§2.5) |
| Looped Flows (2609.11801), EqR, CMM, Scheduling Recursive Reasoning (TAPS, 2609.36653) | Euler/ODE iterations, damping and time conditioning of looped reasoners | Euler-TRM is the vehicle, not a contribution. X3–X5 decide whether "Euler" belongs in the title and whether "unique equilibrium" may be used | X3, X4, X5 | 11–18 |
| Monotone operator equilibrium networks (monDEQ, 2006.08591); Lipschitz-Bounded Equilibrium Networks (LBEN, 2010.01732); Robust Implicit Networks via Non-Euclidean Contractions (NEMON, 2106.03194). Separately: Designing Stable Neural Networks using Convex Analysis and ODEs (2306.17332) | monDEQ, LBEN and NEMON: exact monotone or contractive parameterisations with a unique equilibrium for single implicit layers. 2306.17332: non-expansive explicit networks (Euler steps of a convex-potential gradient flow, f = −Aᵀσ(Ax + b)), with no equilibrium formulation | No monotonicity or uniqueness guarantee may be claimed. The object of study is a soft penalty on a deep, weight-tied, two-level field; B8 (Tier 2) is the exact reference | B8 | Tier 2: 5–12 |
| Learning Maximally Monotone Operators for Image Recovery (2012.13247); Learning truly monotone operators (2404.00390); MOL (2206.04797); DEQ inverse problems (2102.07944); Adversarial Lipschitz Regularization (ALR, 1907.05681) | Learned monotone / firmly nonexpansive operators for imaging, including a hinge on λ_min of a symmetrised Jacobian estimated by power iteration (2404.00390); adversarial pair search by power iteration for a Lipschitz penalty (ALR) | Penalising a symmetric-Jacobian eigenvalue is prior art, and so is an adversarial pair search by power iteration (ALR). What is specific here is the combination: that pair search, at finite radius, applied to a one-sided condition on a weight-tied latent field. X10–X12 test whether the adversarial pairs are necessary. T5 is a transfer and comparison, not a novelty claim | X10, X11 (Tier 1: X12, T5) | 19–34 |
| Stabilizing Recurrent Dynamics in Looped Language Models (STARS, 2605.26733) | Power-iteration penalty on ρ(J) | ρ(J) < 1 and V2 are incomparable conditions; tested by B6 | B6 | Tier 1: 4–8 |
| Unrolled Policy Iteration for Tiny Recursive Models (ICLR 2026 workshop, abstract only) | Assumes the TRM latent update is a contraction (Lipschitz constant L_z < 1), so the truncation bias decays like L_z^n; per the abstract, contraction strength and a latent projection affect stability when training and evaluation unroll depths differ, and the contraction modulus and unroll depth are named as the practical stability controls (4 × 4 and 9 × 9 Sudoku trained from checker feedback) | A contraction (two-sided Lipschitz) condition on TRM's latent map is prior art, so the V1 caps (A1, A4–A7, X17) are comparators, not contributions; only the one-sided condition (V2) and its comparison with two-sided caps may be claimed | A1, A4–A7 (published), X17 | 0 (published runs); X17 in Tier 1: 16–28 |
| GRAM (2605.19376), PTRM (2605.19943), GAR (2610.16427), Guided SMC (2605.25230), C-voting (2604.13521), Speed is Confidence (2601.19085), A Tilted Bowl Is Not a Slippery Slope (2609.39277) | Majority/mode voting, best-Q, answer resampling, token-entropy confidence, test-time augmentation, halt-first selection; the halting head already detects errors with AUROC 0.99–1.00 | Vote share as confidence is close to self-consistency. Only calibration and selective-prediction differences that survive recalibration (H3), benchmarked against E3–E7 on the same pool | E3–E7, M4–M6 (H2, H3) | inside EV-F / EV-TT |
| Input-Conditioned Search (ICS, 2610.18547); Looped Flows; Fractal basins trap latent reasoning (2609.04963); Recursive Reasoning as Attractor Landscape Search (ICLR 2026 workshop, abstract only) | Wrong, low-residual attractors exist in trained TRMs; convergence of EqR and FPRM models depends on the initial latent through fractal settling-time basins; failed TRM runs on Sudoku-Extreme stall at stable wrong attractors set by the initialisation, and initial-answer noise plus more guesses is proposed as a training-free fix (the breadth that E3/E4 test) | Only measured rates of wrong low-residual endpoints (M7) under V2 vs baselines | M7, D4 | 0 (from dumps) |
| Multi-Attractor GNNs (2609.35274) | Contractive equivariant dynamics fail on symmetric multi-solution instances | Invariance may hurt on genuine ambiguity; T2 measures it | T2 (C9) | 2–4 |

**Search boundary.** In our literature search (every paper was checked in full text except three workshop papers
known only from their abstracts: Unrolled Policy Iteration for Tiny Recursive Models, xRRMs and Recursive Reasoning as
Attractor Landscape Search, cited in the rows above and in §11), no paper imposes a one-sided Lipschitz
(monotonicity) penalty on the latent field of TRM, HRM or another looped reasoner; for the three workshop papers this
rests on their abstracts only. The search covered control-theory contraction and monotone-systems work, neural-ODE
stability regularisers and LLM self-consistency only partially, so any "first" wording must be qualified accordingly.

### 1.3 What the current data do not support yet

From `results/sudoku-contractive/report/benchmark.md` and `results/sudoku-contractive/runs/*/diag.json` and
`pop_eval.json`:

- **An accuracy gain at matched test compute.** From 16 starts (1 canonical + 15 random) × 64 ACT steps (1,024 steps
  per puzzle), A0@45k with majority vote reaches 82.37 ± 3.93, A3 with min-residual 81.00 ± 5.74.
- **A unique equilibrium.** At step-16 held-out states A3 has median λ_max(sym J_V) of +2.31 to +2.99 (z_L substep
  times 0 and 5), median ρ(J_Ψ) of 0.99–1.03 and ‖V‖/‖s‖ of 0.66–0.88 over both times (A0: 0.004–0.014). A3 ends on a
  time-embedding-driven orbit, not at a zero of V.
- **Global invariance.** A3's unanimity from 16 starts (1 canonical + 15 random) is 0.61 / 0.60 / 0.49 per seed while
  the mean number of distinct boards is 4.8–5.6, so the non-unanimous puzzles average about 10–11 distinct boards out
  of 16. Invariance tracks whether the puzzle is solved. (The new populations of C3 are all-random.)
- **That the effect comes from V2.** Neither A0 nor A3 saw random starts in training. A0's collapse from random
  starts may be distribution shift, which RI training fixes (EqR).
- **That unanimity means correctness.** A collapsed model is also invariant: A1 (10–12% accuracy) reaches random-start
  unanimity of up to 0.34. Raw unanimity is therefore not an endpoint (M2).
- **State of the art.** Published Sudoku-Extreme results reach 94.2–98.4 single-trajectory and 99.43–99.8 with
  multiple samples (§3.3); our arms reach 62–67 (arm means, exact@16 on the 100k subset) at 18% of the paper's
  training samples (35k steps), and A0 reaches 69 at 45k steps (23%).

### 1.4 Venue fit (CVPR)

- **Two tables per task.** "Same backbone, same budget, same test-time ACT steps" holds all our arms and re-run
  baselines. "Published systems" holds quoted numbers with their budget and protocol (§3.3). The two are never mixed in
  one column.
- **Every multi-start number goes on an accuracy-vs-B_tt plot** that includes the depth-only single trajectory (E1).
  The 16 × 64 population costs 1,024 ACT steps per puzzle.
- **Headline metrics** are invariance (M2, M3), verifier quality (M4–M7) and selective prediction, not raw accuracy.
- **Vision relevance.** T3 (visual Sudoku: perception in the loop), T5 (iterative image restoration, the setting where
  monotone priors are standard), T6 (RAFT, a recurrent refinement operator), T4 (Maze-Hard, 2-D grid planning) and T7
  (ARC). Only T3 fits the CVPR window (Tier 0). On our 100k test subset the number of givens is 25.2 on average
  (17–36), so a per-digit recogniser at 99.2% caps board accuracy at 81.7% (88.1% at 99.5%), assuming independent
  errors: perception errors will be a large error source on T3, which is what the confidence claims need.
- **Wording rules.** Do not claim the Euler/ODE framing, damping, residual halting, majority voting, min-residual
  selection, adversarial pair search by power iteration (ALR) or monotone operators for imaging. Do not write "unique equilibrium" unless X4 supports it.

---

## 2. Cost model and measured anchors

### 2.1 Units and the laptop factor

- **Two regimes on the 5080s.** Plain TRM trains at 5.14 steps/s in the fast regime and 3.12 steps/s in the slow one
  (1.65×). Evaluation throughput is about 2,000–2,110 puzzle-ACT-steps/s in the fast regime and 1,015–1,200/s in
  the slow one (population eval: 2,004–2,029 vs 1,055–1,108/s, slow median ≈1,080/s); on the same evaluation the two
  regimes differ by up to ≈2.0× (1.92× population eval, 1.99× exact@16 on 100k, 2.03× exact@64). The hi end of every
  evaluation range uses the slow-regime median (1,080/s, [n9]), not the worst case, so a single evaluation on the slow
  card can take up to ≈6% longer than its hi value. In the published runs, the seed-1 runs (and A7, A9) were in the slow regime; A0_s1 and A0_s2 were launched at the same time and ran
  in different regimes, which points to one slower card or host; this was not diagnosed. Capacity is planned as one fast and one slow card.
- **Laptop factor (check of the 0.35 assumption).** The 4090 Laptop trains TRM at 1.58 steps/s and evaluates at
  640–680 puzzle-ACT-steps/s: 0.31 of a fast 5080, 0.51–0.61 of a slow one, 0.41 (training) and 0.45–0.47
  (evaluation) of the midpoint. The planning value 0.35 G-h per wall-hour is conservative by ≈15–35%. On the low end
  of a range, laptop wall-time ≈ 3.1–3.2 × G-h. Laptop training rates must be taken from the `steps_per_s` median in
  `metrics.jsonl`: for the resumed laptop runs (A0_s0, A1_s0), `summary.json`'s `train_s` and `wall_s` leave out the
  first 5k-step segment (A0_s0: 1.78 steps/s from `summary.json` vs 1.58 measured).
- **Effective FLOP rates.** One training step costs 42.7 GFLOP per puzzle (35 L-level evaluations of 1.22 GFLOP:
  2 no-grad H-cycles of 7 evaluations, one grad H-cycle of 7 forward plus about 14 backward); one ACT-step forward costs
  25.6 GFLOP per puzzle; both are identical for TRM and Euler-TRM [n1]. That gives 56 / 34 TFLOP/s in training and
  54 / 28 TFLOP/s in evaluation on the fast / slow regime.

### 2.2 Measured anchors

| Quantity | Fast 5080 | Slow 5080 | 4090 Laptop | Source (repository) |
|---|---|---|---|---|
| TRM training, steps/s at B = 256 (median) [n3] | 5.14 | 3.12 | 1.58 | `runs/A0_s*/metrics.jsonl` |
| Euler-TRM (A2) | 5.10 | 3.08 | – | `runs/A2_s*/metrics.jsonl` |
| Euler-TRM + V2 (A3) | 4.00 | 2.44 | – | `runs/A3_s*/metrics.jsonl` |
| V1 cap (A1, A4–A7) | 3.44–3.49 | 2.11–2.14 | 1.13–1.15 | `runs/A1_s*`, `A4_s0`–`A7_s0` (`metrics.jsonl`) |
| Decoder-aware (A8, A10) | 3.13–3.15 | 2.11 (A9) | – | `runs/A8_s0`, `A9_s0`, `A10_s0` (`metrics.jsonl`) |
| In-training eval (test10k), per 5k steps | 78–80 s | 126–140 s | 251–276 s | `metrics.jsonl` (`eval_s`) |
| Peak memory at B = 256, all arms | 9.94–10.24 GB allocated, 10.37–10.65 GB reserved | same | same | `summary.json` |
| exact@16 on the 100k subset (1.6M puzzle-steps) | 762–788 s | 1,375–1,516 s | 2,342–2,522 s | `eval_100k*.json` (`time_s`) |
| exact@64 on the 25k subset (1.6M) | 777–787 s | 1,335–1,577 s | 2,414–2,639 s | `eval_ext64*.json` |
| Full test, 422,786 puzzles × 16 (6.8M) | 3,203–3,244 s | 6,159–6,262 s | 9,959–10,557 s | `eval_full*.json` |
| Population eval, 1k × 16 starts × 64 steps × 2 init modes (2.0M) | 1,009–1,022 s | 1,848–1,942 s | 3,094–3,120 s | `pop_eval*.json` |
| Contraction diagnostics, 1k puzzles (256 Jacobian puzzles) | 452–567 s | 695–851 s | 1,324–1,524 s | `diag*.json` |
| Checkpoint size (EMA or raw weights) | 20.1 MB (TRM), 20.3 MB (Euler-TRM) | | | checkpoint files (not published; measured locally) |
| Optimizer state (`last_state.pt`) | 80 MB for TRM; 284 MB for regularised arms (the adversary's warm-start buffers are included); kept only for A0_s0, A1_s0, A5_s0 | | | checkpoint files (not published; measured locally) |

### 2.3 Training cost per run (G-h, including in-training evaluation)

| Arm type | R (35k) | 2R (70k) | L (195k, paper budget) | Overhead vs TRM |
|---|---|---|---|---|
| TRM, TRM + RI (B1) | 2.0–3.4 (A0 at 45k: 2.6–4.3) | 4.1–6.7 | 11.4–18.7 | 1.00 (RI) |
| Euler-TRM (A2, B3, X3, X6; B2 runs on this path at h = 0.95 with NI, ≤ 1.03) | 2.1–3.4 | 4.1–6.8 | 11.5–19.0 | 1.00 (measured) |
| + V2 (A3, X1, X4, X5, X7, X8) | 2.6–4.2 | 5.2–8.5 | 14.4–23.6 | 1.28 (measured) |
| V1 cap (A1, A4–A7) | 3.0–4.9 | 5.9–9.7 | 16.5–27.1 | 1.47 (measured) |

### 2.4 Overhead model for new regularisers

Let u be one L-level evaluation of one row. A training step costs 35 u per row. A regulariser applied to a fraction
f of the rows (f = 0.25 by default) adds f · c / 35, where c is its per-row cost in u [n2]. The model reproduces the
measured overheads: V2 with P = 3 power steps has c = 2 + 6 + 6P + 12 = 38 → 1.27 (measured 1.28); V1 with one
power step has c = 7 + 21 + 42 = 70 → 1.50 (measured 1.47). Factors used below (on the Euler-TRM base unless noted):

| Variant | c (u per anchor row) | Factor | Rows |
|---|---|---|---|
| Random pairs, no power steps (P = 0) | 12 | 1.05–1.12 | X10 |
| V2 with P = 1 / P = 10 | 26 / 80 | 1.19 / 1.57 | X13 |
| V2 on one block only | ≈19 | 1.14 | X14 |
| V2 with reg_frac 0.125 / 0.5 | 38 | 1.14 / 1.54 | X9 |
| V2 regulariser path in fp32 (small radius; local Jacobian hinge) | 38 × (2–3) | 1.5–1.8 | X11, X12 |
| CMM trace hinge, 2 Hutchinson probes on z_H, double backward | ≈18 | 1.10–1.20 | B4a, B4b |
| Jacobian Frobenius (JR), 1 probe through the cycle map Φ, applied with p ∈ {0.05, 0.2} | ≈28 per applied row | 1.05–1.25 | B5 |
| STARS, 1 JVP power step (fresh Gaussian start in STARS; warm start optional) + ‖J_Φ v‖² with grad | ≈42 | 1.30–1.50 | B6 |
| CoRe-style terminal fixed-point penalty (one extra Φ with grad) | ≈21 | 1.10–1.20 | B7 |
| V2 in another metric (learned diagonal; W_γ): V2 plus metric products and, for the learned diagonal, the gradient of κ | ≈38 + metric products | V2 × 1.0–1.12 (est.) | X16 |
| Shadow carry from a random start + pair penalty (± adversarial mix) | ≈33–48 | 1.24–1.40 | X15 |
| Convex-potential z_L field (gradient inside the forward, double backward) | – | 2.0–3.0 | B8 |
| Visual front-end (per-cell LeNet, ≈32 MFLOP per ACT step vs 42.7 GFLOP) plus rendering | – | 1.05–1.3 | T3 |

### 2.5 Evaluation packages (cost per checkpoint)

One million puzzle-ACT-steps costs 0.139–0.257 G-h (0.42 h on the laptop; the hi end is the slow-regime median)
[n9]. Subsets: pop1k and pop2k are new,
seeded test subsets, nested and disjoint from the test1k used in the published report; every package uses pop1k,
and the other 1k of pop2k is a sealed reserve for a confirmatory rerun if a pop1k result has to be discarded (for
example after a protocol bug). val5k ⊃ val1k ⊃ val500 come from unused puzzles of the Sudoku-Extreme train CSV
(§7.3); EV-V and EV-Vs run on seeds 0–2 only [n10].

| Package | Content | Puzzle-ACT-steps | G-h / checkpoint | Laptop h | Used for |
|---|---|---|---|---|---|
| EV-S0 | exact@16 on 100k (C1); C5-lite: `rand`, N(0,16), other-puzzle state, 16 starts × 64 steps on pop1k | 4.7M | 0.65–1.20 | 2.0 | descriptive arms (A0@T, A1, A5, warm restarts) |
| EV-S | EV-S0 + D1 + D2 [n4] | 4.7M + diag | 1.15–2.20 | 3.6 | 1-seed sweeps and mechanism rows |
| EV-F | exact@16 on 100k; exact@64 on 25k; C3 64 × 64 all-random on pop1k; C2 single unroll to 1,024 on pop1k; E5-core (16 symmetry views × 64); H1 families N(0,16) and other-puzzle 16 × 64 on pop1k; D1 | 11.4M + D1 | 1.83–3.43 | 5.7 | seeds 0–2 of confirmatory and must-baseline arms |
| EV-F6 | EV-F without D1 | 11.4M | 1.58–2.93 | 4.9 | seeds 3–5, reruns, warm restarts |
| EV-G | C4 frontier add-on: 16 × 256 and 4 × 1,024 on pop1k | 8.2M | 1.14–2.11 | 3.5 | frontier figure (seeds 0–2) |
| EV-X | C5-extra: uniform, Laplace, canonical + N(0, σ²) for σ ∈ {0.1, 0.25, 0.5}, z_H-only, z_L-only; 16 × 64 on pop1k; + D2 | 7.2M + D2 | 1.25–2.34 | 3.9 | basin-radius and block analyses |
| EV-TT | E3 PTRM (64 × 64 at σ = 0.3, 16 × 64 at σ = 0.1); E4 GAR (adaptive); E5-ext (64 views × 16; 4 views × 4 starts × 64) | ≈7.5M | 1.05–1.95 | 3.3 | test-time baselines |
| EV-R | C6 attack on 500 puzzles [n5]; C7 mid-trajectory perturbation (2 × 16 × 64 on pop1k); C9 multi-solution (16 × 64 on 1k) | ≈3.1M + attack | 0.63–1.39 | 2.0–2.5 | robustness |
| EV-V | val5k exact@16; on val500: 64 × 64 `rand`, 2 held-out families 16 × 64, E5-core, C2 single unroll to 1,024 (E1) | 4.2M | 0.58–1.07 | 1.8 | selector, (N, K), temperature and recalibration choice; the E1 comparator of H2 |
| EV-Vs | val5k exact@16; `rand` and N(0,16) 16 × 64 on val1k | 2.1M | 0.30–0.55 | 0.9 | choosing σ, weights and sweep winners |
| EV-Full | exact@16 on all 422,786 test puzzles | 6.8M | 0.89–1.74 (measured) | 2.8 (measured) | headline checkpoints |
| (D5, §5.3) | diagnostic D5 as a package: population eval (2 init modes) in fp32 on the published test1k | 2.0M × 2.5 | 0.71–1.32 | 2.2 | fp32 check of residual selectors |
| EV-T3 | EV-S0 on rendered boards; 4 shift sets (pixel noise σ = 0.3 and 0.5, 25% occlusion, writer shift; exact@16 on 25k each); re-render population 16 × 64 on 1k; EV-Vs | 9.4M | 1.31–2.42 | 4.1 | T3 |
| EV-Maze | 1k test mazes: 16 × 64 population and single unroll to 256, at 7.2× the Sudoku cost per step [n13] | 9.2M Sudoku-equiv. | 1.28–2.37 | – | T4 |

### 2.6 Memory and storage

- **Memory.** Every published arm peaks at 10.4–10.65 GB reserved at B = 256 on a 16 GB card, because the
  regulariser runs on stashed anchors before the next main forward (peak = max, not sum). New variants are estimated
  from the extra graph they keep [n6]; five configurations exceed 16 GB and are flagged in bold: the convex-potential
  field (B8), Maze-Hard at B = 128 (T4), ARC fine-tuning at B = 64 (T7), the paper recipe at physical B = 768 (X19,
  about 30 GB) and the init attack without activation checkpointing (C6). Each runs with the workaround given in its
  row (smaller micro-batches with accumulation, or activation checkpointing); this is why the paper budget is reached
  with B = 256 (L) or with 3 × 256 accumulation (X19). Standalone population evaluation at 1,024 rows per batch works
  on the laptop; it comes close to 16 GB only when the laptop is shared with a training run (the published A0_s0
  population eval logged allocator OOM warnings while A1_s0 was training on the same GPU), so use 512 rows when
  sharing. D5 runs in fp32, which roughly doubles activation memory: use 512 rows per batch for D5 (est.).
- **Storage.** An R run with all 5k-step checkpoints, raw final weights, metrics and `last_state.pt` takes ≈0.24 GB
  for unregularised arms and ≈0.45 GB for regularised arms (V1, V2 and other arms with adversary warm-start buffers in
  `last_state.pt`); an L run ≈0.9 GB or ≈1.1 GB. A population dump (per member: board, residuals, q, mean
  max-softmax, per-cell argmax at 4–9 snapshots) takes ≤0.1 GB per checkpoint. Totals: Tier 0 ≈27 GB (59 runs, 23 of
  them regularised, plus 6 warm restarts: ≈20 GB; ≈70 dumps: ≈7 GB), Tier 1 ≈65–70 GB (dominated by optical-flow
  datasets, sizes unverified), Tier 2 ≈23–30 GB. **Keep `last_state.pt` for every new run**, so that 2R and L
  extensions are exact continuations.

---

## 3. Table A: External baselines

### 3.1 Trained baselines (A.1)

Same recipe, data and seeds as the existing arms. Each method gets at most 4 configurations × 1 seed tuned on
validation. Constraint baselines get V2's dual-ascent controller or are reported with both fixed and dual weights.

| ID | Method (source) | Controls for | Configuration | Tier | Train G-h/seed | Seeds (runs) | Train G-h total | Eval G-h | Peak mem (GB) | Eng. days |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | TRM + random-init training, RI (EqR 2605.21488; PI 2211.09961) | Invariance obtained only by seeing random starts (the main rival explanation) | At every ACT reset z_H ~ N(0, σ_H²), z_L ~ N(0, σ_L²) per token from a dedicated generator (data order stays paired); σ_H = 1, σ_L ∈ {1, 8} chosen on validation (EqR's best single run used σ_L = 8) | 0 | 2.0–3.4 | 3 (+1 σ pilot: both σ_L values run at seed 0 and the chosen one is s0) = 4 | 8.2–13.5 | 8.0–15.2 (EV-F ×3, EV-V ×3, EV-Vs ×2, one per σ pilot; C6 on seed 0 of whichever of B1/B2 is not B★, for DP1) | 10.4 | 0.5 |
| B2 | TRM + RI + NI, full EqR recipe (2605.21488) | Strongest published landscape shaping, with Top1Conv | B1 plus z ← z + 0.95 (F(z) − z) + 0.01 ε on every substep (`arch.flow=True arch.flow_h_L=0.95 arch.flow_h_H=0.95 arch.flow_time_embed=False` plus a new noise flag; without `arch.flow=True` the h values are ignored and h = 1, so B2 runs on the Euler code path); evaluate at β = 0 (primary) and β = 0.01 | 0 | 2.0–3.5 | 3 | 6.1–10.4 | 8.1–15.2 (EV-F ×3, EV-V ×3, EV-Vs ×3 at β = 0.01) | 10.4 | 0.25–0.5 |
| B★ | The RI baseline chosen on validation among B1, B2 and B3 (best validation H1 metric); this row adds seeds 3–5 when it is B1 or B2 | Completes the 6-seed comparator for H1 | As B1 or B2; when B3 is chosen, B★ = B3 and this row's runs are not needed (B3 already has 6 seeds and EV-R) [n12] | 0 | 2.0–3.4 | 3 | 6.1–10.1 | 7.5–14.7 (EV-F6 ×3; EV-R ×3 on s0–s2; EV-Full ×1); Tier 1 adds EV-G, EV-X, EV-TT on s0–s2: 10.3–19.2 | 10.4 | 0 |
| B3 | Euler-TRM + RI (EqR recipe on our vehicle) | Whether V2 adds anything beyond Euler steps plus random-start exposure; with A2, A3, X1 completes the {RI} × {V2} 2 × 2 | A2 with B1's init | 0 | 2.1–3.4 | 6 | 12.4–20.5 | 14.8–28.2 (EV-F ×3, EV-F6 ×3, EV-V ×3, EV-R ×3 for the attacked comparison with X1, EV-Full ×1) | 10.4 | 0 |
| B4a | CMM trace hinge, faithful (2603.22871) | Averaged (trace) condition vs V2's worst-direction condition | [ReLU(tr J_F / K − 1)]², weight 1e4, on z_H at segment-end states, Hutchinson with 2 probes; log the activation rate (not known in advance: the joint-field trace is negative at every measured state, §1.2, but the z_H-block trace was not measured) and the gradient spikes a 1e4 weight produces | 0 | 2.3–4.1 | 1 | 2.3–4.1 | 1.4–2.7 (EV-S, EV-Vs) | ≤12 (est.) | 0.5–1 |
| B4b | Binding trace hinge | Same question with a constraint that binds | ReLU(vᵀ J_V v / ‖v‖² + μ_tr), μ_tr = 1.0, Rademacher v, V2's dual ascent to 10% violations | 0 | 2.3–4.1 | 1 | 2.3–4.1 | 1.4–2.7 | ≤12 (est.) | 0.25 |
| B5 | Jacobian Frobenius regularisation (2106.14342) | Two-sided isotropic DEQ stabiliser that DEQ reviewers expect | γ ‖εᵀ J_Φ‖² / d with 1 probe, applied with probability p; 2 configurations on validation (γ = 0.5, p = 0.05, the paper's CIFAR config; γ = 2, p = 0.2), then 2 more seeds at the winner | 0 | 2.2–4.3 | 4 | 8.7–17.0 | 6.1–11.4 (EV-Vs ×2, EV-F ×3) | 11–13 (est.) | 0.5 |
| B6 | STARS spectral-radius penalty (2605.26733) | Local two-sided ρ(J) < 1 | ‖J_Φ v‖² with one power step (STARS: fresh Gaussian start; a warm start is our variant, if used); objective (1 − λ) task + λ penalty, λ = 0.1; 3 seeds only if within the CI of A3 on M2 | 1 | 2.7–5.1 | 1 | 2.7–5.1 | 1.4–2.7 | 11–13 (est.) | 0.5 |
| B7 | CoRe-style terminal fixed-point penalty, our normalised squared form (Think Shallow, Solve Deep, 2608.18222) | "The residual is informative only because training made it small" | ‖Ψ(s_K) − s_K‖² / ‖s_K‖²; weight chosen on validation (EV-Vs) from 2 configurations. CoRe's own penalty is a_T ‖f(z_{T−1}) − z_{T−1}‖ (unsquared, scaled by the learned gate, not normalised), where 0.05 works and 1.0 over-contracts, on 9 × 9 Sudoku with 20–70% holes, not Sudoku-Extreme; those weights do not transfer | 1 | 2.3–4.1 | 2 (configurations) | 4.6–8.2 | 1.8–3.3 (EV-Vs ×2, EV-S) | ≤11 | 0.25 |
| B8 | Exact monotone reference (2306.17332; monDEQ 2006.08591) | Soft penalty vs exact monotonicity | z_L field = −∇_z E_θ(z; z_H + x) − μ z with E convex in z; Euler step with h ‖A‖² L ≤ 1 | 2 | 4.1–10.2 | 1 | 4.1–10.2 | 1.1–2.2 (EV-S) | **15–20 (est.)**: B = 128 × 2 or activation checkpointing | 3–5 |
| B9 | Contractive and firmly nonexpansive imaging priors (2102.07944; 2012.13247) | Standard monotone/contractive priors in imaging | T5 only: spectrally normalised DnCNN (Gilton's model) and the Pesquet hinge λ max{‖∇Q‖², 1 − ε}, Q = 2J − I, ε = 0.05, λ = 1e-5, 5 power iterations | 1 | 0.9–3.75 (est.) | 12 (2 arms × 3 seeds × 2 tasks) | 10.8–45.0 (counted in T5) | 0.6–1.2 (counted in T5) | <8 | 1 (inside T5) |
| B10 | Fast-recipe pilot (Speed is Confidence, 2601.19085) | Whether a cheap high-accuracy regime exists | Its architecture (Muon, SwiGLU, 7M params) for A0 and A3 analogues; only if L runs prove infeasible | 2 | 1.0–2.0 (est.) [n14] | 2 | 2.0–4.0 | 2.3–4.4 (EV-S ×2) | ≈15 (est.) at the source's B = 384 (the source used a 32 GB RTX 5090); fallback 192 × 2 accumulation | 1–2 |

### 3.2 Test-time baselines on the same weights (A.2)

These cost evaluation only. Their compute sits inside the packages of §2.5 except E8 and E9, which are counted
separately.

| ID | Method (source) | Controls for | Configuration | Tier | G-h / checkpoint | Checkpoints | Eval G-h total (counted in) | Peak mem (GB) | Eng. days |
|---|---|---|---|---|---|---|---|---|---|
| E1 | Long single unroll (When Recursive Models Finish Computing, 2609.26487) | "Just iterate longer" | Canonical start, K ∈ {64, 256, 1,024} by prefix of one unroll on pop1k; K = 64 also on 25k; the same unroll to 1,024 on val500 inside EV-V (H2 comparator, C4 selection) | 0 | 0.36–0.67 | 49 | 18–33 (EV-F, EV-F6) | ≤10 | 0.25 |
| E2 | Residual halting (FPRM 2606.18206; Spherical FPRM 2610.25646) | Adaptive depth (prior art, baseline only) | Relative L∞ residual < 0.1 (FPRM); our rule with τ from validation, τ ≥ 0.02 on bf16 trajectories (lower τ only on the fp32 runs of D5, §7.7) | 0 | 0 (same trajectories) | 49 | 0 | – | 0.1 |
| E3 | PTRM per-step noise (2605.19943) | Diversity from noisy trajectories | Gaussian noise on z (not y) every ACT step, σ ∈ {0.1, 0.3}, N ≤ 64; best-Q@N and mode@N | 0 / 1 | 0.71–1.32 | 6 (A0@45k, A3, s0–s2) + 3 (B★, Tier 1) | 4.3–7.9 + 2.1–4.0 (EV-TT) | ≤16 | 0.25 |
| E4 | GAR answer resampling (2610.16427) | Where breadth is best spent | Retry only puzzles with q < 0: reset both states, resample the blank cells of y (z_H) along Gaussian directions rescaled to ‖y_init‖; keep the first retry with q ≥ 0, else max q; R ∈ {16, 64}; count blocks | 0 / 1 | 0.05–0.09 (adaptive) | as E3 | ≤0.5 (EV-TT) | ≤16 | 0.5 |
| E5 | Test-time augmentation vote (2601.19085 aggregates by probability averaging and halt-first; majority vote is our variant) | Breadth over input symmetries vs breadth over z0 | Digit relabelling, band/stack and row-in-band permutations, transpose; alone (16 views × 64) and crossed with random starts at matched B_tt | 0 / 1 | core 0.14–0.26; ext 0.28–0.53 | core 49; ext 6 + 3 | core 7–13 (EV-F); ext 1.7–3.2 + 0.8–1.6 (EV-TT) | ≤16 | 0.5–1 |
| E6 | Seed-ensemble vote (2601.19085 aggregates by probability averaging and halt-first; majority vote is our variant) | Breadth over weights | Vote across 3–6 independently trained seeds at matched B_tt, from C1/C3 outputs | 0 | 0 (CPU) | 6 seeds × 6 arms | 0 | – | 0.1 |
| E7 | Other confidence signals (C-voting 2604.13521; SMC 2605.25230; Tilted Bowl 2609.39277) | "Vote share is just self-consistency" | Mean top-1 probability; token-marginal entropy; halting q; halt-first in time order; residual-weighted vote softmax(−r/T), T from validation | 0 | 0 (logged in C3) | all C3 checkpoints | 0 | – | in M library |
| E8 | External full-budget checkpoints under our protocol: EqR (HF `locuslab/EqR-model`), GAR (`MaximeILL/GAR_ckpts`), community TRM-MLP (`gaoxin492`, third-party) | Undertraining artefact; EqR itself on held-out inits | EV-F + EV-Vs; no official Sudoku TRM checkpoint exists. Their training subsets of the Sudoku-Extreme train CSV cannot be reconstructed (the upstream dataset builder was unseeded), so their training puzzles may overlap our validation rows: E8 uses test-free default selectors (majority vote; min-residual with τ fixed in advance), and its EV-Vs numbers are descriptive only | 0 | 2.13–3.98; download ≈20 MB per 5.03M-parameter fp32 checkpoint (repository sizes not checked) | 3 | 6.4–11.9 (separate) | ≤16 | 0.5–1.5 |
| E9 | Test-time damping without retraining (FPRM-style) | Whether invariance can be had without any training change | h_test ∈ {0.25, 0.5} for TRM, 0.25 for Euler-TRM; 16 random starts × 64 on pop1k (4.1M puzzle-ACT-steps in total) | 0 | 0.28–0.53 (TRM, 2 values); 0.14–0.26 (Euler-TRM) | 3 (A0@45k, A2, A3 s0) | 0.6–1.1 (separate) | ≤16 | 0.1 |

### 3.3 Published numbers quoted, not rerun (A.3)

Kept in a separate block with their protocol, never in the matched table. Compute for every row: **0 G-h (quoted)**.

| Method | Number | Protocol caveat | Compute |
|---|---|---|---|
| TRM, Less is More (2510.04871) | Sudoku-Extreme: MLP 87.4, Att 74.7. Maze-Hard: Att 85.3, MLP 0.0. ARC-1 / ARC-2 (Att, pass@2): 44.6 / 7.8 | Single pass, 16 ACT steps; 50k epochs at B = 768 (≈65,104 steps); upstream README: "around 87% (±2%)", ≈18 h on one L40S | 0 G-h (quoted) |
| TRM-MLP reproductions | PTRM 87.28; GAR 85.90 (1k subset); Guided SMC 85.9 ± 0.6; EqR 84.8; EqR 5-seed baseline 84.33 ± 0.59 (SD); GRAM 83.6; community checkpoint 84.80 (model card; 8 × H200, global B = 4,608) | Full budgets; FPRM's "72.6" is a TRM-Att comparison and is not used | 0 G-h (quoted) |
| EqR (2605.21488) | Sudoku 86.4 (depth 16, breadth 1) → 93.0 (depth 64, breadth 1) → 99.8 (depth 64, breadth 128); 5 seeds 86.18 ± 0.44; σ_L = 8 single run 87.30; Mini-ARC 55.28 (TRM 48.35); Maze-Unique 82.2 / 88.9 / 93.0; Δ_PI 3.58 → 0.10 with RI (Sudoku-Lite, B = 128) | B = 768 × 50k steps; the aggregation behind 99.8 is not stated; EqR's "Maze" is Maze-Unique, not Maze-Hard | 0 G-h (quoted) |
| FPRM (2606.18206) | Sudoku 94.2, Maze-Hard 87.0, ARC-1 47.5, ARC-2 6.2; stabilised TRM 39.46 (with conv) / 19.77 | Single trajectory, up to 35k evaluation iterations; trained 60k epochs | 0 G-h (quoted) |
| Spherical FPRM (2610.25646) | 96.6 / 87.9 / Mini-ARC 59.1 / ARC-1 47.4 / ARC-2 3.8; "72% of puzzles reach the same fixed point" | Single seed; the 72% has no stated sample size, uses x and −x among 8 starts and measures fixed points, not decoded unanimity: not comparable to ours | 0 G-h (quoted) |
| Geometric Fixed-Point Models (2609.33540) | 95.1 on all 422,786 puzzles; Maze-Hard 92.0 | ≤35k damped Picard steps (Sudoku), 1,024 (Maze); seeds not reported | 0 G-h (quoted) |
| Looped Flows (2609.11801) | 97.9 ± 0.4 single, 99.3 ± 0.2 best-Q@5; Maze 86.7; ARC-1 58.8; ARC-2 12.2; TRM failures: 88.3% non-convergence, 11.7% spurious attractors | Counted in flow steps (128), not ACT steps | 0 G-h (quoted) |
| CMM (2603.22871) | 93.7 (CMM512-NSDE); Maze 82.2 | Trace estimator and seed count not stated | 0 G-h (quoted) |
| GAR (2610.16427) | 85.90 → 98.27 (R = 16, 1.68 blocks) → 99.43 (R = 64, 2.03 blocks); PTRM in GAR's harness 98.47 at 8.65 blocks | 1k-puzzle subset | 0 G-h (quoted) |
| PTRM (2605.19943) | best-Q 98.75, pass@K 99.06 (K = 100, D = 64, σ = 0.3); Maze 83.80 → mode@100 86.73 (D = 16, σ = 1.0), best-Q 85.17 | Sudoku test-set size unclear; noise on z only | 0 G-h (quoted) |
| GRAM (2605.19376) | 86.6 (N = 1) → 96.1 (N = 20); TRM with random init 79.9 ± 1.2 → 91.2 ± 0.9 (N = 20, majority) | 10M params; random z0 drawn at inference, training init not stated | 0 G-h (quoted) |
| When Recursive Models Finish Computing (2609.26487) | MLP 74.4 → 91.9; Att-A 59.2 → 87.5 (16 → 512 steps) | Community checkpoints; 1k puzzles from the hardest source | 0 G-h (quoted) |
| ICS (2610.18547) | 99.70 label-free, 99.94 with a task-rule checker; Maze 100.0 (checker) / 99.1 (label-free); EqR re-run on Maze-Hard 95.0 / 87.4 | ≈1.14 TFLOP per puzzle | 0 G-h (quoted) |
| Speed is Confidence (2601.19085) | Baseline 85.5 ± 1.3; 4-view TTA 89.0; 3-seed ensemble 88.5; both 91.5; halt-first over 12 members 97.2 at ≈18.5 ACT steps | Different architecture (Muon, SwiGLU, 7M); baseline 8k steps × B = 384, 48 min on one RTX 5090 | 0 G-h (quoted) |
| Guided SMC (2605.25230); C-voting (2604.13521); Tilted Bowl (2609.39277) | SMC: TRM 85.9 ± 0.6 → 98.0 ± 0.3, entropy AUROC 0.834 ± 0.011. C-voting: 95.2 (Sudoku), 78.6 (Maze), not on TRM. Tilted Bowl: agreement AUROC 0.94–0.98 vs halting head 0.99–1.00; majority loses 3–8 points to the head selector | Different harnesses | 0 G-h (quoted) |
| What Limits Recursive Reasoning Models (2609.39967); Looped Reasoning Finishes Earlier (2610.18174) | 98.41 (Sudoku); QRR Mini-ARC 62.4, Sudoku 97.9 | 2609.39967 reports that gradient accumulation does not reproduce large-batch results | 0 G-h (quoted) |
| TRM on ARC-AGI-1 (2512.11847); ARC Prize checkpoint | 40.00 pass@1 with a 1,000-augmentation vote vs 29.25 single canonical pass; checkpoint card: ARC-1 40%, ARC-2 6.2% | ARC only; no Sudoku or Maze checkpoint | 0 G-h (quoted) |
| Visual Sudoku: SATNet (1905.12149); Assessing SATNet (2312.11522); Reasoning with NCA (2609.36126) | SATNet 63.2% boards (ceiling ≈74.7%); 0% without intermediate digit labels; NCA pixel-space 87.8 easy / 18.9 hard-OOD | SATNet: easy puzzles (≈36 givens). NCA is a different task, not MNIST-rendered Sudoku-Extreme | 0 G-h (quoted) |
| Restoration: 2012.13247; 2102.07944; 2404.00390 | BSD10 kernel (a) 30.86 vs BM3D 30.05 dB (PnP-FB, 1,000 iterations); DE-Grad CelebA Deblur(1)/(2) 32.43 / 37.99 dB; nonlinear deblurring ≈28.3 dB | Pesquet trains on 50k ImageNet images and has no public code; Gilton's denoiser is spectrally normalised | 0 G-h (quoted) |
| Optical flow: RAFT (2003.12039); DEQ-Flow (2204.08442) | RAFT C+T: KITTI-15 train EPE 5.04, F1 17.4%. DEQ-Flow-B: Sintel train clean/final 1.43 / 2.79, KITTI 5.43 / F1 16.67; DEQ-Flow-H-1: 1.45 / 2.58 / 3.97 / 13.41 | RAFT 5.3M params (small ≈1.0M); DEQ-Flow has no "small" variant; DEQ-Flow code is AGPL-3.0, RAFT BSD-3 | 0 G-h (quoted) |
| Jacobian regularisation (2106.14342); Path independence (2211.09961) | JR: CIFAR-10 93.1% at 6 NFE vs 93.6% at 17; PI: the init attack drives non-PI alignment from 0.32 to 0.09 and accuracy to 0 on mazes | Context for B5 and C6 | 0 G-h (quoted) |

---

## 4. Table B: Internal ablations and variations

Flags exist for every row marked "0" engineering days (`scripts/run_benchmark.sh`, `config/arch/trm_contractive.yaml`).
Every existing run and evaluation is reused [n11].
μ defaults to −ln(1 − η)/h_H, so at h = 1 it silently halves to 0.0513: every V2 row without Euler damping passes
`++arch.reg_mu=0.1026` explicitly.

| ID | Arm | Change vs A3 | Question answered | Tier | Train G-h/seed | Seeds (runs) | Train G-h total | Eval G-h | Peak mem (GB) | Eng. days |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | TRM, 45k steps (35k prefix = A0@T) | no Euler, no V2 | Reference; A0@45k is the FLOP-matched control for A3@35k (F_reg ratio 1.28 → 44.8k) | 0 | 2.6–4.3 | 6: reuse s1–s2, rerun s0 on a 5080 (the family's seed 0; the published laptop s0 stays descriptive, §7.2), new s3–s5 (4) | 10.5–17.3 | 31.9–60.1 [n7] | 10.4 | 0 |
| A2 | Euler-TRM, h = 0.5, time embedding τ | no V2 | Vehicle for V2; V2's accuracy cost (H4) | 0 | 2.1–3.4 | 6: reuse s0–s2, new s3–s5 (3) | 6.2–10.2 | 12.7–23.6 (EV-F ×3, EV-V ×3, D5 ×1, EV-F6 ×3) | 10.4 | 0 |
| A3 | Euler-TRM + V2 (main) | none | Main method | 0 | 2.6–4.2 | 6: reuse s0–s2, new s3–s5 (3) | 7.8–12.7 | 24.9–47.0 (EV-F, EV-G, EV-V, EV-X, EV-TT, EV-R ×3; D5; EV-F6 ×3) | 10.6 | 0 |
| A1, A4–A10 | Existing caps and decoder-aware arms | – | A1 and A5 as negative controls in the invariance table (collapse also yields invariance); the rest stay in the supplement unchanged | 0 | 0 | 0 | 0 | 3.3–6.1 (EV-S0 on A1 ×3 and A5; D5 on A1 s0) | 10.6 | 0 |
| X1 | Euler + RI + V2 | + RI | V2 on top of random-init training (H5) | 0 | 2.6–4.2 | 6 | 15.5–25.4 | 14.8–28.2 (EV-F ×3, EV-F6 ×3, EV-V ×3, EV-R ×3, EV-Full); Tier 1: EV-G, EV-X ×3: 7.2–13.4 | 10.6 | 0.25–0.5 |
| X2 | A2 and A0 both warm-restarted from their EMA `step_35000` weights with fresh Adam (and a fresh EMA) to 45k; A0's restart is compared with its true `step_45000` to calibrate the restart. Raw weights are saved only at the end of a run (A2: `step_35000_raw`; A0: `step_45000_raw`), so EMA weights are the only like-for-like starting point (checkpoints not published) | +10k steps | "The gain was bought with compute" (no optimizer state exists, so not an exact continuation) | 0 | 1.2–1.9 (two 10k-step runs) | 3 seeds (6 short runs) | 3.5–5.8 | 6.7–12.4 (EV-F6 ×3, EV-S0 ×3) | 10.4 | 0.25 |
| X3 | Euler without τ (`arch.flow=True arch.flow_time_embed=False`) | no V2, no τ | Splits A2's +2.98 between damping and τ | 0 → 1 | 2.1–3.4 | 1, then +2 | 2.1–3.4 (+4.1–6.8) | 1.1–2.2 (+2.3–4.4) | 10.4 | 0 |
| X4 | X3 + V2 (autonomous field) | no τ | Does an autonomous field strengthen invariance and the residual–correctness link? Required before any "unique equilibrium" wording | 0 → 1 | 2.6–4.2 | 1, then +2 | 2.6–4.2 (+5.2–8.5) | 1.1–2.2 (+2.3–4.4) | 10.6 | 0 |
| X5 | TRM at h = 1 + V2, `++arch.reg_mu=0.1026` | no Euler | Does V2 need damping? | 0 → 1 | 2.6–4.2 | 1, then +2 | 2.6–4.2 (+5.2–8.5) | 1.1–2.2 (+2.3–4.4) | 10.6 | 0 |
| X6 | h = 1 with τ | no damping, no V2 | Extra capacity vs conditioning on time | 1 | 2.1–3.4 | 1 | 2.1–3.4 | 1.1–2.2 | 10.4 | 0 |
| X7 | h ∈ {0.25, 0.75, 0.95} + V2, μ fixed at 0.1026 | h | Damping dose-response at fixed μ (EqR uses h = 0.95) | 1 | 2.6–4.2 | 3 | 7.8–12.7 | 3.4–6.6 | 10.6 | 0 |
| X8 | μ ∈ {0, 0.03, 0.3, 1.0} (A3: 0.103) | margin | Dose-response and the accuracy–uniqueness frontier; μ = 0 tests whether the margin matters | 0 → 1 | 2.6–4.2 | 4, then +2 at μ = 0.3 | 10.3–16.9 (+5.2–8.5) | 5.8–11.0 (+2.3–4.4) | 10.6 | 0 |
| X9 | Controller: violation target {3, 30}%; fixed λ ∈ {0.1, 1} (`reg_dual=False`); V2 only during the 2k warmup; reg_frac ∈ {0.125, 0.5} | enforcement | Is the 10% target arbitrary? Is invariance imprinted early (λ decays from 1.0 to ≈0.09)? | 1 | 2.1–5.3 (mean 2.6–4.2) | 7 | 17.9–29.5 | 8.0–15.4 | 10.6 | 0.25–0.5 |
| X10 | Random secant pairs (`reg_pair=rand`) | no adversary | Is the adversarial pair search necessary? It is ALR's pair search (1907.05681) applied to a one-sided condition on a weight-tied latent field; X10 tests whether that search matters, not a novelty claim | 0 | 2.2–3.8 | 3 | 6.5–11.5 | 3.4–6.6 (EV-S ×3) | 10.6 | 0 |
| X11 | Pair radius 0.5–2% of ‖s‖ (new fp32 field path) and 20–100% (A3: 5–20%) | scale | Local vs coarse monotonicity; random starts lie far beyond 20% of ‖s‖ | 0 → 1 | 3.2–6.1 (small), 2.6–4.2 (large) | 2, then +2 at the winner | 5.8–10.4 (+5.2–12.3) | 2.9–5.5 (+2.3–4.4) | ≤11 (est.) | 0.5–1 |
| X12 | Local Jacobian hinge ReLU(λ_max(sym J_V) + μ), 3 power steps in fp32 | infinitesimal pairs | Enforces exactly what D1 finds violated | 1 | 3.1–6.1 | 1 | 3.1–6.1 | 1.1–2.2 | ≤11 (est.) | 1 |
| X13 | Adversary strength: `reg_v2_power_iters` ∈ {1, 10}; cold start (new `reg_v2_cold_start` flag: the training adversary is always warm-started today, only the monitor cold-starts) | adversary | Is the held-out λ_max > 0 a weak-adversary artefact? | 2 | 2.5–4.1 / 3.2–5.4 / 2.6–4.2 | 3 | 8.3–13.6 | 3.4–6.6 (EV-S ×3) | 10.6 | 0.1 |
| X14 | V2 on one block only (z_L only, z_H only) | block | Which state carries the invariance (cf. TAPS, GAR)? | 1 | 2.3–3.9 | 2 | 4.7–7.8 | 2.3–4.4 | 10.6 | 0.5 |
| X15 | On-trajectory pairs: shadow carry from z0 ~ N(0, I), pair (s_t, s′_t) at the same ACT step; variant mixed 50/50 with adversarial pairs | pair source | Penalises directly what unanimity measures; developed on validation and pre-registered | 1 | 2.6–4.8 | 3 (1 on validation, then 2) | 7.7–14.3 | 3.7–7.2 | ≤11 (est.) | 1–1.5 |
| X16 | Metric variants: learned diagonal metric (κ ∈ {10, 100}); decoder-weighted W_γ = P + γ(I − P), γ ∈ {0.1, 0.3} (cf. semi-contraction with respect to a seminorm, 2005.09774) | metric | Does a weaker metric lower the accuracy cost? | 2 | 2.6–4.7 (V2 × 1.0–1.12, §2.4) | 4 | 10.3–19.0 | 4.6–8.8 (EV-S ×4) | 10.6 | 1–1.5 |
| X17 | V1 caps with more seeds: A7 (cap 4) and A4 (V1 on Euler), s1–s2 | two-sided cap | One-sided vs two-sided with seeds; caps ≤ 2 are decisive at n = 1 | 1 | 3.0–4.9 | 4 | 11.8–19.5 | 4.6–8.8 | 10.6 | 0 |
| X18a | Full budget L (195k at B = 256; 35k and 70k are exact prefixes), seed 0 of A0, A3, B★, X1 | budget | Undertraining artefact? (our TRM: 64–69 vs 83.6–87.3 in reproductions, 87.4 published) | 1 | 11.4–18.7 (TRM, B★), 14.4–23.6 (A3, X1) | 4 | 51.6–84.7 | 21.2–40.0 [n8] | 10.4–10.6 | 0 |
| X18b | L budget, remaining: s1–s2 of A0, A3, B★, X1; A2 s0–s2 | budget | Same, confirmatory at 3 seeds; A3@150k vs A0@195k is the L compute match | 2 | 11.4–23.6 | 11 | 137.6–226.4 | 56.5–106.7 | 10.4–10.6 | 0 |
| X18c | 2R extension of seeds 3–5 (A0 45k → 70k; A2, A3 35k → 70k) | budget | Budget curves at 6 seeds | 2 | 1.5–2.4 / 2.1–3.4 / 2.6–4.2 | 9 | 18.3–30.2 | 10.3–19.8 (EV-S ×9) | 10.4–10.6 | 0 |
| X19 | Batch fidelity: A0, 65,104 steps at B = 768 via 3 × 256 accumulation, separate ACT carries per micro-batch | recipe | Is L at B = 256 a faithful stand-in? (2609.39967: accumulation ≠ large batch) | 1 | 11.4–18.7 | 1 | 11.4–18.7 | 2.0–3.9 (EV-Full, EV-S) | 10.4 per micro-batch; **≈30 (est.) at physical B = 768** | 1 |
| X20 | V2 fine-tune of A2@35k for +10k steps vs plain continuation (X2's A2 arm) | retrofit | Cheap recipe for released checkpoints; decides whether T7's V2 arm is worth running | 2 | 0.7–1.2 | 3 | 2.2–3.6 | 3.4–6.6 | 10.6 | 0.25 |

---

## 5. Table C: Test-time protocols, metrics and diagnostics

B_tt is the total number of ACT steps per puzzle, summed over all starts. Checkpoint counts are for Tier 0 unless
marked.

### 5.1 Protocols (C.1)

| ID | Protocol | Configuration | B_tt | Tier | G-h / checkpoint | Checkpoints | Eval G-h total (counted in) | Peak mem (GB) | Eng. days |
|---|---|---|---|---|---|---|---|---|---|
| C0 | Compute accounting | `torch.utils.flop_counter` per arm (F_main, F_reg), NFE counters every step, batch-1 latency on a 5080 | – | 0 | 0.1 total | one profiled step per arm | 0.1 | – | 0.5 |
| C1 | Single canonical run | K = 16; argmax answer and Q-halt; exact@16 on 100k for every checkpoint; full test only for headline checkpoints | 16 | 0 | 0.21–0.42 (100k); 0.89–1.74 (full) | 77 (100k); 4 new full (A0_s0 rerun, B★, B3, X1 s0) | 16–32 + 3.6–7.0 (packages; EV-Full) | ≤10 | 0 |
| C2 | Long single unroll | E1 + E2; also on val500 inside EV-V (21 checkpoints) | 64–1,024 | 0 | 0.36–0.67 | 49 | 18–33 (EV-F) | ≤10 | 0.25 |
| C3 | Population dump | 64 all-random members × K = 64 on pop1k, snapshots at K = 8/16/32/64; per member: board, residual (last and mean of last 3), q_halt, mean max-softmax, per-cell argmax. N < 64 by averaging 20 random sub-orderings. Selectors: majority vote; vote share; min-residual and Top1Conv; Q-halt best-of-N; halt-first; residual-weighted vote; C-voting; oracle pass@N | ≤4,096 | 0 | 0.57–1.05 | 49 | 28–51 (EV-F) | ≤16 (1,024 rows; 512 when the laptop is also training, and for fp32) | 1.5 (with C5) |
| C4 | Matched-compute frontier | B_tt ∈ {16, 32, …, 4,096} through (N, K) or C2; selector and shape chosen per budget on validation among the shapes EV-V covers (N ≤ 64 with K ≤ 64 from the 64 × 64 dump; N = 1 with K ≤ 1,024 from the single unroll); the EV-G shapes (16 × 256, 4 × 1,024) have no validation run, use the selector chosen on validation for the same N at K = 64, and enter the frontier as fixed points; TTC-AUC over log₂ B ∈ [4, 12]; pre-registered acc@256 and acc@1,024; sequential depth K as latency proxy | 16–4,096 | 0 / 1 | 1.14–2.11 (grid add-on) | 6 (A0@45k, A3) + 6 (X1, B★, Tier 1) | 6.8–12.7 + 6.8–12.7 (EV-G) | ≤16 | 0.25 |
| C5 | Init families | `rand` = N(0, 1) per token on z_H and z_L (the RI training distribution when σ = 1). Held out, relative to the σ chosen for B1 (§3.1): a scaled Gaussian with std ≥ 2× the largest training σ per block, i.e. N(0, 16) on z_H and, on z_L, N(0, 16) if σ_L = 1 or N(0, 256) if σ_L = 8 (written N(0, 16) below for σ_L = 1); uniform and Laplace at unit variance; the converged state of another puzzle; canonical + N(0, σ²), σ ∈ {0.1, 0.25, 0.5}; z_H only, z_L only, both. Headline comparisons use all-random populations for every arm | as C3 | 0 / 1 | H1 pair 0.28–0.53; lite 0.43–0.79; extra 1.0–1.85 | 49 (H1 pair); 28 (lite); 6 + 6 (extra) | in EV-F, EV-S/S0, EV-X | ≤16 | in C3 |
| C6 | Init attack | Adam/PGD, 20 steps on z0 with ‖z0‖_rms ≤ 2, full fp32 backprop through K = 16, maximising CE of the canonical answer; evaluate the attacked z0 at K = 16 and 64; 500 puzzles | n/a | 0 | 0.2–0.6 (est.) [n5] | 16: A0@45k, A3, X1, B3, B★ s0–s2 (EV-R), plus seed 0 of the B1/B2 arm not chosen as B★ (for DP1); the seed-0 attacks of B1, B2, B3, X1 and A3 run before DP1 (Phase C) | 3.2–9.6 (EV-R; the extra seed-0 attack is counted in the B1 row) | ≤12 with activation checkpointing per ACT step; **>16 without** | 0.5–1 |
| C7 | Mid-trajectory perturbation | Perturb the state at ACT step 4 or 16 (Gaussian, or z_H of another puzzle); rate of returning to the same answer | n/a | 0 | 0.28–0.53 | 15 (A0@45k, A3, X1, B3, B★ s0–s2) | 4.2–8.0 (EV-R) | ≤16 | 0.25 |
| C8 | Adaptive breadth | Add starts until 2 agree or vote share ≥ θ ∈ {0.5, 0.75}; N_max = 64; simulated from C3 | adaptive | 0 | 0 (CPU) | all C3 | 0 | – | 0.1 |
| C9 | Multi-solution Sudoku (T2) | Remove clues until ≥ 2 solutions; valid-solution rate, rate of invalid "averaged" boards, distinct valid solutions within pass@16 | 16 × 64 | 0 | 0.14–0.26 (+ CPU < 1 h to build) | 15 (as C7) | 2.1–3.9 (EV-R) | ≤10 | 0.5–1 |
| CV | Validation evaluation | EV-V on seeds 0–2 of every arm that needs selector, (N, K), temperature or recalibration choices; EV-Vs for σ, weights and sweep winners | as C3 | 0 | 0.58–1.07 (EV-V); 0.30–0.55 (EV-Vs) | 21 (EV-V); 18 (EV-Vs) | 12.2–22.5 + 5.4–9.9 | ≤16 | 0.5 (pool and subsets) |

### 5.2 Metrics (C.2)

All metrics are computed on CPU from the outputs of C1–C9; GPU cost 0. CPU cost for the full bootstrap, GLMM and
recalibration pipeline is estimated below 10 CPU-hours per tier.

| ID | Metric | Definition | Role | GPU G-h | Eng. days |
|---|---|---|---|---|---|
| M1 | Exact and cell accuracy | Board fully correct; per-cell accuracy; stratified by `rating` bin and `source` | Everywhere | 0 | 0 |
| M2 | Accuracy-adjusted invariance | UC = unanimous and correct; UW = unanimous but wrong; P(unanimous given canonical correct); P(unanimous given canonical wrong); distinct boards among non-unanimous puzzles | **Primary invariance endpoint.** Raw unanimity is reported but is not an endpoint (A1 at ≈10% accuracy reaches 34% unanimity on one seed) | 0 | 0.25 |
| M3 | Path independence | Δ_PI(B) = mean absolute gap Acc_B − Acc_1 (EqR); AA = cosine between endpoints and the canonical endpoint (PI) | Comparable with EqR and PI | 0 | 0.1 |
| M4 | AUROC | Per confidence signal (E7); DeLong per seed, t-CI across seeds | E7 comparison; invariance–confidence tension (vote share is constant at full invariance) | 0 | in library (1–1.5) |
| M5 | Risk–coverage | AURC, E-AURC; selective accuracy at 50/80/90/95% coverage; coverage at risk ≤ 1% and ≤ 0.1% | H3 | 0 | in library |
| M6 | ECE and Brier | 15 equal-mass bins (vote share takes N + 1 values); reliability diagrams; raw and after isotonic/temperature recalibration fitted on validation | Does the raw ECE gap (A3 0.06 vs A0 0.26 from random starts) survive recalibration? | 0 | in library |
| M7 | Basin count | Distinct boards; latent attractor clusters (single linkage at relative distance 0.05 among endpoints below the fp32-confirmed residual floor); correct-basin mass; wrong-and-low-residual rate; WQD (within-question discrimination: the fraction of (correct, wrong) endpoint pairs of the same puzzle in which the correct endpoint has the higher confidence; from Confidence Improves Self-Consistency in LLMs, 2502.06233) | Spurious-attractor risk | 0 | 0.25 |
| M8 | Recoverable diversity | Oracle pass@N; share of canonical failures recovered by at least one member | Diversity-cost risk | 0 | 0 |
| M9 | Robustness | Attacked accuracy and attacked AA (C6); return rate (C7) | Invariance beyond Gaussian starts | 0 | 0 |

### 5.3 Diagnostics (C.3)

| ID | Diagnostic | Configuration | Tier | G-h / checkpoint | Checkpoints | Eval G-h total (counted in) | Peak mem (GB) | Eng. days |
|---|---|---|---|---|---|---|---|---|
| D1 | Held-out λ_max(sym J_V) profile | 256 puzzles, fp32, 30 shifted power iterations, ACT steps {1, 4, 16, 64} × starts {canonical, `rand`}; median, p90, fraction above −μ and above 0; random Rayleigh quotient (trace proxy over the joint state); the z_H-block trace tr(∂V_H/∂z_H) / d_H by Hutchinson (the quantity CMM penalises, not measured so far); ρ(J_Ψ), ‖V‖/‖s‖ | 0 | 0.25–0.5 (est.) [n4] | 41 (EV-F s0–s2 and EV-S) | 10–21 (EV-F, EV-S) | ≤16 | 1 (with D2) |
| D2 | Secant scale profile | Adversarial quotient q(r) at r/‖s‖ ∈ {0.01, 0.05, 0.2, 0.5, 1}, 3-step and 30-step adversaries, training and test states | 0 | 0.25–0.5 (est.) | 20 (EV-S, EV-X) | 5–10 (EV-S, EV-X) | ≤16 | in D1 |
| D3 | Pair contraction | Slope of log ‖s_t^(m) − s_t^(0)‖ / ‖s_t^(0)‖ over t ∈ [1, 32], time to merge (d_t < 0.02) | 0 | 0 (logged in C3/C5) | all C3 | 0 | – | 0.25 |
| D4 | Residual vs correctness | AUROC(−r), WQD (M7) of −r, P(wrong given r < τ), residual histograms of correct vs wrong selected trajectories | 0 | 0 (CPU) | all C3 | 0 | – | 0 |
| D5 | fp32 re-evaluation of residual selectors | Population eval in fp32 on the published test1k (512 rows per batch); drop every rule with τ < 0.02 unless confirmed in fp32; check A1's inverted residual AUROC | 0 | 0.71–1.32 | 4 (A0@45k, A2, A3, A1 s0) | 2.9–5.3 (arm rows) | ≤16 | 0.1 |
| D6 | Quantization robustness | int8 / int4 post-training weight quantization; exact@16 on 100k and EV-S0-size population at each precision (2 × 4.7M puzzle-ACT-steps) | 1 | 1.30–2.40 | 3 (A0@45k, A2, A3 s0) | 3.9–7.2 | ≤10 | 0.5 |
| D7 | Basin maps | 2D slices of z0 space as in Fractal basins trap latent reasoning (2609.04963), 100 × 100 grid, 5 puzzles, K = 64, A3 and B★ (reduced from the paper's 200 × 200 grids over 100–300 instances per random seed, with at least 300 valid slices per model-task pair); 3.2M puzzle-ACT-steps per checkpoint | 2 | 0.45–0.82 | 2 | 0.9–1.6 | ≤16 | 0.5 |
| D8 | Euler-inheritance check | h ‖J_V‖² / 2μ on held-out states (expected ≫ 1; σ_max(J_Φ) median ≈12 for A3) | 0 | 0 (inside D1) | as D1 | 0 | – | 0 |

---

## 6. Table D: Tasks

| ID | Task | Vision argument | Status | Tier | G-h / run, reduced budget | G-h / run, paper budget | Runs | Train G-h total | Eval G-h | Dataset prep | Peak mem (GB) | Eng. days |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | Sudoku-Extreme (1k × 1,000 augmentations; 422,786 test puzzles) | None by itself; the mechanism testbed | Exists | 0 | 2.0–3.4 (TRM) to 2.6–4.2 (V2) at R | 11.4–18.7 to 14.4–23.6 at L | Tables A, B | Tier 0: 120–202 | Tier 0: 164–310 | Exists (1.3 GB training set); validation pool and new subsets: CPU minutes | 10.4–10.65 | 0.5 |
| T2 | Multi-solution Sudoku (C9; evaluation only) | Invariance vs genuine ambiguity (2609.35274) | Small script | 0 | 0 | 0 | 0 | 0 | 2.1–3.9 (EV-R) | CPU < 1 h (clue removal, exact-cover solution counting) | ≤10 | 0.5–1 |
| T3 | Visual Sudoku-Extreme, grounded: givens rendered from MNIST (train digits for training boards, test digits for test boards); per-cell LeNet pretrained on MNIST and fine-tuned; digit labels are disclosed as used; LeNet pretraining on MNIST costs < 0.1 G-h (est.) and is not counted. Arms A2, A3 and B★ (chosen on validation, never on test) × 3 seeds; shift sets: pixel noise σ ∈ {0.3, 0.5}, 25% occlusion, writer shift | Real pixels, perception uncertainty (SATNet line) | New | 0 | 2.2–4.4 (Euler), 2.7–5.5 (V2) | 12–25 (Euler), 15–31 (V2) | 9 | 21.1–42.9 | 11.8–21.8 (EV-T3) | MNIST download (≈11 MB, unverified), rendering on the GPU from indices; writer-shift set (EMNIST) size unverified | ≈11 (est.) | 2–3 |
| T3+ | Visual Sudoku, ungrounded (loss on non-given cells only, so no digit labels leak; report digit-grounding accuracy) for A3 and B★ × 3, plus X1 on the grounded variant × 3 | Symbol grounding without labels (SATNet fails at 0%) | New | 1 | as T3 | as T3 | 9 | 22.7–46.2 | 11.8–21.8 | none extra | ≈11 (est.) | 1–1.5 |
| T4 | Maze-Hard 30 × 30, TRM-Att; A0 (seed 0 is the DP4 gate run), B★, A3 × 3 seeds | 2-D grid planning; the second standard TRM benchmark | Upstream builder (`aug=False` by default) | 2, gated (DP4) | 15–30 (TRM-Att), 19–38 (V2) at 10M examples (20%) [n13] | 75–150, 95–190 (50M) | 9 + 2k-step pilot | 147.4–294.8 (incl. pilot 0.4–0.8 [n13]) | 12–22 (EV-Maze) | Hugging Face download (few MB); CPU minutes | **≈35 (est.) at B = 128**: micro-batch ≤ 43 with accumulation, or activation checkpointing | 2–4 |
| T5 | Recurrent image restoration: x ← x + h(F(x, y, τ) − x) with a DnCNN-size F. Denoising at σ = 25 (train BSD400; test BSD68, Set12), then BSD10 deblurring with the 8 kernels of 2012.13247. Arms: unconstrained, V2, SN-DnCNN, Pesquet hinge (B9), JR (B5) × 3 seeds × 2 tasks | CVPR-native operator where monotone priors are standard; our point is init invariance plus residual-based uncertainty (AUSE, the area under the sparsification error curve of the residual as an uncertainty score), not the penalty | New | 1 | 0.75–2.5 (unregularised), 0.9–3.75 (regularised) (est.) [n15] | n/a (no reference budget) | 30 | 26.1–105.0 | 1.5–3.0 | BSD400, BSD68, Set12, BSD10, 8 kernels: < 0.5 GB (unverified) | < 8; laptop-friendly | 5–7 |
| T6 | Optical flow: fine-tune the update operator of pretrained RAFT (torchvision weights) for 5–10k steps; plain, + V2, + JR × 2 seeds; invariance to the flow init (zero, random, warm start), residual AUSE (as in T5), EPE at 2–4× the iterations | Most CVPR-native recurrent refinement operator | New | 1 | 2–6 (est.) [n16] | from scratch (RAFT or DEQ-Flow-B): 200–350 (est.), not planned | 6 | 12–36 | 6–12 | RAFT weights (≈20 MB); Sintel (≈6 GB), KITTI-15 (≈2 GB), FlyingChairs (≈30 GB): sizes unverified | 10–14 at B = 5 with AMP (unverified) | 4–8 |
| T7 | ARC-AGI-1, from the ARC Prize TRM checkpoint (ARC-1 40%): 64 augmentations × 8 latent starts vs the 1,000-augmentation vote; fine-tunes (continue-TRM, Euler, Euler + V2, TRM + RI) 10–20k steps | Visual-abstraction benchmark reviewers know | Checkpoint public | 2 | 2–6 per fine-tune (est.) | out of reach (upstream: ≈3 days on 4 × H100 per run) | 4 + 2k-step pilot | 8.2–25.2 (incl. pilot 0.2–1.2 [n13]) | 24–46 [n13] | Rebuild the ARC-1 aug-1000 set with the checkpoint's exact puzzle-ID mapping (a mismatch gives 0%): CPU 1–2 h, several GB (unverified); checkpoint download size not checked | **16–20 (est.) at B = 64**: use 32 × 2 | 3–5 |
| T8 | Mini-ARC from scratch; A0, B★, A3 × 3 seeds; one fixed protocol, TRM rerun as anchor (published TRM numbers disagree: 48.35 vs 36.9) | Cheap ARC-family task | New | 2 | 1–3 (est.) | not defined | 9 | 9–27 | 0.9–2.7 | Small; task count (≈149) and recipe unverified | ≤10 (est.) | 2–4 |

---

## 7. Protocol and statistics

### 7.1 Budgets and checkpoints

- **R / 2R / L.** Constant LR 1e-4 after a 2k-step warmup, EMA 0.999, the seeded 1k × 1,000-augmentation training set
  (`config/cfg_sudoku_bench.yaml`). Data order is Philox(seed + iteration) per 1,280-epoch block and the LR is
  constant, so R and 2R checkpoints are exact prefixes of an L run, and budget curves come for free.
- **Checkpoints** every 5k steps; no checkpoint is ever selected on test. The in-training evaluation on test10k is
  descriptive only.
- **L runs** continue from `last_state.pt` where it exists (the A0_s0 rerun and every new run, e.g. B★ and X1) and
  start from scratch only for the published A2, A3 and A0_s1–s2 runs, which kept no optimizer state; the 35k
  checkpoint of a from-scratch L run doubles as a reproducibility check of the published seed. X18a/X18b are costed
  from scratch throughout, which overstates them by one R-to-45k prefix (≈2–4 G-h) per continued run (≈7–12 G-h in
  X18a, ≈9–15 G-h in X18b); the conservative cost is kept.

### 7.2 Seeds, pairing and hardware

- **Seed counts.** Confirmatory-family arms get 6 seeds (0–5): A0 (35k and 45k checkpoints), A2, A3, B★, B3, X1.
  Other must arms get 3 seeds; sweep points 1 seed, then 3 seeds at the validation winner (Tier 1).
- **What a seed fixes.** Data order and the ACT-exploration RNG; the regulariser and the RI draws use their own
  generators.
- **Hardware.** All training runs on the 5080s, because CUDA initialisation differs by GPU model: rerun A0_s0 (trained
  on the laptop). The 5080 rerun is A0's seed 0 in the confirmatory family and is paired with A2_s0 and A3_s0; the
  published laptop-trained A0_s0 stays descriptive (EV-F6 and D5 only, [n7]). Rerun A1_s0 and A5_s0 only if they
  enter paired tests. Keep GPU-side initialisation: moving to CPU-side initialisation would break pairing with every
  existing run.

### 7.3 Splits

- **Validation.** 5k puzzles from unused rows of the Sudoku-Extreme train CSV, disjoint from the 1k training puzzles
  and from test (the dataset builder already reads that CSV). Never drawn from test.
- **Tuned on validation:** τ, the selector, the RI σ, (N, K), temperatures, recalibration maps, penalty weights, the
  X15/X16 variants and the B★ choice.
- **Test subsets** (nested, seeded): full 422,786 (C1 on headline checkpoints only); 100k (every checkpoint;
  SE ≈0.15 pp); 25k (exact@64); pop2k ⊃ pop1k, new and disjoint from the published test1k: every population
  protocol uses pop1k (SE ≈1.4 pp per seed at 1k; seed-to-seed sd is 4–7 pp, so seeds dominate), and the other 1k of
  pop2k is sealed as a reserve for a confirmatory rerun; 500 puzzles for attacks.
- **External checkpoints.** The validation rows are unused only with respect to our seeded 1k training subset. The
  external checkpoints of E8 were trained on subsets we cannot reconstruct and may have seen some validation rows, so
  no selector for E8 is tuned on validation (E8 row).
- **Stratification.** Every table is broken down by the CSV's `rating` bin and `source`.

### 7.4 Matched compute

- **Training.** C_train = steps × B × (F_main + F_reg), counted with `torch.utils.flop_counter` (C0). F_main = 42.7
  GFLOP per puzzle per step, identical for TRM and Euler-TRM [n1]; F_reg is measured per arm and cross-checked
  against throughput (V2 0.78×, V1 0.68–0.72× of TRM). Wall-clock is reported but never used for matching.
- **Matched pairs** (read off saved checkpoints): A3@35k vs A0@45k; A2@45k warm-restarted (X2); A3@150k vs A0@195k at L.
- **Test time.** B_tt = total ACT steps per puzzle over all starts; one ACT step is 25.6 GFLOP forward for every arm;
  voting, q and residuals are negligible. Full-test inference at K = 16 costs ≈1.7e17 FLOP (≈45% of one R training
  run); a 16 × 64 population on the full test would cost ≈29 R runs, hence the subsets.
- **Tuning.** Every method gets the same budget: at most 4 configurations × 1 seed on validation.

### 7.5 Confirmatory family and statistics

- **Pre-registration.** A dated note committed to this repository on 29 Oct (end of Phase A, §8.4), before any
  evaluation on pop1k or pop2k and before the first test-set evaluation of any new checkpoint (including the external
  checkpoints of E8), naming the endpoints, comparators, selection rules, subsets, seeds, the testing order of the
  family and the p-value that enters it (below). Before the commit only D5 runs, on the already published test1k,
  besides the descriptive in-training test10k evaluation of the Phase A runs (§7.1), which enters no confirmatory
  contrast. The note records that seeds 0–2 of A0/A2/A3 and test1k were already seen and shaped the hypotheses; the
  confirmatory evidence comes mostly from the new arms, seeds 3–5 and the fresh pop subsets.
- **Family (fixed-sequence gatekeeping, FWER α = 0.05).** Tested in the pre-registered order **H5 → H1 → H3 → H2**,
  each at α = 0.05; testing stops at the first hypothesis not rejected, and every later one is reported as not
  confirmed. A fixed sequence controls the family-wise error rate at α without adjustment. Order: H5 isolates V2 on one
  vehicle under one training init distribution and is DP1(a)'s go criterion (§1.1, §9); H1 is the comparison with
  the best RI baseline; H3 precedes H2 because the existing seeds point against H2 (§1.3; per-seed differences below).
  Holm over the four hypotheses was dropped: its first step needs p ≤ 0.0125, which the exact sign-flip test cannot
  reach at n = 6 and at which the t-test has 0.48 power for the H1-type effect (§7.6).
  - **H1** M2 UC rate under the held-out init families (C5: the scaled Gaussian with std ≥ 2× the largest training σ,
    and the other-puzzle state), A3 vs B★ (the RI baseline with the best validation H1 metric among B1, B2, B3).
  - **H2** Accuracy at B_tt = 1,024 with each arm's validation-chosen selector, A3 vs the comparator with the best
    validation accuracy among A0@45k selectors, E1, E5, E6 and the B★ population (all computed on val500 inside EV-V:
    the 64 × 64 dump, the single unroll to 1,024, E5-core, and E6 from the three seeds with EV-V). If E6 (seed-ensemble vote) is the
    comparator, the per-seed pairing uses a leave-one-seed-out ensemble: A3 seed i against the vote of the A0@45k
    seeds other than i at the same B_tt; a puzzle-level paired bootstrap is reported alongside.
  - **H3** AURC of the validation-chosen confidence signal after recalibration, A3 vs A0@45k.
  - **H5** V2 effect under RI training, X1 − B3, on the H1 metric; the 2 × 2 interaction (X1 − B3) − (A3 − A2) is
    exploratory.
- **H4, V2's canonical cost** (A3 − A2, exact@16) is an estimate with its CI, not a test.
- **Per contrast:** per-seed paired differences with mean and 95% t-CI. The p-value that enters the sequence is the
  two-sided paired t-test p on the per-seed differences (df = 5 at n = 6); a seed-level bootstrap p is not used,
  because resampling six clusters is anti-conservative. Sensitivity checks, reported next to every family result: the
  exact sign-flip p (n = 6 gives a minimum two-sided p of 2/64 = 0.031, so it rejects at 0.05 only when all six seeds
  favour the same arm; n = 5 cannot reach 0.05) and the hierarchical bootstrap (seeds, then puzzles, same puzzle
  indices across arms; 10k replicates, BCa). Of the t-CI and the bootstrap interval, report the wider. A puzzle-level
  GLMM logit(correct) ~ arm × budget + (1 | seed) + (1 | puzzle) goes in the supplement.
- **Everything else** is exploratory under BH-FDR at q = 0.05 and labelled as such; full-budget contrasts (A3 − A2 and
  A2 − A0 at L) join this set. Single-seed rows use puzzle-level McNemar tests and never claim significance.
- **Equivalence** only through TOST at ±2 pp. For A3 min-residual vs the A0@45k vote (per-seed differences A3 − A0 of
  +7.0, −5.0, −6.1; mean −1.4, sd 7.3) that needs ≈113 seeds (≈50 at ±3 pp): report the CI and make no equivalence
  claim.

### 7.6 Power (from the measured per-seed sd, two-sided paired t-test, α = 0.05, 80% power)

| Contrast | Measured δ (sd) | Seeds needed | Status at n = 6 |
|---|---|---|---|
| A2 − A0@T, exact@16 | 2.98 (2.30) | ≈7 | borderline, exploratory |
| A3 − A2, exact@16 (H4) | −4.85 (4.65) | ≈10 | estimate only |
| A2 − A0@45k, exact@16 | −2.23 (2.33) | ≈11 | estimate only |
| Invariance gap ≥ 10 pp with sd ≈7 (H5, H1) | – | ≈7 | H5, first in the sequence: 0.797 (0.69 if sd ≈ 8). H1, tested only after H5 is rejected: 0.59–0.80 for an equal effect (0.38–0.69 if sd ≈ 8; Fréchet bounds, the correlation of the two tests is unknown); confirmatory |
| H3 (AURC after recalibration) and H2 (accuracy at B_tt = 1,024) | H2: −1.4 (7.3) on seeds 0–2 (§7.5) | H3: not estimable; H2: none (observed effect < 0) | H3 has no measured effect size; H2 has essentially no power at the observed effect and is placed last |

- **Why not Holm.** At Holm's first step (α = 0.0125) the t-test power for the H1-type effect (d = 10/7) is 0.48 at
  n = 6, 0.62 at n = 7, 0.74 at n = 8, 0.82 at n = 9 and 0.89 at n = 10, so Holm would need ≥ 9 seeds per family arm
  (18 more R runs plus their evaluation), and the exact sign-flip test cannot reject at all at n = 6.
- **Sign-flip sensitivity check.** At α = 0.05 it needs all six seeds to favour the arm: power 0.62 for d = 10/7
  (0.51 if sd ≈ 8). A family result that passes the t-test but not the sign-flip check is reported as such.

### 7.7 Precision

Residuals are computed on bf16 states. In the published diagnostics (`results/sudoku-contractive/runs/*/diag.json`)
the step-64 median relative residual of Ψ is 0.006–0.013 for A0, A2 and A3 (A0_s2 0.012, A2_s0 and A2_s2 0.013) and ≈0.004 for A1 only. In
`pop_eval.json` (`rand`, canonical member), the rule "stop at residual < 0.005" stops at step 62–64 of 64 on average
for most A0/A2/A3 seeds, i.e. those trajectories essentially never reach it. Any threshold below 0.02, and every
diagnostic, therefore runs in fp32 (D5 checks the published selectors).

### 7.8 Curves and figures

Accuracy vs training FLOPs and vs samples per arm (5k checkpoints, R/2R/L); accuracy vs B_tt frontier with the
selector chosen per budget on validation, plus per-selector curves; risk–coverage curves and reliability diagrams;
μ dose-response (X8) as an accuracy–M2 frontier; D1 λ_max profile vs ACT depth; D2 q(r) curves; basin-radius curve
(unanimity vs σ, C5); accuracy vs mean steps for residual halting (E2); per-`rating` breakdowns.

---

## 8. Compute summary

### 8.1 Totals per table and tier (G-h, lo–hi)

| Table | Tier 0 train | Tier 0 eval | Tier 1 train | Tier 1 eval | Tier 2 train | Tier 2 eval |
|---|---|---|---|---|---|---|
| A.1 trained baselines | 46.1–79.7 | 47.3–90.1 | 7.3–13.3 | 3.2–6.0 | 6.1–14.2 | 3.4–6.6 |
| A.2 test-time baselines (E8, E9; the rest is inside packages) | 0 | 7.0–13.0 | 0 | 0 | 0 | 0 |
| A.3 published numbers | 0 | 0 | 0 | 0 | 0 | 0 |
| B internal arms | 73.4–122.0 | 109.7–207.1 | 143.0–241.3 | 58.9–112.7 | 176.7–292.8 | 78.2–148.5 |
| C protocols and diagnostics (C0; Tier 1 extras on X1/B★ and D6; D7) | 0 | 0.1 | 0 | 21.4–39.8 | 0 | 0.9–1.6 |
| D tasks beyond T1/T2 (T3; T3+, T5, T6; T4, T7, T8) | 21.1–42.9 | 11.8–21.8 | 60.8–187.2 | 19.3–36.8 | 164.6–347.0 | 36.9–70.7 |
| **Total** | **140.6–244.6** | **175.9–332.1** | **211.1–441.8** | **102.8–195.3** | **347.4–654.0** | **119.4–227.4** |
| **Tier total (mid)** | **317–577 (447)** | | **314–637 (476)** | | **467–881 (674)** | |
| Engineering (person-days) | 14–20 (rows sum to 15.6–21.6; C2/E1, C9/T2 and T1/CV are the same work) | | 15–22 | | 13–22 | |

All entries are sums of the displayed row values.

### 8.2 Tier contents

- **Tier 0, minimum viable submission.** Training: A0_s0 rerun and A0/A2/A3 seeds 3–5; B1 (σ pilot + 3 seeds), B2 × 3,
  B★ seeds 3–5, B3 × 6, X1 × 6; B4a, B4b; B5 (2 validation configurations + 2 seeds); X2; X3, X4, X5, X8 (4 points),
  X10 × 3, X11 (2 points); T3 (9 runs). Evaluation: EV-F/EV-F6 on the 49 family and must-baseline checkpoints (EV-F
  on 27 including E8's 3, EV-F6 on 22), EV-G/X/TT on A0@45k and A3 s0–s2, EV-R on A0@45k, A3, X1, B3, B★ s0–s2, EV-V
  on s0–s2 of 7 arms, EV-S on 14 sweep checkpoints, D5, E8, E9, full test on 4 new headline checkpoints, T2 inside
  EV-R.
- **Tier 1, strong (camera-ready and ICCV route).** X3–X5 and X8 (μ = 0.3) to 3 seeds, X6, X7, X9, X11 winner × 2,
  X12, X14, X15, X17, B6, B7 (2 configurations), X18a, X19, T3+, T5 (with B9 and JR on T5), T6, D6, Tier-1 evaluation
  extras on X1 and B★.
- **Tier 2, full.** X18b, X18c, X13, X16, X20, B8, B10, T4 (gated), T7, T8, D7.

### 8.3 Capacity and wall-clock

- **Assumed availability:** each 5080 free 60% of the time (sensitivity below); laptop 90% of the time.
- **Capacity:** the 5080 pair delivers ≈2.0 midpoint-G-h per wall-hour when both are free (one fast and one slow card
  give ≈2.1); the laptop 0.35. At 60% / 90%: 28.8 + 7.6 = 36.4 G-h per day. All Tier 0 training (193 G-h midpoint)
  must run on the 5080s to keep seeds paired; evaluation can run anywhere.
- **Engineering staffing.** Tier 0 needs ≈14–20 person-days of engineering, almost all in Phases A–E (11 calendar
  days, 28 Oct–7 Nov). The schedule assumes two people working on it; with one person Tier 0 is engineering-infeasible
  without the cuts of §8.6, even though its compute fits. Tier 1 and 2 estimates assume one person.
- **CVPR 2027 deadlines** (official call for papers, AoE): paper enrollment 10 Nov 2026, submission 16 Nov 2026,
  supplementary material 23 Nov 2026. Freezing test evaluation on 11 Nov leaves 15 compute days (28 Oct–11 Nov),
  ≈546 G-h at 60%. ICCV 2027 has no published call; trackers estimate early March 2027 (unverified).

| 5080 availability | Capacity (G-h/day) | Tier 0 (days) | Fits 28 Oct–11 Nov? | Tiers 0 + 1 (days) | All tiers (days) |
|---|---|---|---|---|---|
| 80% | 46.0 | 9.7 | yes, ≈5 days buffer | 20.0 | 34.7 |
| 60% | 36.4 | 12.3 | yes, ≈2.7 days buffer (days 1–6 are engineering-bound) | 25.3 | 43.9 |
| 40% | 26.8 | 16.7 | no: apply cuts 1–6 of §8.6 (Tier 0-min ≈378 G-h = 14.1 days) | 34.4 | 59.6 |
| laptop only | 7.6 | 59 | no; Tier 0 needs the 5080s at ≥47% on average | – | – |

### 8.4 Schedule for the CVPR window (midpoint G-h)

Each phase is checked per resource: the 5080 pair delivers 28.8 G-h per day and the laptop 7.6 G-h per day (60% / 90%
availability, §8.3). Training stays on the 5080s; the laptop takes evaluation that does not block a decision on the
same day.

| Phase | Dates | 5080 pair | Laptop | Engineering | Load / capacity (G-h): 5080; laptop |
|---|---|---|---|---|---|
| A | 28–29 Oct (days 1–2) | No-code training: A0_s0 rerun, A0/A2/A3 s3–s5, X3, X4, X5, X8 × 4 (55.4) | D5 on the published test1k, before the commit (4.1); after the commit: E9 and EV-Full on the A0_s0 rerun (2.1) | RI and NI flags; validation pool and new subsets; C3/C5 dump with selectors; C0 FLOP/NFE logging; **pre-registration note committed 29 Oct**, before any pop1k/pop2k or new-checkpoint test evaluation | 55.4 / 57.6; 6.2 / 15.1 |
| B | 30–31 Oct (days 3–4) | B1 σ pilots (3–4 h, both cards) and the two EV-Vs (one per pilot) that pick σ → σ chosen on validation → B1 s1–s2, B2, B3 and X1 s0–s2, X10 × 3 (47.4); EV-F on A0@45k s0–s2, EV-V on A0@45k s0 (8.7) | EV-F on A3 s0–s2; EV-V on A0@45k s1–s2, A2 and A3 s0–s2 (14.5) | statistics library; D1–D3; E5; B4a/B4b; B5; fp32 regulariser path; T3 start | 56.1 / 57.6; 14.5 / 15.1 |
| C | 1–2 Nov (days 5–6) | B4a, B4b, B5 pilots, X11 large and small, X2 (25.6); EV-F on B1–B3 s0–s2, EV-Vs on B2, EV-V on X1 s0–s2, C6 on seed 0 of B1, B2, B3, X1 and A3 (29.4) | EV-V on B1–B3 s0–s2 (fixes B★ by 2 Nov); EV-F on A2 s0–s1 (12.7) | T3 renderer and encoder; C6, C7, C9; E3, E4; E8 loaders | 55.0 / 57.6; 12.7 / 15.1 |
| D | 3–4 Nov (days 7–8); **DP0 and DP1 on 4 Nov** | B★, B3, X1 seeds 3–5 (26.5); T3 A2 and A3 × 3 (22.2); EV-F on X1 s0–s2 (7.9) | E8 on 3 external checkpoints (9.2); EV-F on A2 s2; EV-Vs on the B5 pilots (3.5) | T3 debugging; figure scripts | 56.6 / 57.6; 12.6 / 15.1 |
| E | 5–7 Nov (days 9–11); **DP2 on 7 Nov** | T3 B★ arm × 3, B5 seeds (16.4); EV-S/EV-Vs on 14 sweep checkpoints (26.9); EV-G and EV-X on A0@45k and A3 (20.5); EV-T3 on the T3 A2/A3 runs (11.2); EV-F6 on A0 s3–s5 and the published A0_s0 (9.0) | EV-TT and EV-R on A0@45k and A3 (C6 on A3 s0 already done in Phase C) (14.7) | statistics on real outputs | 84.0 / 86.4; 14.7 / 22.7 |
| F | 8–11 Nov (days 12–15); **DP3 on 8–9 Nov**; enrol 10 Nov; freeze 11 Nov | EV-T3 on the T3 B★ arm (first, for DP3); EV-F6 on the other 18 seed-3–5 and warm-restart checkpoints; EV-R on B★, B3, X1 s0–s2 (C6 on s0 already done); EV-F on B5; EV-Full on B★, B3, X1 s0 (65.9) | EV-S0 on A0@T × 7, A1 × 3, A5 and X2's A0 restarts × 3; C0 (13.1) | writing | 65.9 / 115.2; 13.1 / 30.2 (≈66 G-h buffer) |
| Writing | 12–16 Nov | – | – | submission 16 Nov | – |
| Supplement | 17–23 Nov | extra seeds or diagnostics only; no new claims | | | – |

The phases sum to the Tier 0 midpoint (447 G-h) and leave ≈99 G-h (≈2.7 days) of slack, ≈66 G-h of it in Phase F.

### 8.5 Critical path

RI/NI flags (day 1) and the pre-registration commit (day 2) → B1 σ pilots and the validation choice of σ (day 3) →
B2, B3, X1 seeds 0–2 (days 3–4) → EV-V on them (days 5–6; fixes B★) and EV-F (days 5–7) → seeds 3–5 of B★, B3, X1
(days 7–8) → **DP0/DP1** (day 8, 4 Nov) → T3 B★ arm (days 9–11) → EV-F6 and EV-T3 (days 11–13) → statistics and
figures (days 12–14) → freeze (day 15, 11 Nov). Compute slack is ≈2.7 days at 60% availability; with two people,
engineering slack is close to zero in days 1–6. A second chain runs in parallel: T3 code (days 3–6) → T3 A2/A3 runs
(days 7–8) → B★-arm runs (days 9–11) → **DP3** (8–9 Nov).

### 8.6 What to cut first

Tier 0, in order, if availability falls below ≈55% or engineering slips:

| Order | Cut | Saves (G-h, lo–hi) | Consequence |
|---|---|---|---|
| 1 | EV-G, EV-X, EV-TT on A0@45k and A3 → seed 0 only | 13.8–25.6 | Frontier, basin-radius and PTRM/GAR curves become single-seed figures |
| 2 | X2 (warm restarts) | 10.2–18.2 | Matched-compute evidence rests on A3@35k vs A0@45k only |
| 3 | X11 (pair radius) | 8.7–15.9 | Moves to Tier 1; DP2 loses the radius test |
| 4 | B5 → winner seed only | 8.1–15.5 | JR reported at n = 1 |
| 5 | X3 and X5 (keep X4) | 6.9–12.0 | "Euler" and damping questions move to Tier 1 |
| 6 | E9 | 0.6–1.1 | Training-free damping not tested |
| | **Total** | **48.3–88.3** | Tier 0-min ≈268–488 G-h (mid 378) |

Tier 1 and 2, in order: (1) T7, B8, X13, X16, X20, T8, B10, D7; (2) T4 unless its gate passes cheaply; (3) X18b
reduced to A0, A3 and B★ (saves 89–153 G-h); (4) X6, X7, X9 to the supplement (40–70); (5) X12, X14, X15 (23–42);
(6) T6 before T5 (18–48).

**Never cut:** B1–B3 and X1 (with B★); six seeds for the family arms; E1, E5, E6, E8; the C5 held-out inits and M2;
the validation split; D1 on the family arms; X8 and X10; B4a/B4b; T3, the only pixel task in Tier 0.

### 8.7 Storage and evaluation budgets

| Tier | New training runs | Checkpoint storage | Population dumps | Datasets | Eval G-h (share of tier) |
|---|---|---|---|---|---|
| 0 | 59 R-scale runs (23 regularised) + 6 warm restarts | ≈20 GB | ≈7 GB (≈70 checkpoints) | MNIST (≈11 MB); validation pool from the existing CSV | 176–332 (57%) |
| 1 | 42 R-scale (35 regularised) + 5 L runs + 30 restoration + 6 flow | ≈23 GB | ≈4 GB | restoration sets (< 0.5 GB), optical-flow sets (≈40 GB, unverified) | 103–195 (31%) |
| 2 | 11 L + 9 extensions + 9 Maze + 4 ARC (plus a 2k-step pilot each for Maze and ARC) + 9 Mini-ARC + 13 others | ≈18–23 GB | ≈3 GB | Maze (few MB), ARC aug-1000 (several GB, unverified) | 119–227 (26%) |

### 8.8 Dollar cost

[COST_ESTIMATE.md](COST_ESTIMATE.md) prices every row of this plan in US dollars at a 27 Oct 2026 snapshot of cloud
prices, including setup, storage, egress, a failure allowance and VAT where it applies; the scripts that print its
numbers are in [cost/](cost/). Its recommended package differs from the tiers above mainly in one respect: the
confirmatory core factorial (A0, A2, A3, B★, B3, X1 × 6 seeds) runs at the paper budget L, which subsumes X18a/b/c.
It also uses exact@16 instead of EV-S0 at 70k on those runs (a deviation from note [n8]), runs T7, T8, B8 and B10
only if DP1–DP3 leave the paper alive, and adds an optional +P (seeds 6–8 at L for X1, B3, A3, B★). All Sudoku
training stays on RTX 5080s (the owned pair plus rented 1-GPU 5080s on Vast.ai verified hosts, about $0.345/h with
disk), which keeps the pairing rule of §7.2.

| All tiers, core factorial at L | Central | Pessimistic |
|---|---|---|
| Recommended package (the owned 5080s and the laptop take part of the work) | $428 | $1,195 |
| Same, everything rented | $561 | $1,312 |
| Option +P (seeds 6–8 at L for X1, B3, A3, B★), rented | +$71 | +$162 |

| Rented 4-GPU node | Break-even price against rented 5080s | Market price, 27 Oct 2026 |
|---|---|---|
| 4× RTX PRO 6000 | ≤ $2.33 per node-hour ($2.89 if its speed-up over a fast 5080 reaches 2.6) | $5.13–9.96 per node-hour |
| 4× RTX 6000 Ada | ≤ $1.33 per node-hour | $2.40–3.96 per node-hour |

Per unit of work, a rented 4× RTX PRO 6000 node costs 2.2× (Vast marketplace) to 4.3× (managed on-demand, with VAT)
as much as rented 5080s. Even the pessimistic all-rented case with +P ($1,474) stays under a $2,000 budget with a 20%
reserve, so the binding constraints are engineering person-days and the 11 Nov freeze (§8.3), not compute dollars.
The cost estimate's own spending order, stop/go checkpoints and cut order are in its §6.

---

## 9. Decision points and kill criteria

| DP | When | Compute spent before the decision (G-h, mid) | Test | Result → consequence |
|---|---|---|---|---|
| DP0 | ≈4 Nov, with DP1 (E8 and the A2 EV-F finish in Phase D) | ≈37 (D5, EV-F on A0@45k/A2/A3, E8) | fp32 re-evaluation (D5) | If A3's min-residual advantage shrinks in fp32: restrict the residual-selection claim and drop every bf16 rule with τ < 0.02 |
| | | | E5 and E6 on A0@45k at B_tt = 1,024 vs A3's best selector | If the baselines are at least as good: drop "accuracy at matched compute"; the claim reduces to invariance plus selective prediction |
| | | | M2 on the existing arms | If A0@45k's P(unanimous given correct) ≈ A3's: the raw unanimity gap was confounded with accuracy; rewrite the headline |
| | | | E8 under held-out inits | If a full-budget TRM, or EqR itself, already reaches A3-level M2: the effect is undertraining or EqR already has it; X18a becomes mandatory and the ICCV route is taken |
| DP1 | ≈4 Nov (end of Phase D; B★ itself is fixed on validation by 2 Nov) | ≈83 (B1–B3, X1 s0–s2 and their evaluation; C6 on A3 s0) | B1–B3 vs A3 and X1 vs B3 under C5 held-out inits (seeds 0–2) and C6 (seed 0 only before DP1, A3 s0 included; the other seeds follow in Phases E–F) | (a) X1 > B3 on M2, Δ_PI and attacked AA with ≤ 2 pp canonical cost: **go**; "V2 adds invariance beyond random-init training" is the headline. (b) A3 wins only on held-out or adversarial inits, not on N(0, 1): narrow the claim to zero-shot invariance beyond the training init distribution. (c) RI matches or beats A3 everywhere and X1 ≈ B3: V2 is a costly substitute for RI; no CVPR submission; analysis paper on what makes the landscape single-basin |
| DP2 | ≈7 Nov | ≈81 (X3–X5, X8, X10, X11, B4a/B4b, B5 pilots and EV-S) | X8, X10 vs A3, X11, D1–D3, X3–X5; B4b and B5 on validation M2 (EV-V/EV-Vs; B5's test-set EV-F comes after DP2) | **Strength.** No monotone relation of M2 to μ and random pairs ≈ adversarial pairs: drop the mechanism and adversarial-pair claims; retitle "monotonicity-regularised". **Held-out monotonicity.** A3 no more monotone than A0 at any radius on held-out states: claim at most "coarse-scale secant monotonicity on visited states" if D2 supports it. **Time embedding.** X4 loses invariance: non-autonomous story, never "unique equilibrium"; X4 gains: lead with the autonomous field. **Euler.** X5 ≈ A3: drop "Euler" from the title. **Penalty type.** B4b or B5 matches A3 on validation M2 at equal or better validation accuracy: the one-sided worst-direction condition is not the active ingredient |
| DP3 | ≈8–9 Nov | ≈49 (T3 runs and EV-T3) | T3 | If V2's M2 or AURC advantage disappears with pixel input, or residual-based confidence is no better than voting over re-rendered inputs under perception noise: the CVPR vision argument fails; move to ICCV with T5 and T6 |
| DP4 | Tier 1–2 | ≈100 (X18a); ≈23 (T4 pilot and first A0 run) | X18a; T4 gate | **Full budget.** A0 at L reaches A3's M2 within CI: undertraining artefact, kill the method claim. V2 costs > 5 pp at L with no population gain: kill. **Maze gate.** Continue T4 only if the first reduced-budget A0 run (a single run) reaches ≥ 43% (half of TRM-Att's 85.3) with the lower bound of its puzzle-level 95% CI (1,000 test mazes, ±3 pp) at ≥ 40%; EqR's maze numbers are Maze-Unique and cannot serve as the gate |

**CVPR rule.** Submit to CVPR 2027 only if DP1 lands on (a) or (b) and DP3 passes. Otherwise take the ICCV route with
Tier 1.

---

## 10. Estimation notes

- **[n1]** F_main and the ACT-step forward cost were measured with `torch.utils.flop_counter` on CPU in fp32 with the
  benchmark config (TRM 5.03M parameters, Euler-TRM 5.06M); the counting script will be added to the repository with
  C0. 42.7 / 25.6 = 35 / 21 L-level evaluations of 1.22 GFLOP.
- **[n2]** Overhead model: per-row costs in units u of one L-level evaluation; a field evaluation is 2 u (z_H and z_L
  blocks), a cycle map Φ is 7 u, a backward pass with respect to inputs ≈2× its forward, with respect to parameters
  ≈2×. The factor applies to the whole R cost (in-training evaluation is ≈7% of it, so the error is small). Validated
  on V2 (1.27 predicted vs 1.28 measured) and V1 (1.50 vs 1.47).
- **[n3]** Training ranges: steps/s fast / slow from the median `steps_per_s` in `metrics.jsonl`; in-training
  evaluation 79 / 130 s per 5k steps. An R run is 7 blocks of 5k steps, A0 9, L 39 (195k), 2R 14.
- **[n4]** D1 and D2 at 0.25–0.5 G-h each are scaled from the measured diagnostics (0.13–0.24 G-h for 1k puzzles,
  256 Jacobian puzzles, 20 power iterations): D1 has 8 state sets × 30 fp32 iterations on 256 puzzles (≈2× the
  measured Jacobian work), D2 5 radii × 2 adversaries × 2 state sets on 128 puzzles.
- **[n5]** C6: 500 puzzles × 20 PGD steps × 16 ACT steps × (forward + recomputation + backward ≈ 4 ACT-step
  equivalents) ≈ 0.64M ACT-step equivalents, × 2–3 for fp32 ≈ 1.3–1.9M bf16 equivalents, plus evaluation of the
  attacked starts. Without activation checkpointing per ACT step the graph is ≈336 L-level evaluations per puzzle
  (≈3–4 GB per puzzle in fp32), which does not fit.
- **[n6]** Memory: the main graph at B = 256 is ≈10 GB; regulariser graphs on 64 anchor rows hold 2–7 L-level
  evaluations with double backward, i.e. well under the main graph; activation memory scales ≈linearly with rows ×
  tokens (Maze/ARC: 916 tokens per row vs 97).
- **[n7]** A0 evaluation: the confirmatory seeds 0–2 are the 5080 rerun of s0 and the published s1–s2; they get EV-F,
  EV-G, EV-V, EV-X, EV-TT and EV-R, and the rerun also gets EV-Full. The published laptop-trained s0 is descriptive
  and gets EV-F6 and D5 (which checks the published selectors). s3–s5 get EV-F6; the 35k checkpoints (A0@T) of all 7
  A0 runs (published s0, its rerun, s1–s5) get EV-S0.
- **[n8]** Per L run: EV-F, EV-Full, EV-G at 195k; EV-S0 at 70k; exact@16 on 100k at 35k (reproducibility check);
  V2 arms also EV-S at 150k for the compute match (EV-S rather than EV-S0, so that the compute-matched checkpoint
  also gets D1/D2).
- **[n9]** Evaluation unit: 2,000 / 1,080 puzzle-ACT-steps/s → 0.139 / 0.257 G-h per 1M. The lo end is about the
  slowest fast-regime population eval (2.05M steps in 1,009–1,022 s, i.e. 2,004–2,029/s); the hi end is the median
  slow-regime population eval (1,848–1,942 s, i.e. 1,055–1,108/s, median ≈1,080/s), not the worst case. Other
  slow-regime evaluations run at 1,015–1,199/s (the slowest is exact@64 on A7_s0, 1,577 s; exact@16 on 100k reaches
  1,055/s on A2_s1); at 1,015/s one million steps costs 0.274 G-h, ≈6% above the hi value. The laptop runs 640–680/s
  → 0.42 h per 1M. Rollout bookkeeping (residuals, rules, storage) is inside the measured population-eval time.
- **[n10]** EV-V and EV-Vs are computed on seeds 0–2 only: selectors and recalibration maps are chosen per arm, not
  per seed.
- **[n11]** Tier 0 reuses every published checkpoint and evaluation; the published test1k numbers are not
  recomputed, the new pop subsets are fresh.
- **[n12]** The B★ row's cost (seeds 3–5, their EV-F6, EV-R on s0–s2 and one EV-Full) is the worst case (B★ ≠ B3);
  if validation picks B3, B★ = B3, whose row already carries 6 seeds, EV-R and EV-Full, so the B★ row is saved. Rows
  that use B★ downstream (C4/E3 Tier 1 extras, T3, T3+, T4, T8, X18a/b, D7) then use B3; the costs change by < 2%.
- **[n13]** Maze/ARC: TRM-Att costs 307 GFLOP per puzzle per training step (7.2× Sudoku; our FLOP count, script to
  be added with C0). 10M examples = 3.07 EFLOP; at the measured 35–56 TFLOP/s that is 15–24 h, widened to 15–30 for
  the small micro-batches needed at 16 GB. ARC evaluation: ≈420 test inputs × augmentations × starts × 16 ACT steps × 7.2. Both are
  unmeasured; T4 and T7 start with a 2k-step pilot (throughput and memory at 16 GB) that replaces these numbers. The
  pilots are separate runs and are costed in the rows: T4's is 2k steps × 128 × 307 GFLOP ≈ 79 PFLOP, 0.4–0.6 h at
  35–56 TFLOP/s and 0.4–0.8 G-h with the same small-micro-batch widening; T7's is 2k steps of a 10–20k-step
  fine-tune at 2–6 G-h, i.e. 0.2–1.2 G-h.
- **[n14]** B10: the source reports 48 min on one RTX 5090 for its baseline (8k steps × B = 384); a 5080 is assumed
  at 0.5–0.8 of a 5090 (unverified).
- **[n15]** T5: 20–30k steps of 64 patches (50–64 px), 16 operator applications with a TRM-style 1-step gradient
  (≈18 forward equivalents), DnCNN-size F (≈1.1 MFLOP per pixel), at an assumed 20–40 TFLOP/s for convolutions;
  regularised arms × 1.2–1.5. Unmeasured. This arithmetic gives 0.44–2.2 h per unregularised run; the
  row's 0.75–2.5 G-h adds a margin for the slow regime (up to 1.65×), periodic validation (PSNR on BSD68/Set12 or
  BSD10) and data loading, until a pilot replaces it.
- **[n16]** T6: RAFT reports no training time; 5–10k fine-tuning steps at B = 5 (the 1-GPU mixed-precision setting)
  assumed at 0.3–0.6 s per step, V2 through the correlation lookup × 1.3–2. Unmeasured; pilot first. This arithmetic
  gives 0.4–1.7 h per plain run and up to 3.3 h for V2 on a fast card (≈3–14 G-h for the 6 runs); the row keeps 2–6 G-h
  per run (12–36 in total) as a deliberate margin for the slow regime (up to 1.65×), periodic Sintel/KITTI validation,
  data loading from the ≈40 GB of flow datasets and pilot restarts.
- **[n17]** Wall-clock = midpoint G-h / capacity per day; capacity = 24 × (2.0 × a + 0.35 × b) with 5080 availability
  a and laptop availability b = 0.9. The midpoint is appropriate because the measured fast and slow regimes each carry
  about half of the work.

---

## 11. References

Papers are cited by title and arXiv id; the three workshop papers without an arXiv version are cited by their
OpenReview forum and were read only as abstracts. Code and checkpoint locations are public repositories.

- Less is More: Recursive Reasoning with Tiny Networks (TRM), 2510.04871. Code: `github.com/SamsungSAILMontreal/TinyRecursiveModels`.
- Hierarchical Reasoning Model, 2506.21734.
- Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning, 2605.21488. Code: `github.com/locuslab/EqR`; checkpoints: `huggingface.co/locuslab/EqR-model`.
- Path Independent Equilibrium Models Can Better Exploit Test-Time Computation, 2211.09961.
- Dynamical Systems Theory Behind a Hierarchical Reasoning Model, 2603.22871.
- Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers, 2606.18206.
- Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere, 2610.25646.
- Reasoning on the Simplex: Geometric Fixed-Point Models, 2609.33540.
- Thinking with Looped Flows, 2609.11801.
- Scheduling Recursive Reasoning in Looped Transformers, 2609.36653.
- Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth (CoRe), 2608.18222.
- Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models (STARS), 2605.26733.
- Stabilizing Equilibrium Models by Jacobian Regularization, 2106.14342. Code: `github.com/locuslab/deq`.
- Monotone operator equilibrium networks, 2006.08591.
- Lipschitz-Bounded Equilibrium Networks (LBEN), 2010.01732.
- Robust Implicit Networks via Non-Euclidean Contractions (NEMON), 2106.03194.
- Designing Stable Neural Networks using Convex Analysis and ODEs, 2306.17332.
- Weak and Semi-Contraction for Network Systems and Diffusively-Coupled Oscillators, 2005.09774.
- Adversarial Lipschitz Regularization (ALR), 1907.05681.
- Learning Maximally Monotone Operators for Image Recovery, 2012.13247.
- Learning truly monotone operators with applications to nonlinear inverse problems, 2404.00390.
- Memory-efficient model-based deep learning with convergence and robustness guarantees (MOL), 2206.04797.
- Deep Equilibrium Architectures for Inverse Problems in Imaging, 2102.07944. Code: `github.com/dgilton/deep_equilibrium_inverse`.
- Resample the Answer: Where to Spend Test-Time Compute in Recursive Reasoning (GAR), 2610.16427. Checkpoints: `huggingface.co/MaximeILL/GAR_ckpts`.
- Probabilistic Tiny Recursive Model (PTRM), 2605.19943.
- Generative Recursive Reasoning (GRAM), 2605.19376.
- Boosting Inference with Guided Reasoning: Stochastic Exploration for Recursive Models (Guided SMC), 2605.25230.
- C-voting: Confidence-Based Test-Time Voting without Explicit Energy Functions, 2604.13521.
- Confidence Improves Self-Consistency in LLMs (CISC; defines WQD), 2502.06233.
- Speed is Confidence, 2601.19085.
- A Tilted Bowl Is Not a Slippery Slope: Compressing Looped Models, 2609.39277.
- When Recursive Models Finish Computing, 2609.26487.
- Input-Conditioned Search for Efficient Test-Time Scaling of Recursive Reasoners (ICS), 2610.18547.
- Multi-Attractor GNNs: Set-Valued Expressivity Beyond Unique Equilibria, 2609.35274.
- Fractal basins trap latent reasoning, 2609.04963.
- Unrolled Policy Iteration for Tiny Recursive Models, ICLR 2026 Workshop on AI with Recursive Self-Improvement; abstract only. `openreview.net/forum?id=rIzREKws05`.
- xRRMs: Explorative Recurrent Reasoning Models, NeurIPS 2026 Workshop AXIOM; abstract only. `openreview.net/forum?id=x3FOpcswLP`.
- Recursive Reasoning as Attractor Landscape Search: Mechanistic Dynamics of the Tiny Recursive Model, ICLR 2026 Workshop on Latent & Implicit Thinking (LIT); abstract only. `openreview.net/forum?id=kKps9W1K7n`.
- What Limits Recursive Reasoning Models: Optimization, Architecture and Test-Time Scaling, 2609.39967.
- Looped Reasoning Finishes Earlier Than You Think, 2610.18174.
- Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute, 2512.11847. Checkpoint: `huggingface.co/arcprize/trm_arc_prize_verification`.
- Reasoning with Neural Cellular Automata, 2609.36126.
- SATNet, 1905.12149; Assessing SATNet's Ability to Solve the Symbol Grounding Problem, 2312.11522.
- RAFT: Recurrent All-Pairs Field Transforms for Optical Flow, 2003.12039. Code: `github.com/princeton-vl/RAFT`.
- Deep Equilibrium Optical Flow Estimation, 2204.08442. Code: `github.com/locuslab/deq-flow`.
- Community TRM-MLP Sudoku checkpoint (third-party): `huggingface.co/gaoxin492/TinyRecursiveModels-Sudoku-Extreme-mlp`.
- CVPR 2027 call for papers: `cvpr.thecvf.com/Conferences/2027/CallForPapers`.
