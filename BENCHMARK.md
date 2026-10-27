# Contractive recursive operators for TRM — Sudoku-Extreme benchmark

Built on [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
(`main` @ c011037). Upstream model and config files are unchanged (`models/losses.py` only gains code paths for the
new model) and the additions are opt-in; `pretrain.py` and `dataset/build_sudoku_dataset.py` have new defaults
(seeded data, offline wandb, auto-resume, NaN abort, raw final checkpoint; see the README, *Credit*). Results, caveats
and the full reproduction guide are in the top-level [README.md](README.md); the run files are in
[results/sudoku-contractive/](results/sudoku-contractive/).

## Hypothesis and arms

TRM's inference can be read as iterating its ACT-step map Ψ towards a fixed point. We test soft contraction-type
penalties on its recursive latent operator: V1 on TRM's cycle map (Lipschitz caps ρ from 0.95 to 4), V2 on the
Euler-TRM vector field, and the decoder-aware D2 / Dγ, which constrain only what the answer decoder reads.
State `s = (z_H, z_L)`, joint Euclidean norm.

| arm | model | regularizer |
|---|---|---|
| A0 | TRM-MLP (Sudoku recipe) | none (monitoring only) |
| A1 | TRM-MLP | **V1** `cycle_lip`: `ReLU(log(‖Φ(s)−Φ(s̃)‖/‖s−s̃‖) − log ρ)`, ρ = 1−η |
| A2 | Euler-TRM: `z ← z + h·(F(z, ctx+τ(t)) − z)`, h = 0.5, time embedding τ | none |
| A3 | Euler-TRM | **V2** `flow_mono`: `ReLU(⟨V(s)−V(s̃), s−s̃⟩/‖s−s̃‖² + μ)`, μ = −ln(1−η)/h |
| A4 | Euler-TRM | V1, ρ = 0.95 |
| A5 / A6 / A7 | TRM-MLP | V1 with ρ = 1 / 2 / 4 (`reg_eta=0` / `++arch.reg_rho=2.0` / `4.0`) |
| A8 | TRM-MLP | **Dγ** `dec_semi`: `‖P(Φ(s)−Φ(s̃))‖ ≤ ρ·sqrt(‖PΔ‖² + γ²‖(I−P)Δ‖²)`, ρ = 0.95, γ = 0.3 |
| A9 | TRM-MLP | **D2** `dec_out`: `‖P(Φ(s)−Φ(s̃))‖ ≤ ρ‖s−s̃‖`, ρ = 1 |
| A10 | Euler-TRM | Dγ, ρ = 0.95, γ = 0.3 |

A0–A3 ran with seeds 0, 1, 2; A4–A10 with seed 0. P projects onto the decoder-visible subspace (the z_H answer
tokens on the centered row space of the output head, z_L → 0), Δ = s − s̃. Exact overrides: `arm_overrides` in
`scripts/run_benchmark.sh`.

* Φ = one full H-cycle (6 z_L updates + 1 z_H update); one ACT step Ψ = 3 cycles.
* V(s) = `(F(z_H, z_L+τ_H) − z_H, F(z_L, z_H+x+τ_L) − z_L)` = the joint vector field of Euler-TRM
  (for TRM itself this is the field of `ds/dt = F(s) − s`). h = 1 with τ = 0 recovers TRM exactly.
* Pairs are **adversarial** (ALR-style power iteration, warm-started per puzzle across ACT steps), on a fraction
  `reg_frac` of the batch rows (0.25; 0.125 for A8 / A10): V1, D2 and Dγ take one power step per training step
  (J^T J, or its projected / metric-mapped version), V2 one gain step plus three shifted sym(J_V) steps.
  Radius 5–20% of ‖s‖ (of ‖W_γ s‖ for Dγ).
* η = 0.05; the strength λ follows dual ascent toward a 10% violation rate (frozen during LR warmup).
* The regularizer runs outside the main compiled graph on detached anchor states, with its own RNG,
  so the main training path and ACT exploration are seed-paired across arms.

## Files

| file | purpose |
|---|---|
| `models/recursive_reasoning/trm_contractive.py` | TRM fork: Euler-TRM flags, operator API (`cycle`, `field`), exported anchors |
| `models/regularizers.py` | V1 / V2 / D2 / Dγ (and D1) penalties, adversarial pair search, dual ascent, training-time contraction monitor |
| `models/adam_atan2_torch.py` | pure-PyTorch port of `adam-atan2` 0.0.3 (no nvcc needed; Apache-2.0, see `LICENSES/`) |
| `pretrain.py` | + regularizer hook, full resume, JSONL metrics, offline wandb, `summary.json` |
| `config/cfg_sudoku_bench.yaml`, `config/arch/trm_contractive.yaml` | upstream README Sudoku MLP-T recipe at the benchmark schedule (B=256, 35k steps) + added flags |
| `dataset/build_sudoku_dataset.py` (`--seed`, `--source-revision`), `dataset/make_test_subset.py`, `dataset/sudoku-extreme-s0.sha256` | reproducible data, nested test subsets, data checksums |
| `scripts/evaluate.py` | checkpoint eval, accuracy at every ACT step up to K (test-time extrapolation), Q-halt accuracy |
| `scripts/population_eval.py` | residual-based stopping time (stop when ‖Ψ(s)−s‖/‖s‖ < τ) and multi-init populations (M=16, noisy or random inits): majority vote, vote-share confidence (AUROC, selective accuracy, ECE), min-residual selection, endpoint uniqueness |
| `scripts/fixed_point_diagnostics.py` | residuals, σ_max / spectral radius of J_Ψ and J_Φ, λ_max(sym J_V), fp32 convergence rate, init robustness |
| `scripts/run_benchmark.sh`, `scripts/post_run.sh`, `scripts/smoke.sh`, `scripts/summarize.py` | resumable run queue, final evals, smoke test, report |
| `tests/` | CPU test suite (equivalence with upstream, penalty toys, optimizer, pipeline, diagnostics) |
| `results/sudoku-contractive/report/` | `benchmark.md` + `figs/` (from `summarize.py`), `cap_sweep.png`, standalone HTML summary |
| `results/sudoku-contractive/runs/<ARM>_s<seed>/` | per-run `summary.json`, `all_config.yaml`, `metrics.jsonl`, `eval_*.json`, `diag*.json`, `pop_eval*.json` (no weights) |

## Reproduce

```bash
conda create -n trm-bench python=3.12 && conda activate trm-bench
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core huggingface_hub numba pytest matplotlib

python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000-s0 \
    --subsample-size 1000 --num-aug 1000 --seed 0 --source-revision 58942f96baeb572ca3127e2a9e9c70f330783d6b
for n in 1000 10000 25000 100000; do
  python dataset/make_test_subset.py --source data/sudoku-extreme-1k-aug-1000-s0 \
    --output data/sudoku-extreme-test$((n/1000))k --n $n --seed 0
done
sha256sum -c dataset/sudoku-extreme-s0.sha256

CUDA_VISIBLE_DEVICES="" DISABLE_COMPILE=1 python -m pytest tests -q      # CPU test suite

# one iteration = 1280 epochs = 5000 steps at B=256; A0 runs 9 iterations (approximately compute-matched control), others 7
FULL_TEST_SEEDS=0 SEEDS="0 1 2" ARMS="A0 A1 A2 A3" ITERS=7 A0_ITERS=9 scripts/run_benchmark.sh
FULL_TEST_SEEDS=0 SEEDS=0 ARMS="A4 A5 A6 A7 A8 A9 A10" ITERS=7 scripts/run_benchmark.sh
python scripts/summarize.py --root checkpoints/sudoku-contractive --out reports/sudoku-contractive

# regenerate the published report (benchmark.md, identical, + figures) from the published run files
python scripts/summarize.py --root results/sudoku-contractive/runs --out reports/sudoku-contractive
```

The shell scripts activate the conda env `trm-bench` (`TRM_CONDA_ENV`); set `CONDA_SH` to your `conda.sh` if conda
is not on `PATH`, or `TRM_CONDA_ENV=none` to use the current Python.

Single run with the settings of `A1_s0` (the config defaults are the benchmark schedule; keep the directory outside
`checkpoints/sudoku-contractive/`, which the queue resumes from):
`python pretrain.py --config-name cfg_sudoku_bench seed=0 arch.reg_mode=cycle_lip arch.reg_eta=0.05 run_name=A1_s0 checkpoint_path=checkpoints/single/A1_s0`.

## Deviations from the paper setup (all arms identical)

* Batch 256 (16 GB GPU) instead of 768, and about 18% of the upstream training examples (35k steps at B=256 vs
  50,000 epochs ≈ 65k steps at B=768; A0 to 45k steps). LR schedule (1e-4 constant after 2,000 warmup steps) and EMA
  0.999 as upstream.
* Seeded 1k-puzzle training subset (the paper's subset is unknown), so absolute accuracies differ from the paper's 87%.
* Final accuracy on a fixed 100k subset of the 423k test set (SE ≈ 0.15 pp) for every run; full test set for seed 0.
* Not evaluated (paused): `A1_s0` and `A5_s0` raw-weight diagnostics (`diag_raw.json`); `A5_s0` 100k, 64-step,
  diagnostics and population evaluations (only its full-test accuracy and training metrics exist).
