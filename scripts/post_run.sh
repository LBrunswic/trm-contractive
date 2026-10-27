#!/usr/bin/env bash
# Final evaluation + diagnostics of a finished run (resumable: skips outputs that exist).
#   scripts/post_run.sh <run_dir> [extra_step ...]
# EMA checkpoint `step_N` is primary; `step_N_raw` (raw weights) also gets diagnostics.
# Extra steps (e.g. A0's checkpoint at the common budget T) get the same evaluations with a _stepN suffix.
set -euo pipefail
cd "$(dirname "$0")/.."
# Python env: conda env ${TRM_CONDA_ENV:-trm-bench} (CONDA_SH = path to conda.sh), or TRM_CONDA_ENV=none for the current python.
# conda.sh lookup: $CONDA_SH, else the conda on PATH, else ~/miniconda3, ~/miniforge3, ~/anaconda3.
if [[ ${TRM_CONDA_ENV:-trm-bench} != none ]]; then
  CONDA_SH=${CONDA_SH:-$(conda info --base 2>/dev/null || true)/etc/profile.d/conda.sh}
  for c in "$HOME"/miniconda3 "$HOME"/miniforge3 "$HOME"/anaconda3; do [[ -f $CONDA_SH ]] || CONDA_SH=$c/etc/profile.d/conda.sh; done
  if [[ ! -f $CONDA_SH ]]; then
    echo "conda.sh not found: set CONDA_SH=/path/to/conda.sh, or TRM_CONDA_ENV=none to use the current python" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$CONDA_SH" && conda activate "${TRM_CONDA_ENV:-trm-bench}" \
    || { echo "cannot activate conda env ${TRM_CONDA_ENV:-trm-bench} (set TRM_CONDA_ENV)" >&2; exit 1; }
fi
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DIR=$1; shift
FULL=data/sudoku-extreme-1k-aug-1000-s0      # full 422,786-puzzle test split (only with FULL_TEST=1; about 2 h per run)
MAIN=data/sudoku-extreme-test100k             # fixed 100k subset (SE ~0.15 pp), used for every run
EXT=data/sudoku-extreme-test25k
DIAG=data/sudoku-extreme-test1k
final=$(python -c "import json;print(json.load(open('$DIR/summary.json'))['final_step'])")

evaluate_ckpt() {  # <ckpt> <suffix>
  local ck=$1 sfx=$2
  [[ -f $DIR/eval_100k$sfx.json ]] || python scripts/evaluate.py --ckpt "$ck" --data $MAIN --act_steps 16 --batch_size 1024 --compile --out "$DIR/eval_100k$sfx.json"
  # Full-test pass: skipped if logs/skip_full_test exists (e.g. eval_full*.json is computed elsewhere and copied in)
  if [[ ${FULL_TEST:-0} == 1 && ! -f logs/skip_full_test && ! -f $DIR/eval_full$sfx.json ]]; then
    python scripts/evaluate.py --ckpt "$ck" --data $FULL --act_steps 16 --batch_size 1024 --compile --out "$DIR/eval_full$sfx.json"
  fi
  [[ -f $DIR/eval_ext64$sfx.json ]] || python scripts/evaluate.py --ckpt "$ck" --data $EXT --act_steps 64 --batch_size 1024 --compile --out "$DIR/eval_ext64$sfx.json"
  [[ -f $DIR/diag$sfx.json ]] || python scripts/fixed_point_diagnostics.py --ckpt "$ck" --data $DIAG --n 1000 --n_jac 256 --jac_batch 16 --K 64 --out "$DIR/diag$sfx.json"
  # residual stopping time + multi-init majority vote / confidence / uniqueness
  [[ -f $DIR/pop_eval$sfx.json ]] || python scripts/population_eval.py --ckpt "$ck" --data $DIAG --n 1000 --M 16 --K 64 --puzzles_per_batch 64 --compile --out "$DIR/pop_eval$sfx.json"
}

echo "[$(date '+%F %T')] post $DIR (final step $final)"
evaluate_ckpt "$DIR/step_$final" ""
if [[ ${FULL_TEST:-0} == 1 && -f $DIR/step_${final}_raw && ! -f $DIR/diag_raw.json ]]; then  # raw-weight diagnostics: seed 0 only
  python scripts/fixed_point_diagnostics.py --ckpt "$DIR/step_${final}_raw" --data $DIAG --n 1000 --n_jac 256 --jac_batch 16 --K 64 --out "$DIR/diag_raw.json"
fi
for st in "$@"; do
  [[ -f $DIR/step_$st && $st != "$final" ]] && evaluate_ckpt "$DIR/step_$st" "_step$st"
done
echo "[$(date '+%F %T')] post done $DIR"
