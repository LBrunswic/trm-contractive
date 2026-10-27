#!/usr/bin/env bash
# Usage: scripts/smoke.sh <tag> <batch> <steps> [arms...]   (compiled unless DISABLE_COMPILE is set)
set -uo pipefail
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
export WANDB_MODE=offline HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=$1; B=$2; STEPS=$3; shift 3
ETA=${ETA:-0.05}
mkdir -p logs/smoke
for arm in "$@"; do
  case $arm in  # same overrides as arm_overrides in scripts/run_benchmark.sh
    A0) O=();;
    A1) O=(arch.reg_mode=cycle_lip arch.reg_eta=$ETA);;
    A2) O=(arch.flow=True);;
    A3) O=(arch.flow=True arch.reg_mode=flow_mono arch.reg_eta=$ETA);;
    A4) O=(arch.flow=True arch.reg_mode=cycle_lip arch.reg_eta=$ETA);;
    A5) O=(arch.reg_mode=cycle_lip arch.reg_eta=0.0);;
    A6) O=(arch.reg_mode=cycle_lip ++arch.reg_rho=2.0);;
    A7) O=(arch.reg_mode=cycle_lip ++arch.reg_rho=4.0);;
    A8) O=(arch.reg_mode=dec_semi ++arch.reg_rho=0.95 ++arch.reg_gamma=0.3 arch.reg_frac=0.125);;
    A9) O=(arch.reg_mode=dec_out ++arch.reg_rho=1.0);;
    A10) O=(arch.flow=True arch.reg_mode=dec_semi ++arch.reg_rho=0.95 ++arch.reg_gamma=0.3 arch.reg_frac=0.125);;
    *) echo "unknown arm $arm (expected A0-A10)" >&2; exit 1;;
  esac
  dir=checkpoints/smoke/${TAG}_$arm; rm -rf "$dir"
  start=$(date +%s)
  python pretrain.py --config-name cfg_sudoku_bench run_name=${TAG}_$arm checkpoint_path=$dir global_batch_size=$B \
    epochs=1000 eval_interval=1000 max_train_steps=$STEPS log_every=20 data_paths_test="[data/sudoku-extreme-test1k]" \
    resume=False "${O[@]}" ${EXTRA:-} > logs/smoke/${TAG}_$arm.log 2>&1
  rc=$?
  echo "== $arm rc=$rc wall=$(( $(date +%s) - start ))s"
  grep -E "OutOfMemory|Error" logs/smoke/${TAG}_$arm.log | grep -v "^wandb" | head -3
  [ -f "$dir/metrics.jsonl" ] && python - "$dir/metrics.jsonl" <<'PY'
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r["split"] == "train":
        print(r["step"], "sps=%.2f" % r["steps_per_s"], "mem=%.1f/%.1f" % (r["mem_alloc_peak_gb"], r["mem_reserved_peak_gb"]),
              "lm=%.3f" % r["train/lm_loss"], "reg=%s lam=%s viol=%s" % (r.get("reg/value"), r.get("reg/lambda"), r.get("reg/viol")),
              "mon_v1=%s mon_v2=%s" % (r.get("mon/v1_ratio"), r.get("mon/v2_q")))
    elif r["split"] == "eval":
        print("eval", r["step"], "eval_s=%.1f" % r["eval_s"], "exact=%.4f" % r["all/exact_accuracy"])
PY
done
