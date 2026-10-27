#!/usr/bin/env bash
# Resumable benchmark queue. Each run trains (resuming from last_state.pt) up to its epoch budget,
# then, once training is complete, runs the final evaluations and diagnostics if the scripts exist.
#
#   SEEDS="0 1 2" ARMS="A0 A1 A2 A3" ITERS=7 A0_ITERS=9 scripts/run_benchmark.sh
#
# One iteration = EPI epochs (= 5000 optimizer steps at B=256); in-training eval on test10k after each.
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

PROJECT=${PROJECT:-sudoku-contractive}
SEEDS=${SEEDS:-"0"}
ARMS=${ARMS:-"A0 A1 A2 A3"}
B=${B:-256}
EPI=${EPI:-1280}            # epochs per iteration (eval interval)
ITERS=${ITERS:-7}           # iterations for every arm (T)
A0_ITERS=${A0_ITERS:-$ITERS} # A0 runs longer for the compute-matched control (checkpoint at T kept)
ETA=${ETA:-0.05}
POST=${POST:-1}             # run final evals/diagnostics after a run completes

arm_overrides() {
  case $1 in
    A0) echo "";;
    A1) echo "arch.reg_mode=cycle_lip arch.reg_eta=$ETA";;
    A2) echo "arch.flow=True";;
    A3) echo "arch.flow=True arch.reg_mode=flow_mono arch.reg_eta=$ETA";;
    A4) echo "arch.flow=True arch.reg_mode=cycle_lip arch.reg_eta=$ETA";;
    A5) echo "arch.reg_mode=cycle_lip arch.reg_eta=0.0";;   # V1 non-expansive: Lip(Phi) <= 1
    A6) echo "arch.reg_mode=cycle_lip ++arch.reg_rho=2.0";;  # V1 cap sweep: Lip(Phi) <= 2
    A7) echo "arch.reg_mode=cycle_lip ++arch.reg_rho=4.0";;  # V1 cap sweep: Lip(Phi) <= 4
    A8) echo "arch.reg_mode=dec_semi ++arch.reg_rho=0.95 ++arch.reg_gamma=0.3 arch.reg_frac=0.125";;  # D-gamma: semi-contraction onto decoder fibres, leak 0.3
    A9) echo "arch.reg_mode=dec_out ++arch.reg_rho=1.0";;    # D2: decoded output 1-Lipschitz in the full state
    A10) echo "arch.flow=True arch.reg_mode=dec_semi ++arch.reg_rho=0.95 ++arch.reg_gamma=0.3 arch.reg_frac=0.125";;  # Euler-TRM + D-gamma
    *) echo "unknown arm $1" >&2; exit 1;;
  esac
}

mkdir -p logs/runs
for seed in $SEEDS; do
  for arm in $ARMS; do
    run="${arm}_s${seed}"
    dir="checkpoints/$PROJECT/$run"
    iters=$ITERS; [[ $arm == A0 ]] && iters=$A0_ITERS
    epochs=$(( EPI * iters ))
    target_step=$(( epochs * 1000 / B ))

    done_step=$(python -c "import json,sys;print(json.load(open('$dir/summary.json'))['final_step'])" 2>/dev/null || echo 0)
    if (( done_step < target_step )); then
      echo "[$(date '+%F %T')] train $run -> $epochs epochs ($target_step steps)"
      rm -f "$dir/summary.json"
      # shellcheck disable=SC2046
      python pretrain.py --config-name cfg_sudoku_bench project_name=$PROJECT run_name=$run checkpoint_path=$dir \
        seed=$seed global_batch_size=$B epochs=$epochs eval_interval=$EPI $(arm_overrides $arm) ${EXTRA:-} \
        >> "logs/runs/$run.log" 2>&1 || { echo "[$(date '+%F %T')] FAILED $run (see logs/runs/$run.log)"; continue; }
      echo "[$(date '+%F %T')] done $run"
    fi

    if [[ $POST == 1 && -f scripts/post_run.sh && -f $dir/summary.json ]]; then
      extra=""
      [[ $arm == A0 && $A0_ITERS != $ITERS ]] && extra=$(( EPI * ITERS * 1000 / B ))   # A0 checkpoint at the common budget T
      echo "[$(date '+%F %T')] post $run"
      full=0; [[ " ${FULL_TEST_SEEDS:-} " == *" $seed "* ]] && full=1   # full 423k test only for these seeds
      FULL_TEST=$full bash scripts/post_run.sh "$dir" $extra >> "logs/runs/$run.post.log" 2>&1 || echo "[$(date '+%F %T')] POST FAILED $run"
    fi
  done
done
echo "[$(date '+%F %T')] queue finished"
