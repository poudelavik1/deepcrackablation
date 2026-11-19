#!/bin/bash
# Ablation of DeepCrack's training recipe (augmentation x optimizer). Run on the login node, from the repo root.
#
#   bash slurm/ablation.sh check      # build every variant on the login node (CPU, seconds)
#   bash slurm/ablation.sh screen     # stage 1: one change per run + the reference x 3 seeds (~53 runs)
#   bash slurm/ablation.sh report     # any time: tables, figures, decisions and the next step
#   bash slurm/ablation.sh combine    # stage 2, after stage 1: the positive changes combined (<= 4 runs)
#   bash slurm/ablation.sh final      # stage 3, after stage 2: seeds 1 2 for the best candidates (<= 4 runs)
#   bash slurm/ablation.sh test       # ONCE, at the very end: final recipe + reference on the test set
#
# Every training task also evaluates its run on val. Resubmitting a stage only queues runs that are
# not finished and evaluated yet, and leaves out runs written to in the last 30 min (running now),
# so it is always safe (e.g. after a time-out).
#   GPUS=4 bash slurm/ablation.sh final                   # 4 runs at once, one per GPU (4 GPUs: jobs <= 1 day)
#   PACK=2 bash slurm/ablation.sh screen                  # 2 runs at once sharing one GPU (no faster for DeepCrack)
#   DRY_RUN=1 bash slurm/ablation.sh screen               # print the job list, submit nothing
#   bash slurm/ablation.sh screen --only 'opt_*' ref      # a subset of variants (globs)
#   PLAN=configs/ablation/<model>.yaml bash slurm/ablation.sh ...   # another model's plan
# With PACK=n a job asks for n x CPUS_PER_RUN CPUs (default 8) and n x MEM_PER_RUN_GB memory (default 24).
# Nothing is submitted while tasks of an earlier ablation array still wait in the queue (FORCE=1 overrides).

set -euo pipefail
cd "$(dirname "$0")/.."
[ -f slurm/env.sh ] || { echo "Run: cp slurm/env.example.sh slurm/env.sh  and edit it." >&2; exit 1; }
source slurm/env.sh
export PATH="$ENV_DIR/bin:$PATH"
STAGE="${1:-}"
[ -n "$STAGE" ] || { sed -n '2,20p' "$0"; exit 1; }
shift
PLAN="${PLAN:-configs/ablation/deepcrack.yaml}"
GPUS="${GPUS:-1}"                 # GPUs per job; with GPUS=n, n runs train at once, one per GPU
PACK="${PACK:-$GPUS}"             # runs per job (more runs than GPUs share them)
ABLATION=(python scripts/ablation.py --plan "$PLAN" --runs-root "$RUNS_ROOT")

# One waiting array at a time: a second one would queue the same runs again, and the older one
# (e.g. unpacked) would usually start first. Running tasks do not count: the job list leaves out
# runs in progress, so switching to PACK=2 only needs `scancel --state=PENDING <old id>`.
if [ "$STAGE" != "check" ] && [ "$STAGE" != "report" ] && [ "${FORCE:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ] \
        && command -v squeue > /dev/null; then
    if [ "$STAGE" = "test" ]; then NAME=ablation_test; else NAME=ablation; fi
    WAITING=$(squeue -u "${USER:-$(id -un)}" -h -t PENDING -n "$NAME" -o "  %i  (%r)" 2> /dev/null || true)
    if [ -n "$WAITING" ]; then
        echo "Not submitted: tasks of an earlier '$NAME' array are still waiting in the queue:" >&2
        echo "$WAITING" >&2
        echo "A second array would queue the same runs again. Cancel the waiting tasks first" >&2
        echo "(a task that is running keeps running, and its run is left out of the new list):" >&2
        echo "    scancel --state=PENDING <job id>" >&2
        echo "then run this command again, or submit anyway with FORCE=1." >&2
        exit 1
    fi
fi

case "$STAGE" in
    check|report) exec "${ABLATION[@]}" "$STAGE" "$@" ;;
    screen)       "${ABLATION[@]}" check > /dev/null || { "${ABLATION[@]}" check | grep FAIL; exit 1; } ;;
    combine|final|test) ;;
    *) echo "Unknown stage '$STAGE' (check | screen | combine | final | report | test)" >&2; exit 1 ;;
esac

mkdir -p logs "$RUNS_ROOT/joblists"
JOBLIST="$RUNS_ROOT/joblists/ablation_${STAGE}_$(date +%Y%m%d_%H%M%S).txt"
COUNT=$("${ABLATION[@]}" jobs --stage "$STAGE" --out "$JOBLIST" "$@" | tail -n 1)
if [ "$COUNT" -eq 0 ]; then
    echo "Nothing to submit for stage '$STAGE'."
    exit 0
fi

export JOBLIST PLAN
if [ "$STAGE" = "test" ]; then
    echo "Job list: $JOBLIST ($COUNT evaluations)"
    [ "${DRY_RUN:-0}" = "1" ] && exit 0
    JOB_ID=$(sbatch --parsable --partition="$PARTITION" --export=ALL ${SBATCH_ARGS:-} slurm/ablation_test.sbatch)
    echo "Submitted test evaluation job ${JOB_ID%%;*}: logs/ablation_test_${JOB_ID%%;*}.out"
    exit 0
fi

[ "$GPUS" -le "$COUNT" ] || GPUS=$COUNT     # never hold (or wait for) more GPUs than there are runs
[ "$PACK" -le "$COUNT" ] || PACK=$COUNT
TASKS=$(( (COUNT + PACK - 1) / PACK ))
RESOURCES=""
if [ "$PACK" -gt 1 ]; then
    RESOURCES="--cpus-per-task=$(( ${CPUS_PER_RUN:-8} * PACK )) --mem=$(( ${MEM_PER_RUN_GB:-24} * PACK ))G"
fi
if [ "$GPUS" -gt 1 ]; then
    RESOURCES="$RESOURCES --gres=gpu:$GPUS"
fi
echo "Job list: $JOBLIST ($COUNT runs in $TASKS array tasks, $PACK per task${RESOURCES:+: $RESOURCES})"
[ "${DRY_RUN:-0}" = "1" ] && exit 0

export PACK TRAIN_ARGS="${TRAIN_ARGS:-}" EVAL_AFTER=1
JOB_ID=$(sbatch --parsable --job-name=ablation --partition="$PARTITION" $RESOURCES \
    --array="0-$((TASKS - 1))%${MAX_PARALLEL:-4}" --export=ALL ${SBATCH_ARGS:-} slurm/train_array.sbatch)
JOB_ID="${JOB_ID%%;*}"
echo "Submitted array job $JOB_ID (stage $STAGE)"
echo "  monitor : squeue -u ${USER:-$(id -un)}  |  python scripts/progress.py --runs-root $RUNS_ROOT --protocol ablation_deepcrack"
if [ "$PACK" -gt 1 ]; then
    echo "  logs    : logs/ablation_${JOB_ID}_<task>.out (GPU load) and logs/ablation_${JOB_ID}_<task>.<run>.out (one per run)"
else
    echo "  logs    : tail -f logs/ablation_${JOB_ID}_0.out"
fi
echo "  then    : bash slurm/ablation.sh report     (prints the next step)"
