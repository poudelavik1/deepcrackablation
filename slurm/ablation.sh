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
# not finished and evaluated yet, so it is always safe (e.g. after a time-out).
#   DRY_RUN=1 bash slurm/ablation.sh screen               # print the job list, submit nothing
#   bash slurm/ablation.sh screen --only 'opt_*' ref      # a subset of variants (globs)
#   PLAN=configs/ablation/<model>.yaml bash slurm/ablation.sh ...   # another model's plan

set -euo pipefail
cd "$(dirname "$0")/.."
[ -f slurm/env.sh ] || { echo "Run: cp slurm/env.example.sh slurm/env.sh  and edit it." >&2; exit 1; }
source slurm/env.sh
export PATH="$ENV_DIR/bin:$PATH"
STAGE="${1:-}"
[ -n "$STAGE" ] || { sed -n '2,16p' "$0"; exit 1; }
shift
PLAN="${PLAN:-configs/ablation/deepcrack.yaml}"
ABLATION=(python scripts/ablation.py --plan "$PLAN" --runs-root "$RUNS_ROOT")

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
echo "Job list: $JOBLIST ($COUNT tasks, at most ${MAX_PARALLEL:-4} at once)"
[ "${DRY_RUN:-0}" = "1" ] && exit 0

export JOBLIST
if [ "$STAGE" = "test" ]; then
    JOB_ID=$(sbatch --parsable --partition="$PARTITION" --export=ALL ${SBATCH_ARGS:-} slurm/ablation_test.sbatch)
    echo "Submitted test evaluation job ${JOB_ID%%;*}: logs/ablation_test_${JOB_ID%%;*}.out"
    exit 0
fi
export TRAIN_ARGS="${TRAIN_ARGS:-}" EVAL_AFTER=1
JOB_ID=$(sbatch --parsable --job-name=ablation --partition="$PARTITION" \
    --array="0-$((COUNT - 1))%${MAX_PARALLEL:-4}" --export=ALL ${SBATCH_ARGS:-} slurm/train_array.sbatch)
JOB_ID="${JOB_ID%%;*}"
echo "Submitted array job $JOB_ID (stage $STAGE)"
echo "  monitor : squeue -u $USER  |  python scripts/progress.py --runs-root $RUNS_ROOT --protocol ablation_deepcrack"
echo "  logs    : tail -f logs/ablation_${JOB_ID}_0.out"
echo "  then    : bash slurm/ablation.sh report     (prints the next step)"
