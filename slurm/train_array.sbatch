#!/bin/bash
# ONE array job over (model, seed). Do not submit this file directly: use
#     bash slurm/submit.sh [filters]
# which writes the job list and sets --array=0-(N-1)%K to its length.
#
# Task k trains line k of $JOBLIST ("model seed config"). A finished run is skipped and
# an interrupted one resumes from last.pt, so resubmitting the same list is always safe.
#SBATCH --job-name=crackbench
#SBATCH --partition=ASL-gpu          # <PARTITION> (slurm/submit.sh overrides it from slurm/env.sh)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # = DataLoader workers
#SBATCH --mem=32G
#SBATCH --time=12:00:00              # per task; a task killed by the limit resumes when resubmitted
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out

set -uo pipefail
source slurm/common.sh
: "${JOBLIST:?Submit with slurm/submit.sh (JOBLIST is not set)}"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBLIST")
[ -n "$LINE" ] || { echo "No line $SLURM_ARRAY_TASK_ID in $JOBLIST" >&2; exit 1; }
read -r MODEL SEED CONFIG <<< "$LINE"
echo "task   : $MODEL seed $SEED ($CONFIG)"

# Run in the batch step (not srun): since Slurm 22.05 an srun step ignores --cpus-per-task.
python -u train.py --config "$CONFIG" --seed "$SEED" --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" ${TRAIN_ARGS:-}
STATUS=$?
# EVAL_AFTER=1 (slurm/ablation.sh): score this run on val right away (threshold chosen on val).
if [ "$STATUS" -eq 0 ] && [ "${EVAL_AFTER:-0}" = "1" ]; then
    python -u evaluate.py --split val --config "$CONFIG" --seed "$SEED" --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT"
    STATUS=$?
fi
echo "===== finished $(date) | exit $STATUS ====="
exit $STATUS
