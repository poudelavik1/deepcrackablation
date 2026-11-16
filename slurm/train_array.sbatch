#!/bin/bash
# ONE array job over (model, seed). Do not submit this file directly: use
#     bash slurm/submit.sh [filters]        or        bash slurm/ablation.sh <stage>
# which write the job list and set --array to its length.
#
# Task k trains line k of $JOBLIST ("name seed config"). A finished run is skipped and
# an interrupted one resumes from last.pt, so resubmitting the same list is always safe.
#
# PACK=n (slurm/ablation.sh): task k trains lines k*n .. k*n+n-1 AT THE SAME TIME on its one GPU,
# each with 1/n of the task's CPUs as DataLoader workers and its own log
# logs/<job name>_<array id>_<task>.<name>_s<seed>.out. For accounts that may run one job at a time.
#SBATCH --job-name=crackbench
#SBATCH --partition=ASL-gpu          # <PARTITION> (slurm/submit.sh overrides it from slurm/env.sh)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # = DataLoader workers (slurm/ablation.sh asks for more with PACK > 1)
#SBATCH --mem=32G
#SBATCH --time=12:00:00              # per task; a task killed by the limit resumes when resubmitted
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out

set -uo pipefail
source slurm/common.sh
: "${JOBLIST:?Submit with slurm/submit.sh (JOBLIST is not set)}"
PACK="${PACK:-1}"

# run_line <line number (1-based)> [workers]: train one job-list line, then (EVAL_AFTER=1) score it on val.
run_line() {
    local LINE NAME SEED CONFIG STATUS WORKERS_ARG=""
    LINE=$(sed -n "${1}p" "$JOBLIST")
    read -r NAME SEED CONFIG <<< "$LINE"
    if [ -n "${2:-}" ]; then WORKERS_ARG="--workers $2"; fi
    echo "task   : $NAME seed $SEED ($CONFIG)"
    # Run in the batch step (not srun): since Slurm 22.05 an srun step ignores --cpus-per-task.
    python -u train.py --config "$CONFIG" --seed "$SEED" --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" \
        $WORKERS_ARG ${TRAIN_ARGS:-}
    STATUS=$?
    # EVAL_AFTER=1 (slurm/ablation.sh): score this run on val right away (threshold chosen on val).
    if [ "$STATUS" -eq 0 ] && [ "${EVAL_AFTER:-0}" = "1" ]; then
        python -u evaluate.py --split val --config "$CONFIG" --seed "$SEED" --data-root "$DATA_ROOT" \
            --runs-root "$RUNS_ROOT" $WORKERS_ARG
        STATUS=$?
    fi
    echo "===== $NAME seed $SEED finished $(date) | exit $STATUS ====="
    return $STATUS
}

FIRST=$((SLURM_ARRAY_TASK_ID * PACK + 1))
[ -n "$(sed -n "${FIRST}p" "$JOBLIST")" ] || { echo "No line $FIRST in $JOBLIST" >&2; exit 1; }

if [ "$PACK" -le 1 ]; then
    run_line "$FIRST"
    STATUS=$?
else
    WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} / PACK ))
    [ "$WORKERS" -ge 1 ] || WORKERS=1
    PIDS=()
    for LANE in $(seq 0 $((PACK - 1))); do
        NUMBER=$((FIRST + LANE))
        LINE=$(sed -n "${NUMBER}p" "$JOBLIST")
        [ -n "$LINE" ] || continue                  # the last task of a list may have fewer lines
        read -r NAME SEED _ <<< "$LINE"
        LOG="logs/${SLURM_JOB_NAME:-job}_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID}.${NAME}_s${SEED}.out"
        echo "lane $LANE : $NAME seed $SEED, $WORKERS workers -> $LOG"
        run_line "$NUMBER" "$WORKERS" > "$LOG" 2>&1 &
        PIDS+=($!)
        sleep 20                                    # stagger the start (dataset scan, model build)
    done
    # GPU load every 10 min in this log: shows whether the GPU has room for more lanes.
    echo "gpu    : time, utilization, memory used, memory total"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader -l 600 &
    MONITOR=$!
    STATUS=0
    for PID in "${PIDS[@]}"; do
        wait "$PID" || STATUS=1
    done
    kill "$MONITOR" 2>/dev/null
fi
echo "===== finished $(date) | exit $STATUS ====="
exit $STATUS
