#!/bin/bash
# Test-set evaluation of the ablation's final recipe and of the reference. Submit with
#     bash slurm/ablation.sh test
# ONCE, after the recipe was chosen on validation. Each run uses the threshold chosen on its val set;
# evaluate.py never recomputes an existing test result (no --overwrite here).
#SBATCH --job-name=ablation_test
#SBATCH --partition=ASL-gpu          # <PARTITION> (slurm/ablation.sh overrides it from slurm/env.sh)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
source slurm/common.sh
: "${JOBLIST:?Submit with slurm/ablation.sh test (JOBLIST is not set)}"
STATUS=0
while read -r NAME SEED CONFIG; do
    [ -n "$NAME" ] || continue
    echo "----- $NAME seed $SEED"
    python -u evaluate.py --split test --config "$CONFIG" --seed "$SEED" --data-root "$DATA_ROOT" \
        --runs-root "$RUNS_ROOT" || STATUS=1
done < "$JOBLIST"
python -u scripts/ablation.py report --plan "${PLAN:-configs/ablation/deepcrack.yaml}" --runs-root "$RUNS_ROOT" || STATUS=1
echo "===== finished $(date) | exit $STATUS ====="
exit $STATUS
