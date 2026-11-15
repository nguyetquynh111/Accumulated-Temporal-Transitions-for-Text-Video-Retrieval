#!/usr/bin/env bash
# Train the transition model for one seed per array task, then evaluate it on test.
# Usage: RUN_NAME=transition_model sbatch scripts/train_eval.sbatch [extra train.py args]
# Writes outputs/checkpoints/<run>_seed<seed>_best.pt and outputs/results/<run>_seed<seed>.json.
# The epoch limit, early stopping, and seeds follow the shared protocol in config.py.
#SBATCH --job-name=attvr-train-eval
#SBATCH --partition=farsightQ
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

SEEDS=(42 43 44)
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
WORKERS="$((SLURM_CPUS_PER_TASK - 1))"
source scripts/env.sh
echo "run=$RUN_NAME seed=$SEED"

python -m src.models.transition_model.pipelines.train \
  --run-name "$RUN_NAME" --seed "$SEED" --num-workers "$WORKERS" "$@"
python -m src.models.transition_model.pipelines.evaluate \
  --run-name "$RUN_NAME" --seeds "$SEED" --num-workers "$WORKERS"
