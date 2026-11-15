#!/usr/bin/env bash
# Train the transition model on cached features. Extra args go to train.py,
# e.g. sbatch scripts/train.sbatch --run-name no_rerank --no-query-conditioning
#SBATCH --job-name=attvr-train
#SBATCH --partition=farsightQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source scripts/env.sh
python -m src.models.transition_model.pipelines.train \
  --num-workers "$((SLURM_CPUS_PER_TASK - 1))" "$@"
