#!/usr/bin/env bash
# Evaluate a checkpoint on the test split. Extra args go to evaluate.py,
# e.g. sbatch scripts/evaluate.sbatch --checkpoint outputs/checkpoints/no_rerank_best.pt
#SBATCH --job-name=attvr-eval
#SBATCH --partition=farsightQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source scripts/env.sh
python -m src.models.transition_model.pipelines.evaluate \
  --num-workers "$((SLURM_CPUS_PER_TASK - 1))" \
  --transition-dir outputs/features/transitions "$@"
