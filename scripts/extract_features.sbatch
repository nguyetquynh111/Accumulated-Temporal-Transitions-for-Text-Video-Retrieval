#!/usr/bin/env bash
# Cache V-JEPA 2 states for all splits. Each array task is one GPU shard.
# Change the shard count with: sbatch --array=0-7 scripts/extract_features.sbatch
#SBATCH --job-name=attvr-extract
#SBATCH --partition=farsightQ
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source scripts/env.sh
ANNOTATIONS="${ATTVR_ANNOTATION_ROOT:-data/annotations}"
python -m src.models.transition_model.pipelines.extract_features \
  --annotations "$ANNOTATIONS/train.jsonl" "$ANNOTATIONS/val.jsonl" "$ANNOTATIONS/test.jsonl" \
  --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --batch-size 16 \
  --num-workers "$((SLURM_CPUS_PER_TASK - 1))" \
  "$@"
