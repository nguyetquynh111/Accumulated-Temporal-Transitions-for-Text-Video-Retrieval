#!/usr/bin/env bash
# B2, the T3.3 ablations, and rankings for the main model. One GPU per variant.
# Every trained variant uses the same data, epochs, and evaluator as the main model.
# full_transition only evaluates the main checkpoint (for error-analysis rankings).
#SBATCH --job-name=attvr-variant
#SBATCH --partition=farsightQ
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

VARIANTS=(
  pooled
  no_transition
  mean_transition
  final_minus_initial
  full_transition_without_query_conditioning
  full_transition
  shuffled_full_transition
  shuffled_no_transition
)
source scripts/env.sh
variant="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "variant=$variant"
python -m src.models.baselines.vjepa_baseline --variant "$variant" "$@"
