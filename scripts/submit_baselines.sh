#!/usr/bin/env bash
# Submit CLIP, the V-JEPA variants (seeds 42, 43, 44), their reverse-time evaluation,
# then the comparison table once all finish.
# Logs go to logs/.
# Usage: bash scripts/submit_baselines.sh   (needs cached V-JEPA features and the main checkpoint)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

clip=$(sbatch --parsable scripts/eval_clip_baseline.sbatch)
echo "clip:     $clip"
variants=$(sbatch --parsable scripts/train_vjepa_variants.sbatch)
echo "variants: $variants"
# Reverse time needs the trained checkpoints, so it waits for every variant to succeed.
reversed=$(sbatch --parsable --dependency="afterok:$variants" scripts/train_vjepa_variants.sbatch --reverse-time)
echo "reversed: $reversed"
# afterany: the table is still built (with missing models noted) if one job fails.
compare=$(sbatch --parsable --dependency="afterany:$clip:$variants:$reversed" scripts/build_tables.sbatch)
echo "compare:  $compare"
echo "Watch with: squeue -u $USER   Logs: logs/"
