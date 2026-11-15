#!/usr/bin/env bash
# Submit CLIP, the V-JEPA variants, then the comparison table once both finish.
# Logs go to logs/.
# Usage: bash scripts/submit_baselines.sh   (needs cached V-JEPA features and the main checkpoint)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

clip=$(sbatch --parsable scripts/eval_clip_baseline.sbatch)
echo "clip:     $clip"
variants=$(sbatch --parsable scripts/train_vjepa_variants.sbatch)
echo "variants: $variants"
# afterany: the table is still built (with missing models noted) if one job fails.
compare=$(sbatch --parsable --dependency="afterany:$clip:$variants" scripts/build_tables.sbatch)
echo "compare:  $compare"
echo "Watch with: squeue -u $USER   Logs: logs/"
