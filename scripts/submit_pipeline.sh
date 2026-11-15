#!/usr/bin/env bash
# Submit extract -> train+evaluate (3 seeds), each starting only if the previous one succeeded.
# Usage: bash scripts/submit_pipeline.sh [--skip-extract]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

dependency=()
if [[ "${1:-}" != "--skip-extract" ]]; then
  extract=$(sbatch --parsable scripts/extract_features.sbatch)
  echo "extract:  $extract"
  dependency=(--dependency="afterok:$extract")
fi
# One array task per seed; each trains and then evaluates its own checkpoint.
train_eval=$(RUN_NAME=transition_model sbatch --parsable "${dependency[@]}" scripts/train_eval.sbatch)
echo "train+evaluate (seeds 42 43 44): $train_eval"
echo "Watch with: squeue -u $USER   Logs: logs/"
