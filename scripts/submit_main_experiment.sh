#!/usr/bin/env bash
# Main experiment (T3.1), E1 (states only) vs E2 (states + dz), both two-stage with the
# token-level matcher. E2 is already trained; this submits what is missing:
#   1. E1 stage 1: cosine-only encoder on [z, 0] tokens (pipelines.train), one array task
#      per seed, each evaluated on test afterwards (states-only encoder baseline row).
#   2. E1 stage 2: token-level matcher on that encoder (after its seed's stage 1 succeeds),
#      then E1 reverse-time evaluation of that seed.
#   3. E2 re-evaluation of the existing checkpoints (per-query ranks and rankings, no
#      training), then E2 reverse-time evaluation.
# Usage: bash scripts/submit_main_experiment.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

SEEDS=(42 43 44)  # must match the array order in train_eval_single_stage.sbatch
s1=$(RUN_NAME=ablation_state_only_without_query_conditioning \
     sbatch --parsable scripts/train_eval_single_stage.sbatch \
     --video-input state_only --no-query-conditioning)
echo "E1 stage 1 (array, seeds ${SEEDS[*]}): $s1"
for task in 0 1 2; do
  seed="${SEEDS[$task]}"
  s2=$(sbatch --parsable --dependency="afterok:${s1}_${task}" scripts/train_token_matcher.sbatch \
       --encoder state_only --seeds "$seed")
  rev=$(sbatch --parsable --dependency="afterok:$s2" scripts/train_token_matcher.sbatch \
       --encoder state_only --reverse-time --seeds "$seed")
  echo "seed $seed: E1 stage 2 $s2 -> E1 reverse-time $rev"
done
e2=$(sbatch --parsable scripts/train_token_matcher.sbatch --encoder full_transition --eval-only --seeds 42 43 44)
e2rev=$(sbatch --parsable --dependency="afterok:$e2" scripts/train_token_matcher.sbatch \
  --encoder full_transition --reverse-time --seeds 42 43 44)
echo "E2 re-evaluation: $e2 -> E2 reverse-time: $e2rev"
echo "Watch with: squeue -u $USER   Logs: logs/"
