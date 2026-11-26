#!/usr/bin/env bash
# Rebuild every generated file from the SSv2 metadata and videos, on one machine, in order:
# splits and candidate sets, V-JEPA 2 and CLIP features, baselines and ablations, the
# single-stage model, and the main experiment (E1 and E2, both stages, reverse time).
# Needs data/raw_metadata/ and data/videos/ (see README "Prepare data") and a GPU.
# Steps whose outputs already exist are skipped by the underlying tools where they
# support it (feature extraction resumes; candidate sets are kept unless --overwrite).
# Usage: bash scripts/reproduce_all.sh [first_step]   (first_step 1-6, default 1)
set -euo pipefail
cd "$(dirname "$0")/.."
test -f config.py || { echo "run from the project root" >&2; exit 1; }
mkdir -p logs

SEEDS=(42 43 44)
start="${1:-1}"
step() { (( start <= $1 )) && echo "== step $1: $2"; }

if step 1 "splits and candidate sets -> data/annotations/"; then
  python -m src.data.prepare_data --raw-metadata data/raw_metadata --seed 42
  python -m src.data.build_candidate_sets
fi

if step 2 "V-JEPA 2 features -> outputs/features/vjepa/"; then
  python -m src.models.transition_model.pipelines.extract_features \
    --annotations data/annotations/train.jsonl data/annotations/val.jsonl data/annotations/test.jsonl
fi

if step 3 "CLIP baseline (B1), V-JEPA baseline (B2), ablations, E2 stage 1"; then
  python -m src.models.baselines.clip_baseline
  python -m src.models.baselines.vjepa_baseline --variant pooled all_ablations --seeds "${SEEDS[@]}"
  python -m src.models.baselines.vjepa_baseline --variant all_ablations --reverse-time \
    --seeds "${SEEDS[@]}"
fi

if step 4 "single-stage transition model"; then
  for seed in "${SEEDS[@]}"; do
    python -m src.models.transition_model.pipelines.train --seed "$seed"
  done
  python -m src.models.transition_model.pipelines.evaluate
fi

if step 5 "E1 stage 1 (states-only encoder)"; then
  for seed in "${SEEDS[@]}"; do
    python -m src.models.transition_model.pipelines.train --seed "$seed" \
      --run-name ablation_state_only_without_query_conditioning \
      --video-input state_only --no-query-conditioning
  done
  python -m src.models.transition_model.pipelines.evaluate \
    --run-name ablation_state_only_without_query_conditioning
fi

if step 6 "stage 2 (token-level matcher) for E2 and E1, then reverse time"; then
  for encoder in full_transition state_only; do
    python -m src.models.transition_model.pipelines.train_token_reranker \
      --encoder "$encoder" --seeds "${SEEDS[@]}"
    python -m src.models.transition_model.pipelines.train_token_reranker \
      --encoder "$encoder" --reverse-time --seeds "${SEEDS[@]}"
  done
fi

echo "done: data/annotations/, outputs/features/, outputs/checkpoints/, outputs/results/, logs/"
