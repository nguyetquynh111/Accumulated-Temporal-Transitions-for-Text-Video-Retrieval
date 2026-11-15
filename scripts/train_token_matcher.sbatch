#!/usr/bin/env bash
# Stage 2 of the final method: train/evaluate the query-conditioned token-level matcher
# on a frozen stage-1 encoder. Extra args go to train_token_reranker.py, e.g.
#   sbatch scripts/train_token_matcher.sbatch --encoder state_only --seeds 42        (E1)
#   sbatch scripts/train_token_matcher.sbatch --encoder full_transition --eval-only --seeds 42 43 44
#SBATCH --job-name=attvr-tokrr
#SBATCH --partition=farsightQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source scripts/env.sh
python -m src.models.transition_model.pipelines.train_token_reranker "$@"
