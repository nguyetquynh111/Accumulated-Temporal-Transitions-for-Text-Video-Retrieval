#!/usr/bin/env bash
# B1: zero-shot CLIP with frame averaging on the test split (decodes test videos).
#SBATCH --job-name=attvr-clip
#SBATCH --partition=farsightQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source scripts/env.sh
python -m src.models.baselines.clip_baseline "$@"
