#!/usr/bin/env bash
# Pack every shared artifact into one zip for Google Drive: artifacts/attvr_artifacts.zip.
# Paths inside the zip are relative to the repository root, so `unzip` restores them in place.
# Usage: bash scripts/pack_artifacts.sbatch   or   sbatch [--dependency=afterany:<job>] scripts/pack_artifacts.sbatch
#SBATCH --job-name=attvr-pack
#SBATCH --partition=farsightQ
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
test -f config.py || { echo "run from the project root" >&2; exit 1; }

# Older runs left per-epoch training logs next to the checkpoints.
shopt -s nullglob
for log in outputs/checkpoints/*_train_log.jsonl; do mv -n "$log" logs/; done

paths=(
  data/annotations
  outputs/features/vjepa
  outputs/features/clip
  outputs/checkpoints
  outputs/results
  logs
)
for path in "${paths[@]}"; do
  test -e "$path" || { echo "missing: $path" >&2; exit 1; }
done

mkdir -p artifacts
out=artifacts/attvr_artifacts.zip
tmp="$out.tmp.zip"
rm -f "$tmp"
# -n .npz stores the already-compressed feature files as is; -q keeps the log short.
zip -r -q -n .npz "$tmp" "${paths[@]}" -x '*/.gitkeep' 'logs/attvr-pack_*'
mv "$tmp" "$out"
unzip -tq "$out"
ls -lh "$out"
echo "entries: $(unzip -Z1 "$out" | wc -l)"
