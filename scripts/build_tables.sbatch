#!/usr/bin/env bash
# Build the comparison CSV and the error analysis from whatever results exist.
#SBATCH --job-name=attvr-compare
#SBATCH --partition=farsightQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source scripts/env.sh
R=outputs/results
results=()
for name in clip_baseline vjepa_baseline ablation_no_transition ablation_transition_only \
    ablation_shuffled_states ablation_mean_transition ablation_final_minus_initial \
    ablation_full_transition_without_query_conditioning transition_model; do
  # Trained models write one <name>_seed<seed>.json per seed; CLIP writes <name>.json.
  # Reverse-time results (<name>_seed<seed>_reversed.json) are kept out of this table.
  found=("$R/$name".json "$R/$name"_seed*.json)
  found=($(ls "${found[@]}" 2>/dev/null | grep -v '_reversed\.json$' || true))
  if (( ${#found[@]} )); then results+=("${found[@]}"); else echo "missing: $R/$name[_seed*].json"; fi
done
python -m src.eval.evaluate "${results[@]}" --output "$R/comparison_table.csv"
column -s, -t "$R/comparison_table.csv"
python -m src.models.baselines.error_analysis --figure-dir "$R/qualitative"
