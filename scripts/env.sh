# Shared setup sourced by every sbatch script in this folder.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
test -f config.py || { echo "submit from the project root" >&2; exit 1; }

# Activate the project env (override with ATTVR_CONDA_ENV=<name>).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ATTVR_CONDA_ENV:-attvr}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
echo "job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none}"
echo "python=$(command -v python) started=$(date -Is)"
