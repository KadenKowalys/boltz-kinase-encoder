#!/bin/bash
#SBATCH --job-name=boltz-12proteins
#SBATCH --account=def-hpcg1856_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/boltz-full-%j.out
#SBATCH --error=logs/boltz-full-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.2
module load rdkit/2024.03.5

cd /global/scratch/hpc6240/boltz_encoder_project/boltz-kinase-encoder
source .venv-gpu/bin/activate

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/gpu/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "Allocated GPU is unavailable"
print("GPU:", torch.cuda.get_device_name(0))
PY

inputs="gpu/full_inputs_${SLURM_JOB_ID}"
run_dir="gpu/full_run_${SLURM_JOB_ID}"
features="data/features/boltz2_${SLURM_JOB_ID}.npz"
results="results/local-boltz-${SLURM_JOB_ID}"
model="models/boltz_regressor_${SLURM_JOB_ID}.npz"

kinase-encoder prepare-gpu \
  --targets data/processed/targets.csv \
  --output "$inputs"

python scripts/run_encoder.py \
  --inputs "$inputs" \
  --output "$run_dir" \
  --cache gpu/cache

kinase-encoder collect \
  --inputs "$inputs" \
  --run "$run_dir" \
  --output "$features"

kinase-encoder evaluate \
  --embeddings "$features" \
  --output "$results"

kinase-encoder fit \
  --method ligand_boltz \
  --embeddings "$features" \
  --output "$model"

echo "FULL RUN COMPLETE"
echo "Results: $results/report.md"
echo "Embeddings: $features"
echo "Trained regressor: $model"
