#!/bin/bash
#SBATCH --job-name=kinase-encoder
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=encoder-%j.log
set -euo pipefail
# Submit from the repository root after activating your dedicated Boltz environment.
# Add the account/partition directives required by your cluster before submission.
# Resource requests are starting estimates, not measured requirements.
export OMP_NUM_THREADS=4
python scripts/run_encoder.py --inputs gpu/inputs --output gpu/run --cache gpu/cache
kinase-encoder collect --inputs gpu/inputs --run gpu/run --output data/features/boltz2.npz
