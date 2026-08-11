#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=48:00:00

# Load modules (modify these as needed)
module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

if [ -f .env ]; then set -a; source .env; set +a; fi

export DATA_PATH="./synergy_plus"

srun -n 1 python ./src/feature_matrix_scripts/mxbai.py --data-path "$DATA_PATH"
srun -n 1 python ./src/feature_matrix_scripts/multilingual_e5.py --data-path "$DATA_PATH"
