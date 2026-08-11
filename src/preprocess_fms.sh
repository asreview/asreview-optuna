#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00

# Load modules (modify these as needed)
module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

# DATA_PATH must point at the synergy_plus data directory, e.g.:
#   sbatch --export=DATA_PATH="/path/to/synergy_plus" ./src/preprocess_fms.sh

srun -n 1 python feature_matrix_scripts/mxbai.py --data-path "$DATA_PATH"
srun -n 1 python feature_matrix_scripts/multilingual_e5.py --data-path "$DATA_PATH"
