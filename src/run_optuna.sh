#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=48:00:00

# Load modules (modify these as needed)
module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

echo "Running..."
srun -n 1 python ./hpc/run_optuna.py

# Run Optuna script
#echo "Running Optuna optimization..."
#
#srun --ntasks=47 --cpus-per-task=1 \
#     python ./hpc/run_optuna.py --n_trials 5000 --n_jobs 1 &
#
#wait

cleanup
