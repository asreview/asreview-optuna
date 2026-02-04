#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=48:00:00

# Load modules (modify these as needed)
module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

# --pre-processed-fms \
srun -n 1 python main.py \
            --metric loss \
            --study-set demo \
            --classifier svm \
            --feature-extractor tfidf \
            --n-trials 20 \
            --parallelize-objective \
            --n-workers 47 \
            --data-path ./data \
            --fms-path ./preprocessed_fms \
            --studies-path ./studies