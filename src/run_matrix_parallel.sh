#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=48:00:00

# Load modules (modify these as needed)
module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

# DATA_PATH must point at the synergy_plus data directory, e.g.:
#   sbatch --export=DATA_PATH="/path/to/synergy_plus",DB_URI="..." ./src/run_matrix_parallel.sh

CLASSIFIERS=("svm" "rf" "log" "nb")
FEATURES=("tfidf" "mxbai" "multilingual-e5")

for clf in "${CLASSIFIERS[@]}"; do
    for feat in "${FEATURES[@]}"; do
        echo "Starting study with classifier=$clf and feature-extractor=$feat"

        # --pre-processed-fms \
        srun -n 1 python main.py \
            --metric loss \
            --study-set train \
            --classifier "$clf" \
            --feature-extractor "$feat" \
            --n-trials 5 \
            --parallelize-objective \
            --n-workers 47 \
            --data-path "$DATA_PATH" &
        
        # Limit to 4 parallel jobs at a time
        while [ "$(jobs -r | wc -l)" -ge 4 ]; do
            sleep 5
        done
    done
done

wait
echo "All studies completed!"