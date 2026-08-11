#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=48:00:00

# Load modules (modify these as needed)
module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export DATA_PATH="./synergy_plus"
export DB_URI=""

# ---- Edit these, then submit: sbatch ./src/run_single.sh ----
STUDY_SET="train"
CLASSIFIER="svm"
FEATURE_EXTRACTOR="tfidf"
METRIC="loss"
N_TRIALS=2
# ------------------------------------------------------------------------------------------------

# Derived from --cpus-per-task above, not hardcoded, so it can't drift out of sync.
N_WORKERS=$((SLURM_CPUS_PER_TASK - 1))

# mxbai/multilingual-e5 have no on-the-fly implementation and require
# precomputed embeddings; tfidf is tuned every trial and must NOT be
# pointed at (nonexistent) precomputed features.
EXTRA_ARGS=()
if [[ "$FEATURE_EXTRACTOR" == "mxbai" || "$FEATURE_EXTRACTOR" == "multilingual-e5" ]]; then
    EXTRA_ARGS+=(--pre-processed-fms)
fi

srun -n 1 python main.py \
            --metric "$METRIC" \
            --study-set "$STUDY_SET" \
            --classifier "$CLASSIFIER" \
            --feature-extractor "$FEATURE_EXTRACTOR" \
            --n-trials "$N_TRIALS" \
            --parallelize-objective \
            --n-workers "$N_WORKERS" \
            --data-path "$DATA_PATH" \
            "${EXTRA_ARGS[@]}"