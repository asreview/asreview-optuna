#!/bin/bash
#SBATCH --job-name=run_tfidf
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00

# Load modules (modify these as needed)
module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

if [ -f .env ]; then set -a; source .env; set +a; fi

export DATA_PATH="./synergy_plus"

# ---- Edit these, then submit: sbatch ./src/run_single.sh ----
STUDY_SET="train"
CLASSIFIER="svm"
FEATURE_EXTRACTOR="tfidf"
METRIC="loss"
N_TRIALS=500
# Leave empty to start a fresh study. To continue a previous run, paste its exact
# study_name here (printed at the top of that run's log) and set N_TRIALS to the
# number of ADDITIONAL trials to run, not the new total.
STUDY_NAME="[Aug-11-16:40] svm-tfidf-train-loss"
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
if [[ -n "$STUDY_NAME" ]]; then
    EXTRA_ARGS+=(--study-name "$STUDY_NAME")
fi

srun -n 1 python ./src/main.py \
            --metric "$METRIC" \
            --study-set "$STUDY_SET" \
            --classifier "$CLASSIFIER" \
            --feature-extractor "$FEATURE_EXTRACTOR" \
            --n-trials "$N_TRIALS" \
            --parallelize-objective \
            --n-workers "$N_WORKERS" \
            --data-path "$DATA_PATH" \
            "${EXTRA_ARGS[@]}"