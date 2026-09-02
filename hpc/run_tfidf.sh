#!/bin/bash
#SBATCH --job-name=run_tfidf
#SBATCH --output=logs/run_tfidf_%j.out
#SBATCH --error=logs/run_tfidf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00

# Load modules (modify these as needed)
module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

DATA_PATH="./synergy_plus"

# ---- Edit these, then submit: sbatch ./src/run_single.sh ----
STUDY_SET="train"
CLASSIFIER="log"
FEATURE_EXTRACTOR="tfidf"
BALANCER="ratio"
METRIC="loss"
N_TRIALS=500
# Leave empty to start a fresh study. To continue a previous run, paste its exact
# study_name here (printed at the top of that run's log) and set N_TRIALS to the
# number of ADDITIONAL trials to run, not the new total. Must be empty when switching
# BALANCER to a value that study wasn't tuned with -- resuming under a different
# balancer mixes incompatible hyperparameters into one study.
STUDY_NAME=""
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
            --balancer "$BALANCER" \
            --n-trials "$N_TRIALS" \
            --parallelize-objective \
            --n-workers "$N_WORKERS" \
            --data-path "$DATA_PATH" \
            "${EXTRA_ARGS[@]}"