#!/bin/bash
#SBATCH --job-name=run_stratum
#SBATCH --output=logs/run_stratum_%A_%a.out
#SBATCH --error=logs/run_stratum_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --array=0-10

# Load modules (modify these as needed)
module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

DATA_PATH="./synergy_plus"

# ---- Edit these, then submit ONCE: sbatch ./hpc/run_stratum.sh ----
# Every entry below runs as its own array task -- its own job allocation, its
# own log file -- in a single submission. No need to edit-and-resubmit per
# stratum. To retry/rerun just one, submit with an override, e.g.
# `sbatch --array=3 ./hpc/run_stratum.sh`.
#
# IMPORTANT: keep #SBATCH --array above equal to 0-(N-1) for N entries here --
# Slurm parses #SBATCH lines before this script runs, so it can't be derived
# from the list automatically. Currently 16 entries -> 0-15.
STUDY_SETS=(
    #"train-domain-health"
    #"train-domain-nonhealth"
    #"train-size-small"
    #"train-size-medium"
    #"train-size-large"
    "train-inclusion_ratio-low"
    "train-inclusion_ratio-mid"
    "train-inclusion_ratio-high"
    "train-n_databases-low"
    "train-n_databases-mid"
    "train-n_databases-high"
    "train-protocol-protocol"
    "train-protocol-no_protocol"
    "train-baseline_loss-low"
    "train-baseline_loss-mid"
    "train-baseline_loss-high"
)
STUDY_SET="${STUDY_SETS[$SLURM_ARRAY_TASK_ID]}"
CLASSIFIER="svm"
FEATURE_EXTRACTOR="tfidf"
BALANCER="ratio"
METRIC="loss"
N_TRIALS=500
# Shared across every array task -- leave empty to always start fresh studies.
# Per-task resuming isn't supported here; fall back to hpc/run_tfidf.sh's
# single-job pattern if you need to resume one specific stratum's study.
STUDY_NAME=""
# ------------------------------------------------------------------------------------------------

# Derived from --cpus-per-task above, not hardcoded, so it can't drift out of sync.
N_WORKERS=$((SLURM_CPUS_PER_TASK - 1))

EXTRA_ARGS=()
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
