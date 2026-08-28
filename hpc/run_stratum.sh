#!/bin/bash
#SBATCH --job-name=run_stratum
#SBATCH --output=logs/run_stratum_%j.out
#SBATCH --error=logs/run_stratum_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00

# Load modules (modify these as needed)
module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

DATA_PATH="./synergy_plus"

# ---- Edit these, then submit: sbatch ./hpc/run_stratum.sh ----
# STUDY_SET must name a stratum produced by generate_studies.py --stratify-by, e.g.
# 
STUDY_SET="train-domain-health" # train-domain-health, train-domain-nonhealth, train-size-small, train-size-medium, train-size-large
CLASSIFIER="svm"
FEATURE_EXTRACTOR="tfidf"
BALANCER="ratio"
METRIC="loss"
N_TRIALS=500
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
