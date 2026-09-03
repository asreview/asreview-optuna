#!/bin/bash
#SBATCH --job-name=run_stratum_resume_log
#SBATCH --output=logs/run_stratum_resume_log_%A_%a.out
#SBATCH --error=logs/run_stratum_resume_log_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --array=0-5

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# One-off top-up for the `log` classifier strata that hit the 24h wall-clock
# limit before reaching 500 trials (checked against the DB on 2026-09-02).
# STUDY_NAME resumes the exact existing study; N_TRIALS is the number of
# ADDITIONAL trials needed to reach 500 total, not the new total itself.
DATA_PATH="./synergy_plus"
STUDY_SETS=(
    "train-domain-health"
    "train-domain-nonhealth"
    "train-size-large"
    "train-inclusion_ratio-low"
    "train-inclusion_ratio-high"
    "train-protocol-no_protocol"
)
STUDY_NAMES=(
    "[Sep-02-15:24] log-tfidf-ratio-train-domain-health-loss"
    "[Sep-02-15:24] log-tfidf-ratio-train-domain-nonhealth-loss"
    "[Sep-02-15:24] log-tfidf-ratio-train-size-large-loss"
    "[Sep-02-15:24] log-tfidf-ratio-train-inclusion_ratio-low-loss"
    "[Sep-02-15:24] log-tfidf-ratio-train-inclusion_ratio-high-loss"
    "[Sep-02-15:24] log-tfidf-ratio-train-protocol-no_protocol-loss"
)
N_TRIALS_LIST=(13 143 141 294 241 299)

STUDY_SET="${STUDY_SETS[$SLURM_ARRAY_TASK_ID]}"
STUDY_NAME="${STUDY_NAMES[$SLURM_ARRAY_TASK_ID]}"
N_TRIALS="${N_TRIALS_LIST[$SLURM_ARRAY_TASK_ID]}"
CLASSIFIER="log"
FEATURE_EXTRACTOR="tfidf"
BALANCER="ratio"
METRIC="loss"
N_WORKERS=$((SLURM_CPUS_PER_TASK - 1))

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
            --study-name "$STUDY_NAME"
