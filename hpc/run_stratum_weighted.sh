#!/bin/bash
#SBATCH --job-name=run_stratum_weighted
#SBATCH --output=logs/run_stratum_weighted_%A_%a.out
#SBATCH --error=logs/run_stratum_weighted_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=48:00:00
#SBATCH --array=0-14

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATA_PATH="./synergy_plus"
# Pilot: fixed upweight factors on two axes before committing to a weighting
# rule for all strata. Task i -> stratum i / N_FACTORS, factor i % N_FACTORS.
STUDY_SETS=(
    "train-domain-health"
    "train-domain-nonhealth"
    "train-n_databases-low"
    "train-n_databases-mid"
    "train-n_databases-high"
)
UPWEIGHT_FACTORS=(2 5 10)
N_FACTORS=${#UPWEIGHT_FACTORS[@]}
UPWEIGHT_STUDY_SET="${STUDY_SETS[$((SLURM_ARRAY_TASK_ID / N_FACTORS))]}"
UPWEIGHT_FACTOR="${UPWEIGHT_FACTORS[$((SLURM_ARRAY_TASK_ID % N_FACTORS))]}"
# ---- EDIT THIS before every submission: "svm", "nb", or "log" ----
CLASSIFIER="svm"
# --------------------------------------------------------------------
FEATURE_EXTRACTOR="tfidf"
BALANCER="ratio"
METRIC="loss"
N_TRIALS=500
STUDY_NAME=""
N_WORKERS=$((SLURM_CPUS_PER_TASK - 1))

EXTRA_ARGS=()
if [[ -n "$STUDY_NAME" ]]; then
    EXTRA_ARGS+=(--study-name "$STUDY_NAME")
fi

srun -n 1 python ./src/main.py \
            --metric "$METRIC" \
            --study-set train \
            --upweight-study-set "$UPWEIGHT_STUDY_SET" \
            --upweight-factor "$UPWEIGHT_FACTOR" \
            --classifier "$CLASSIFIER" \
            --feature-extractor "$FEATURE_EXTRACTOR" \
            --balancer "$BALANCER" \
            --n-trials "$N_TRIALS" \
            --parallelize-objective \
            --n-workers "$N_WORKERS" \
            --data-path "$DATA_PATH" \
            "${EXTRA_ARGS[@]}"
