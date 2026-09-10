#!/bin/bash
#SBATCH --job-name=run_seed_robustness_resume
#SBATCH --output=logs/run_seed_robustness_resume_%A_%a.out
#SBATCH --error=logs/run_seed_robustness_resume_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --array=0-7

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Resume the 8 seed-robustness studies that timed out on their first
# 24h run (checked against the live DB on 2026-09-10). N_TRIALS is the
# exact shortfall to reach 500 completed trials as of that check -- may
# need another top-up round if any further trials fail/timeout, same as
# the rest of this project's history with `log`.
DATA_PATH="./synergy_plus"

# classifier|study_set|seed|n_trials_needed
COMBOS=(
    "log|train-inclusion_ratio-high|43|258"
    "log|train-inclusion_ratio-low|43|277"
    "log|train|43|213"
    "nb|train|43|33"
    "log|train-baseline_loss-high|44|1"
    "log|train-inclusion_ratio-high|44|115"
    "log|train-inclusion_ratio-low|44|300"
    "log|train|44|382"
)

IFS='|' read -r CLASSIFIER STUDY_SET SEED N_TRIALS <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

FEATURE_EXTRACTOR="tfidf"
BALANCER="ratio"
METRIC="loss"
N_WORKERS=$((SLURM_CPUS_PER_TASK - 1))

STUDY_NAME="[seed${SEED}] ${CLASSIFIER}-tfidf-ratio-${STUDY_SET}-loss"

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
            --seed "$SEED" \
            --study-name "$STUDY_NAME"
