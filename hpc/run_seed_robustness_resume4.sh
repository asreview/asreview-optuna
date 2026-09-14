#!/bin/bash
#SBATCH --job-name=run_seed_robustness_resume4
#SBATCH --output=logs/run_seed_robustness_resume4_%A_%a.out
#SBATCH --error=logs/run_seed_robustness_resume4_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --array=0-3

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Fourth resume round: these 4 finished their previous resume run without
# quite reaching 500 (small trial-count shortfalls) -- checked against the
# live DB on 2026-09-14. Should be quick given how close they are.
DATA_PATH="./synergy_plus"

# classifier|study_set|seed|n_trials_needed
COMBOS=(
    "log|train-inclusion_ratio-high|43|21"
    "log|train-inclusion_ratio-low|43|77"
    "log|train-inclusion_ratio-low|44|133"
    "log|train|44|42"
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
