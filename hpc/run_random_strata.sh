#!/bin/bash
#SBATCH --job-name=run_random_strata
#SBATCH --output=logs/run_random_strata_%A_%a.out
#SBATCH --error=logs/run_random_strata_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --array=0-35

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Random-strata control (RERUNS.md): tune on size-matched random subsets of the
# train split, to test whether metadata strata do any better than random
# subsets of the same size. Study sets come from src/generate_random_strata.py.
# 3 classifiers x 4 sizes x 3 draws = 36 combos, sampler seed 42 to match the
# main design.
DATA_PATH="./synergy_plus"

CLASSIFIERS=(svm nb log)
SIZES=(21 30 40 50)
N_DRAWS=3

PER_CLASSIFIER=$((${#SIZES[@]} * N_DRAWS))
CLF_IDX=$((SLURM_ARRAY_TASK_ID / PER_CLASSIFIER))
REM=$((SLURM_ARRAY_TASK_ID % PER_CLASSIFIER))
SIZE_IDX=$((REM / N_DRAWS))
DRAW=$((REM % N_DRAWS))

CLASSIFIER="${CLASSIFIERS[$CLF_IDX]}"
STUDY_SET="train-random-n${SIZES[$SIZE_IDX]}-r${DRAW}"

FEATURE_EXTRACTOR="tfidf"
BALANCER="ratio"
METRIC="loss"
N_TRIALS=500
SEED=42
N_WORKERS=$((SLURM_CPUS_PER_TASK - 1))

STUDY_NAME="[random] ${CLASSIFIER}-tfidf-ratio-${STUDY_SET}-loss"

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
