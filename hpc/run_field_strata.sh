#!/bin/bash
#SBATCH --job-name=run_field_strata
#SBATCH --output=logs/run_field_strata_%A_%a.out
#SBATCH --error=logs/run_field_strata_%A_%a.err
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

# New `field` axis (medicine / non_medicine) x all 3 classifiers already run
# for the other 6 axes. No new pooled baseline needed -- compare_strata.py
# reuses each classifier's existing baseline study. N_TRIALS per task matches
# that classifier's established budget for the other axes (svm=1000,
# nb=500, log=500), so field is apples-to-apples comparable within each
# classifier rather than standardized across them.
DATA_PATH="./synergy_plus"
STUDY_SETS=(
    "train-field-medicine"
    "train-field-non_medicine"
    "train-field-medicine"
    "train-field-non_medicine"
    "train-field-medicine"
    "train-field-non_medicine"
)
CLASSIFIERS=(svm svm nb nb log log)
N_TRIALS_LIST=(500 500 500 500 500 500)

STUDY_SET="${STUDY_SETS[$SLURM_ARRAY_TASK_ID]}"
CLASSIFIER="${CLASSIFIERS[$SLURM_ARRAY_TASK_ID]}"
N_TRIALS="${N_TRIALS_LIST[$SLURM_ARRAY_TASK_ID]}"
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
            --data-path "$DATA_PATH"
