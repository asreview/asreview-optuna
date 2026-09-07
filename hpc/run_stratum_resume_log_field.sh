#!/bin/bash
#SBATCH --job-name=run_stratum_resume_log_field
#SBATCH --output=logs/run_stratum_resume_log_field_%A_%a.out
#SBATCH --error=logs/run_stratum_resume_log_field_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --array=0-1

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Resume for `log`'s 2 field strata (medicine / non_medicine), both timed out
# with an orphaned RUNNING trial (~2 days old). Checked against the DB on
# 2026-09-07.
DATA_PATH="./synergy_plus"
STUDY_SETS=(
    "train-field-medicine"
    "train-field-non_medicine"
)
STUDY_NAMES=(
    "[Sep-04-11:03] log-tfidf-ratio-train-field-medicine-loss"
    "[Sep-04-11:03] log-tfidf-ratio-train-field-non_medicine-loss"
)
N_TRIALS_LIST=(84 200)

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
