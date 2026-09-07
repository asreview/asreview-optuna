#!/bin/bash
#SBATCH --job-name=run_stratum_resume_log_3
#SBATCH --output=logs/run_stratum_resume_log_3_%A_%a.out
#SBATCH --error=logs/run_stratum_resume_log_3_%A_%a.err
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

# Third top-up round for `log`: the last 2 studies (of the original 6 axes)
# still short of 500 completed trials, unrelated to the trial-cap change --
# these were just never resumed after the last DB-maintenance casualty round.
# Checked against the DB on 2026-09-07.
DATA_PATH="./synergy_plus"
STUDY_SETS=(
    "train-inclusion_ratio-low"
    "train-protocol-no_protocol"
)
STUDY_NAMES=(
    "[Sep-02-15:24] log-tfidf-ratio-train-inclusion_ratio-low-loss"
    "[Sep-02-15:24] log-tfidf-ratio-train-protocol-no_protocol-loss"
)
N_TRIALS_LIST=(201 40)

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
