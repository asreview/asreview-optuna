#!/bin/bash
#SBATCH --job-name=run_stratum_resume_log_2
#SBATCH --output=logs/run_stratum_resume_log_2_%A_%a.out
#SBATCH --error=logs/run_stratum_resume_log_2_%A_%a.err
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

# Second top-up round for `log`: covers the pooled baseline (never resumed
# after its own 24h timeout) plus the 3 stratum studies whose first resume
# attempt (run_stratum_resume_log.sh) got killed by the overnight Exoscale
# DB maintenance window (~05:00-05:02 on 2026-09-04, visible in the DB as a
# FAIL trial on each of the 3). Checked against the DB on 2026-09-04.
DATA_PATH="./synergy_plus"
STUDY_SETS=(
    "train"
    "train-inclusion_ratio-high"
    "train-inclusion_ratio-low"
    "train-protocol-no_protocol"
)
STUDY_NAMES=(
    "[Sep-02-15:13] log-tfidf-ratio-train-loss"
    "[Sep-02-15:24] log-tfidf-ratio-train-inclusion_ratio-high-loss"
    "[Sep-02-15:24] log-tfidf-ratio-train-inclusion_ratio-low-loss"
    "[Sep-02-15:24] log-tfidf-ratio-train-protocol-no_protocol-loss"
)
N_TRIALS_LIST=(276 127 202 211)

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
