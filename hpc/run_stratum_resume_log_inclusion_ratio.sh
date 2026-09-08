#!/bin/bash
#SBATCH --job-name=run_stratum_resume_log_inclusion_ratio
#SBATCH --output=logs/run_stratum_resume_log_inclusion_ratio_%j.out
#SBATCH --error=logs/run_stratum_resume_log_inclusion_ratio_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Resume for `log`'s last remaining straggler: inclusion_ratio-low, dead again
# (364/500 complete) after a dropped DB connection during Optuna's
# trial-finalization commit killed the whole process (see .err log:
# psycopg2.OperationalError: SSL SYSCALL error: EOF detected). Checked
# against the DB on 2026-09-08.
DATA_PATH="./synergy_plus"
STUDY_SET="train-inclusion_ratio-low"
STUDY_NAME="[Sep-02-15:24] log-tfidf-ratio-train-inclusion_ratio-low-loss"
N_TRIALS=136
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
