#!/bin/bash
#SBATCH --job-name=run_stratum
#SBATCH --output=logs/run_stratum_%A_%a.out
#SBATCH --error=logs/run_stratum_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --array=0-15

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATA_PATH="./synergy_plus"
STUDY_SETS=(
    "train-domain-health"
    "train-domain-nonhealth"
    "train-size-small"
    "train-size-medium"
    "train-size-large"
    "train-inclusion_ratio-low"
    "train-inclusion_ratio-mid"
    "train-inclusion_ratio-high"
    "train-n_databases-low"
    "train-n_databases-mid"
    "train-n_databases-high"
    "train-protocol-protocol"
    "train-protocol-no_protocol"
    "train-baseline_loss-low"
    "train-baseline_loss-mid"
    "train-baseline_loss-high"
)
STUDY_SET="${STUDY_SETS[$SLURM_ARRAY_TASK_ID]}"
CLASSIFIER="log"
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
            --study-set "$STUDY_SET" \
            --classifier "$CLASSIFIER" \
            --feature-extractor "$FEATURE_EXTRACTOR" \
            --balancer "$BALANCER" \
            --n-trials "$N_TRIALS" \
            --parallelize-objective \
            --n-workers "$N_WORKERS" \
            --data-path "$DATA_PATH" \
            "${EXTRA_ARGS[@]}"
