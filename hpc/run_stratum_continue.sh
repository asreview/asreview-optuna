#!/bin/bash
#SBATCH --job-name=run_stratum_continue
#SBATCH --output=logs/run_stratum_continue_%A_%a.out
#SBATCH --error=logs/run_stratum_continue_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --array=0-15

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

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
STUDY_NAMES=(
    "[Aug-28-10:46] svm-tfidf-ratio-train-domain-health-loss"
    "[Aug-28-10:48] svm-tfidf-ratio-train-domain-nonhealth-loss"
    "[Aug-28-10:49] svm-tfidf-ratio-train-size-small-loss"
    "[Aug-28-10:50] svm-tfidf-ratio-train-size-medium-loss"
    "[Aug-28-10:50] svm-tfidf-ratio-train-size-large-loss"
    "[Aug-31-12:28] svm-tfidf-ratio-train-inclusion_ratio-low-loss"
    "[Aug-31-12:28] svm-tfidf-ratio-train-inclusion_ratio-mid-loss"
    "[Aug-31-12:28] svm-tfidf-ratio-train-inclusion_ratio-high-loss"
    "[Aug-31-12:28] svm-tfidf-ratio-train-n_databases-low-loss"
    "[Aug-31-12:28] svm-tfidf-ratio-train-n_databases-mid-loss"
    "[Aug-31-12:28] svm-tfidf-ratio-train-n_databases-high-loss"
    "[Aug-31-12:28] svm-tfidf-ratio-train-protocol-protocol-loss"
    "[Aug-31-12:28] svm-tfidf-ratio-train-protocol-no_protocol-loss"
    "[Aug-31-12:28] svm-tfidf-ratio-train-baseline_loss-low-loss"
    "[Aug-31-12:28] svm-tfidf-ratio-train-baseline_loss-mid-loss"
    "[Aug-31-12:28] svm-tfidf-ratio-train-baseline_loss-high-loss"
)
STUDY_SET="${STUDY_SETS[$SLURM_ARRAY_TASK_ID]}"
STUDY_NAME="${STUDY_NAMES[$SLURM_ARRAY_TASK_ID]}"
CLASSIFIER="svm"
FEATURE_EXTRACTOR="tfidf"
BALANCER="ratio"
METRIC="loss"
N_TRIALS=500
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
