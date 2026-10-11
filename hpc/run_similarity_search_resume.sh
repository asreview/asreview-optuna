#!/bin/bash
#SBATCH --job-name=run_similarity_search_resume
#SBATCH --output=logs/run_similarity_search_resume_%A_%a.out
#SBATCH --error=logs/run_similarity_search_resume_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=12:00:00
#SBATCH --array=0-45

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Resume for hpc/run_similarity_search.sh: same tasks and study names, but
# --n-trials-total continues each study only until it has 500 COMPLETED trials
# (studies already there exit immediately), so this is safe to resubmit until
# everything is done. The first round reached 163-500 trials (knn30) and
# 120-274 (soft_ess30) in 5h, so the two halves need different time limits:
#   sbatch --array=0-22  --time=12:00:00 hpc/run_similarity_search_resume.sh   # knn30
#   sbatch --array=23-45 --time=24:00:00 hpc/run_similarity_search_resume.sh   # soft_ess30
DATA_PATH="./synergy_plus"
TARGETS=($(cat src/studies/similarity_search_targets.txt))  # ids contain no spaces
N_TARGETS=${#TARGETS[@]}
TARGET="${TARGETS[$((SLURM_ARRAY_TASK_ID % N_TARGETS))]}"

CLASSIFIER="svm"
FEATURE_EXTRACTOR="tfidf"
BALANCER="ratio"
METRIC="loss"
N_TRIALS_TOTAL=500
SEED=42
N_WORKERS=$((SLURM_CPUS_PER_TASK - 1))

if (( SLURM_ARRAY_TASK_ID < N_TARGETS )); then
    SCHEME="knn30_eligibility"
    SCHEME_ARGS=(--study-set "train-knn30_eligibility-${TARGET}")
else
    SCHEME="soft_ess30_eligibility"
    SCHEME_ARGS=(--study-set train --dataset-weights "src/studies/weights/soft_ess30_eligibility-${TARGET}.json")
fi
STUDY_NAME="[simsearch] ${CLASSIFIER}-tfidf-ratio-${SCHEME}-${TARGET}-loss"

srun -n 1 python ./src/main.py \
            --metric "$METRIC" \
            "${SCHEME_ARGS[@]}" \
            --classifier "$CLASSIFIER" \
            --feature-extractor "$FEATURE_EXTRACTOR" \
            --balancer "$BALANCER" \
            --n-trials-total "$N_TRIALS_TOTAL" \
            --parallelize-objective \
            --n-workers "$N_WORKERS" \
            --data-path "$DATA_PATH" \
            --seed "$SEED" \
            --study-name "$STUDY_NAME"
