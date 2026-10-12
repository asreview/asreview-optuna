#!/bin/bash
#SBATCH --job-name=run_similarity_search_resume2
#SBATCH --output=logs/run_similarity_search_resume2_%A_%a.out
#SBATCH --error=logs/run_similarity_search_resume2_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --array=20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45%12

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Second resume round (2026-10-12) for the studies still under 500 completed
# trials: knn30 tasks 20-22 and soft_ess30 tasks 23-45 (24 and 38 included in
# case they are not finished; done studies exit immediately). Identical to
# run_similarity_search_resume.sh otherwise: --n-trials-total continues each
# study to 500 COMPLETED trials, so this is safe to resubmit.
# Several of these made no progress in the first resume round: check that
# round's logs for errors before submitting.
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
