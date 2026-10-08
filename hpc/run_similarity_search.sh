#!/bin/bash
#SBATCH --job-name=run_similarity_search
#SBATCH --output=logs/run_similarity_search_%A_%a.out
#SBATCH --error=logs/run_similarity_search_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=5:00:00
#SBATCH --array=0-45

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Real searches for the similarity-weighted schemes that
# analysis/build_similarity_reselection.py only re-selected, one search per
# test review (inputs from analysis/generate_similarity_studies.py, eligibility
# similarity). Tasks 0-22: knn30 (search on the review's 30 nearest train
# reviews, ~1/3 of a pooled run). Tasks 23-45: soft_ess30 (search on all 91
# train reviews with softmax weights, about a pooled run; may need a resume).
# Fixed study names, so a resubmit with the same name continues the study.
DATA_PATH="./synergy_plus"
TARGETS=($(cat src/studies/similarity_search_targets.txt))  # ids contain no spaces
N_TARGETS=${#TARGETS[@]}
TARGET="${TARGETS[$((SLURM_ARRAY_TASK_ID % N_TARGETS))]}"

CLASSIFIER="svm"
FEATURE_EXTRACTOR="tfidf"
BALANCER="ratio"
METRIC="loss"
N_TRIALS=500
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
            --n-trials "$N_TRIALS" \
            --parallelize-objective \
            --n-workers "$N_WORKERS" \
            --data-path "$DATA_PATH" \
            --seed "$SEED" \
            --study-name "$STUDY_NAME"
