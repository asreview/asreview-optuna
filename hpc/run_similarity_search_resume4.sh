#!/bin/bash
#SBATCH --job-name=run_similarity_search_resume4
#SBATCH --output=logs/run_similarity_search_resume4_%A_%a.out
#SBATCH --error=logs/run_similarity_search_resume4_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --exclude=tcn530
#SBATCH --time=8:00:00
#SBATCH --array=25,26,27,29,30,35,36

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Fourth resume round: the 7 soft_ess30 searches still under 500 completed
# trials after round 3 (Boersma-van_Dam_2024, Bos_2018, Bosch_2021, Chueca_2023,
# Donners_2021, Lewowski_2021, Menon_2022; 290-348 trials). They gain ~7
# trials/hour, so the slowest may need one more resubmit of this script.
#
# Third resume round (2026-10-12): as round 2, minus tasks 44-45 (soft Walker,
# van der Waal), which are still running from round 2. Round 2 lost its other
# tasks to instant system cancellations (CANCELLED by 0 after 1s, no logs; task
# 32 on tcn530), so that node is excluded.
#
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
