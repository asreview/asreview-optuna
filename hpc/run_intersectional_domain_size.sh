#!/bin/bash
#SBATCH --job-name=run_intersectional_domain_size
#SBATCH --output=logs/run_intersectional_domain_size_%A_%a.out
#SBATCH --error=logs/run_intersectional_domain_size_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --array=0-5

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Intersectional domain x size HPO -- "are we missing anything" check from
# paper/results.md's open items: 6 cells, SVM only, scoped down from the
# full 3-classifier x 2-pair version to keep this at roughly the cost of one
# already-completed single-axis experiment (~3000 trials). Cells generated
# via src/generate_intersectional_studies.py --axis-a domain --axis-b size.
DATA_PATH="./synergy_plus"
STUDY_SETS=(
    "train-domain_size-health-small"
    "train-domain_size-health-medium"
    "train-domain_size-health-large"
    "train-domain_size-nonhealth-small"
    "train-domain_size-nonhealth-medium"
    "train-domain_size-nonhealth-large"
)
STUDY_SET="${STUDY_SETS[$SLURM_ARRAY_TASK_ID]}"
CLASSIFIER="svm"
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
