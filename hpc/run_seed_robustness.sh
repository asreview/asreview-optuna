#!/bin/bash
#SBATCH --job-name=run_seed_robustness
#SBATCH --output=logs/run_seed_robustness_%A_%a.out
#SBATCH --error=logs/run_seed_robustness_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --array=0-41

module load 2025 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/optuna/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Seed-robustness check (paper/discussion.md / results.md open question): does
# the headline pattern hold under a different TPE sampler seed? Scoped to
# baseline + the best-performing axis (inclusion_ratio) + the
# worst-performing axis (baseline_loss), all 3 classifiers, seeds 43 and 44 --
# bracketing the observed effect range rather than an arbitrary/convenience
# sample. 3 classifiers x 7 study-sets x 2 seeds = 42 combos.
DATA_PATH="./synergy_plus"

CLASSIFIERS=(svm nb log)
STUDY_SETS=(
    "train"
    "train-inclusion_ratio-low"
    "train-inclusion_ratio-mid"
    "train-inclusion_ratio-high"
    "train-baseline_loss-low"
    "train-baseline_loss-mid"
    "train-baseline_loss-high"
)
SEEDS=(43 44)

# Flatten (classifier x study_set x seed) into one 42-entry array, indexed
# the same way regardless of which loop order -- classifier outermost,
# study_set middle, seed innermost.
N_STUDY_SETS=${#STUDY_SETS[@]}
N_SEEDS=${#SEEDS[@]}
PER_CLASSIFIER=$((N_STUDY_SETS * N_SEEDS))

CLF_IDX=$((SLURM_ARRAY_TASK_ID / PER_CLASSIFIER))
REM=$((SLURM_ARRAY_TASK_ID % PER_CLASSIFIER))
STUDY_SET_IDX=$((REM / N_SEEDS))
SEED_IDX=$((REM % N_SEEDS))

CLASSIFIER="${CLASSIFIERS[$CLF_IDX]}"
STUDY_SET="${STUDY_SETS[$STUDY_SET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

FEATURE_EXTRACTOR="tfidf"
BALANCER="ratio"
METRIC="loss"
N_TRIALS=500
N_WORKERS=$((SLURM_CPUS_PER_TASK - 1))

STUDY_NAME="[seed${SEED}] ${CLASSIFIER}-tfidf-ratio-${STUDY_SET}-loss"

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
            --seed "$SEED" \
            --study-name "$STUDY_NAME"
