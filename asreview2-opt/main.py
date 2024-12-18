import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

import asreview
from asreview.metrics import loss
from asreview.models.balance import Balanced
from asreview.models.feature_extraction import Tfidf
from asreview.models.query import MaxQuery
from models import optuna_studies_params, optuna_studies_models

# Study variables
PICKLE_FOLDER_PATH = Path("synergy-dataset", "pickles")
N_STUDIES = 260
CLASSIFIER_TYPE = "nb"  # Options: "nb", "log", "svm", "rf"
STUDY_NAME = "ASReview2 " + datetime.now().strftime("%Y-%m-%d at %H.%M.%S")
PARALLELIZE_OBJECTIVE = True

# Optuna variables
OPTUNA_N_TRIALS = 500
OPTUNA_TIMEOUT = None  # in seconds
OPTUNA_N_JOBS = 1 if PARALLELIZE_OBJECTIVE else mp.cpu_count() - 2

# Early stopping condition variables
MIN_TRIALS = 100
N_HISTORY = 5
STOPPING_THRESHOLD = 0.001

# list of studies
studies = pd.read_json("synergy_studies_1000.jsonl", lines=True).head(n=N_STUDIES)


# Function to run the loop in parallel
def run_parallel(studies, *args, **kwargs):
    losses = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count() - 2) as executor:
        # Submit tasks
        futures = {
            executor.submit(process_row, row, *args, **kwargs): i
            for i, row in studies.iterrows()
        }
        # Collect results
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                losses.append(result)
    return np.mean(losses)


# Function to run the loop in parallel
def run_sequential(studies, *args, **kwargs):
    losses = []
    for _, row in studies.iterrows():
        losses.append(process_row(row, *args, **kwargs))

    return np.mean(losses)


# Function to process each row
def process_row(row, params, ratio):
    with open(PICKLE_FOLDER_PATH / f"{row['dataset_id']}.pkl", "rb") as f:
        fm, labels = pickle.load(f)

    priors = row["prior_inclusions"] + row["prior_exclusions"]

    # Create balancer with optuna value
    blc = Balanced(ratio=ratio)

    # Create classifier with params
    clf = optuna_studies_models[CLASSIFIER_TYPE](**params)

    # Setup simulation
    n_query = 1 if row["dataset_id"] != "Walker_2018" else 50

    simulate = asreview.Simulate(
        fm=fm,
        labels=labels,
        classifier=clf,
        balance_strategy=blc,
        query_strategy=MaxQuery(),
        feature_extraction=Tfidf(),
        n_query=n_query,
    )

    # Set priors
    simulate.label(priors, prior=True)

    # Start simulation
    simulate.review()

    # Return loss
    padded_results = list(simulate._results["label"]) + [0] * (
        len(simulate.labels) - len(simulate._results["label"])
    )
    return loss(padded_results)


def objective(trial):
    # Use normal distribution for ratio (ratio effect is linear)
    ratio = trial.suggest_float("ratio", 1.0, 30.0)
    params = optuna_studies_params[CLASSIFIER_TYPE](trial)

    if PARALLELIZE_OBJECTIVE:
        return run_parallel(studies, params=params, ratio=ratio)
    else:
        return run_sequential(studies, params=params, ratio=ratio)


class StopWhenOptimumReached:
    def __init__(self, min_trials: int, threshold: float, n_history: int):
        self.min_trials = min_trials
        self.threshold = threshold
        self.n_history = n_history

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        # If there are more than min_trials completed, check early stopping condition
        if trial.number >= self.min_trials:
            # Take latest n_history trial loss values
            prev_trial_losses = [
                prev_trial.value for prev_trial in study.trials[-self.n_history :]
            ]

            # If the difference is smaller than threshold, we stop the entire study
            if max(prev_trial_losses) - min(prev_trial_losses) < self.threshold:
                study.stop()


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    study_stop_cb = StopWhenOptimumReached(
        min_trials=MIN_TRIALS, threshold=STOPPING_THRESHOLD, n_history=N_HISTORY
    )

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name=STUDY_NAME,
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        callbacks=[study_stop_cb],
        n_jobs=OPTUNA_N_JOBS,
    )

    print(f"Best value: {study.best_value} (params: {study.best_params})")
