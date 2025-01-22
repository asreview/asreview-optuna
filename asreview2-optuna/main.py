import os
from collections import defaultdict
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import synergy_dataset as sd

import asreview
from asreview.metrics import loss
from asreview.models.balance import Balanced
from asreview.models.query import MaxQuery
from asreview.learner import ActiveLearningCycle
from classifiers import classifier_params, classifiers
from feature_extractors import feature_extractor_params, feature_extractors

# Study variables
VERSION = 1
STUDY_SET = "demo"
PICKLE_FOLDER_PATH = Path("synergy-dataset", "pickles")
CLASSIFIER_TYPE = "svm"  # Options: "nb", "log", "svm", "rf"
FEATURE_EXTRACTOR_TYPE = "tfidf"  # Options: "tfidf", "onehot"
PRE_PROCESSED_FMS = False  # False = on the fly
PARALLELIZE_OBJECTIVE = True

# Optuna variables
OPTUNA_N_TRIALS = 500
OPTUNA_TIMEOUT = None  # in seconds
OPTUNA_N_JOBS = 1 if PARALLELIZE_OBJECTIVE else mp.cpu_count() - 1

# Early stopping condition variables
MIN_TRIALS = 400
N_HISTORY = 5
STOPPING_THRESHOLD = 0.0001

dataset_sizes = {
    dataset.name: dataset.metadata["data"]["n_records"]
    for dataset in sd.iter_datasets()
}


def load_dataset(dataset_id):
    if dataset_id == "Moran_2021":
        return pd.read_csv(Path("datasets", "Moran_2021_corrected_shuffled_raw.csv"))

    return sd.Dataset(dataset_id).to_frame().reset_index()


def n_query(results, n_records):
    if len(results) >= max(1000, round(0.1 * n_records)):
        return 100
    elif len(results) >= max(500, round(0.05 * n_records)):
        return 10
    else:
        return 1


def n_query_extreme(results, n_records):
    if n_records >= 10000:
        if len(results) >= 10000:
            return 10**5  # finish the run
        if len(results) >= 1000:
            return 1000
        elif len(results) >= 100:
            return 25
        else:
            return 1
    else:
        if len(results) >= 1000:
            return 100
        elif len(results) >= 100:
            return 5
        else:
            return 1


# Function to run the loop in parallel
def run_parallel(studies, *args, **kwargs):
    losses = defaultdict(list)
    with ProcessPoolExecutor(max_workers=mp.cpu_count() - 1) as executor:
        # Submit tasks
        futures = {
            executor.submit(process_row, row, *args, **kwargs): i
            for i, row in studies.iterrows()
        }
        # Collect results
        for future in as_completed(futures):
            dataset_id, result = future.result()
            if result is not None:
                losses[dataset_id].append(result)
    return losses


# Function to run the loop in parallel
def run_sequential(studies, *args, **kwargs):
    losses = defaultdict(list)
    for _, row in studies.iterrows():
        dataset_id, result = process_row(row, *args, **kwargs)
        losses[dataset_id].append(result)

    return losses


# Function to process each row
def process_row(row, clf_params, fe_params, ratio):
    priors = row["prior_inclusions"] + row["prior_exclusions"]

    # Create balancer with optuna value
    blc = Balanced(ratio=ratio)

    # Create classifier and feature extractor with params
    clf = classifiers[CLASSIFIER_TYPE](**clf_params)
    fe = feature_extractors[FEATURE_EXTRACTOR_TYPE](**fe_params)

    if PRE_PROCESSED_FMS:
        with open(PICKLE_FOLDER_PATH / f"{row['dataset_id']}.pkl", "rb") as f:
            X, labels = pickle.load(f)

        alc = ActiveLearningCycle(
            query_strategy=MaxQuery(),
            classifier=clf,
            balance_strategy=blc,
            n_query=lambda results: n_query_extreme(results, X.shape[0]),
        )
    else:
        X = load_dataset(row["dataset_id"])

        labels = X["label_included"]
        fe = feature_extractors[FEATURE_EXTRACTOR_TYPE](**fe_params)

        alc = ActiveLearningCycle(
            query_strategy=MaxQuery(),
            classifier=clf,
            balance_strategy=blc,
            feature_extraction=fe,
            n_query=lambda results: n_query_extreme(results, X.shape[0]),
        )

    simulate = asreview.Simulate(
        X=X,
        labels=labels,
        learners=[alc],
    )

    # Set priors
    simulate.label(priors)

    # Start simulation
    simulate.review()

    # Return loss
    padded_results = list(simulate._results["label"]) + [0] * (
        len(simulate.labels) - len(simulate._results["label"])
    )
    return row["dataset_id"], loss(padded_results)


def objective_report(report_order):
    def objective(trial):
        # Use normal distribution for ratio (ratio effect is linear)
        ratio = trial.suggest_float("ratio", 1.0, 5.0)
        # ratio = 1.5
        clf_params = classifier_params[CLASSIFIER_TYPE](trial)
        fe_params = feature_extractor_params[FEATURE_EXTRACTOR_TYPE](trial)

        if PARALLELIZE_OBJECTIVE:
            result = run_parallel(
                studies, clf_params=clf_params, fe_params=fe_params, ratio=ratio
            )
        else:
            result = run_sequential(
                studies, clf_params=clf_params, fe_params=fe_params, ratio=ratio
            )

        all_losses = []
        for i, dataset_id in enumerate(report_order):
            losses = result[dataset_id] if dataset_id in result else [0]
            trial.report(np.mean(losses), i)
            all_losses += losses

        return np.mean(all_losses)

    return objective


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
    # list of studies
    studies = pd.read_json(f"synergy_studies_{STUDY_SET}.jsonl", lines=True)
    report_order = sorted(set(studies["dataset_id"]))

    sampler = optuna.samplers.TPESampler()
    study_stop_cb = StopWhenOptimumReached(
        min_trials=MIN_TRIALS, threshold=STOPPING_THRESHOLD, n_history=N_HISTORY
    )

    study = optuna.create_study(
        storage=os.getenv(
            "DB_URI", "sqlite:///db.sqlite3"
        ),  # Specify the storage URL here.
        study_name=f"ASReview2-{STUDY_SET}-{CLASSIFIER_TYPE}-{VERSION}",
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    study.optimize(
        objective_report(report_order),
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        callbacks=[study_stop_cb],
        n_jobs=OPTUNA_N_JOBS,
    )

    print(f"Best value: {study.best_value} (params: {study.best_params})")
