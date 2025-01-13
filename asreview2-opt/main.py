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
from classifiers import classifier_params, classifiers
from feature_extractors import feature_extractor_params, feature_extractors

# Study variables
VERSION = 1
PICKLE_FOLDER_PATH = Path("synergy-dataset", "pickles")
N_STUDIES = 260
CLASSIFIER_TYPE = "nb"  # Options: "nb", "log", "svm", "rf"
FEATURE_EXTRACTOR_TYPE = "tfidf"
PRE_PROCESSED_FMS = False  # False = on the fly
PARALLELIZE_OBJECTIVE = True

# Optuna variables
OPTUNA_N_TRIALS = 500
OPTUNA_TIMEOUT = None  # in seconds
OPTUNA_N_JOBS = 1 if PARALLELIZE_OBJECTIVE else mp.cpu_count() - 2

# Early stopping condition variables
MIN_TRIALS = 100
N_HISTORY = 5
STOPPING_THRESHOLD = 0.001

dataset_sizes = {
    dataset.name: dataset.metadata["data"]["n_records"]
    for dataset in sd.iter_datasets()
}


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


def sort_studies(studies, dataset_sizes):
    studies_sorter = sorted(dataset_sizes.items(), key=lambda x: x[1], reverse=True)

    return studies.sort_values(
        "dataset_id",
        key=lambda x: x.map({val[0]: i for i, val in enumerate(studies_sorter)}),
    )


# Function to run the loop in parallel
def run_parallel(studies, *args, **kwargs):
    losses = defaultdict(list)
    with ProcessPoolExecutor(max_workers=mp.cpu_count() - 2) as executor:
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
    # If True, use pre processed feature matrices. Else determine fm on the fly
    if PRE_PROCESSED_FMS:
        with open(PICKLE_FOLDER_PATH / f"{row['dataset_id']}.pkl", "rb") as f:
            fm, labels = pickle.load(f)
    else:
        df = sd.Dataset(row["dataset_id"]).to_frame().reset_index()
        fe = feature_extractors[FEATURE_EXTRACTOR_TYPE](**fe_params)

        fm = fe.fit_transform(
            df["title"].fillna("").values, df["abstract"].fillna("").values
        )
        labels = df["label_included"]

    priors = row["prior_inclusions"] + row["prior_exclusions"]

    # Create balancer with optuna value
    blc = Balanced(ratio=ratio)

    # Create classifier and feature extractor with params
    clf = classifiers[CLASSIFIER_TYPE](**clf_params)
    fe = feature_extractors[FEATURE_EXTRACTOR_TYPE](**fe_params)

    simulate = asreview.Simulate(
        fm=fm,
        labels=labels,
        classifier=clf,
        balance_strategy=blc,
        query_strategy=MaxQuery(),
        feature_extraction=fe,
        n_query=lambda x: n_query_extreme(x, dataset_sizes[row["dataset_id"]]),
    )

    # Set priors
    simulate.label(priors, prior=True)

    # Start simulation
    simulate.review()

    # Return loss
    padded_results = list(simulate._results["label"]) + [0] * (
        len(simulate.labels) - len(simulate._results["label"])
    )
    return row["dataset_id"], loss(padded_results)


def objective(trial):
    # Use normal distribution for ratio (ratio effect is linear)
    # ratio = trial.suggest_float("ratio", 1.0, 5.0)
    ratio = 1.5
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
    for i, dataset_id in enumerate(dataset_sizes.keys()):
        losses = result[dataset_id] if dataset_id in result else [0]
        trial.report(np.mean(losses), i)
        all_losses += losses

    return np.mean(all_losses)


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
    studies = pd.read_json("synergy_studies_1000.jsonl", lines=True).head(n=N_STUDIES)

    studies = (
        sort_studies(studies=studies, dataset_sizes=dataset_sizes)
        .groupby("dataset_id")
        .head(2)
    )

    sampler = optuna.samplers.TPESampler()
    study_stop_cb = StopWhenOptimumReached(
        min_trials=MIN_TRIALS, threshold=STOPPING_THRESHOLD, n_history=N_HISTORY
    )

    study = optuna.create_study(
        storage=os.getenv(
            "DB_URI", "sqlite:///db.sqlite3"
        ),  # Specify the storage URL here.
        study_name=f"ASReview2-{CLASSIFIER_TYPE}-{len(studies)}-{VERSION}",
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
