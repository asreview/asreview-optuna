import multiprocessing as mp
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import synergy_dataset as sd

import asreview
from asreview.metrics import loss
from asreview.models.balance import Balanced
from asreview.models.feature_extraction import Tfidf
from asreview.models.query import MaxQuery

from asreview.models.classifiers import (
    RandomForestClassifier,
)

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


def _sort_studies(studies, dataset_sizes, ascending=True):
    studies_sorter = sorted(dataset_sizes.items(), key=lambda x: x[1], reverse=True)

    return studies.sort_values(
        "dataset_id",
        key=lambda x: x.map({val[0]: i for i, val in enumerate(studies_sorter)}),
        ascending=ascending,
    )


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


# Function to process each row
def run_sysrev_sim(row):
    with open(PICKLE_FOLDER_PATH / f"{row['dataset_id']}.pkl", "rb") as f:
        fm, labels = pickle.load(f)

    priors = row["prior_inclusions"] + row["prior_exclusions"]

    blc = Balanced(ratio=1.5)

    simulate = asreview.Simulate(
        fm=fm,
        labels=labels,
        classifier=RandomForestClassifier(),
        balance_strategy=blc,
        query_strategy=MaxQuery(),
        feature_extraction=Tfidf(),
        n_query=lambda x: n_query_extreme(x, dataset_sizes[row["dataset_id"]]),
    )

    simulate.label(priors, prior=True)
    simulate.review()

    padded_results = list(simulate._results["label"]) + [0] * (
        len(simulate.labels) - len(simulate._results["label"])
    )
    return loss(padded_results)


if __name__ == "__main__":
    dataset_sizes = {
        dataset.name: dataset.metadata["data"]["n_records"]
        for dataset in sd.iter_datasets()
    }

    with open("asreview2-opt/synergy_studies_1000.jsonl", "r") as f:
        studies = pd.read_json(f, lines=True).groupby("dataset_id").head(1)

    studies = _sort_studies(
        studies=studies, dataset_sizes=dataset_sizes, ascending=True
    )

    for index, study in studies.iterrows():
        run_sysrev_sim(study)
