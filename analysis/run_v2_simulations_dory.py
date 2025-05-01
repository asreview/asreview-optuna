import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor

import asreview
import numpy as np
import pandas as pd
from asreview.models.balancers import Balanced
from asreview.models.classifiers import SVM
from asreview.models.queriers import Max
from asreviewcontrib.nemo.feature_extractors.sentence_transformers import (
    MXBAI,
    MultilingualE5Large,
)

NUM_WORKERS = mp.cpu_count() - 1


def pad_labels(labels, num_priors, num_records):
    return pd.Series(
        labels.tolist() + np.zeros(num_records - len(labels) - num_priors).tolist()
    )


def n_query_extreme(results, n_records):
    if n_records >= 10000:
        if len(results) >= 10000:
            return 10**5
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


def process_study(study, dataset_name, params=None):
    priors = study["prior_inclusions"] + study["prior_exclusions"]

    if params["fe"] == "e5":
        with open(f"./fms/pickles_e5/{dataset_name}.pkl", "rb") as f:
            X, labels = pickle.load(f)

        alc = asreview.ActiveLearningCycle(
            querier=Max(),
            classifier=SVM(C=0.106, loss="squared_hinge", max_iter=5000),
            balancer=Balanced(ratio=9.707),
            feature_extractor=MultilingualE5Large(),
            n_query=lambda results: n_query_extreme(results, X.shape[0]),
        )
    elif params["fe"] == "mxbai":
        with open(f"./fms/pickles_mxbai/{dataset_name}.pkl", "rb") as f:
            X, labels = pickle.load(f)

        alc = asreview.ActiveLearningCycle(
            querier=Max(),
            classifier=SVM(C=0.067, loss="squared_hinge", max_iter=5000),
            balancer=Balanced(ratio=9.724),
            feature_extractor=MXBAI(),
            n_query=lambda results: n_query_extreme(results, X.shape[0]),
        )

    simulate = asreview.Simulate(X=X, labels=labels, cycles=[alc], skip_transform=True)
    simulate.label(priors)
    simulate.review()

    df_results = simulate._results.dropna(axis=0, subset="training_set")
    labels_processed = pad_labels(
        df_results["label"].reset_index(drop=True), len(priors), len(X)
    )

    return labels_processed.cumsum()


def run_simulation(
    report_order, studies_filtered, output_file, params=None, n_workers=NUM_WORKERS
):
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for dataset_name in report_order:
            dataset_studies = studies_filtered[
                studies_filtered["dataset_id"] == dataset_name
            ]
            for _, study in dataset_studies.iterrows():
                futures.append(
                    executor.submit(process_study, study, dataset_name, params)
                )

        for future in futures:
            results.append(future.result())

    pd.DataFrame(results).to_csv(output_file, index=False)


def main():
    studies = pd.read_json("synergy_studies_validation.jsonl", lines=True)
    studies_filtered = studies.sort_values("dataset_id").reset_index(drop=True)
    report_order = studies_filtered["dataset_id"].unique()

    print("Running ASReview v.2 E5 Simulations")
    run_simulation(
        report_order,
        studies_filtered,
        params={"fe": "e5"},
        output_file="recalls_new2_e5_svm.csv",
    )
    print("E5 simulations complete\n")

    print("Running ASReview v.2 mxbai Simulations")
    run_simulation(
        report_order,
        studies_filtered,
        params={"fe": "mxbai"},
        output_file="recalls_new2_mxbai_svm.csv",
    )
    print("mxbai simulations complete")


if __name__ == "__main__":
    main()
