import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import asreview
import numpy as np
import pandas as pd
import synergy_dataset as sd
from asreview.models.balancers import Balanced
from asreview.models.classifiers import SVM, NaiveBayes
from asreview.models.feature_extractors import Tfidf
from asreview.models.queriers import Max

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


def process_study(study, dataset_name, clf):
    if dataset_name == "Moran_2021_corrected":
        X = pd.read_csv("../src/datasets/Moran_2021_corrected_shuffled_raw.csv")
    elif dataset_name == "Muthu_2021_corrected":
        X = pd.read_csv("../src/datasets/Muthu_2021_corrected_shuffled_raw.csv")
    else:
        X = sd.Dataset(dataset_name).to_frame().reset_index()

    labels = X["label_included"]
    priors = study["prior_inclusions"] + study["prior_exclusions"]

    if clf == "nb":
        tfidf_kwargs = {
            "ngram_range": (1, 2),
            "sublinear_tf": True,
            "max_df": 0.93,
            "min_df": 7,
        }

        alc = asreview.ActiveLearningCycle(
            querier=Max(),
            classifier=NaiveBayes(alpha=1.48),
            balancer=Balanced(ratio=1.58),
            feature_extractor=Tfidf(stop_words="english"),
            n_query=lambda results: n_query_extreme(results, X.shape[0]),
        )

    elif clf == "svm":
        tfidf_kwargs = {
            "ngram_range": (1, 2),
            "sublinear_tf": True,
            "max_df": 0.95,
            "min_df": 1,
        }

        alc = asreview.ActiveLearningCycle(
            querier=Max(),
            classifier=SVM(C=0.11, loss="squared_hinge"),
            balancer=Balanced(ratio=9.8),
            feature_extractor=Tfidf(**tfidf_kwargs),
            n_query=lambda results: n_query_extreme(results, X.shape[0]),
        )

    # Run simulation
    simulate = asreview.Simulate(X=X, labels=labels, cycles=[alc])
    simulate.label(priors)
    simulate.review()

    df_results = simulate._results.dropna(axis=0, subset="training_set")
    labels_processed = pad_labels(
        df_results["label"].reset_index(drop=True), len(priors), len(X)
    )

    return labels_processed.cumsum()


def run_simulation(
    report_order, studies_filtered, output_file, clf, n_workers=NUM_WORKERS
):
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for dataset_name in report_order:
            dataset_studies = studies_filtered[
                studies_filtered["dataset_id"] == dataset_name
            ]
            for _, study in dataset_studies.iterrows():
                futures.append(executor.submit(process_study, study, dataset_name, clf))

        for future in futures:
            results.append(future.result())

    # Save results
    pd.DataFrame(results).to_csv(output_file, index=False)


def main():
    studies = pd.read_json("synergy_studies_validation.jsonl", lines=True)
    studies_filtered = studies.sort_values("dataset_id").reset_index(drop=True)
    report_order = studies_filtered["dataset_id"].unique()

    print("Running ASReview v.2 NB Simulations")
    run_simulation(
        report_order,
        studies_filtered,
        clf="nb",
        output_file="recalls_new2_nb.csv",
    )
    print("NB simulations complete\n")
    
    print("Running ASReview v.2 SVM Simulations")
    run_simulation(
        report_order,
        studies_filtered,
        clf="svm",
        output_file="recalls_new2_svm.csv",
    )
    print("SVM simulations complete")


if __name__ == "__main__":
    main()
