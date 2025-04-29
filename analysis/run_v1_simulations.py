import concurrent.futures
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import synergy_dataset as sd
from asreview import ASReviewData, ASReviewProject, open_state
from asreview.models.balance import DoubleBalance
from asreview.models.classifiers import NaiveBayesClassifier, SVMClassifier
from asreview.models.feature_extraction import Tfidf
from asreview.models.query import MaxQuery
from asreview.review import ReviewSimulate

NUM_WORKERS = mp.cpu_count() - 1


def pad_labels(labels, num_priors, num_records):
    return pd.Series(
        labels.tolist() + np.zeros(num_records - len(labels) - num_priors).tolist()
    )


def process_study(dataset_name, study, index, clf):
    priors = study["prior_inclusions"] + study["prior_exclusions"]

    project_path = Path(f"{dataset_name}-{index}")
    project_path.mkdir(exist_ok=True, parents=True)

    if dataset_name == "Moran_2021_corrected":
        file_path = "../src/datasets/Moran_2021_corrected_shuffled_raw.csv"
    elif dataset_name == "Muthu_2021_corrected":
        file_path = "../src/datasets/Muthu_2021_corrected_shuffled_raw.csv"
    else:
        file_path = f"./datasets/synergy_dataset/{dataset_name}.csv"

    data_obj = ASReviewData.from_file(file_path)

    project = ASReviewProject.create(
        project_path=project_path / "api_simulation",
        project_id="api_example",
        project_mode="simulate",
        project_name="api_example",
    )

    project.add_dataset("../../../" + file_path)

    # Define models
    train_model = SVMClassifier() if clf == "svm" else NaiveBayesClassifier()
    query_model = MaxQuery()
    balance_model = DoubleBalance()
    feature_model = Tfidf()

    if len(data_obj) >= 10000:
        n_instances = [1, 25, 1000, 10**5]
        stop_ifs = [100, 1000, 10000]
    else:
        n_instances = [1, 5, 100]
        stop_ifs = [100, 1000]

    # Common reviewer args (shared across all ReviewSimulate calls)
    common_args = dict(
        as_data=data_obj,
        model=train_model,
        query_model=query_model,
        balance_model=balance_model,
        feature_model=feature_model,
        project=project,
        n_prior_included=len(study["prior_inclusions"]),
        n_prior_excluded=len(study["prior_exclusions"]),
        prior_indices=priors,
    )

    # Run review steps with computed stop_ifs
    prev_stop_if = stop_ifs[0]
    for i in range(len(stop_ifs)):
        stop_if = stop_ifs[i]
        instances = n_instances[i]
        if i > 0:
            queries_needed = (stop_ifs[i] - stop_ifs[i - 1]) // instances
            stop_if = prev_stop_if + queries_needed
            prev_stop_if = stop_if
        reviewer = ReviewSimulate(
            **common_args,
            n_instances=instances,
            stop_if=stop_if,
        )
        reviewer.review()

    reviewer = ReviewSimulate(
        **common_args,
        n_instances=n_instances[-1],
        stop_if="min",
    )
    reviewer.review()

    # Export results and cleanup
    project.export(f"asreview_old1_{clf}/{dataset_name}-{index}.asreview")
    shutil.rmtree(project_path)


def create_csv(report_order, studies_filtered, clf):
    recalls_old = []

    for dataset_name in report_order:
        # Load dataset
        if dataset_name == "Moran_2021_corrected":
            X = pd.read_csv("../src/datasets/Moran_2021_corrected_shuffled_raw.csv")
        elif dataset_name == "Muthu_2021_corrected":
            X = pd.read_csv("../src/datasets/Muthu_2021_corrected_shuffled_raw.csv")
        else:
            X = sd.Dataset(dataset_name).to_frame().reset_index()

        num_records = len(X)

        for i, study in studies_filtered[
            studies_filtered["dataset_id"] == dataset_name
        ].iterrows():
            priors = study["prior_inclusions"] + study["prior_exclusions"]

            with open_state(
                f"asreview_old1_{clf}/{dataset_name}-{i}.asreview"
            ) as state:
                df = state.get_dataset()

                df.drop(df[df["training_set"] < 0].index, axis=0, inplace=True)
                labels_old = pad_labels(
                    df["label"].reset_index(drop=True),
                    len(priors),
                    num_records,
                )
                recalls_old.append(labels_old.cumsum())

    pd.DataFrame(recalls_old).to_csv(f"recalls_old1_{clf}.csv", index=False)


# Load studies and filter
studies = pd.read_json("synergy_studies_validation.jsonl", lines=True)
studies_filtered = studies.sort_values("dataset_id").reset_index(drop=True)
report_order = studies_filtered["dataset_id"].unique()

print("Running ASReview v.1 NB Simulations")
clf = "nb"
os.mkdir(f"./asreview_old1_{clf}")
# Run in parallel using ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = []
    for dataset_name in report_order:
        for i, study in studies_filtered[
            studies_filtered["dataset_id"] == dataset_name
        ].iterrows():
            futures.append(executor.submit(process_study, dataset_name, study, i, clf))

    # Wait for all tasks to complete
    concurrent.futures.wait(futures)
print("NB simulations complete.\nCreating cumsum CSV")
create_csv(report_order=report_order, studies_filtered=studies_filtered, clf=clf)

print("Running ASReview v.1 SVM Simulations")
clf = "svm"
os.mkdir(f"./asreview_old1_{clf}")
# Run in parallel using ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = []
    for dataset_name in report_order:
        for i, study in studies_filtered[
            studies_filtered["dataset_id"] == dataset_name
        ].iterrows():
            futures.append(executor.submit(process_study, dataset_name, study, i, clf))

    # Wait for all tasks to complete
    concurrent.futures.wait(futures)

print("SVM simulations complete.\nCreating cumsum CSV")
create_csv(report_order=report_order, studies_filtered=studies_filtered, clf=clf)
