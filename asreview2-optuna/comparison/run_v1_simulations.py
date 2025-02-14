import concurrent.futures
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from asreview import ASReviewData, ASReviewProject, open_state
from asreview.models.balance import DoubleBalance
from asreview.models.classifiers import NaiveBayesClassifier
from asreview.models.feature_extraction import Tfidf
from asreview.models.query import MaxQuery
from asreview.review import ReviewSimulate


def pad_labels(labels, num_priors, num_records):
    return pd.Series(
        labels.tolist() + np.zeros(num_records - len(labels) - num_priors).tolist()
    )


def process_study(dataset_name, study, index):
    """Function to process a single study"""

    priors = study["prior_inclusions"] + study["prior_exclusions"]

    project_path = Path(f"{dataset_name}-{index}")
    project_path.mkdir(exist_ok=True, parents=True)

    if dataset_name == "Moran_2021_corrected":
        file_path = "./datasets/Moran_2021_corrected_shuffled_raw.csv"
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
    train_model = NaiveBayesClassifier()
    query_model = MaxQuery()
    balance_model = DoubleBalance()
    feature_model = Tfidf()

    # Initialize and run the simulation
    reviewer = ReviewSimulate(
        as_data=data_obj,
        model=train_model,
        query_model=query_model,
        balance_model=balance_model,
        feature_model=feature_model,
        n_instances=1,
        project=project,
        n_prior_included=len(study["prior_inclusions"]),
        n_prior_excluded=len(study["prior_exclusions"]),
        prior_indices=priors,
    )

    reviewer.review()

    # Export results and cleanup
    project.export(f"asreview_old_files/{dataset_name}-{index}.asreview")
    shutil.rmtree(project_path)


# Load studies and filter
studies = pd.read_json("synergy_studies_full_val.jsonl", lines=True)
studies_filtered = (
    studies.sort_values("dataset_id")
    .groupby("dataset_id")
    .head(5)
    .reset_index(drop=True)
)
report_order = studies_filtered["dataset_id"].unique()

# Run in parallel using ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count() - 1) as executor:
    futures = []
    for dataset_name in report_order:
        for i, study in studies_filtered[
            studies_filtered["dataset_id"] == dataset_name
        ].iterrows():
            futures.append(executor.submit(process_study, dataset_name, study, i))

    # Wait for all tasks to complete
    concurrent.futures.wait(futures)

print("Processing complete.")

recalls_old = []

for dataset_name in report_order:
    for i, study in studies_filtered[
        studies_filtered["dataset_id"] == dataset_name
    ].iterrows():
        priors = study["prior_inclusions"] + study["prior_exclusions"]

        with open_state(f"asreview_old_files/{dataset_name}-{i}.asreview") as state:
            df = state.get_dataset()
            num_records = len(df)
            df.drop(df[df["training_set"] < 0].index, axis=0, inplace=True)
            labels_old = pad_labels(
                df["label"].reset_index(drop=True),
                len(priors),
                num_records,
            )
            recalls_old.append(labels_old.cumsum())

pd.DataFrame(recalls_old).to_csv("recalls_old.csv", index=False)
