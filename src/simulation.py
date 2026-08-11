import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import asreview
import pandas as pd
from asreview.learner import ActiveLearningCycle
from asreview.metrics import loss, ndcg
from asreview.models.balancers import Balanced
from asreview.models.queriers import Max

from classifiers import classifiers
from feature_extractors import feature_extractors


def load_dataset(data_path: str, dataset_id: str) -> pd.DataFrame:
    """
    Load a dataset by ID from the local synergy_plus mirror.

    Args:
        data_path (str): Path to the synergy_plus data directory.
        dataset_id (str): Identifier for the dataset (matches its filename stem).

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    df = pd.read_csv(Path(data_path) / f"{dataset_id}.csv")
    return df.drop(columns=["label_abstract_included"], errors="ignore")


def n_query(results: list, n_records: int) -> int:
    """
    Determine the number of items to query in each active learning cycle.

    Args:
        results (list): List of current simulation results.
        n_records (int): Total number of records in the dataset.

    Returns:
        int: Number of items to query.
    """
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


def featurize_dataset(
    dataset_id: str,
    feature_extractor: str,
    fe_params: dict,
    pre_processed_fms: bool,
    data_path: str,
    fms_path: str,
):
    """
    Load or compute the feature matrix and labels for one dataset, once.

    Loading the raw dataset and fitting/loading its feature matrix is
    independent of which prior draw a simulation is about to use, so this is
    factored out and called once per (trial, dataset) rather than once per
    prior draw.

    Args:
        dataset_id (str): Identifier for the dataset.
        feature_extractor (str): Name of the feature extractor to use.
        fe_params (dict): Hyperparameters for the feature extractor.
        pre_processed_fms (bool): Whether to use pre-processed feature matrices.
        data_path (str): The path to the raw data.
        fms_path (str): The path to the preprocessed fms.

    Returns:
        tuple: (feature matrix, labels)
    """
    if pre_processed_fms:
        with open(Path(fms_path) / feature_extractor / f"{dataset_id}.pkl", "rb") as f:
            X, labels = pickle.load(f)
        return X, pd.Series(labels)

    df = load_dataset(data_path, dataset_id)
    fe = feature_extractors[feature_extractor](**fe_params)
    return fe.fit_transform(df), df["label_included"]


def process_row(
    row: pd.Series,
    X,
    labels,
    clf_params: dict,
    ratio: float,
    classifier: str,
    metric: str,
) -> tuple[str, float]:
    """
    Run a single ASReview simulation for one study row, against an
    already-featurized dataset.

    This function:
    - Initializes the classifier and balancer
    - Runs an ASReview simulation with this row's priors
    - Computes and returns the chosen evaluation metric

    Args:
        row (pd.Series): A single row from the studies DataFrame.
        X: Feature matrix for the dataset (from `featurize_dataset`).
        labels: Labels for the dataset (from `featurize_dataset`).
        clf_params (dict): Hyperparameters for the classifier.
        ratio (float): Class balance ratio for the Balanced balancer.
        classifier (str): Name of the classifier to use.
        metric (str): Metric to compute ("loss" or "ndcg").

    Returns:
        tuple[str, float]: Dataset ID and computed metric value.
    """
    priors = row["prior_inclusions"] + row["prior_exclusions"]

    blc = Balanced(ratio=ratio)
    clf = classifiers[classifier](**clf_params)

    alc = ActiveLearningCycle(
        querier=Max(),
        classifier=clf,
        balancer=blc,
        n_query=lambda results: n_query(results, X.shape[0]),
    )

    simulate = asreview.Simulate(
        X=X,
        labels=labels,
        cycles=[alc],
        skip_transform=True,
        print_progress=False,
    )

    # Set priors
    simulate.label(priors)

    # Start simulation
    simulate.review()

    # Return loss
    padded_results = list(simulate._results["label"]) + [0] * (
        len(simulate.labels) - len(simulate._results["label"])
    )
    calculated_metric = (
        loss(padded_results) if metric == "loss" else ndcg(padded_results)
    )
    return row["dataset_id"], calculated_metric


def run_studies(
    studies: pd.DataFrame,
    parallel: bool,
    n_workers: int,
    clf_params: dict,
    fe_params: dict,
    ratio: float,
    classifier: str,
    feature_extractor: str,
    metric: str,
    pre_processed_fms: bool,
    data_path: str,
    fms_path: str,
) -> dict[str, list[float]]:
    """
    Run ASReview simulations on a set of studies, either in parallel or sequentially.

    Each unique dataset in `studies` is featurized once (see
    `featurize_dataset`), then every study row is simulated against its
    dataset's already-computed feature matrix.

    Args:
        studies (pd.DataFrame): DataFrame containing study rows.
        parallel (bool): If True, runs studies in parallel using ProcessPoolExecutor.
        n_workers (int): Number of workers used to parallelize the objective.
        clf_params (dict): Hyperparameters for the classifier.
        fe_params (dict): Hyperparameters for the feature extractor.
        ratio (float): Class balance ratio for the Balanced balancer.
        classifier (str): Name of the classifier to use.
        feature_extractor (str): Name of the feature extractor to use.
        metric (str): Metric to compute ("loss" or "ndcg").
        pre_processed_fms (bool): Whether to use pre-processed feature matrices.
        data_path (str): The path to the raw data.
        fms_path (str): The path to the preprocessed fms.

    Returns:
        dict[str, list[float]]: Mapping from dataset_id to metric values.
    """
    featurized = {
        dataset_id: featurize_dataset(
            dataset_id,
            feature_extractor,
            fe_params,
            pre_processed_fms,
            data_path,
            fms_path,
        )
        for dataset_id in studies["dataset_id"].unique()
    }

    losses = defaultdict(list)

    if parallel:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    process_row,
                    row,
                    *featurized[row["dataset_id"]],
                    clf_params,
                    ratio,
                    classifier,
                    metric,
                ): i
                for i, row in studies.iterrows()
            }
            for future in as_completed(futures):
                dataset_id, result = future.result()
                if result is not None:
                    losses[dataset_id].append(result)
    else:
        for _, row in studies.iterrows():
            dataset_id, result = process_row(
                row,
                *featurized[row["dataset_id"]],
                clf_params,
                ratio,
                classifier,
                metric,
            )
            if result is not None:
                losses[dataset_id].append(result)

    return losses
