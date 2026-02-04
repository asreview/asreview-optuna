import argparse
import datetime
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import asreview
import numpy as np
import optuna
import pandas as pd
import synergy_dataset as sd
from asreview.learner import ActiveLearningCycle
from asreview.metrics import loss, ndcg
from asreview.models.balancers import Balanced
from asreview.models.queriers import Max

from classifiers import classifier_params, classifiers
from feature_extractors import feature_extractor_params, feature_extractors


def load_dataset(data_path: str, dataset_id: str) -> pd.DataFrame:
    """
    Load a dataset by ID.

    Args:
        path (str): Path to raw data files.
        dataset_id (str): Identifier for the dataset.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    if dataset_id == "Appenzeller-Herzog_2019":
        return pd.read_csv(Path(data_path) / f"{dataset_id}.csv")
    return sd.Dataset(dataset_id).to_frame().reset_index()


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


def run_studies(
    studies: pd.DataFrame,
    parallel: bool,
    n_workers: int,
    *args,
    **kwargs,
) -> dict[str, list[float]]:
    """
    Run ASReview simulations on a set of studies, either in parallel or sequentially.

    Args:
        studies (pd.DataFrame): DataFrame containing study rows.
        parallel (bool): If True, runs studies in parallel using ProcessPoolExecutor.
        n_workers (int): Number of workers used to parallelize the objective.
        *args: Positional arguments passed to `process_row`.
        **kwargs: Keyword arguments passed to `process_row`.

    Returns:
        dict[str, list[float]]: Mapping from dataset_id to metric values.
    """

    losses = defaultdict(list)

    if parallel:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_row, row, *args, **kwargs): i
                for i, row in studies.iterrows()
            }
            for future in as_completed(futures):
                dataset_id, result = future.result()
                if result is not None:
                    losses[dataset_id].append(result)
    else:
        for _, row in studies.iterrows():
            dataset_id, result = process_row(row, *args, **kwargs)
            if result is not None:
                losses[dataset_id].append(result)

    return losses


def process_row(
    row: pd.Series,
    clf_params: dict,
    fe_params: dict,
    ratio: float,
    classifier: str,
    feature_extractor: str,
    metric: str,
    pre_processed_fms: bool,
    data_path: str,
    fms_path: str,
) -> tuple[str, float]:
    """
    Run a single ASReview simulation for one study row.

    This function:
    - Loads the dataset (raw or pre-processed features)
    - Initializes the classifier, feature extractor, and balancer
    - Runs an ASReview simulation with predefined priors
    - Computes and returns the chosen evaluation metric

    Args:
        row (pd.Series): A single row from the studies DataFrame.
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
        tuple[str, float]: Dataset ID and computed metric value.
    """
    priors = row["prior_inclusions"] + row["prior_exclusions"]

    # Create balancer with optuna value
    blc = Balanced(ratio=ratio)

    # Create classifier and feature extractor with params
    clf = classifiers[classifier](**clf_params)

    if pre_processed_fms:
        with open(
            Path(fms_path) / f"{feature_extractor}" / f"{row['dataset_id']}.pkl",
            "rb",
        ) as f:
            X, labels = pickle.load(f)

        labels = pd.Series(labels)

        alc = ActiveLearningCycle(
            querier=Max(),
            classifier=clf,
            balancer=blc,
            n_query=lambda results: n_query(results, X.shape[0]),
        )
    else:
        X = load_dataset(data_path, row["dataset_id"])

        labels = X["label_included"]
        fe = feature_extractors[feature_extractor](**fe_params)

        alc = ActiveLearningCycle(
            querier=Max(),
            classifier=clf,
            balancer=blc,
            feature_extractor=fe,
            n_query=lambda results: n_query(results, X.shape[0]),
        )

    simulate = asreview.Simulate(
        X=X,
        labels=labels,
        cycles=[alc],
        skip_transform=pre_processed_fms,
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


def objective_report(
    studies: pd.DataFrame,
    classifier: str,
    feature_extractor: str,
    parallelize_objective: bool,
    metric: str,
    pre_processed_fms: bool,
    n_workers: int,
    data_path: str,
    fms_path: str,
) -> Callable[[optuna.trial.Trial], float]:
    """
    Create an Optuna objective function for hyperparameter optimization.

    The returned objective function:
    - Samples hyperparameters using Optuna
    - Runs ASReview simulations for all studies
    - Reports intermediate results per dataset
    - Returns the aggregated metric across all studies

    Args:
        studies (pd.DataFrame): DataFrame containing all study configurations.
        classifier (str): Name of the classifier to optimize.
        feature_extractor (str): Name of the feature extractor to optimize.
        parallelize_objective (bool): Whether to run studies in parallel.
        metric (str): Optimization metric ("loss" or "ndcg").
        pre_processed_fms (bool): Whether to use pre-processed feature matrices.
        n_workers (int): Number of workers to use when parallelize_objective is True.
        data_path (str): The path to the raw data.
        fms_path (str): The path to the preprocessed fms.

    Returns:
        callable: Optuna-compatible objective function.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        # Use normal distribution for ratio (ratio effect is linear)
        ratio = trial.suggest_float("ratio", 1.0, 10.0)

        clf_params = classifier_params[classifier](trial)
        fe_params = (
            feature_extractor_params[feature_extractor](trial)
            if feature_extractor in feature_extractor_params
            else {}
        )

        result = run_studies(
            studies,
            parallel=parallelize_objective,
            n_workers=n_workers,
            classifier=classifier,
            clf_params=clf_params,
            feature_extractor=feature_extractor,
            fe_params=fe_params,
            ratio=ratio,
            metric=metric,
            pre_processed_fms=pre_processed_fms,
            data_path=data_path,
            fms_path=fms_path,
        )

        report_order = sorted(set(studies["dataset_id"]))
        all_losses = []
        for i, dataset_id in enumerate(report_order):
            losses = result[dataset_id] if dataset_id in result else [0]
            trial.report(np.mean(losses), i)
            trial.report(np.std(losses), len(report_order) + i)
            all_losses += losses

        return np.mean(all_losses)

    return objective


class StopWhenOptimumReached:
    """
    Optuna callback for early stopping based on convergence.

    The study is stopped when the variation in objective values
    over the last `n_history` trials is smaller than `threshold`,
    provided at least `min_trials` have been completed.
    """

    def __init__(self, min_trials: int, threshold: float, n_history: int):
        """
        Initialize the early stopping callback.

        Args:
            min_trials (int): Minimum number of trials before checking convergence.
            threshold (float): Maximum allowed difference between recent trial values.
            n_history (int): Number of recent trials to consider.
        """
        self.min_trials = min_trials
        self.threshold = threshold
        self.n_history = n_history

    def __call__(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        """
        Evaluate whether the study should be stopped early.

        Args:
            study (optuna.study.Study): The current Optuna study.
            trial (optuna.trial.FrozenTrial): The most recently completed trial.
        """
        # If there are more than min_trials completed, check early stopping condition
        if trial.number >= self.min_trials:
            # Take latest n_history trial loss values
            prev_trial_losses = [
                t.value for t in study.trials[-self.n_history :] if t.value is not None
            ]
            # If the difference is smaller than threshold, we stop the entire study
            if max(prev_trial_losses) - min(prev_trial_losses) < self.threshold:
                study.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ASReview Optuna",
        description="Program that helps running exhaustive parameter optimization studies for ASReview and SYNERGY+",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="If set, the program will only create the DB, not run any trials.",
    )
    parser.add_argument(
        "--metric",
        default="loss",
        choices=["loss", "ndcg"],
        help="The metric used as objective during optimization.",
    )
    parser.add_argument(
        "--study-set",
        default="demo",
        choices=["demo", "full"],
        help="The study set that is used.",
    )
    parser.add_argument(
        "--classifier",
        default="svm",
        choices=["log", "nb", "rf", "svm"],
        help="The classifier to optimize.",
    )
    parser.add_argument(
        "--feature-extractor",
        default="tfidf",
        choices=["tfidf", "onehot"],
        help="The feature extractor to optimize.",
    )
    parser.add_argument(
        "--pre-processed-fms",
        action="store_true",
        help="If set, use the pre-processed feature matrices.",
    )
    parser.add_argument(
        "--n-trials",
        default=500,
        type=int,
        help="Set the maximum number of trials that will be ran.",
    )
    parser.add_argument(
        "--parallelize-objective",
        action="store_true",
        help='If set, run one trial with several threads. Each thread will run 1 study set row at a time. Useful if you have a lot of studies (e.g., study-set="full").',
    )
    parser.add_argument(
        "--n-workers",
        default=1,
        type=int,
        help="Set the number of workers used for parallelizing the objective.",
    )
    parser.add_argument(
        "--data-path",
        default="./data",
        help="The path to the raw data.",
    )
    parser.add_argument(
        "--fms-path",
        default="./preprocessed_fms",
        help="The path to the preprocessed feature matrices.",
    )
    parser.add_argument(
        "--studies-path",
        default="./studies",
        help='The path to the studies JSON files, "demo" and "full".',
    )
    args = parser.parse_args()

    sampler = optuna.samplers.TPESampler()
    study_stop_cb = StopWhenOptimumReached(
        min_trials=400, threshold=0.0001, n_history=5
    )

    if args.init_db:
        optuna.create_study(
            sampler=sampler,
            direction="minimize" if args.metric == "loss" else "maximize",
            study_name=f"[{datetime.datetime.now().strftime('%b-%d-%H:%M')}] {args.classifier}-{args.feature_extractor}-{args.study_set}-{args.metric}",
            storage=os.getenv("DB_URI", "sqlite:///db.sqlite3"),
            load_if_exists=True,
        )
        print("Database initialized, exiting.")
        sys.exit(0)

    study = optuna.create_study(
        sampler=sampler,
        direction="minimize" if args.metric == "loss" else "maximize",
        study_name=f"[{datetime.datetime.now().strftime('%b-%d-%H:%M')}] {args.classifier}-{args.feature_extractor}-{args.study_set}-{args.metric}",
        storage=os.getenv("DB_URI", "sqlite:///db.sqlite3"),
        load_if_exists=True,
    )

    # list of studies
    studies = pd.read_json(
        Path(args.studies_path) / f"synergy_studies_{args.study_set}.jsonl", lines=True
    )

    n_studies = len(studies)
    n_datasets = studies["dataset_id"].nunique()

    print(f"""
    === ASReview Optuna run ===
    study_name         : {study.study_name}
    study_set          : {args.study_set}
    studies            : {n_studies} row(s) / {n_datasets} dataset(s)
    classifier         : {args.classifier}
    feature_extractor  : {args.feature_extractor}
    metric             : {args.metric}
    preprocessed_fms   : {args.pre_processed_fms}
    parallel_objective : {args.parallelize_objective}
    max_workers        : {args.n_workers if args.parallelize_objective else 1}
    n_trials           : {args.n_trials}
    data_path          : {args.data_path}
    fms_path           : {args.fms_path}
    studies_path       : {args.studies_path}
    DB                 : {"local" if os.getenv("DB_URI", "sqlite:///db.sqlite3") == "sqlite:///db.sqlite3" else "remote"}
    ===========================
    """)

    study.optimize(
        objective_report(
            studies=studies,
            classifier=args.classifier,
            feature_extractor=args.feature_extractor,
            parallelize_objective=args.parallelize_objective,
            metric=args.metric,
            pre_processed_fms=args.pre_processed_fms,
            n_workers=args.n_workers,
            data_path=args.data_path,
            fms_path=args.fms_path,
        ),
        n_trials=args.n_trials,
        callbacks=[study_stop_cb],
    )
