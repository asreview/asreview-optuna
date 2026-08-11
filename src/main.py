import argparse
import datetime
import os
from pathlib import Path
from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd

from classifiers import classifier_params
from feature_extractors import feature_extractor_params
from simulation import run_studies


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
            all_losses.extend(losses)

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
        "--metric",
        default="loss",
        choices=["loss", "ndcg"],
        help="The metric used as objective during optimization.",
    )
    parser.add_argument(
        "--study-set",
        default="demo",
        choices=["demo", "train"],
        help="The study set that is used. Test-split data is intentionally not selectable here; use evaluate_test.py.",
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
        choices=["tfidf", "onehot", "mxbai", "multilingual-e5"],
        help="The feature extractor to optimize. mxbai and multilingual-e5 have no on-the-fly "
        "implementation and require --pre-processed-fms.",
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
        help="If set, run one trial with several processes. Each process will run 1 study set row at a time. Useful if you have a lot of studies.",
    )
    parser.add_argument(
        "--n-workers",
        default=1,
        type=int,
        help="Set the number of workers used for parallelizing the objective.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="The path to the synergy_plus data directory.",
    )
    parser.add_argument(
        "--fms-path",
        default=str(Path(__file__).resolve().parent / "preprocessed_fms"),
        help="The path to the preprocessed feature matrices.",
    )
    parser.add_argument(
        "--studies-path",
        default=str(Path(__file__).resolve().parent / "studies"),
        help='The path to the studies JSON files, "demo" and "train".',
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed for the Optuna sampler (TPESampler), for a reproducible hyperparameter search order.",
    )
    args = parser.parse_args()

    if (
        args.feature_extractor not in feature_extractor_params
        and not args.pre_processed_fms
    ):
        parser.error(
            f"--feature-extractor {args.feature_extractor} has no on-the-fly implementation; "
            "it can only be used together with --pre-processed-fms."
        )

    timestamp = datetime.datetime.now().strftime("%b-%d-%H:%M")
    study_name = f"[{timestamp}] {args.classifier}-{args.feature_extractor}-{args.study_set}-{args.metric}"
    studies = pd.read_json(
        Path(args.studies_path) / f"synergy_studies_{args.study_set}.jsonl", lines=True
    )
    n_studies = len(studies)
    n_datasets = studies["dataset_id"].nunique()

    print(f"""
=== ASReview Optuna run ===
study_name         : {study_name}
study_set          : {args.study_set}
studies            : {n_studies} row(s) / {n_datasets} dataset(s)
classifier         : {args.classifier}
feature_extractor  : {args.feature_extractor}
metric             : {args.metric}
preprocessed_fms   : {args.pre_processed_fms}
parallel_objective : {args.parallelize_objective}
max_workers        : {args.n_workers if args.parallelize_objective else 1}
n_trials           : {args.n_trials}
seed               : {args.seed}
data_path          : {args.data_path}
fms_path           : {args.fms_path}
studies_path       : {args.studies_path}
DB                 : {"local" if os.getenv("DB_URI", "sqlite:///db.sqlite3") == "sqlite:///db.sqlite3" else "remote"}
===========================
    """)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        direction="minimize" if args.metric == "loss" else "maximize",
        study_name=study_name,
        storage=os.getenv("DB_URI", "sqlite:///db.sqlite3"),
        load_if_exists=True,
    )

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
        callbacks=[
            StopWhenOptimumReached(min_trials=400, threshold=0.0001, n_history=5)
        ],
    )
