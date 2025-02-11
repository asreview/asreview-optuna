import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import asreview
import numpy as np
import optuna
import pandas as pd
import synergy_dataset as sd
from asreview.models.balancers import Balanced
from asreview.models.classifiers import NaiveBayes
from asreview.models.feature_extractors import Tfidf
from asreview.models.queriers import Max

RUN_OLD = True
DB_PATH = ""
NDCG_STUDY = "ASReview2-full-tfidf-nb-3"
LOSS_STUDY = "ASReview2-full-nb-1"
NUM_WORKERS = mp.cpu_count() - 1


# ------------------ Helper Functions ------------------


def pad_labels(labels, num_priors, num_records):
    """Pad labels to match the dataset size."""
    return pd.Series(
        labels.tolist() + np.zeros(num_records - len(labels) - num_priors).tolist()
    )


def n_query_extreme(results, n_records):
    """Dynamic query function for active learning."""
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


# ------------------ Core Processing Logic ------------------


def process_study(study, dataset_name, params=None):
    """Processes a single study, handling dataset loading and active learning."""

    # Load dataset
    if dataset_name == "Moran_2021_corrected":
        X = pd.read_csv("./datasets/Moran_2021_corrected_shuffled_raw.csv")
    else:
        X = sd.Dataset(dataset_name).to_frame().reset_index()

    labels = X["label_included"]
    priors = study["prior_inclusions"] + study["prior_exclusions"]

    # Set parameters for different configurations
    alpha = params.get("nb__alpha", 3.822) if params else 3.822
    ratio = params.get("ratio", 3) if params else 3
    tfidf_kwargs = {
        "stop_words": None if params else "english",
        "ngram_range": (1, 2) if params else (1, 1),
        "sublinear_tf": params.get("sublinear_tf", False) if params else False,
        "max_df": params.get("tfidf__max_df", 1.0) if params else 1.0,
        "min_df": params.get("tfidf__min_df", 1) if params else 1,
    }

    # Setup Active Learning Cycle
    alc = asreview.ActiveLearningCycle(
        querier=Max(),
        classifier=NaiveBayes(alpha=alpha),
        balancer=Balanced(ratio=ratio),
        feature_extractor=Tfidf(**tfidf_kwargs),
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
    report_order, studies_filtered, output_file, params=None, n_workers=NUM_WORKERS
):
    """Runs the simulation for all datasets and saves results in parallel."""
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

    # Save results
    pd.DataFrame(results).to_csv(output_file, index=False)


# ------------------ Main Function ------------------


def main():
    """Main execution function."""
    # Load studies and filter top 5 per dataset
    studies = pd.read_json("synergy_studies_full_val.jsonl", lines=True)
    studies_filtered = (
        studies.sort_values("dataset_id")
        .groupby("dataset_id")
        .head(5)
        .reset_index(drop=True)
    )
    report_order = studies_filtered["dataset_id"].unique()

    # Load optimized parameters
    opt_study = optuna.load_study(study_name=NDCG_STUDY, storage=DB_PATH)
    params_ndcg = opt_study.best_trial.params

    opt_study = optuna.load_study(study_name=LOSS_STUDY, storage=DB_PATH)
    params_loss = opt_study.best_trial.params
    params_loss["nb__alpha"] = params_loss["alpha"]

    # Run different simulations
    run_simulation(
        report_order,
        studies_filtered,
        params=params_ndcg,
        output_file="recalls_ndcg.csv",
    )
    run_simulation(
        report_order,
        studies_filtered,
        params=params_loss,
        output_file="recalls_loss.csv",
    )
    if RUN_OLD:
        run_simulation(
            report_order, studies_filtered, output_file="recalls_old.csv"
        )  # Defaults to old settings


if __name__ == "__main__":
    main()
