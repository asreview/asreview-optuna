import argparse
import os
import re
from pathlib import Path

import optuna
import pandas as pd
from asreview.models.models import get_ai_config

from classifiers import classifier_kwargs_from_trial_params
from feature_extractors import feature_extractor_kwargs_from_trial_params
from simulation import run_studies


BASELINE_MODELS = {
    "tfidf": ["elas_u3", "elas_u4"],
    "mxbai": ["elas_h3"],
    "multilingual-e5": ["elas_l2"],
}

ELAS_FMS_FOLDER = {
    "tfidf": "tfidf",
    "mxbai": "mxbai",
    "multilingual-e5-large": "multilingual-e5",
}


def sanitize_for_filename(name: str) -> str:
    """Turn a study name (which may contain [, ], :, spaces) into a safe filename fragment."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def build_report(
    test_studies: pd.DataFrame,
    result: dict,
    metric: str,
    study_name: str,
    run: str,
    classifier: str,
    feature_extractor: str,
) -> pd.DataFrame:
    """
    Build a per-dataset breakdown report, plus a final OVERALL row using the
    same flat-mean-across-everything formula as the Optuna objective.

    `run` distinguishes the tuned result from any ELAS baseline rows (e.g.
    "Tuned" vs "ELAS u4") within the same report; `study_name` stays constant
    across all rows of one evaluate_test.py invocation for traceability.
    """
    rows = []
    dataset_ids = sorted(test_studies["dataset_id"].unique())
    all_values = []

    for dataset_id in dataset_ids:
        values = result.get(dataset_id, [])
        all_values.extend(values)
        series = pd.Series(values, dtype=float)
        rows.append(
            {
                "study_name": study_name,
                "run": run,
                "classifier": classifier,
                "feature_extractor": feature_extractor,
                "dataset_id": dataset_id,
                "n_priors": len(values),
                f"{metric}_mean": series.mean(),
                f"{metric}_std": series.std(),
                f"{metric}_min": series.min(),
                f"{metric}_max": series.max(),
            }
        )

    overall_series = pd.Series(all_values, dtype=float)
    rows.append(
        {
            "study_name": study_name,
            "run": run,
            "classifier": classifier,
            "feature_extractor": feature_extractor,
            "dataset_id": "OVERALL",
            "n_priors": len(all_values),
            f"{metric}_mean": overall_series.mean(),
            f"{metric}_std": overall_series.std(),
            f"{metric}_min": overall_series.min(),
            f"{metric}_max": overall_series.max(),
        }
    )

    return pd.DataFrame(rows)


def run_baseline(
    elas_name: str,
    test_studies: pd.DataFrame,
    metric: str,
    data_path: str,
    fms_path: str,
    parallel: bool,
    n_workers: int,
) -> tuple[str, str, str, dict]:
    """
    Run one ASReview-shipped baseline model (e.g. "elas_u4") against the test
    split, using its exact shipped hyperparameters (not this repo's own
    tuning defaults).

    Returns (label, classifier, feature_extractor, result) where result is
    the same dict[dataset_id -> list[metric]] shape run_studies returns.
    """
    config = get_ai_config(elas_name)
    alc = config["value"]

    pre_processed = alc.feature_extractor != "tfidf"
    fe_folder = ELAS_FMS_FOLDER.get(alc.feature_extractor, alc.feature_extractor)

    result = run_studies(
        test_studies,
        parallel=parallel,
        n_workers=n_workers,
        classifier=alc.classifier,
        clf_params=dict(alc.classifier_param),
        feature_extractor=fe_folder,
        fe_params={} if pre_processed else dict(alc.feature_extractor_param),
        ratio=alc.balancer_param["ratio"],
        metric=metric,
        pre_processed_fms=pre_processed,
        data_path=data_path,
        fms_path=fms_path,
    )
    return config["label"], alc.classifier, alc.feature_extractor, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ASReview Optuna test evaluation",
        description="Evaluate a finished Optuna study's best hyperparameters against the held-out test split.",
    )
    parser.add_argument(
        "--storage",
        default=os.getenv("DB_URI", "sqlite:///db.sqlite3"),
        help="Optuna storage URI (defaults to the same DB_URI/sqlite convention as main.py).",
    )
    parser.add_argument(
        "--study-name",
        required=True,
        help="Exact study name, as printed in main.py's run banner.",
    )
    parser.add_argument(
        "--classifier",
        required=True,
        choices=["log", "nb", "rf", "svm"],
        help="Must match what the study was tuned with.",
    )
    parser.add_argument(
        "--feature-extractor",
        required=True,
        choices=["tfidf", "onehot", "mxbai", "multilingual-e5"],
        help="Must match what the study was tuned with.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        choices=["loss", "ndcg"],
        help="Defaults to whichever the study optimized (inferred from study.direction).",
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
        "--pre-processed-fms",
        action="store_true",
        help="If set, use the pre-processed feature matrices.",
    )
    parser.add_argument(
        "--studies-path",
        default=str(Path(__file__).resolve().parent / "studies"),
        help="Directory containing the studies JSON files (always reads synergy_studies_test.jsonl from here).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="If set, run simulations in parallel.",
    )
    parser.add_argument(
        "--n-workers",
        default=1,
        type=int,
        help="Number of workers used for parallelization.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: ./test_results_<study_name>.csv).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip comparing against the relevant ASReview-shipped ELAS baseline model(s).",
    )
    args = parser.parse_args()

    if (
        args.feature_extractor in ("mxbai", "multilingual-e5")
        and not args.pre_processed_fms
    ):
        parser.error(
            f"--feature-extractor {args.feature_extractor} has no on-the-fly implementation; "
            "it can only be used together with --pre-processed-fms."
        )

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    best = study.best_trial
    metric = args.metric or (
        "loss" if study.direction == optuna.study.StudyDirection.MINIMIZE else "ndcg"
    )

    ratio = best.params["ratio"]
    clf_params = classifier_kwargs_from_trial_params(args.classifier, best.params)
    fe_params = feature_extractor_kwargs_from_trial_params(
        args.feature_extractor, best.params
    )

    test_studies = pd.read_json(
        Path(args.studies_path) / "synergy_studies_test.jsonl", lines=True
    )

    baseline_names = (
        [] if args.skip_baseline else BASELINE_MODELS.get(args.feature_extractor, [])
    )
    baseline_desc = (
        ", ".join(get_ai_config(n)["label"] for n in baseline_names)
        if baseline_names
        else "none"
    )

    print(f"""
=== ASReview Optuna test evaluation ===
study_name         : {args.study_name}
best_trial         : #{best.number} (value={best.value})
classifier         : {args.classifier} ({clf_params})
feature_extractor  : {args.feature_extractor} ({fe_params})
ratio              : {ratio}
metric             : {metric}
test studies       : {len(test_studies)} row(s) / {test_studies["dataset_id"].nunique()} dataset(s)
baseline           : {baseline_desc}
========================================
    """)

    result = run_studies(
        test_studies,
        parallel=args.parallel,
        n_workers=args.n_workers,
        classifier=args.classifier,
        clf_params=clf_params,
        feature_extractor=args.feature_extractor,
        fe_params=fe_params,
        ratio=ratio,
        metric=metric,
        pre_processed_fms=args.pre_processed_fms,
        data_path=args.data_path,
        fms_path=args.fms_path,
    )
    reports = [
        build_report(
            test_studies,
            result,
            metric,
            args.study_name,
            "Tuned",
            args.classifier,
            args.feature_extractor,
        )
    ]

    for elas_name in baseline_names:
        label, b_classifier, b_feature_extractor, b_result = run_baseline(
            elas_name,
            test_studies,
            metric,
            args.data_path,
            args.fms_path,
            args.parallel,
            args.n_workers,
        )
        reports.append(
            build_report(
                test_studies,
                b_result,
                metric,
                args.study_name,
                label,
                b_classifier,
                b_feature_extractor,
            )
        )

    report = pd.concat(reports, ignore_index=True)
    print(report.to_string(index=False))

    if len(reports) > 1:
        direction = "lower is better" if metric == "loss" else "higher is better"
        summary = report.loc[
            report["dataset_id"] == "OVERALL",
            ["run", "classifier", "feature_extractor", f"{metric}_mean"],
        ]
        print(f"\n=== Summary (OVERALL, {metric}, {direction}) ===")
        print(summary.to_string(index=False))

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"./test_results_{sanitize_for_filename(args.study_name)}.csv")
    )
    report.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")
