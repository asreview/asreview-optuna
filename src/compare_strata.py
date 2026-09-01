import argparse
import json
import os
from pathlib import Path

import optuna
import pandas as pd
from dotenv import load_dotenv

from balancers import balancer_kwargs_from_trial_params
from classifiers import classifier_kwargs_from_trial_params
from feature_extractors import feature_extractor_kwargs_from_trial_params
from simulation import run_studies

load_dotenv()


def find_study_name(storage: str, must_contain: list[str]) -> str:
    """
    Find the single study name in `storage` containing every substring in
    `must_contain`.

    Args:
        storage (str): Optuna storage URI.
        must_contain (list[str]): Substrings the study name must contain.

    Returns:
        str: The single matching study name.
    """
    all_names = optuna.study.get_all_study_names(storage=storage)
    matches = [n for n in all_names if all(s in n for s in must_contain)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one study name containing {must_contain}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def dataset_ids_for_stratum(
    manifest: dict, axis: str, stratum: str, split: str
) -> list[str]:
    """
    Look up dataset_ids in `manifest` matching a given stratum on `axis`,
    restricted to one split ("train" or "test").

    Args:
        manifest (dict): Parsed stratification_manifest.json.
        axis (str): Stratification axis, e.g. "domain" or "size".
        stratum (str): Stratum value, e.g. "health" or "small".
        split (str): "train" or "test".

    Returns:
        list[str]: Matching dataset_ids.
    """
    key = f"{axis}_stratum"
    return [
        dataset_id
        for dataset_id, entry in manifest["datasets"].items()
        if entry.get(key) == stratum and entry["split"] == split
    ]


def best_hyperparams(
    storage: str, study_name: str, classifier: str, feature_extractor: str, balancer: str
) -> tuple[dict, dict, dict, int, float]:
    """
    Load a study and reconstruct its best trial's classifier/feature-extractor/
    balancer kwargs.

    Returns:
        tuple: (clf_params, fe_params, balancer_kwargs, n_completed_trials, best_value)
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    best = study.best_trial
    n_completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    clf_params = classifier_kwargs_from_trial_params(classifier, best.params)
    fe_params = feature_extractor_kwargs_from_trial_params(feature_extractor, best.params)
    balancer_kwargs = balancer_kwargs_from_trial_params(balancer, best.params)
    return clf_params, fe_params, balancer_kwargs, n_completed, best.value


DEFAULT_STRATA = {
    "domain": ["health", "nonhealth"],
    "size": ["small", "medium", "large"],
    "inclusion_ratio": ["low", "mid", "high"],
    "n_databases": ["low", "mid", "high"],
    "protocol": ["protocol", "no_protocol"],
    "baseline_loss": ["low", "mid", "high"],
}

DEFAULT_DATE_TAGS = {
    "domain": "Aug-28",
    "size": "Aug-28",
}


def mean_metric(
    test_studies: pd.DataFrame,
    classifier: str,
    clf_params: dict,
    feature_extractor: str,
    fe_params: dict,
    balancer: str,
    balancer_kwargs: dict,
    metric: str,
    data_path: str,
    fms_path: str,
    pre_processed_fms: bool,
    parallel: bool,
    n_workers: int,
) -> float:
    """Run `test_studies` with the given hyperparameters and return the flat mean metric."""
    result = run_studies(
        test_studies,
        parallel=parallel,
        n_workers=n_workers,
        classifier=classifier,
        clf_params=clf_params,
        feature_extractor=feature_extractor,
        fe_params=fe_params,
        balancer=balancer,
        balancer_kwargs=balancer_kwargs,
        metric=metric,
        pre_processed_fms=pre_processed_fms,
        data_path=data_path,
        fms_path=fms_path,
    )
    all_values = [v for values in result.values() for v in values]
    return float(pd.Series(all_values, dtype=float).mean())


def run_axis_comparison(
    args: argparse.Namespace,
    axis: str,
    strata: list[str],
    manifest: dict,
    test_studies: pd.DataFrame,
    study_name_overrides: dict,
    output_path: str,
    date_tag: str,
) -> pd.DataFrame:
    """Run the baseline-vs-strata comparison for one axis and write its report CSV."""
    studies_to_compare = {"baseline": args.baseline_study_name}
    for stratum in strata:
        studies_to_compare[stratum] = study_name_overrides.get(
            stratum
        ) or find_study_name(args.storage, [date_tag, f"-{axis}-{stratum}-"])

    eval_subsets = {"all_test": test_studies}
    for stratum in strata:
        eval_subsets[f"{stratum}_test"] = test_studies[
            test_studies["dataset_id"].isin(
                dataset_ids_for_stratum(manifest, axis, stratum, "test")
            )
        ]

    subset_counts = ", ".join(
        f"{name}={subset['dataset_id'].nunique()}" for name, subset in eval_subsets.items()
    )
    study_lines = "\n".join(
        f"{tuned_on:15s}: {name}" for tuned_on, name in studies_to_compare.items()
    )

    print(f"""
=== ASReview Optuna strata comparison ===
storage            : {"local" if args.storage == "sqlite:///db.sqlite3" else "remote"}
metric             : {args.metric}
classifier         : {args.classifier}
feature_extractor  : {args.feature_extractor}
balancer           : {args.balancer}
axis               : {axis}
strata             : {strata}
date_tag           : {date_tag}
studies:
{study_lines}
eval subsets       : {subset_counts} dataset(s)
==========================================
    """)

    rows = []
    for tuned_on, study_name in studies_to_compare.items():
        clf_params, fe_params, balancer_kwargs, n_completed, best_value = best_hyperparams(
            args.storage, study_name, args.classifier, args.feature_extractor, args.balancer
        )
        print(
            f"[{tuned_on}] {study_name}: {n_completed} completed trial(s), "
            f"best_trial.value={best_value:.4f}"
        )
        print(f"  classifier params  : {clf_params}")
        print(f"  feature_extractor  : {fe_params}")
        print(f"  balancer params    : {balancer_kwargs}")

        for evaluated_on, subset in eval_subsets.items():
            value = mean_metric(
                subset,
                args.classifier,
                clf_params,
                args.feature_extractor,
                fe_params,
                args.balancer,
                balancer_kwargs,
                args.metric,
                args.data_path,
                args.fms_path,
                args.pre_processed_fms,
                args.parallel,
                args.n_workers,
            )
            print(f"    evaluated on {evaluated_on:15s}: {args.metric}={value:.4f}")
            rows.append(
                {
                    "tuned_on": tuned_on,
                    "study_name": study_name,
                    "n_completed_trials": n_completed,
                    "best_trial_value": best_value,
                    "evaluated_on": evaluated_on,
                    f"{args.metric}_mean": value,
                }
            )

    report = pd.DataFrame(rows)
    print(f"\n=== Transfer matrix (tuned_on x evaluated_on) [{axis}] ===")
    print(
        report.pivot(index="tuned_on", columns="evaluated_on", values=f"{args.metric}_mean")
        .to_string()
    )

    report.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ASReview Optuna strata comparison",
        description="Compare a baseline (pooled-train) study against stratified studies "
        "for a chosen axis (domain: health/nonhealth, or size: small/medium/large): for "
        "each study's best hyperparameters, evaluate against the full test split and "
        "against each stratum's test subset, to see whether stratum-specific tuning "
        "helps on its own stratum and how it transfers to the others.",
    )
    parser.add_argument(
        "--storage",
        default=os.getenv("DB_URI", "sqlite:///db.sqlite3"),
        help="Optuna storage URI (defaults to the same DB_URI/sqlite convention as main.py).",
    )
    parser.add_argument(
        "--baseline-study-name",
        default="[Aug-11-16:40] svm-tfidf-train-loss",
        help="Exact study name of the pooled/global baseline study.",
    )
    parser.add_argument(
        "--axis",
        default="domain",
        help="Stratification axis to compare against the baseline (any axis "
        "generate_studies.py --stratify-by produced, e.g. domain, size, "
        "inclusion_ratio, n_databases, protocol, baseline_loss). Pass --strata "
        "explicitly for axes without a built-in default stratum list. Ignored "
        "when --all-axes is set.",
    )
    parser.add_argument(
        "--all-axes",
        action="store_true",
        help="Run-and-forget mode: loop over every axis in DEFAULT_STRATA (16 "
        "strata across 6 axes) instead of just --axis, auto-discovering each "
        "stratum's study name via --date-tag and writing one CSV per axis into "
        "--output-dir. Ignores --axis/--strata/--study-name/--output.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write the per-axis CSVs into when --all-axes is set "
        "(default: current directory). Files are named strata_comparison_<axis>.csv.",
    )
    parser.add_argument(
        "--strata",
        nargs="+",
        default=None,
        help="Stratum values to compare for --axis (default: health nonhealth for "
        "domain; small medium large for size).",
    )
    parser.add_argument(
        "--study-name",
        action="append",
        default=[],
        metavar="STRATUM=NAME",
        help="Override auto-discovery for one stratum's study name, e.g. "
        "'health=[Aug-28-10:46] svm-tfidf-ratio-train-domain-health-loss'. "
        "Repeatable. Strata not overridden are auto-discovered via --date-tag.",
    )
    parser.add_argument(
        "--date-tag",
        default=None,
        help="Substring used to auto-discover stratum study names. If passed "
        "explicitly, it overrides DEFAULT_DATE_TAGS for every axis; if omitted, "
        "each axis falls back to its DEFAULT_DATE_TAGS entry (domain/size: "
        "Aug-28), or 'Aug-31' for axes with no entry.",
    )
    parser.add_argument(
        "--axis-date-tag",
        action="append",
        default=[],
        metavar="AXIS=TAG",
        help="Override the date-tag used to auto-discover study names for one "
        "axis, e.g. 'domain=Aug-28'. Repeatable. Takes precedence over "
        "--date-tag and DEFAULT_DATE_TAGS. Only relevant with --all-axes "
        "(single-axis mode already lets --date-tag cover the one axis).",
    )
    parser.add_argument("--classifier", default="svm", choices=["log", "nb", "svm", "rf"])
    parser.add_argument(
        "--feature-extractor", default="tfidf", choices=["tfidf", "onehot", "mxbai", "multilingual-e5"]
    )
    parser.add_argument("--balancer", default="ratio", choices=["ratio", "double"])
    parser.add_argument("--metric", default="loss", choices=["loss", "ndcg"])
    parser.add_argument("--data-path", required=True, help="Path to the synergy_plus data directory.")
    parser.add_argument(
        "--pre-processed-fms",
        action="store_true",
        help="If set, use the pre-processed feature matrices. Required for "
        "--feature-extractor mxbai/multilingual-e5.",
    )
    parser.add_argument(
        "--fms-path", default=str(Path(__file__).resolve().parent / "preprocessed_fms")
    )
    parser.add_argument(
        "--studies-path", default=str(Path(__file__).resolve().parent / "studies")
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Path to stratification_manifest.json (default: <studies-path>/stratification_manifest.json).",
    )
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n-workers", default=1, type=int)
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: ./strata_comparison_<axis>.csv, so "
        "different axes don't overwrite each other's results).",
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

    manifest_path = Path(
        args.manifest_path or Path(args.studies_path) / "stratification_manifest.json"
    )
    with open(manifest_path) as f:
        manifest = json.load(f)

    test_studies = pd.read_json(
        Path(args.studies_path) / "synergy_studies_test.jsonl", lines=True
    )

    axis_date_tag_overrides = dict(s.split("=", 1) for s in args.axis_date_tag)

    def date_tag_for(axis: str) -> str:
        return (
            axis_date_tag_overrides.get(axis)
            or args.date_tag
            or DEFAULT_DATE_TAGS.get(axis)
            or "Aug-31"
        )

    if args.all_axes:
        for axis, strata in DEFAULT_STRATA.items():
            output_path = str(Path(args.output_dir) / f"strata_comparison_{axis}.csv")
            run_axis_comparison(
                args, axis, strata, manifest, test_studies, {}, output_path,
                date_tag_for(axis),
            )
    else:
        strata = args.strata or DEFAULT_STRATA.get(args.axis)
        if strata is None:
            parser.error(
                f"--axis {args.axis!r} has no built-in default stratum list; "
                "pass --strata explicitly (e.g. --strata low mid high)."
            )
        overrides = dict(s.split("=", 1) for s in args.study_name)
        output_path = args.output or f"./strata_comparison_{args.axis}.csv"
        run_axis_comparison(
            args, args.axis, strata, manifest, test_studies, overrides, output_path,
            date_tag_for(args.axis),
        )
