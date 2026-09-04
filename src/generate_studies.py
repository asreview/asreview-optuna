import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulation import load_dataset


def sample_priors_for_dataset(
    dataset_id: str, df: pd.DataFrame, n_priors: int, seed: int
) -> list[dict]:
    """
    Draw n_priors random priors for one dataset.

    Each draw is 1 randomly sampled included record + 1 randomly sampled
    excluded record (positional indices into `df`, matching the row order
    `load_dataset` produces). Uses a per-dataset deterministic RNG derived
    from `seed` and `dataset_id`, so regenerating is reproducible and
    unaffected by the order datasets happen to be processed in.

    Args:
        dataset_id (str): Identifier for the dataset.
        df (pd.DataFrame): The dataset, as returned by `load_dataset`.
        n_priors (int): Number of prior draws to generate (X).
        seed (int): Base seed for reproducible generation.

    Returns:
        list[dict]: Rows in the same schema as the existing studies files.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, *dataset_id.encode("utf-8")])
    )
    labels = df["label_included"].to_numpy()
    included = np.flatnonzero(labels == 1)
    excluded = np.flatnonzero(labels == 0)

    if len(included) == 0 or len(excluded) == 0:
        raise ValueError(
            f"{dataset_id}: cannot sample priors, included={len(included)} excluded={len(excluded)}"
        )

    inc_draws = rng.choice(included, size=n_priors, replace=len(included) < n_priors)
    exc_draws = rng.choice(excluded, size=n_priors, replace=len(excluded) < n_priors)

    return [
        {
            "dataset_id": dataset_id,
            "prior_inclusions": [int(inc_draws[i])],
            "prior_exclusions": [int(exc_draws[i])],
        }
        for i in range(n_priors)
    ]


def compute_tertile_strata(
    values: pd.Series, train_mask: pd.Series, labels: list[str]
) -> tuple[pd.Series, tuple[float, float]]:
    """
    Label every entry in `values` by tertile, using boundaries computed only
    from the entries where `train_mask` is True.

    Args:
        values (pd.Series): Numeric values to stratify (aligned to train_mask).
        train_mask (pd.Series): Boolean mask selecting the rows whose
            distribution defines the tertile cut points.
        labels (list[str]): The 3 stratum labels, low-to-high.

    Returns:
        tuple[pd.Series, tuple[float, float]]: Labels aligned to
        values.index, and the (q1, q2) tertile boundary values.
    """
    q1, q2 = values[train_mask].quantile([1 / 3, 2 / 3])
    strata = pd.cut(values, bins=[-np.inf, q1, q2, np.inf], labels=labels)
    return strata, (float(q1), float(q2))


def load_external_column(
    path: Path, key_column: str, value_column: str, base_metadata: pd.DataFrame
) -> pd.Series:
    """
    Read a CSV with `key_column`/`value_column` columns and return
    `value_column`'s values aligned to base_metadata.index, joined against
    base_metadata's "key" column.

    Args:
        path (Path): Path to a CSV containing `key_column` and `value_column`.
        key_column (str): Name of the join-key column in that CSV (its values
            must match base_metadata["key"]; the column itself may be named
            differently, e.g. "dataset_id").
        value_column (str): Name of the column to pull.
        base_metadata (pd.DataFrame): Must contain a "key" column.

    Returns:
        pd.Series: `value_column`'s values aligned to base_metadata.index.
    """
    if not path.exists():
        raise FileNotFoundError(f"External column file not found: {path}")
    ext = pd.read_csv(path)[[key_column, value_column]].rename(columns={key_column: "key"})
    merged = base_metadata[["key"]].merge(ext, on="key", how="left", validate="one_to_one")
    missing = merged.loc[merged[value_column].isna(), "key"].tolist()
    if missing:
        raise ValueError(f"{value_column} missing in {path} for key(s): {missing}")
    return merged.set_index(base_metadata.index)[value_column]


def load_domain_labels(
    extended_metadata_path: Path, base_metadata: pd.DataFrame
) -> pd.Series:
    """
    Read the extended metadata CSV and return "health"/"nonhealth" labels
    aligned to base_metadata.index, joined on `key`.

    Args:
        extended_metadata_path (Path): Path to the extended review_metadata.csv
            containing a "primary_topic_domain" column.
        base_metadata (pd.DataFrame): Must contain a "key" column.

    Returns:
        pd.Series: "health"/"nonhealth" labels aligned to base_metadata.index.
    """
    if not extended_metadata_path.exists():
        raise FileNotFoundError(
            f"--extended-metadata-path not found: {extended_metadata_path}"
        )
    ext = pd.read_csv(extended_metadata_path)[["key", "primary_topic_domain"]]
    merged = base_metadata[["key"]].merge(
        ext, on="key", how="left", validate="one_to_one"
    )
    missing = merged.loc[merged["primary_topic_domain"].isna(), "key"].tolist()
    if missing:
        raise ValueError(
            f"primary_topic_domain missing in {extended_metadata_path} for key(s): {missing}"
        )
    return merged.set_index(base_metadata.index)["primary_topic_domain"].map(
        lambda v: "health" if v == "Health Sciences" else "nonhealth"
    )


def load_field_labels(
    extended_metadata_path: Path, base_metadata: pd.DataFrame
) -> pd.Series:
    """
    Read the extended metadata CSV and return "medicine"/"non_medicine" labels
    aligned to base_metadata.index, joined on `key`.

    Finer-grained than `domain`: "Health Sciences" (the `domain` axis's
    "health") is itself ~87% "Medicine" at the OpenAlex field level, with the
    remainder split across fields too small to stratify on individually
    (Health Professions, Nursing, Dentistry). "Medicine" is the only
    non-domain field large enough to be its own stratum (~38 train datasets);
    every other field has well under half that.

    Args:
        extended_metadata_path (Path): Path to the extended review_metadata.csv
            containing a "primary_topic_field" column.
        base_metadata (pd.DataFrame): Must contain a "key" column.

    Returns:
        pd.Series: "medicine"/"non_medicine" labels aligned to base_metadata.index.
    """
    if not extended_metadata_path.exists():
        raise FileNotFoundError(
            f"--extended-metadata-path not found: {extended_metadata_path}"
        )
    ext = pd.read_csv(extended_metadata_path)[["key", "primary_topic_field"]]
    merged = base_metadata[["key"]].merge(
        ext, on="key", how="left", validate="one_to_one"
    )
    missing = merged.loc[merged["primary_topic_field"].isna(), "key"].tolist()
    if missing:
        raise ValueError(
            f"primary_topic_field missing in {extended_metadata_path} for key(s): {missing}"
        )
    return merged.set_index(base_metadata.index)["primary_topic_field"].map(
        lambda v: "medicine" if v == "Medicine" else "non_medicine"
    )


def write_stratum_files(
    metadata: pd.DataFrame,
    stratum_column: str,
    axis_name: str,
    data_path: str,
    n_priors: int,
    seed: int,
    studies_path: Path,
    split_value: str = "train",
) -> None:
    """
    Partition metadata[split_value]'s dataset_ids by `stratum_column` and
    write one JSONL file per stratum value, reusing generate_rows/write_jsonl.

    Filename: f"synergy_studies_{split_value}-{axis_name}-{stratum}.jsonl"
    (e.g. "synergy_studies_train-domain-health.jsonl").
    """
    split_rows = metadata.loc[metadata["split"] == split_value]
    for stratum_value in sorted(split_rows[stratum_column].dropna().unique().astype(str)):
        dataset_ids = sorted(
            split_rows.loc[split_rows[stratum_column].astype(str) == stratum_value, "key"]
        )
        rows = generate_rows(dataset_ids, data_path, n_priors, seed)
        out_path = (
            studies_path
            / f"synergy_studies_{split_value}-{axis_name}-{stratum_value}.jsonl"
        )
        write_jsonl(rows, out_path)
        print(f"{out_path}: {len(rows)} row(s) / {len(dataset_ids)} dataset(s)")


def generate_rows(
    dataset_ids: list, data_path: str, n_priors: int, seed: int
) -> list[dict]:
    """Load each dataset and sample its prior rows."""
    rows = []
    for dataset_id in dataset_ids:
        df = load_dataset(data_path, dataset_id)
        rows.extend(sample_priors_for_dataset(dataset_id, df, n_priors, seed))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows to a JSONL file, one JSON object per line."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Generate ASReview Optuna studies",
        description="Generate random-prior study rows for the train/test/demo splits from the local synergy_plus mirror.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the synergy_plus data directory (contains {dataset_id}.csv files and metadata/review_metadata.csv).",
    )
    parser.add_argument(
        "--studies-path",
        default=None,
        help="Output directory for the studies JSONL files (default: <this script's directory>/studies).",
    )
    parser.add_argument(
        "--n-priors",
        default=10,
        type=int,
        help="Number of random prior draws per dataset (X), applied to both train and test.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Base seed for reproducible generation."
    )
    parser.add_argument(
        "--demo-dataset-id",
        default="Appenzeller-Herzog_2019",
        help="Dataset to use for the demo studies file.",
    )
    parser.add_argument(
        "--demo-n-priors",
        default=2,
        type=int,
        help="Number of random prior draws for the demo dataset (kept small/independent of --n-priors).",
    )
    parser.add_argument(
        "--stratify-by",
        nargs="+",
        choices=[
            "domain",
            "field",
            "search_size",
            "inclusion_ratio",
            "n_databases",
            "protocol",
            "baseline_loss",
        ],
        default=[],
        help="Opt-in: also partition the TRAIN split by the given axis/axes "
        "(each axis partitioned independently, not crossed) into extra "
        "synergy_studies_train-<axis>-<stratum>.jsonl files, plus a "
        "stratification_manifest.json covering all datasets (train and test). "
        "Omit to keep current behavior unchanged.",
    )
    parser.add_argument(
        "--extended-metadata-path",
        default=None,
        help="Path to the extended review_metadata.csv containing "
        "primary_topic_domain/primary_topic_field (joined on `key`). Defaults "
        "to '<data-path>_extended/metadata/review_metadata.csv'. Only read "
        "when --stratify-by includes 'domain' and/or 'field'.",
    )
    parser.add_argument(
        "--baseline-loss-path",
        default=None,
        help="Path to a CSV with 'dataset_id'/'loss_mean' columns covering ALL "
        "datasets in both splits (e.g. concatenating `evaluate_test.py "
        "--study-set train --skip-baseline` output for train with an existing "
        "test_results_*.csv's 'Tuned' rows for test) -- each dataset's loss "
        "under a baseline study's tuned hyperparameters. Required when "
        "--stratify-by includes 'baseline_loss'; has no default since there's "
        "no fixed baseline study to derive one from.",
    )
    args = parser.parse_args()

    if "baseline_loss" in args.stratify_by and not args.baseline_loss_path:
        parser.error(
            "--stratify-by baseline_loss requires --baseline-loss-path (a CSV "
            "with dataset_id/loss_mean columns for the train split)."
        )

    studies_path = (
        Path(args.studies_path)
        if args.studies_path
        else Path(__file__).resolve().parent / "studies"
    )
    studies_path.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(Path(args.data_path) / "metadata" / "review_metadata.csv")

    unknown_splits = set(metadata["split"]) - {"train", "test"}
    if unknown_splits:
        raise ValueError(
            f"Unexpected split value(s) in review_metadata.csv: {unknown_splits}"
        )

    missing = [
        key
        for key in metadata["key"]
        if not (Path(args.data_path) / f"{key}.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset CSV(s) for key(s) in review_metadata.csv: {missing}"
        )

    for split_name in ["train", "test"]:
        dataset_ids = sorted(metadata.loc[metadata["split"] == split_name, "key"])
        rows = generate_rows(dataset_ids, args.data_path, args.n_priors, args.seed)
        out_path = studies_path / f"synergy_studies_{split_name}.jsonl"
        write_jsonl(rows, out_path)
        print(f"{out_path}: {len(rows)} row(s) / {len(dataset_ids)} dataset(s)")

    demo_rows = generate_rows(
        [args.demo_dataset_id], args.data_path, args.demo_n_priors, args.seed
    )
    demo_path = studies_path / "synergy_studies_demo.jsonl"
    write_jsonl(demo_rows, demo_path)
    print(f"{demo_path}: {len(demo_rows)} row(s) / 1 dataset(s)")

    if args.stratify_by:
        manifest_path = studies_path / "stratification_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest["seed"] = args.seed
            manifest["n_priors"] = args.n_priors
            manifest["axes_computed"] = sorted(
                set(manifest.get("axes_computed", [])) | set(args.stratify_by)
            )
        else:
            manifest = {
                "seed": args.seed,
                "n_priors": args.n_priors,
                "axes_computed": sorted(args.stratify_by),
                "datasets": {},
            }
        strat_metadata = metadata.copy()

        train_mask = strat_metadata["split"] == "train"

        if "search_size" in args.stratify_by:
            strat_metadata["size_stratum"], (q1, q2) = compute_tertile_strata(
                strat_metadata["n_records"], train_mask, ["small", "medium", "large"]
            )
            manifest["size_axis"] = {
                "computed_from_split": "train",
                "column": "n_records",
                "tertile_boundaries": [q1, q2],
                "labels": ["small", "medium", "large"],
            }
            write_stratum_files(
                strat_metadata,
                "size_stratum",
                "size",
                args.data_path,
                args.n_priors,
                args.seed,
                studies_path,
            )

        if "inclusion_ratio" in args.stratify_by:
            inclusion_ratio = strat_metadata["n_records_included"] / strat_metadata["n_records"]
            strat_metadata["inclusion_ratio_stratum"], (q1, q2) = compute_tertile_strata(
                inclusion_ratio, train_mask, ["low", "mid", "high"]
            )
            manifest["inclusion_ratio_axis"] = {
                "computed_from_split": "train",
                "column": "n_records_included / n_records",
                "tertile_boundaries": [q1, q2],
                "labels": ["low", "mid", "high"],
            }
            write_stratum_files(
                strat_metadata,
                "inclusion_ratio_stratum",
                "inclusion_ratio",
                args.data_path,
                args.n_priors,
                args.seed,
                studies_path,
            )

        if "n_databases" in args.stratify_by:
            strat_metadata["n_databases_stratum"], (q1, q2) = compute_tertile_strata(
                strat_metadata["number_of_databases"], train_mask, ["low", "mid", "high"]
            )
            manifest["n_databases_axis"] = {
                "computed_from_split": "train",
                "column": "number_of_databases",
                "tertile_boundaries": [q1, q2],
                "labels": ["low", "mid", "high"],
            }
            write_stratum_files(
                strat_metadata,
                "n_databases_stratum",
                "n_databases",
                args.data_path,
                args.n_priors,
                args.seed,
                studies_path,
            )

        if "protocol" in args.stratify_by:
            strat_metadata["protocol_stratum"] = strat_metadata["protocol"].map(
                {1: "protocol", 0: "no_protocol"}
            )
            manifest["protocol_axis"] = {
                "column": "protocol",
                "protocol_value": 1,
                "labels": ["protocol", "no_protocol"],
            }
            write_stratum_files(
                strat_metadata,
                "protocol_stratum",
                "protocol",
                args.data_path,
                args.n_priors,
                args.seed,
                studies_path,
            )

        if "baseline_loss" in args.stratify_by:
            baseline_loss_path = Path(args.baseline_loss_path)
            baseline_loss = load_external_column(
                baseline_loss_path, "dataset_id", "loss_mean", strat_metadata
            )
            strat_metadata["baseline_loss_stratum"], (q1, q2) = compute_tertile_strata(
                baseline_loss, train_mask, ["low", "mid", "high"]
            )
            manifest["baseline_loss_axis"] = {
                "computed_from_split": "train",
                "source_path": str(baseline_loss_path),
                "tertile_boundaries": [q1, q2],
                "labels": ["low", "mid", "high"],
            }
            write_stratum_files(
                strat_metadata,
                "baseline_loss_stratum",
                "baseline_loss",
                args.data_path,
                args.n_priors,
                args.seed,
                studies_path,
            )

        if "domain" in args.stratify_by:
            extended_metadata_path = Path(
                args.extended_metadata_path
                or f"{str(args.data_path).rstrip('/')}_extended/metadata/review_metadata.csv"
            )
            strat_metadata["domain_stratum"] = load_domain_labels(
                extended_metadata_path, strat_metadata
            )
            manifest["domain_axis"] = {
                "extended_metadata_path": str(extended_metadata_path),
                "health_label_value": "Health Sciences",
                "labels": ["health", "nonhealth"],
            }
            write_stratum_files(
                strat_metadata,
                "domain_stratum",
                "domain",
                args.data_path,
                args.n_priors,
                args.seed,
                studies_path,
            )

        if "field" in args.stratify_by:
            extended_metadata_path = Path(
                args.extended_metadata_path
                or f"{str(args.data_path).rstrip('/')}_extended/metadata/review_metadata.csv"
            )
            strat_metadata["field_stratum"] = load_field_labels(
                extended_metadata_path, strat_metadata
            )
            manifest["field_axis"] = {
                "extended_metadata_path": str(extended_metadata_path),
                "medicine_label_value": "Medicine",
                "labels": ["medicine", "non_medicine"],
            }
            write_stratum_files(
                strat_metadata,
                "field_stratum",
                "field",
                args.data_path,
                args.n_priors,
                args.seed,
                studies_path,
            )

        stratum_columns = [c for c in strat_metadata.columns if c.endswith("_stratum")]
        for _, row in strat_metadata.iterrows():
            entry = manifest["datasets"].setdefault(row["key"], {})
            entry["split"] = row["split"]
            entry["n_records"] = int(row["n_records"])
            for col in stratum_columns:
                if pd.notna(row[col]):
                    entry[col] = str(row[col])

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(
            f"{manifest_path}: {len(manifest['datasets'])} dataset(s), "
            f"axes={manifest['axes_computed']}"
        )
