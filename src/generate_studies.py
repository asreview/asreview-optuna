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


def compute_size_strata(
    metadata: pd.DataFrame, train_split_value: str = "train"
) -> tuple[pd.Series, tuple[float, float]]:
    """
    Label every row in `metadata` (all splits) small/medium/large by
    n_records, using tertile boundaries computed only from the rows where
    metadata["split"] == train_split_value.

    Args:
        metadata (pd.DataFrame): Must contain "split" and "n_records" columns.
        train_split_value (str): The split label whose n_records distribution
            defines the tertile cut points.

    Returns:
        tuple[pd.Series, tuple[float, float]]: Labels aligned to
        metadata.index, and the (q1, q2) tertile boundary values.
    """
    train_n_records = metadata.loc[metadata["split"] == train_split_value, "n_records"]
    q1, q2 = train_n_records.quantile([1 / 3, 2 / 3])
    labels = pd.cut(
        metadata["n_records"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["small", "medium", "large"],
    )
    return labels, (float(q1), float(q2))


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
        choices=["domain", "search_size"],
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
        "primary_topic_domain (joined on `key`). Defaults to "
        "'<data-path>_extended/metadata/review_metadata.csv'. Only read "
        "when --stratify-by includes 'domain'.",
    )
    args = parser.parse_args()

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

        if "search_size" in args.stratify_by:
            strat_metadata["size_stratum"], (q1, q2) = compute_size_strata(strat_metadata)
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

        for _, row in strat_metadata.iterrows():
            entry = manifest["datasets"].setdefault(row["key"], {})
            entry["split"] = row["split"]
            entry["n_records"] = int(row["n_records"])
            if "size_stratum" in strat_metadata.columns:
                entry["size_stratum"] = str(row["size_stratum"])
            if "domain_stratum" in strat_metadata.columns:
                entry["domain_stratum"] = str(row["domain_stratum"])

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(
            f"{manifest_path}: {len(manifest['datasets'])} dataset(s), "
            f"axes={manifest['axes_computed']}"
        )
