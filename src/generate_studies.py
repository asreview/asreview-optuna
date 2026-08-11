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
