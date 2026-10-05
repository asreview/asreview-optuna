"""
Generate size-matched *random* train-split strata, as a control for the
metadata strata: does tuning on a metadata-defined subset do any better than
tuning on a random subset of the same size?

Each random stratum is a subset of synergy_studies_train.jsonl's rows, so the
prior draws are identical to the pooled and metadata-stratum studies (priors are
seeded per dataset in generate_studies.py, so subsetting rows is equivalent to
regenerating them).

Writes synergy_studies_train-random-n<size>-r<draw>.jsonl and records the
dataset ids per random stratum in random_strata_manifest.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate size-matched random train strata.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[21, 30, 40, 50])
    parser.add_argument("--n-draws", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--studies-path", default=str(Path(__file__).resolve().parent / "studies")
    )
    args = parser.parse_args()

    studies_path = Path(args.studies_path)
    train = pd.read_json(studies_path / "synergy_studies_train.jsonl", lines=True)
    dataset_ids = sorted(train["dataset_id"].unique())
    rng = np.random.default_rng(args.seed)

    manifest = {"seed": args.seed, "source": "synergy_studies_train.jsonl", "strata": {}}
    for size in args.sizes:
        for draw in range(args.n_draws):
            chosen = sorted(rng.choice(dataset_ids, size=size, replace=False).tolist())
            name = f"train-random-n{size}-r{draw}"
            subset = train[train["dataset_id"].isin(chosen)]
            out_path = studies_path / f"synergy_studies_{name}.jsonl"
            subset.to_json(out_path, orient="records", lines=True)
            manifest["strata"][name] = chosen
            print(f"{out_path}: {len(subset)} row(s) / {len(chosen)} dataset(s)")

    with open(studies_path / "random_strata_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
