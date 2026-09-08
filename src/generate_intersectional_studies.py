"""
Generate train-split JSONL study files for intersectional strata (e.g.
domain x size: 'health-small', 'nonhealth-large', ...), for the cross-strata
HPO check in paper/results.md's open items.

Reuses stratification_manifest.json's existing per-dataset axis labels (no
new tertile/threshold computation) and generate_studies.py's
sample_priors_for_dataset/write_jsonl, matching its seed/n_priors so the
generated priors are consistent with the rest of the project.

Also writes the joint stratum label back into the manifest (as
f"{axis_a}_{axis_b}_stratum") for later reuse by evaluation scripts.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_studies import generate_rows, write_jsonl  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate train-split JSONL files for one intersectional "
        "(axis_a x axis_b) stratification, e.g. domain x size."
    )
    parser.add_argument("--axis-a", required=True, help="e.g. domain")
    parser.add_argument("--axis-b", required=True, help="e.g. size")
    parser.add_argument(
        "--data-path", required=True, help="Path to the synergy_plus data directory."
    )
    parser.add_argument(
        "--studies-path", default=str(Path(__file__).resolve().parent / "studies")
    )
    args = parser.parse_args()

    studies_path = Path(args.studies_path)
    manifest_path = studies_path / "stratification_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    axis_a, axis_b = args.axis_a, args.axis_b
    joint_axis_name = f"{axis_a}_{axis_b}"
    col_a, col_b = f"{axis_a}_stratum", f"{axis_b}_stratum"

    joint_by_stratum: dict[str, list[str]] = {}
    for dataset_id, entry in manifest["datasets"].items():
        if entry.get("split") != "train":
            continue
        stratum_a, stratum_b = entry.get(col_a), entry.get(col_b)
        if stratum_a is None or stratum_b is None:
            continue
        joint = f"{stratum_a}-{stratum_b}"
        entry[f"{joint_axis_name}_stratum"] = joint
        joint_by_stratum.setdefault(joint, []).append(dataset_id)

    manifest["axes_computed"] = sorted(set(manifest.get("axes_computed", [])) | {joint_axis_name})
    manifest[f"{joint_axis_name}_axis"] = {
        "derived_from": [axis_a, axis_b],
        "labels": sorted(joint_by_stratum),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Updated {manifest_path} with {joint_axis_name}_stratum labels")

    for joint_stratum, dataset_ids in sorted(joint_by_stratum.items()):
        dataset_ids = sorted(dataset_ids)
        rows = generate_rows(dataset_ids, args.data_path, manifest["n_priors"], manifest["seed"])
        out_path = studies_path / f"synergy_studies_train-{joint_axis_name}-{joint_stratum}.jsonl"
        write_jsonl(rows, out_path)
        print(f"{out_path}: {len(rows)} row(s) / {len(dataset_ids)} dataset(s)")
