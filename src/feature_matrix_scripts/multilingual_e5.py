import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "intfloat/multilingual-e5-large"


def main():
    parser = argparse.ArgumentParser(
        description="Precompute multilingual-e5 embeddings for synergy_plus datasets."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the synergy_plus data directory.",
    )
    parser.add_argument(
        "--fms-path",
        default=str(Path(__file__).resolve().parent.parent / "preprocessed_fms"),
        help="Output root for preprocessed feature matrices.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=None,
        help="Restrict to specific dataset id(s); repeatable. Default: all datasets in metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a pickle already exists.",
    )
    args = parser.parse_args()

    out_dir = Path(args.fms_path) / "multilingual-e5"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_id:
        dataset_ids = args.dataset_id
    else:
        metadata = pd.read_csv(Path(args.data_path) / "metadata" / "review_metadata.csv")
        dataset_ids = sorted(metadata["key"])

    model = SentenceTransformer(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for dataset_id in tqdm(dataset_ids):
        pickle_file_path = out_dir / f"{dataset_id}.pkl"
        if not args.force and pickle_file_path.exists():
            print(f"Skipping {dataset_id}, pickle file already exists.")
            continue

        df = pd.read_csv(Path(args.data_path) / f"{dataset_id}.csv")
        # E5 models expect an asymmetric "passage: " prefix for corpus/document text.
        combined_texts = (
            "passage: " + df["title"].fillna("") + " " + df["abstract"].fillna("")
        ).tolist()

        X = model.encode(
            combined_texts,
            batch_size=512,
            show_progress_bar=False,
            device=device,
            normalize_embeddings=True,
        )

        with open(pickle_file_path, "wb") as f:
            pickle.dump((X, df["label_included"].tolist()), f)


if __name__ == "__main__":
    main()
