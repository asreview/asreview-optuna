import pickle
from pathlib import Path

import synergy_dataset as sd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

FORCE = True

# Folder to save embeddings
folder_pickle_files = Path("..", "preprocessed_fms", "pickles_mxbai")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

# Load LaBSE model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Loop through datasets
for dataset in tqdm(sd.iter_datasets(), total=26):
    df = dataset.to_frame().reset_index()
    dataset_name = dataset.name

    # Combine 'title' and 'abstract' text
    combined_texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

    pickle_file_path = folder_pickle_files / f"{dataset_name}.pkl"

    # Check if the pickle file already exists
    if not FORCE and pickle_file_path.exists():
        print(f"Skipping {dataset_name}, pickle file already exists.")
        continue

    # Generate embeddings
    X = model.encode(
        combined_texts,
        batch_size=512,
        show_progress_bar=False,
        device=device,
        normalize_embeddings=True,
    )

    # Save embeddings and labels as a pickle file
    with open(pickle_file_path, "wb") as f:
        pickle.dump(
            (
                X,
                df["label_included"].tolist(),
            ),
            f,
        )
