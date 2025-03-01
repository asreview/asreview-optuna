import pickle
from pathlib import Path

import pandas as pd
import synergy_dataset as sd  # Assuming this is your custom dataset handler
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

FORCE = False

# Folder to save embeddings
folder_pickle_files = Path("synergy-dataset", "pickles_lajavaness")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

# Load LaBSE model
model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Loop through datasets
for dataset in tqdm(sd.iter_datasets(), total=26):
    if dataset.name == "Moran_2021":
        df = pd.read_csv("../datasets/Moran_2021_corrected_shuffled_raw.csv")
    else:
        # Convert dataset to a DataFrame and reset index
        df = dataset.to_frame().reset_index()

    # Combine 'title' and 'abstract' text
    combined_texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

    dataset_name = (
        dataset.name if dataset.name != "Moran_2021" else "Moran_2021_corrected"
    )
    pickle_file_path = folder_pickle_files / f"{dataset_name}.pkl"

    # Check if the pickle file already exists
    if not FORCE and pickle_file_path.exists():
        print(f"Skipping {dataset_name}, pickle file already exists.")
        continue

    # Generate embeddings
    X = model.encode(
        combined_texts, batch_size=128, show_progress_bar=False, device=device
    )

    # Save embeddings and labels as a pickle file
    with open(folder_pickle_files / f"{dataset_name}.pkl", "wb") as f:
        pickle.dump(
            (
                X,
                df["label_included"].tolist(),
            ),
            f,
        )
