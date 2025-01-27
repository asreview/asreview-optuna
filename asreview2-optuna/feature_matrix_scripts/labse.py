from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import synergy_dataset as sd  # Assuming this is your custom dataset handler
import torch

# Folder to save embeddings
folder_pickle_files = Path("synergy-dataset", "pickles_labse")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

# Load LaBSE model
model = SentenceTransformer('sentence-transformers/LaBSE')

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Loop through datasets
for dataset in tqdm(sd.iter_datasets(), total=26):
    # Convert dataset to a DataFrame and reset index
    df = dataset.to_frame().reset_index()

    # Combine 'title' and 'abstract' text
    combined_texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

    # Generate embeddings
    X = model.encode(combined_texts, batch_size=64, show_progress_bar=False, device=device)

    # Save embeddings and labels as a pickle file
    with open(folder_pickle_files / f"{dataset.name}.pkl", "wb") as f:
        pickle.dump((X, df["label_included"].tolist()), f)