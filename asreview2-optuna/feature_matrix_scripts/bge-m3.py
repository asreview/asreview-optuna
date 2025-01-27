from pathlib import Path
import pickle
from FlagEmbedding import BGEM3FlagModel

import synergy_dataset as sd
from tqdm import tqdm
import pandas as pd

FORCE = False

folder_pickle_files = Path("synergy-dataset", "pickles_bge-m3")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

model = BGEM3FlagModel("BAAI/bge-m3", devices=["cuda:0"])

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

    # Generate embeddings using the LLM embedder
    X = model.encode(
        combined_texts,
        batch_size=128,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    # Save embeddings and labels as a pickle file
    with open(folder_pickle_files / f"{dataset_name}.pkl", "wb") as f:
        pickle.dump((X["dense_vecs"], df["label_included"].tolist()), f)

    with open(folder_pickle_files / f"sparse-{dataset_name}.pkl", "wb") as f:
        pickle.dump((X["lexical_weights"], df["label_included"].tolist()), f)
