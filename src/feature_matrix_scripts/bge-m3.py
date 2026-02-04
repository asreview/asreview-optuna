import pickle
from pathlib import Path

import synergy_dataset as sd
from FlagEmbedding import BGEM3FlagModel
from sklearn.preprocessing import normalize
from tqdm import tqdm

FORCE = False

folder_pickle_files = Path("..", "spreprocessed_fms", "bge-m3")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

model = BGEM3FlagModel("BAAI/bge-m3", devices=["cuda:0"])

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

    # Generate embeddings using the LLM embedder
    X = model.encode(
        combined_texts,
        batch_size=128,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    X["dense_vecs_norm"] = normalize(X["dense_vecs"], norm="l2")

    # Save embeddings and labels as a pickle file
    with open(pickle_file_path, "wb") as f:
        pickle.dump((X["dense_vecs_norm"], df["label_included"].tolist()), f)
