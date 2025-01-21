from pathlib import Path
import pickle
from FlagEmbedding import BGEM3FlagModel

import synergy_dataset as sd
from tqdm import tqdm

folder_pickle_files = Path("synergy-dataset", "pickles")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

model = BGEM3FlagModel("BAAI/bge-m3", devices=["cuda:0"])

# Loop through datasets
for dataset in tqdm(sd.iter_datasets(), total=26):
    # Convert dataset to a DataFrame and reset index
    df = dataset.to_frame().reset_index()

    # Combine 'title' and 'abstract' text
    combined_texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

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
    with open(folder_pickle_files / f"dense-{dataset.name}.pkl", "wb") as f:
        pickle.dump((X["dense_vecs"], df["label_included"].tolist()), f)

    with open(folder_pickle_files / f"sparse-{dataset.name}.pkl", "wb") as f:
        pickle.dump((X["lexical_weights"], df["label_included"].tolist()), f)
