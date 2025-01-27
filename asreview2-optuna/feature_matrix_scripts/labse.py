from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer

import synergy_dataset as sd
from tqdm import tqdm

folder_pickle_files = Path("synergy-dataset", "pickles_labse")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer('sentence-transformers/LaBSE')

# Loop through datasets
for dataset in tqdm(sd.iter_datasets(), total=26):
    # Convert dataset to a DataFrame and reset index
    df = dataset.to_frame().reset_index()

    # Combine 'title' and 'abstract' text
    combined_texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

    # Generate embeddings using the LLM embedder
    X = model.encode(combined_texts)

    # Save embeddings and labels as a pickle file
    with open(folder_pickle_files / f"{dataset.name}.pkl", "wb") as f:
        pickle.dump((X["dense_vecs"], df["label_included"].tolist()), f)