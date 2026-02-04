import pickle
from pathlib import Path

import synergy_dataset as sd
from asreview.models.feature_extractors import Tfidf
from tqdm import tqdm

FORCE = False

folder_pickle_files = Path("..", "preprocessed_fms", "tfidf")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

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

    X = Tfidf(ngram_range=(1,2), sublinear_tf=True, min_df=1, max_df=0.95).fit_transform(df)

    with open(pickle_file_path, "wb") as f:
        pickle.dump((X, df["label_included"]), f)
