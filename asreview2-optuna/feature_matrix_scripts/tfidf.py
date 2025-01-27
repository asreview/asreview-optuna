from pathlib import Path
import pickle

import synergy_dataset as sd
import pandas as pd
from tqdm import tqdm

from asreview.models.feature_extraction import Tfidf

FORCE = False

folder_pickle_files = Path("synergy-dataset", "pickles_tfidf")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

for dataset in tqdm(sd.iter_datasets(), total=26):
    if dataset.name == "Moran_2021":
        df = pd.read_csv("../datasets/Moran_2021_corrected_shuffled_raw.csv")
    else:
        # Convert dataset to a DataFrame and reset index
        df = dataset.to_frame().reset_index()

    dataset_name = (
        dataset.name if dataset.name != "Moran_2021" else "Moran_2021_corrected"
    )
    pickle_file_path = folder_pickle_files / f"{dataset_name}.pkl"

    # Check if the pickle file already exists
    if not FORCE and pickle_file_path.exists():
        print(f"Skipping {dataset_name}, pickle file already exists.")
        continue

    X = Tfidf().fit_transform(df)

    with open(folder_pickle_files / f"{dataset_name}.pkl", "wb") as f:
        pickle.dump((X, df["label_included"]), f)
