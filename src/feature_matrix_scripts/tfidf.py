import pickle
from pathlib import Path

import pandas as pd
import synergy_dataset as sd
from asreview.models.feature_extractors import Tfidf
from tqdm import tqdm

FORCE = False

folder_pickle_files = Path("synergy-dataset", "pickles_tfidf")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

for dataset in tqdm(sd.iter_datasets(), total=26):
    # Load dataset
    if dataset.name == "Moran_2021_corrected":
        df = pd.read_csv("./datasets/Moran_2021_corrected_shuffled_raw.csv")
    elif dataset.name == "Muthu_2021_corrected":
        df = pd.read_csv("./datasets/Muthu_2021_corrected_shuffled_raw.csv")
    else:
        df = dataset.to_frame().reset_index()

    if dataset.name == "Moran_2021":
        dataset_name = "Moran_2021_corrected"
    elif dataset.name == "Muthu_2021":
        dataset_name = "Muthu_2021_corrected"
    else:
        dataset_name = dataset.name

    pickle_file_path = folder_pickle_files / f"{dataset_name}.pkl"

    # Check if the pickle file already exists
    if not FORCE and pickle_file_path.exists():
        print(f"Skipping {dataset_name}, pickle file already exists.")
        continue

    X = Tfidf().fit_transform(df)

    with open(folder_pickle_files / f"{dataset_name}.pkl", "wb") as f:
        pickle.dump((X, df["label_included"]), f)
