from pathlib import Path
import pickle

import synergy_dataset as sd
import asreview as asr

from asreview.models.feature_extraction import Tfidf

print("synergy-dataset version", sd.__version__)
print("asreview version:", asr.__version__)

folder_pickle_files = Path("synergy-dataset", "pickles")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

for dataset in sd.iter_datasets():
    df = dataset.to_frame().reset_index()

    X = Tfidf().fit_transform(df)

    with open(folder_pickle_files / f"{dataset.name}.pkl", "wb") as f:
        pickle.dump((X, df["label_included"]), f)
