import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import synergy_dataset as sd
from gensim.models.doc2vec import Doc2Vec as GenSimDoc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.preprocessing import normalize as SKNormalize
from tqdm import tqdm


class Doc2Vec:
    """
    Doc2Vec feature extraction technique (``doc2vec``).

    Feature extraction technique provided by the `gensim
    <https://radimrehurek.com/gensim/>`__ package. It trains a model to generate
    document embeddings, which can reduce dimensionality and accelerate modeling.

    .. note::

        For fully reproducible runs, limit the model to a single worker thread
        (`n_jobs=1`) to eliminate potential variability due to thread scheduling.

    Parameters
    ----------
    vector_size : int, optional
        Dimensionality of the feature vectors. Default: 40
    epochs : int, optional
        Number of epochs to train the model. Default: 33
    min_count : int, optional
        Ignores all words with total frequency lower than this. Default: 1
    n_jobs : int, optional
        Number of threads to use during training. Default: 1
    window : int, optional
        Maximum distance between the current and predicted word. Default: 7
    dm_concat : bool, optional
        If True, concatenate word vectors. Default: False
    dm : int, optional
        Training model:
        - 0: Distributed Bag of Words (DBOW)
        - 1: Distributed Memory (DM)
        - 2: Both DBOW and DM (concatenated embeddings). Default: 2
    dbow_words : bool, optional
        Train word vectors alongside DBOW. Default: False
    normalize : bool, optional
        Normalize embeddings using min-max scaling. Default: True
    verbose : bool, optional
        Print progress and status updates. Default: True
    """

    def __init__(
        self,
        vector_size=40,
        epochs=33,
        min_count=1,
        n_jobs=1,
        window=7,
        dm_concat=False,
        dm=2,
        dbow_words=False,
        normalize=True,
        norm="l2",
        verbose=True,
    ):
        self.vector_size = int(vector_size)
        self.epochs = int(epochs)
        self.min_count = int(min_count)
        self.n_jobs = int(n_jobs)
        self.window = int(window)
        self.dm_concat = 1 if dm_concat else 0
        self.dm = int(dm)
        self.dbow_words = 1 if dbow_words else 0
        self.normalize = normalize
        self.norm = norm
        self.verbose = verbose
        self._model_instance = None

        self._tagged_document = TaggedDocument
        self._simple_preprocess = simple_preprocess
        self._model = GenSimDoc2Vec

    def fit(self, X, y=None):
        if self.verbose:
            print("Preparing corpus...")
        corpus = [
            self._tagged_document(self._simple_preprocess(text), [i])
            for i, text in enumerate(X)
        ]

        model_param = {
            "vector_size": self.vector_size,
            "epochs": self.epochs,
            "min_count": self.min_count,
            "workers": self.n_jobs,
            "window": self.window,
            "dm_concat": self.dm_concat,
            "dbow_words": self.dbow_words,
        }

        if self.dm == 2:
            # Train both DM and DBOW models
            model_param["vector_size"] = int(self.vector_size / 2)
            if self.verbose:
                print("Training DM model...")
            self._model_dm = self._train_model(corpus, **model_param, dm=1)
            if self.verbose:
                print("Training DBOW model...")
            self._model_dbow = self._train_model(corpus, **model_param, dm=0)
        else:
            if self.verbose:
                print(f"Training single model with dm={self.dm}...")
            self._model_instance = self._train_model(corpus, **model_param, dm=self.dm)

    def transform(self, texts):
        if self.verbose:
            print("Preparing corpus for transformation...")
        corpus = [
            self._tagged_document(self._simple_preprocess(text), [i])
            for i, text in enumerate(texts)
        ]

        if self.dm == 2:
            X_dm = self._infer_vectors(self._model_dm, corpus)
            X_dbow = self._infer_vectors(self._model_dbow, corpus)
            X = np.concatenate((X_dm, X_dbow), axis=1)
        else:
            X = self._infer_vectors(self._model_instance, corpus)

        if self.verbose:
            print("Finished transforming texts to vectors.")

        if self.normalize:
            if self.verbose:
                print("Normalizing embeddings.")
            X = SKNormalize(X, norm=self.norm)

        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _train_model(self, corpus, *args, **kwargs):
        model = self._model(*args, **kwargs)
        if self.verbose:
            print("Building vocabulary...")
        model.build_vocab(corpus)
        if self.verbose:
            print("Training model...")
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        if self.verbose:
            print("Model training complete.")
        return model

    def _infer_vectors(self, model, corpus):
        if self.verbose:
            print("Inferring vectors for documents...")
        X = [model.infer_vector(doc.words) for doc in corpus]
        if self.verbose:
            print("Vector inference complete.")
        return np.array(X)


FORCE = True

# Folder to save embeddings
folder_pickle_files = Path("synergy-dataset", "pickles_doc2vec")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

model = Doc2Vec(n_jobs=10)

# Loop through datasets
for dataset in tqdm(sd.iter_datasets(), total=26):
    if dataset.name == "Chou_2004" or dataset.name == "Jeyaraman_2020":
        continue
    elif dataset.name == "Moran_2021":
        df = pd.read_csv("./datasets/Moran_2021_corrected_shuffled_raw.csv")
        dataset_name = "Moran_2021_corrected"
    elif dataset.name == "Muthu_2021":
        df = pd.read_csv("./datasets/Muthu_2021_corrected_shuffled_raw.csv")
        dataset_name = "Muthu_2021_corrected"
    else:
        df = dataset.to_frame().reset_index()
        dataset_name = dataset.name

    # Combine 'title' and 'abstract' text
    combined_texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

    pickle_file_path = folder_pickle_files / f"{dataset_name}.pkl"

    # Check if the pickle file already exists
    if not FORCE and pickle_file_path.exists():
        print(f"Skipping {dataset_name}, pickle file already exists.")
        continue

    # Generate embeddings
    X = model.fit_transform(combined_texts, [])

    # Save embeddings and labels as a pickle file
    with open(folder_pickle_files / f"{dataset_name}.pkl", "wb") as f:
        pickle.dump(
            (
                X,
                df["label_included"].tolist(),
            ),
            f,
        )
