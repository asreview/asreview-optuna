from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
import optuna


PICKLE_FOLDER_PATH = Path("synergy-dataset", "pickles")
STUDY_NAME = "ASReview2.x " + datetime.now().strftime("%Y-%m-%d at %H.%M.%S")
NUMBER_OF_STUDIES = 5

# list of studies
studies = pd.read_json("synergy_studies_1000.jsonl", lines=True).head(
    n=NUMBER_OF_STUDIES
)


# Function to run the loop in parallel
def run_parallel(studies, alpha, ratio):
    losses = []
    num_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        futures = {
            executor.submit(process_row, row, alpha, ratio): i
            for i, row in studies.iterrows()
        }
        # Collect results
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                losses.append(result)
    return np.mean(losses) if losses else None


# Function to process each row
def process_row(row, alpha, ratio):
    import pickle  # Ensure imports within the worker function to avoid multiprocessing issues
    import asreview
    from asreview.metrics import loss
    from asreview.models.balance import Balanced
    from asreview.models.classifiers import NaiveBayesClassifier
    from asreview.models.query import MaxQuery
    from asreview.models.feature_extraction import Tfidf

    try:
        with open(PICKLE_FOLDER_PATH / f"{row['dataset_id']}.pkl", "rb") as f:
            fm, labels = pickle.load(f)
        
        priors = row["prior_inclusions"] + row["prior_exclusions"]
        
        # Create classifier and balancer with optuna value
        clf = NaiveBayesClassifier(alpha=alpha)
        blc = Balanced(ratio=ratio)  # Peter's balancer
        
        # Setup simulation
        simulate = asreview.Simulate(
            fm=fm,
            labels=labels,
            classifier=clf,
            balance_strategy=blc,
            query_strategy=MaxQuery(),
            feature_extraction=Tfidf(),
        )
        
        # Set priors
        simulate.label(priors, prior=True)
        
        # Start simulation
        simulate.review()

        # Return loss
        padded_results = list(simulate._results["label"]) + [0] * (
            len(simulate.labels) - len(simulate._results["label"])
        )
        return loss(padded_results)
    
    except Exception as e:
        print(f"Error processing dataset {row['dataset_id']}: {e}")
        return None


def objective(trial):
    # Use logarithmic normal distribution for alpha (alpha effect is non-linear)
    alpha = trial.suggest_float("alpha", 1e-3, 100, log=True)
    
    # Use normal distribution for ratio (ratio effect is linear)
    ratio = trial.suggest_float("ratio", 0.0, 30.0)

    return run_parallel(studies, alpha=alpha, ratio=ratio)


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name=STUDY_NAME,
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=5)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
