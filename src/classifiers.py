import optuna
from asreview.models.classifiers import (
    SVM,
    Logistic,
    NaiveBayes,
    RandomForest,
)


def naive_bayes_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for alpha (alpha effect is non-linear)
    alpha = trial.suggest_float("nb__alpha", 0.01, 15, log=True)
    return {"alpha": alpha}


def logistic_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for C (C effect is non-linear)
    C = trial.suggest_float("log__C", 1e-3, 100, log=True)
    return {"C": C, "solver": "lbfgs"}


def svm_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for C (C effect is non-linear)
    C = trial.suggest_float("svm__C", 1e-3, 100, log=True)

    return {"C": C, "loss": "squared_hinge", "max_iter": 5000}


def random_forest_params(trial: optuna.trial.FrozenTrial):
    # Use normal distribution for n_estimators (n_estimators effect is linear)
    n_estimators = trial.suggest_categorical("rf__n_estimators", [100, 200, 500, 1000])
    min_samples_split = trial.suggest_categorical("rf__min_samples_split", [2, 3, 4, 5])
    return {
        "n_estimators": n_estimators,
        "max_features": "sqrt",
        "min_samples_split": min_samples_split,
    }


classifier_params = {
    "nb": naive_bayes_params,
    "log": logistic_params,
    "svm": svm_params,
    "rf": random_forest_params,
}


classifiers = {
    "nb": NaiveBayes,
    "log": Logistic,
    "svm": SVM,
    "rf": RandomForest,
}


classifier_static_params = {
    "nb": {},
    "log": {"solver": "lbfgs"},
    "svm": {"loss": "squared_hinge", "max_iter": 5000},
    "rf": {"max_features": "sqrt"},
}


def classifier_kwargs_from_trial_params(classifier: str, params: dict) -> dict:
    """
    Reconstruct classifier constructor kwargs from a flat trial params dict
    (e.g. study.best_trial.params), reversing the "{classifier}__{kwarg}"
    naming convention used by the *_params(trial) functions above.

    Args:
        classifier (str): Name of the classifier (key into `classifiers`).
        params (dict): Flat params dict, e.g. optuna trial.params.

    Returns:
        dict: Kwargs usable as `classifiers[classifier](**kwargs)`.
    """
    prefix = f"{classifier}__"
    tuned = {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)}
    return {**tuned, **classifier_static_params[classifier]}
