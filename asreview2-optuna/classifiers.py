import optuna
from asreview.models.classifiers import (
    Logistic,
    NaiveBayes,
    RandomForest,
    SVM,
)

from sklearn.ensemble import RandomForestClassifier


def naive_bayes_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for alpha (alpha effect is non-linear)
    alpha = trial.suggest_float("nb__alpha", 0.1, 100, log=True)
    return {"alpha": alpha}


def logistic_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for C (C effect is non-linear)
    C = trial.suggest_float("log__C", 1e-3, 100, log=True)
    return {"C": C, "solver": "lbfgs"}


def svm_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for C (C effect is non-linear)
    C = trial.suggest_float("svm__C", 1e-3, 100, log=True)
    return {"C": C}


def random_forest_params(trial: optuna.trial.FrozenTrial):
    # Use normal distribution for n_estimators (n_estimators effect is linear)
    n_estimators = trial.suggest_int("rf__n_estimators", 50, 200)

    # Use normal distribution for max_features (max_features effect is linear)
    max_features = trial.suggest_categorical("rf__max_features", ["sqrt", "log2"])

    return {"n_estimators": n_estimators, "max_features": max_features}


classifier_params = {
    "nb": naive_bayes_params,
    "log": logistic_params,
    "svm": svm_params,
    "rf": random_forest_params,
}


class RFClassifier(RandomForestClassifier):
    """Random forest classifier.

    Based on the sklearn implementation of the random forest
    sklearn.ensemble.RandomForestClassifier.
    """

    name = "rf"
    label = "Random forest"

    def __init__(self, n_estimators=100, max_features=10, **kwargs):
        super().__init__(
            n_estimators=int(n_estimators),
            max_features=max_features,
            **kwargs,
        )


classifiers = {
    "nb": NaiveBayes,
    "log": Logistic,
    "svm": SVM,
    "rf": RandomForest,
}
