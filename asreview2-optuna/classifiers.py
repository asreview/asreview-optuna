import optuna
from asreview.models.classifiers import (
    Logistic,
    NaiveBayes,
    RandomForest,
    SVM,
)

from sklearn.svm import LinearSVC


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
    C = trial.suggest_float("svm__C", 0.001, 100, log=True)
    
    return {"C": C, "loss": "squared_hinge"}


def random_forest_params(trial: optuna.trial.FrozenTrial):
    # Use normal distribution for n_estimators (n_estimators effect is linear)
    n_estimators = trial.suggest_int("rf__n_estimators", 50, 500)

    # Use normal distribution for max_features (max_features effect is linear)
    max_features = trial.suggest_int("rf__max_features", 2, 20)
    return {"n_estimators": n_estimators, "max_features": max_features}


classifier_params = {
    "nb": naive_bayes_params,
    "log": logistic_params,
    "svm": svm_params,
    "rf": random_forest_params,
}


class LinearSVMClassifier(LinearSVC):
    """Support vector machine classifier.

    Based on the sklearn implementation of the support vector machine
    sklearn.svm.LinearSVC.
    """

    name = "svm"
    label = "Support vector machine"

    def __init__(self, C=15.4, **kwargs):
        super().__init__(
            C=C,
            **kwargs,
        )

classifiers = {
    "nb": NaiveBayes,
    "log": Logistic,
    "svm": SVM,
    "rf": RandomForest,
}
