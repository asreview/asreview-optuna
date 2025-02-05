import optuna
from asreview.models.classifiers import (
    Logistic,
    NaiveBayes,
    RandomForest,
    SVM,
)

from xgboost import XGBClassifier


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
    n_estimators = trial.suggest_int("rf__n_estimators", 50, 500)

    # Use normal distribution for max_features (max_features effect is linear)
    max_features = trial.suggest_int("rf__max_features", 2, 20)
    return {"n_estimators": n_estimators, "max_features": max_features}


def xgboost_params(trial: optuna.trial.FrozenTrial):
    # Use normal distribution for n_estimators (n_estimators effect is linear)
    n_estimators = trial.suggest_int("xgboost__n_estimators", 50, 500)

    # Use normal distribution for max_depth (max_depth effect is linear)
    max_depth = trial.suggest_int("xgboost__max_depth", 2, 20)
    return {"n_estimators": n_estimators, "max_depth": max_depth}


classifier_params = {
    "nb": naive_bayes_params,
    "log": logistic_params,
    "svm": svm_params,
    "rf": random_forest_params,
    "xgboost": xgboost_params,
}

class XGBoost(XGBClassifier):
    """XGBoost classifier.

    """

    name = "xgboost"
    label = "XGBoost"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

classifiers = {
    "nb": NaiveBayes,
    "log": Logistic,
    "svm": SVM,
    "rf": RandomForest,
    "xgboost": XGBoost,
}
