import optuna

from asreview.models.classifiers import (
    NaiveBayesClassifier,
    LogisticClassifier,
    SVMClassifier,
    RandomForestClassifier,
)


def naive_bayes_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for alpha (alpha effect is non-linear)
    alpha = trial.suggest_float("alpha", 0.1, 100, log=True)
    return {"alpha": alpha}


def logistic_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for C (C effect is non-linear)
    C = trial.suggest_float("C", 1e-3, 100, log=True)
    return {"C": C}


def svm_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for C (C effect is non-linear)
    C = trial.suggest_float("C", 1e-3, 100, log=True)

    # Use categorical for kernel
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])

    # Only set gamma to a value if we use rbf kernel
    gamma = None
    if kernel == "rbf":
        # Use logarithmic normal distribution for gamma (gamma effect is non-linear)
        gamma = trial.suggest_float("gamma", 1e-4, 10, log=True)
    return {"C": C, "kernel": kernel, "gamma": gamma}


def random_forest_params(trial: optuna.trial.FrozenTrial):
    # Use normal distribution for n_estimators (n_estimators effect is linear)
    n_estimators = trial.suggest_int("n_estimators", 50, 500)

    # Use normal distribution for max_features (max_features effect is linear)
    max_features = trial.suggest_int("max_depth", 2, 20)
    return {"n_estimators": n_estimators, "max_features": max_features}


optuna_studies_params = {
    "nb": naive_bayes_params,
    "log": logistic_params,
    "svm": svm_params,
    "rf": random_forest_params,
}


def naive_bayes_model(params):
    return NaiveBayesClassifier(alpha=params["alpha"])


def logistic_model(params):
    return LogisticClassifier(C=params["C"])


def svm_model(params):
    return SVMClassifier(C=params["C"], kernel=params["kernel"], gamma=params["gamma"])


def random_forest_model(params):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"], max_features=params["max_features"]
    )


optuna_studies_models = {
    "nb": naive_bayes_model,
    "log": logistic_model,
    "svm": svm_model,
    "rf": random_forest_model,
}
