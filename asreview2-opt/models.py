import optuna

from asreview.models.classifiers import (
    NaiveBayesClassifier,
    LogisticClassifier,
    SVMClassifier,
    RandomForestClassifier,
)

from custom_models import CustomRandomForestClassifier


def naive_bayes_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for alpha (alpha effect is non-linear)
    alpha = trial.suggest_float("alpha", 0.1, 100, log=True)
    return {"alpha": alpha}


def logistic_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for C (C effect is non-linear)
    C = trial.suggest_float("C", 1e-3, 100, log=True)
    return {"C": C, "solver": "lbfgs"}


def svm_params(trial: optuna.trial.FrozenTrial):
    # Use logarithmic normal distribution for C (C effect is non-linear)
    C = trial.suggest_float("C", 1e-3, 100, log=True)

    # Use categorical for kernel
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])

    # Only set gamma to a value if we use rbf kernel
    gamma = "scale"
    if kernel == "rbf":
        # Use logarithmic normal distribution for gamma (gamma effect is non-linear)
        gamma = trial.suggest_float("gamma", 1e-4, 10, log=True)
    return {"C": C, "kernel": kernel, "gamma": gamma}


def random_forest_params(trial: optuna.trial.FrozenTrial):
    # Use normal distribution for n_estimators (n_estimators effect is linear)
    n_estimators = trial.suggest_int("n_estimators", 50, 500)

    # Use normal distribution for max_features (max_features effect is linear)
    max_features = trial.suggest_int("max_features", 2, 20)
    return {"n_estimators": n_estimators, "max_features": max_features}


def random_forest_custom_params(trial: optuna.trial.FrozenTrial):
    n_estimators = 100
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    min_weight_fraction_leaf = 0.0
    max_features = "sqrt"
    max_leaf_nodes = None
    min_impurity_decrease = 0.0
    random_state = None
    ccp_alpha = 0.0
    monotonic_cst = None

    return {
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "min_weight_fraction_leaf": min_weight_fraction_leaf,
        "max_features": max_features,
        "max_leaf_nodes": max_leaf_nodes,
        "min_impurity_decrease": min_impurity_decrease,
        "random_state": random_state,
        "ccp_alpha": ccp_alpha,
        "monotonic_cst": monotonic_cst,
    }


optuna_studies_params = {
    "nb": naive_bayes_params,
    "log": logistic_params,
    "svm": svm_params,
    "rf": random_forest_params,
    "rf_c": random_forest_custom_params,
}


optuna_studies_models = {
    "nb": NaiveBayesClassifier,
    "log": LogisticClassifier,
    "svm": SVMClassifier,
    "rf": RandomForestClassifier,
    "rf_c": CustomRandomForestClassifier,
}
