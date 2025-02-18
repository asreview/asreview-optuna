import optuna
from asreview.models.classifiers import (
    Logistic,
    NaiveBayes,
    RandomForest,
    SVM,
)

from sklearn.neural_network import MLPClassifier


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

def mlp_params(trial: optuna.trial.FrozenTrial):
    alpha = trial.suggest_categorical("mlp__alpha", [1e-5, 1e-4, 1e-3, 1e-2])
    learning_rate_init = trial.suggest_categorical("mlp__lr_init", [0.001, 0.01])
    activation = 'relu'#trial.suggest_categorical("mlp__activation", ['relu', 'tanh'])
    solver = 'adam'#trial.suggest_categorical("mlp__solver", ['adam', 'lbfgs'])
    batch_size = trial.suggest_categorical("mlp__batch_size", [256, 512])
    max_iter = trial.suggest_categorical("mlp__max_iter", [200, 500, 1000])
    early_stopping = True #trial.suggest_categorical("mlp__early_stopping", [True, False])

    return {"alpha": alpha, "learning_rate_init": learning_rate_init, "activation": activation, "solver": solver, "batch_size": batch_size, "max_iter": max_iter, "early_stopping": early_stopping}


classifier_params = {
    "nb": naive_bayes_params,
    "log": logistic_params,
    "svm": svm_params,
    "rf": random_forest_params,
    "nn": mlp_params,
}

class MLP(MLPClassifier):
    """Multi Layer Perceptron classifier.

    Based on the sklearn implementation of the MLP
    sklearn.neural_network
    """

    name = "nn"
    label = "Multi Layer Perceptron"

    def __init__(self, n_dims=1024, alpha=0.001, activation="relu", solver="adam", max_iter=200, **kwargs):
        self.n_dims = n_dims
        hidden_layer_1 = max(self.n_dims // 4, 64)  # At least 64 neurons
        hidden_layer_2 = max(self.n_dims // 8, 32)  # At least 32 neurons
        layer_sizes = (hidden_layer_1, hidden_layer_2)
        super().__init__(
            hidden_layer_sizes=layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            **kwargs,
        )

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y)

classifiers = {
    "nb": NaiveBayes,
    "log": Logistic,
    "svm": SVM,
    "rf": RandomForest,
    "nn": MLP,
}
