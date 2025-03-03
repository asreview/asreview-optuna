import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["KERAS_BACKEND"] = "torch"
import keras
import optuna
from asreview.models.classifiers import (
    SVM,
    Logistic,
    NaiveBayes,
    RandomForest,
)
from keras.src.regularizers import L2
from keras.src.layers import Dense, Input, BatchNormalization, Dropout


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


def nn_params(trial: optuna.trial.FrozenTrial):
    # Regularization parameters
    l2_alpha = trial.suggest_float("l2_alpha", 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    batch_norm = trial.suggest_categorical("batch_norm", [True, False])

    # Learning parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # Architecture parameters
    activation = trial.suggest_categorical(
        "activation", ["relu", "selu", "swish", "gelu"]
    )

    hidden_layers_1 = trial.suggest_categorical("hidden_layers_1", [2, 4])
    hidden_layer_2 = trial.suggest_categorical("hidden_layer_2", [1, 2])

    loss = trial.suggest_categorical("loss", ["focal", "crossEntropy"])

    epochs = trial.suggest_int("epochs", 10, 100, step=10)

    return {
        "l2_alpha": l2_alpha,
        "dropout_rate": dropout_rate,
        "batch_norm": batch_norm,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "activation": activation,
        "hidden_layers_1": hidden_layers_1,
        "hidden_layers_2": hidden_layer_2,
        "loss_func": loss,
        "epochs": epochs,
    }


classifier_params = {
    "nb": naive_bayes_params,
    "log": logistic_params,
    "svm": svm_params,
    "rf": random_forest_params,
    "nn": nn_params,
}


class NN(keras.wrappers.SKLearnClassifier):
    """Multi Layer Perceptron classifier based on KerasClassifier."""

    name = "nn"
    label = "Multi Layer Perceptron"

    def __init__(self, **kwargs):
        self.epochs = kwargs.pop("epochs")
        self.batch_size = kwargs.pop("batch_size")

        def build_nn_model(
            X,
            y,
            batch_norm,
            activation,
            dropout_rate,
            l2_alpha,
            learning_rate,
            hidden_layers_1,
            hidden_layers_2,
            loss_func,
        ):
            # Creates a basic MLP model dynamically choosing the input and
            # output shapes.
            n_features_in = X.shape[1]
            inp = Input(shape=(n_features_in,))

            if batch_norm:
                inp = BatchNormalization()(inp)

            layers = [
                n_features_in // hidden_layers_1,
                n_features_in // (hidden_layers_1 * hidden_layers_2),
            ]
            hidden = inp
            for layer_size in layers:
                hidden = Dense(
                    layer_size,
                    activation=activation,
                    kernel_regularizer=L2(l2_alpha),
                    kernel_initializer="he_normal",
                )(hidden)

                hidden = Dropout(rate=dropout_rate)(hidden)

                # Batch normalization (if enabled)
                if batch_norm:
                    hidden = BatchNormalization()(hidden)

            n_outputs = y.shape[1] if len(y.shape) > 1 else 1
            out = [Dense(n_outputs, activation="softmax")(hidden)]
            model = keras.Model(inp, out)

            # Learning rate scheduler
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=1000,
                decay_rate=0.9,
                staircase=True,
            )

            model.compile(
                loss=keras.losses.BinaryFocalCrossentropy()
                if loss_func == "focal"
                else keras.losses.BinaryCrossentropy(),
                optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                metrics=["Accuracy"],
            )

            return model

        super().__init__(model=build_nn_model, model_kwargs=kwargs)

    def fit(self, X, y, sample_weight):
        print(self.epochs)
        super().fit(X, y, sample_weight=sample_weight, epochs=self.epochs, verbose=0, batch_size=self.batch_size)

    def predict_proba(self, X):
        return self.model_.predict(X, verbose=0)


classifiers = {
    "nb": NaiveBayes,
    "log": Logistic,
    "svm": SVM,
    "rf": RandomForest,
    "nn": NN,
}
