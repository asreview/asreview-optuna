import numpy as np
import optuna
from asreview.models.balancers import Balanced
from sklearn.base import BaseEstimator


def _rel_weight(n_one, n_zero, a, alpha):
    """Get the weight of the ones."""
    return a * (n_one / n_zero) ** (-alpha)


def _irrel_weight(n_read, b, beta):
    """Get the weight of the zeros."""
    return 1 - (1 - b) * (1 + np.log(n_read)) ** (-beta)


class DoubleBalance(BaseEstimator):
    """Sample-weight port of ASReview's old "double dynamic resampling" balancer.

    `DoubleBalance` was removed from ASReview in 2024 when the engine moved to a
    `compute_sample_weight`-only balancer API (it originally resampled the training
    set via `sample(labeled_idx, y)`, returning duplicated/undersampled indices). This
    reuses its exact schedule (`_rel_weight`/`_irrel_weight` and the
    `n_rel_train`/`n_irrel_train` derivation are unchanged from the original) but
    stops short of resampling: each record's weight is set to
    `target_count_for_its_class / actual_count_in_that_class`, the expected number of
    times the original's resampling would have copied/kept it. This is exact in
    expectation for the original's tiling behavior; the only mechanic dropped is the
    random-undersampling noise (a fixed weight can't reproduce per-sample
    inclusion/exclusion coin flips), which lets this plug directly into the current
    `ActiveLearningCycle`/`Simulate` pipeline instead of needing a bespoke resampling
    training loop.

    Parameters are the original `DoubleBalance` defaults, not re-tuned here.

    Parameters
    ----------
    a: float
        Governs the weight of the 1's. Higher values mean linearly more 1's in the
        (expected) training sample.
    alpha: float
        Governs the scaling of the weight of the 1's as a function of the ratio of
        ones to zeros.
    b: float
        Governs how strongly sampling depends on the total number of records read so
        far. 1 means no dependence; lower means stronger dependence.
    beta: float
        Governs the scaling of the weight of the zeros as a function of the number of
        records read so far.
    """

    name = "double"
    label = "Dynamic resampling (Double)"

    def __init__(self, a=2.155, alpha=0.94, b=0.789, beta=1.0):
        self.a = a
        self.alpha = alpha
        self.b = b
        self.beta = beta

    def compute_sample_weight(self, y):
        y = np.asarray(y)
        if len(set(y.tolist())) != 2:
            raise ValueError("Only binary classification is supported.")

        n_one = int((y == 1).sum())
        n_zero = int((y == 0).sum())
        n_read = n_one + n_zero

        rel_w = _rel_weight(n_one, n_zero, self.a, self.alpha)
        irrel_w = _irrel_weight(n_read, self.b, self.beta)
        tot_zo_weight = rel_w * n_one + irrel_w * n_zero

        n_rel_train = rel_w * n_one * n_read / tot_zo_weight
        # max(1, ...) must be outermost, so tiny n_read (e.g. 2, straight after the 2
        # priors) still guarantees at least 1 "slot" for the relevant class instead of
        # clamping it to 0.
        n_rel_train = max(1, min(n_read - 2, n_rel_train))
        n_irrel_train = n_read - n_rel_train

        weights = np.empty(n_read)
        weights[y == 1] = n_rel_train / n_one
        weights[y == 0] = n_irrel_train / n_zero

        # Match `Balanced`'s convention: rescale so weights sum to len(y).
        return weights * (n_read / weights.sum())


def ratio_params(trial: optuna.trial.Trial) -> dict:
    # Flat, unprefixed key name -- unchanged from before --balancer existed, so
    # completed studies stay readable by `balancer_kwargs_from_trial_params`.
    return {"ratio": trial.suggest_float("ratio", 1.0, 10.0)}


def double_params(trial: optuna.trial.Trial) -> dict:
    return {
        "a": trial.suggest_float("double__a", 0.1, 8.0),
        "alpha": trial.suggest_float("double__alpha", 0.05, 3.0),
        "b": trial.suggest_float("double__b", 0.02, 1.0),
        "beta": trial.suggest_float("double__beta", 0.05, 4.0),
    }


balancer_params = {
    "ratio": ratio_params,
    "double": double_params,
}


balancers = {
    "ratio": Balanced,
    "double": DoubleBalance,
}


def balancer_kwargs_from_trial_params(balancer: str, params: dict) -> dict:
    """
    Reconstruct balancer constructor kwargs from a flat trial params dict (e.g.
    `study.best_trial.params`), reversing the search-space naming used by the
    `*_params(trial)` functions above.

    `"ratio"` is special-cased to read the flat, unprefixed `"ratio"` key directly
    (rather than a `"ratio__ratio"`-style prefix) for backward compatibility with
    studies completed before `--balancer` existed, which always tuned
    `Balanced(ratio=...)` under that exact key.

    Args:
        balancer (str): Name of the balancer (key into `balancers`).
        params (dict): Flat params dict, e.g. `study.best_trial.params`.

    Returns:
        dict: Kwargs usable as `balancers[balancer](**kwargs)`.
    """
    if balancer == "ratio":
        return {"ratio": params["ratio"]}
    prefix = f"{balancer}__"
    return {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)}
