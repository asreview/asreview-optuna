import optuna
from asreview.models.feature_extractors import OneHot, Tfidf


def tfidf_params(trial: optuna.trial.FrozenTrial):
    max_df = trial.suggest_float("tfidf__max_df", 0.5, 1.0)

    min_df = trial.suggest_int("tfidf__min_df", 1, 10)

    ngram_range = trial.suggest_categorical("tfidf__ngram_range", [1, 2, 3])

    sublinear_tf = trial.suggest_categorical("tfidf__sublinear_tf", [True, False])

    return {
        "max_df": max_df,
        "min_df": min_df,
        "ngram_range": (1, ngram_range),
        "sublinear_tf": sublinear_tf,
    }


def onehot_params(trial: optuna.trial.FrozenTrial):
    max_df = trial.suggest_float("onehot__max_df", 0.5, 1.0)

    min_df = trial.suggest_int("onehot__min_df", 1, 10)

    return {
        "max_df": max_df,
        "min_df": min_df,
        "ngram_range": (1, 2),
    }


feature_extractor_params = {
    "tfidf": tfidf_params,
    "onehot": onehot_params,
}

feature_extractors = {
    "tfidf": Tfidf,
    "onehot": OneHot,
}

feature_extractor_static_params = {
    "tfidf": {},
    "onehot": {"ngram_range": (1, 2)},
}


def feature_extractor_kwargs_from_trial_params(
    feature_extractor: str, params: dict
) -> dict:
    """
    Reconstruct feature extractor constructor kwargs from a flat trial params
    dict (e.g. study.best_trial.params), reversing the
    "{feature_extractor}__{kwarg}" naming convention used by the
    *_params(trial) functions above.

    Args:
        feature_extractor (str): Name of the feature extractor.
        params (dict): Flat params dict, e.g. optuna trial.params.

    Returns:
        dict: Kwargs usable as `feature_extractors[feature_extractor](**kwargs)`.
    """
    prefix = f"{feature_extractor}__"
    tuned = {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)}
    if feature_extractor == "tfidf" and "ngram_range" in tuned:
        tuned["ngram_range"] = (1, tuned["ngram_range"])
    return {**tuned, **feature_extractor_static_params.get(feature_extractor, {})}
