import optuna
from asreview.models.feature_extractors import OneHot, Tfidf


def tfidf_params(trial: optuna.trial.FrozenTrial):
    max_df = trial.suggest_float("tfidf__max_df", 0.5, 1.0)

    min_df = trial.suggest_int("tfidf__min_df", 1, 10)

    return {
        "max_df": max_df,
        "min_df": min_df,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
    }


def onehot_params(trial: optuna.trial.FrozenTrial):
    max_df = trial.suggest_float("onehot__max_df", 0.5, 1.0)

    min_df = trial.suggest_int("onehot__min_df", 1, 10)

    # trial.suggest_categorical does not support tuples, so choose max_ngram_range first, then create a tuple.
    max_ngram_range = trial.suggest_int("onehot__max_ngram_range", 1, 3)
    ngram_range = (1, max_ngram_range)
    return {
        "max_df": max_df,
        "min_df": min_df,
        "ngram_range": ngram_range,
    }


feature_extractor_params = {
    "tfidf": tfidf_params,
    "onehot": onehot_params,
}

feature_extractors = {
    "tfidf": Tfidf,
    "onehot": OneHot,
}
