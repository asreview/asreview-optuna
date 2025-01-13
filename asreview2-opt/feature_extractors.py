import optuna

from asreview.models.feature_extraction import Tfidf


def tfidf_params(trial: optuna.trial.FrozenTrial):
    max_features = trial.suggest_int("tfidf__max_features", 200, 20_000)

    max_df = trial.suggest_float("tfidf__max_df", 0.5, 1.0)

    min_df = trial.suggest_int("tfidf__min_df", 1, 10)

    ngram_range = trial.suggest_categorical(
        "tfidf__ngram_range", [(1, 1), (1, 2), (1, 3)]
    )

    sublinear_tf = trial.suggest_categorical("tfidf__sublinear_tf", [True, False])

    return {
        "max_features": max_features,
        "max_df": max_df,
        "min_df": min_df,
        "ngram_range": ngram_range,
        "sublinear_tf": sublinear_tf,
    }


feature_extractor_params = {
    "tfidf": tfidf_params,
}


feature_extractors = {
    "tfidf": Tfidf,
}
