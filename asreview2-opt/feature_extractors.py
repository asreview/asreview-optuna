import optuna

from asreview.models.feature_extraction import Tfidf


def tfidf_params(trial: optuna.trial.FrozenTrial):
    max_features = trial.suggest_int("tfidf__max_features", 15_000, 50_000)

    max_df = trial.suggest_float("tfidf__max_df", 0.7, 0.9)

    min_df = 1

    ngram_range = (1, 2)

    sublinear_tf = False

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
