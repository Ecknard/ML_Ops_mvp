"""Construction du Pipeline scikit-learn : ColumnTransformer + estimateur.

Un seul endroit décide du prétraitement ; train.py, evaluate.py, predict.py
et l'API appellent tous build_pipeline() pour ne jamais désynchroniser le
prétraitement d'entraînement et celui de l'inférence.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MODEL_REGISTRY = {
    "logreg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": lambda: RandomForestClassifier(random_state=42, class_weight="balanced"),
    "histgradientboosting": lambda: HistGradientBoostingClassifier(random_state=42),
}


def build_preprocessor(feat_cfg: dict) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    numeric_log_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=feat_cfg.get("categorical_min_frequency", 0.01),sparse_output=False
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, feat_cfg["numeric"]),
            ("num_log", numeric_log_pipe, feat_cfg["numeric_log"]),
            ("cat", categorical_pipe, feat_cfg["categorical"]),
        ]
    )


def build_pipeline(model_type: str, feat_cfg: dict) -> Pipeline:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Modèle inconnu : {model_type}. Options : {list(MODEL_REGISTRY)}")
    return Pipeline(
        [
            ("preprocess", build_preprocessor(feat_cfg)),
            ("classifier", MODEL_REGISTRY[model_type]()),
        ]
    )
