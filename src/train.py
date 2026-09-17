"""Entraînement : split -> GridSearchCV -> seuil OOF -> log MLflow ->
enregistrement dans le registre sous l'alias `challenger`.

Chaque exécution crée UN run MLflow qui trace : la lignée des données
(hash + digest), les hyperparamètres retenus, les métriques CV et test,
et le seuil de décision. Le modèle n'est jamais promu `champion`
automatiquement ici : c'est le rôle de promote.py.
"""

from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from metrics import best_threshold_oof, compute_metrics
from pipeline import build_pipeline
from utils import file_sha256, get_logger, load_config

log = get_logger(__name__)


def load_dataset(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    data_cfg = config["data"]
    df = pd.read_csv(data_cfg["raw_path"])
    for col in config["features"]["drop"]:
        if col in df.columns:
            df = df.drop(columns=col)
    y = (df[data_cfg["target"]] == data_cfg["positive_label"]).astype(int)
    X = df.drop(columns=data_cfg["target"])
    return X, y


def run(config: dict) -> str:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    X, y = load_dataset(config)
    split_cfg = config["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y if split_cfg["stratify"] else None,
    )

    model_type = config["model"]["type"]
    pipe = build_pipeline(model_type, config["features"])
    param_grid = config["model"]["param_grid"][model_type]
    cv = StratifiedKFold(n_splits=config["model"]["cv_folds"], shuffle=True, random_state=42)

    with mlflow.start_run(run_name=f"{model_type}") as run_ctx:
        mlflow.log_params({f"grid__{k}": str(v) for k, v in param_grid.items()})
        mlflow.log_param("model_type", model_type)

        raw_path = config["data"]["raw_path"]
        mlflow.log_param("data_sha256", file_sha256(raw_path))
        mlflow.log_param("data_rows", len(pd.read_csv(raw_path)))
        mlflow.log_param("data_positive_rate", round(float(y.mean()), 4))

        log.info("GridSearchCV sur %s (%d combinaisons)", model_type, _grid_size(param_grid))
        search = GridSearchCV(pipe, param_grid, cv=cv, scoring=config["model"]["scoring"], n_jobs=-1)
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})
        mlflow.log_metric("cv_best_roc_auc", search.best_score_)

        threshold = best_threshold_oof(best_pipe, X_train, y_train, cv)
        mlflow.log_param("decision_threshold", round(threshold, 4))

        y_proba_test = best_pipe.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_proba_test, threshold)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        log.info("Test ROC-AUC=%.4f  F1(seuil optimisé)=%.4f", test_metrics["roc_auc"], test_metrics["f1"])

        signature = mlflow.models.infer_signature(X_train, best_pipe.predict_proba(X_train))
        model_info = mlflow.sklearn.log_model(
            best_pipe,
            "model",
            signature=signature,
            registered_model_name=config["mlflow"]["registered_model_name"],
            serialization_format="cloudpickle",  # plus fiable que skops pour un ColumnTransformer
        )

        client = mlflow.MlflowClient()
        version = model_info.registered_model_version
        model_name = config["mlflow"]["registered_model_name"]
        client.set_registered_model_alias(model_name, "challenger", version)
        client.set_model_version_tag(model_name, version, "decision_threshold", str(round(threshold, 4)))
        client.set_model_version_tag(
            model_name, version, "test_roc_auc", str(round(test_metrics["roc_auc"], 4))
        )
        client.set_model_version_tag(model_name, version, "model_type", model_type)

        log.info("Version %s enregistrée avec l'alias 'challenger'", version)
        return run_ctx.info.run_id


def _grid_size(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(load_config(args.config))
