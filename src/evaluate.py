"""Évaluation détaillée d'un modèle enregistré (par défaut : le champion).

Génère les courbes ROC/PR/confusion et un export des prédictions, et
logue tout dans un run MLflow lié au run d'entraînement d'origine via
`train_run_id`, pour remonter du rapport d'évaluation au run qui a produit
le modèle.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from metrics import compute_metrics
from train import load_dataset
from utils import get_logger, load_config

log = get_logger(__name__)


def evaluate(config: dict, model_uri: str | None = None) -> dict:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    model_name = config["mlflow"]["registered_model_name"]
    model_uri = model_uri or f"models:/{model_name}@champion"

    client = mlflow.MlflowClient()
    version = (
        client.get_model_version_by_alias(model_name, model_uri.split("@")[-1]) if "@" in model_uri else None
    )
    threshold = float(version.tags.get("decision_threshold", 0.5)) if version else 0.5

    model = mlflow.sklearn.load_model(model_uri)
    X, y = load_dataset(config)
    proba = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, proba, threshold)

    artifacts_dir = config["paths"]["artifacts_dir"]
    import os

    os.makedirs(artifacts_dir, exist_ok=True)

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("decision_threshold", threshold)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y, proba, ax=ax)
        fig.savefig(f"{artifacts_dir}/roc_curve.png")
        mlflow.log_artifact(f"{artifacts_dir}/roc_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_predictions(y, proba, ax=ax)
        fig.savefig(f"{artifacts_dir}/pr_curve.png")
        mlflow.log_artifact(f"{artifacts_dir}/pr_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y, (proba >= threshold).astype(int), ax=ax)
        fig.savefig(f"{artifacts_dir}/confusion_matrix.png")
        mlflow.log_artifact(f"{artifacts_dir}/confusion_matrix.png")
        plt.close(fig)

        pd.DataFrame({"y_true": y, "y_proba": proba, "y_pred": (proba >= threshold).astype(int)}).to_csv(
            f"{artifacts_dir}/predictions.csv", index=False
        )
        mlflow.log_artifact(f"{artifacts_dir}/predictions.csv")

    log.info("Évaluation terminée : ROC-AUC=%.4f F1=%.4f", metrics["roc_auc"], metrics["f1"])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-uri", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.model_uri)
