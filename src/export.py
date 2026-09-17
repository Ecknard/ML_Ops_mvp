"""Exporte le modèle `champion` du registre MLflow vers artifacts/.

C'est le pont entre MLflow (lourd, pas embarqué en prod) et le service
(API légère, ou export Hugging Face) : model_meta.json est le contrat
qui dit exactement quelle version, quel run et quel seuil sont servis.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from joblib import dump

from utils import get_logger, load_config, write_json

log = get_logger(__name__)


def export_champion(config: dict) -> dict:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    model_name = config["mlflow"]["registered_model_name"]
    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(model_name, "champion")
    model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")

    model_path = config["paths"]["model_path"]
    os.makedirs(Path(model_path).parent, exist_ok=True)
    dump(model, model_path)

    meta = {
        "model_name": model_name,
        "model_version": version.version,
        "run_id": version.run_id,
        "decision_threshold": float(version.tags.get("decision_threshold", 0.5)),
        "test_roc_auc": float(version.tags.get("test_roc_auc", 0.0)),
        "model_type": version.tags.get("model_type", "unknown"),
    }
    write_json(config["paths"]["model_meta_path"], meta)
    log.info("Champion v%s exporté -> %s", version.version, model_path)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    export_champion(load_config(args.config))
