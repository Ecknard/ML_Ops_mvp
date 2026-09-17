"""Inférence batch : lit un CSV de features, écrit un CSV avec score/label,
en utilisant le même artefact (artifacts/model.joblib) et le même seuil
que l'API — jamais de logique de prétraitement dupliquée ailleurs."""

from __future__ import annotations

import argparse

import pandas as pd
from joblib import load

from utils import get_logger, load_config, read_json

log = get_logger(__name__)


def predict_batch(config: dict, input_path: str, output_path: str) -> str:
    model = load(config["paths"]["model_path"])
    meta = read_json(config["paths"]["model_meta_path"])
    threshold = meta["decision_threshold"]

    df = pd.read_csv(input_path)
    proba = model.predict_proba(df)[:, 1]
    df["score"] = proba
    df["label"] = [">50K" if p >= threshold else "<=50K" for p in proba]
    df["model_version"] = meta["model_version"]

    df.to_csv(output_path, index=False)
    log.info("Prédictions écrites : %s (%d lignes)", output_path, len(df))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predict_batch(load_config(args.config), args.input, args.output)
