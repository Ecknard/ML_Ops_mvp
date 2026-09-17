"""API FastAPI : sert le modèle exporté (artifacts/model.joblib), pas
MLflow directement (image de service plus légère). Chaque réponse porte
la version du registre et le run_id qui l'ont produite, pour tracer
n'importe quelle prédiction jusqu'à son run d'entraînement.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel

from utils import load_config, read_json

_config = load_config()
_model = None
_meta = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="Adult Income Classifier API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    age: float
    workclass: str | None = None
    education_num: float
    marital_status: str | None = None
    occupation: str | None = None
    relationship: str | None = None
    race: str | None = None
    sex: str | None = None
    capital_gain: float = 0.0
    capital_loss: float = 0.0
    hours_per_week: float
    native_country: str | None = None


def load_artifacts():
    global _model, _meta
    try:
        _model = load(_config["paths"]["model_path"])
        _meta = read_json(_config["paths"]["model_meta_path"])
    except FileNotFoundError:
        _model, _meta = None, None


@app.get("/health")
def health():
    if _meta is None:
        raise HTTPException(503, "Aucun modèle chargé — lancer `make export`")
    return {"status": "ok", **_meta}


@app.post("/predict")
def predict(req: PredictionRequest):
    if _model is None:
        raise HTTPException(503, "Aucun modèle chargé — lancer `make export`")
    import pandas as pd

    start = time.perf_counter()
    row = pd.DataFrame([req.model_dump()])
    proba = float(_model.predict_proba(row)[:, 1][0])
    threshold = _meta["decision_threshold"]
    label = ">50K" if proba >= threshold else "<=50K"
    latency = time.perf_counter() - start
    return {
        "label": label,
        "score": round(proba, 4),
        "threshold": threshold,
        "model_name": _meta["model_name"],
        "model_version": _meta["model_version"],
        "run_id": _meta["run_id"],
        "latency_ms": round(latency * 1000, 2),
    }
