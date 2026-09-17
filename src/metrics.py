"""Métriques d'évaluation et choix du seuil de décision.

Le seuil par défaut de 0.5 n'a aucune raison d'être optimal sur un
dataset déséquilibré (~24% de classe positive). On choisit le seuil qui
maximise le F1 sur des prédictions out-of-fold (cross_val_predict), donc
sans jamais toucher au split de test tenu à l'écart.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict


def best_threshold_oof(estimator, X, y, cv) -> float:
    """Seuil qui maximise le F1 sur des probabilités out-of-fold."""
    proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, proba >= t) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def compute_metrics(y_true, y_proba, threshold: float) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_at_0.5": f1_score(y_true, (y_proba >= 0.5).astype(int), zero_division=0),
    }
