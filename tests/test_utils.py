import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from metrics import best_threshold_oof, compute_metrics  # noqa: E402
from utils import file_sha256, read_json, write_json  # noqa: E402


def test_compute_metrics_keys():
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.6, 0.4, 0.8])
    m = compute_metrics(y_true, y_proba, threshold=0.5)
    for key in [
        "roc_auc",
        "average_precision",
        "brier",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_at_0.5",
    ]:
        assert key in m
        assert 0.0 <= m[key] <= 1.0 or key == "brier"


def test_best_threshold_oof_in_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    threshold = best_threshold_oof(LogisticRegression(), X, y, cv)
    assert 0.05 <= threshold <= 0.95


def test_file_sha256_deterministic(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n")
    h1 = file_sha256(str(f))
    h2 = file_sha256(str(f))
    assert h1 == h2
    assert len(h1) == 64


def test_write_read_json_roundtrip(tmp_path):
    path = tmp_path / "meta.json"
    payload = {"model_version": "3", "threshold": 0.42}
    write_json(str(path), payload)
    assert read_json(str(path)) == payload
