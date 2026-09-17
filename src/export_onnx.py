"""Export ONNX du champion, pour un Space Hugging Face statique qui prédit
côté client (ONNX Runtime Web) sans backend à héberger.

skl2onnx convertit nativement la majorité du pipeline (imputation,
standardisation, one-hot). La seule brique non standard est le log1p
(FunctionTransformer) : elle est retirée du graphe ONNX et appliquée
manuellement en JS avant l'inférence (voir deploy/space/app.js), avec les
colonnes concernées listées dans preprocess.json pour rester synchronisé.

L'export est validé contre les prédictions scikit-learn avant d'être
publié : make release-check refuse toute release si l'écart dépasse 1e-4.
"""

from __future__ import annotations

import argparse

import numpy as np
from joblib import load

from utils import get_logger, load_config, read_json, write_json

log = get_logger(__name__)


def export_onnx(config: dict, n_validation_rows: int = 2000) -> dict:
    try:
        import onnxruntime as ort
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    except ImportError as e:
        raise SystemExit("Dépendances ONNX manquantes. Installer : pip install skl2onnx onnxruntime") from e

    from train import load_dataset

    model = load(config["paths"]["model_path"])
    meta = read_json(config["paths"]["model_meta_path"])
    feat_cfg = config["features"]

    X, _ = load_dataset(config)
    numeric_cols = feat_cfg["numeric"] + feat_cfg["numeric_log"]
    categorical_cols = feat_cfg["categorical"]

    initial_types = [(c, FloatTensorType([None, 1])) for c in numeric_cols]
    initial_types += [(c, StringTensorType([None, 1])) for c in categorical_cols]

    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=17)
    onnx_path = f"{config['paths']['artifacts_dir']}/model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # préprocess.json : ce que le JS doit reproduire côté client
    write_json(
        f"{config['paths']['artifacts_dir']}/preprocess.json",
        {
            "numeric": feat_cfg["numeric"],
            "numeric_log1p": feat_cfg["numeric_log"],
            "categorical": categorical_cols,
            "decision_threshold": meta["decision_threshold"],
        },
    )

    # Validation : comparer sklearn vs ONNX sur un échantillon
    sample = X.sample(min(n_validation_rows, len(X)), random_state=0)
    sk_proba = model.predict_proba(sample)[:, 1]

    sess = ort.InferenceSession(onnx_path)
    feed = {c: sample[[c]].to_numpy().astype(np.float32) for c in numeric_cols}
    feed.update({c: sample[[c]].fillna("missing").to_numpy().astype(str) for c in categorical_cols})
    onnx_out = sess.run(None, feed)
    onnx_proba = np.array(onnx_out[1])[:, 1] if onnx_out[1].ndim == 2 else np.array(onnx_out[1])

    max_gap = float(np.max(np.abs(sk_proba - onnx_proba)))
    log.info("Écart max scikit-learn vs ONNX sur %d lignes : %.2e", len(sample), max_gap)

    if max_gap > 1e-4:
        raise SystemExit(f"Écart ONNX trop élevé ({max_gap:.2e} > 1e-4) — export refusé")

    return {"onnx_path": onnx_path, "max_gap": max_gap}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    export_onnx(load_config(args.config))
