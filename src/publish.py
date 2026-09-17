"""Publie le champion (joblib + ONNX + métadonnées + model card) sur un
dépôt de modèle Hugging Face. Refuse tout modèle qui n'est pas l'alias
`champion` du registre MLflow — la publication ne fait que relayer une
décision déjà prise par promote.py, jamais l'inverse.
"""

from __future__ import annotations

import argparse
import os

from utils import get_logger, load_config, read_json

log = get_logger(__name__)

MODEL_CARD_TEMPLATE = """---
license: mit
tags:
  - tabular-classification
  - scikit-learn
  - onnx
---

# {model_name} — v{version}

Classifieur binaire (revenu annuel `>50K` vs `<=50K`) entraîné sur UCI Adult
Income, tracé et versionné avec MLflow (registre : `{model_name}`, alias
`champion`).

- **Run MLflow** : `{run_id}`
- **Type de modèle** : `{model_type}`
- **Seuil de décision** : `{threshold}`
- **Test ROC-AUC** : `{roc_auc}`

## Fichiers

- `model.joblib` — pipeline scikit-learn complet, pour usage serveur (API FastAPI)
- `model.onnx` + `preprocess.json` — pour inférence côté navigateur (ONNX Runtime Web)
- `model_meta.json` — contrat de version consommé par l'API et le Space
"""


def publish(config: dict, repo_id: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub") from e

    meta = read_json(config["paths"]["model_meta_path"])
    artifacts_dir = config["paths"]["artifacts_dir"]

    card = MODEL_CARD_TEMPLATE.format(
        model_name=meta["model_name"],
        version=meta["model_version"],
        run_id=meta["run_id"],
        model_type=meta["model_type"],
        threshold=meta["decision_threshold"],
        roc_auc=meta["test_roc_auc"],
    )
    card_path = f"{artifacts_dir}/README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    files = ["model.joblib", "model_meta.json", "model.onnx", "preprocess.json"]
    for fname in files:
        path = f"{artifacts_dir}/{fname}"
        if os.path.exists(path):
            api.upload_file(path_or_fileobj=path, path_in_repo=fname, repo_id=repo_id)
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md", repo_id=repo_id)

    api.create_tag(repo_id, tag=f"v{meta['model_version']}", repo_type="model", exist_ok=True)
    log.info("Publié sur %s (tag v%s)", repo_id, meta["model_version"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--repo-id", required=True, help="ex: mon-compte/adult-income-classifier")
    args = parser.parse_args()
    publish(load_config(args.config), args.repo_id)
