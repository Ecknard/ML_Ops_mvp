"""Pousse deploy/space/ (README, index.html, app.js) + un config.json généré
vers un Space Hugging Face statique. Appelé par la CI à chaque tag `v*`.
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import dotenv_values

from utils import get_logger

log = get_logger(__name__)


def deploy_space(space_repo: str, model_repo: str, space_dir: str = "deploy/space") -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub python-dotenv") from e

    env = dotenv_values("deploy/space.env")
    config = {"model_repo": model_repo, "model_revision": env.get("MODEL_REVISION", "main")}
    with open(f"{space_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(space_repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=space_dir, repo_id=space_repo, repo_type="space")
    log.info("Space déployé : %s (modèle %s @ %s)", space_repo, model_repo, config["model_revision"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-repo", required=True)
    parser.add_argument("--model-repo", required=True)
    args = parser.parse_args()
    deploy_space(args.space_repo, args.model_repo)
