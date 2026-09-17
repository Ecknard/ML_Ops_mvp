"""Promotion champion/challenger.

Compare le `test_roc_auc` du challenger (dernière version enregistrée) à
celui du champion courant, sur le même split de test / seed. L'alias
`champion` ne bouge que si le gain dépasse mlflow.promotion.min_improvement.
Aucun autre script ne déplace cet alias : c'est le seul point de décision.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import mlflow

from utils import get_logger, load_config

log = get_logger(__name__)


def promote(config: dict, force: bool = False) -> bool:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    model_name = config["mlflow"]["registered_model_name"]
    min_improvement = config["mlflow"]["promotion"]["min_improvement"]

    challenger = client.get_model_version_by_alias(model_name, "challenger")
    challenger_auc = float(challenger.tags.get("test_roc_auc", 0.0))

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_auc = float(champion.tags.get("test_roc_auc", 0.0))
    except mlflow.exceptions.MlflowException:
        champion = None
        champion_auc = -1.0  # pas de champion -> le premier challenger gagne toujours

    promoted = force or (challenger_auc - champion_auc >= min_improvement)

    with mlflow.start_run(run_name=f"promote-v{challenger.version}"):
        mlflow.log_param("challenger_version", challenger.version)
        mlflow.log_param("challenger_test_roc_auc", challenger_auc)
        mlflow.log_param("champion_version", champion.version if champion else None)
        mlflow.log_param("champion_test_roc_auc", champion_auc if champion else None)
        mlflow.log_param("promoted", promoted)

        if promoted:
            client.set_registered_model_alias(model_name, "champion", challenger.version)
            client.set_model_version_tag(
                model_name, challenger.version, "promoted_at", datetime.now(UTC).isoformat()
            )
            log.info(
                "Version %s promue champion (AUC %.4f > %.4f)",
                challenger.version,
                challenger_auc,
                champion_auc,
            )
        else:
            client.set_model_version_tag(model_name, challenger.version, "promotion_rejected", "true")
            log.info(
                "Version %s NON promue (AUC %.4f ne bat pas le champion %.4f d'au moins %.3f)",
                challenger.version,
                challenger_auc,
                champion_auc,
                min_improvement,
            )

    return promoted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true", help="Promeut sans comparaison")
    args = parser.parse_args()
    promote(load_config(args.config), force=args.force)
