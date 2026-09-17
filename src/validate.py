"""Validation du CSV brut avant tout entraînement.

Objectif : faire échouer en quelques millisecondes sur un CSV corrompu,
plutôt qu'après plusieurs minutes de GridSearchCV. Ce n'est pas une
validation de qualité de modèle, c'est un contrôle de contrat de données.
"""

from __future__ import annotations

import argparse

import pandas as pd

from utils import get_logger, load_config

log = get_logger(__name__)


class DataValidationError(Exception):
    pass


def validate(df: pd.DataFrame, config: dict) -> None:
    data_cfg = config["data"]
    val_cfg = config["validation"]
    target = data_cfg["target"]
    errors: list[str] = []

    missing_cols = set(data_cfg["columns"]) - set(df.columns)
    if missing_cols:
        errors.append(f"Colonnes manquantes : {sorted(missing_cols)}")

    if len(df) < val_cfg["min_rows"]:
        errors.append(f"Trop peu de lignes : {len(df)} < {val_cfg['min_rows']}")

    if target in df.columns:
        missing_rate = df.isna().mean().max()
        if missing_rate > val_cfg["max_missing_rate"]:
            errors.append(f"Taux de valeurs manquantes trop élevé : {missing_rate:.2%}")

        positive_label = data_cfg["positive_label"]
        positive_rate = (df[target] == positive_label).mean()
        if not (val_cfg["min_positive_rate"] <= positive_rate <= val_cfg["max_positive_rate"]):
            errors.append(
                f"Taux de classe positive hors bornes : {positive_rate:.2%} "
                f"(attendu entre {val_cfg['min_positive_rate']:.0%} et {val_cfg['max_positive_rate']:.0%})"
            )
    else:
        errors.append(f"Colonne cible absente : {target}")

    if errors:
        raise DataValidationError("Validation des données échouée :\n- " + "\n- ".join(errors))

    log.info("Validation OK : %d lignes, taux positif %.2f%%", len(df), positive_rate * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(cfg["data"]["raw_path"])
    validate(df, cfg)
