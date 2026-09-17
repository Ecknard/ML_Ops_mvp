"""Téléchargement du dataset UCI Adult Income et normalisation minimale.

Le dataset brut a trois particularités qu'il faut corriger *avant* même
la validation de schéma, sinon la validation elle-même donne de faux
positifs :
  - pas d'en-tête -> colonnes fournies manuellement (config.data.columns)
  - un espace devant chaque valeur ("<=50K" est en fait " <=50K")
  - le fichier de test a un "." final sur la cible (">50K.")
  - les valeurs manquantes sont notées "?" plutôt que vides
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import get_logger, load_config

log = get_logger(__name__)


def _fetch(url: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(url, header=None, names=columns, skipinitialspace=True, na_values="?")
    return df


def build_raw_csv(config: dict) -> str:
    cfg = config["data"]
    out_path = cfg["raw_path"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("Téléchargement train : %s", cfg["train_url"])
    train = _fetch(cfg["train_url"], cfg["columns"])

    log.info("Téléchargement test : %s", cfg["test_url"])
    # adult.test a une ligne d'en-tête parasite à ignorer (skiprows=1)
    test = pd.read_csv(
        cfg["test_url"],
        header=None,
        names=cfg["columns"],
        skipinitialspace=True,
        na_values="?",
        skiprows=1,
    )

    full = pd.concat([train, test], ignore_index=True)
    target = cfg["target"]
    full[target] = full[target].astype(str).str.rstrip(".").str.strip()

    full.to_csv(out_path, index=False)
    log.info("Écrit %s (%d lignes) -> %s", target, len(full), out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    build_raw_csv(load_config(args.config))
