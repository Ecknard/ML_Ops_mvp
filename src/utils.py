"""Utilitaires transverses : config, logging, hash de données, IO JSON."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml


def get_logger(name: str) -> logging.Logger:
    """Logger uniforme pour tout le projet, niveau piloté par LOG_LEVEL."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    """Charge le YAML de config. Toute la pipeline dépend de ce seul fichier."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha256(path: str) -> str:
    """Empreinte du fichier de données : sert à tracer dans MLflow *quel*
    CSV exact a produit *quel* run (deux runs sur des données différentes
    sont ainsi distinguables même si le nom de fichier est identique)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
