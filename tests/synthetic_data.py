"""Génère un CSV synthétique au même schéma que UCI Adult Income, pour
faire tourner tests et CI sans dépendre d'un accès réseau à archive.ics.uci.edu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_adult(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 75, n),
            "workclass": rng.choice(
                ["Private", "Self-emp", "Government", np.nan], n, p=[0.7, 0.15, 0.1, 0.05]
            ),
            "fnlwgt": rng.integers(10000, 500000, n),
            "education": rng.choice(["HS-grad", "Bachelors", "Masters", "Some-college"], n),
            "education_num": rng.integers(1, 16, n),
            "marital_status": rng.choice(["Married-civ-spouse", "Never-married", "Divorced"], n),
            "occupation": rng.choice(["Exec-managerial", "Craft-repair", "Sales", "Other-service"], n),
            "relationship": rng.choice(["Husband", "Not-in-family", "Own-child"], n),
            "race": rng.choice(["White", "Black", "Asian-Pac-Islander"], n),
            "sex": rng.choice(["Male", "Female"], n),
            "capital_gain": rng.choice([0, 0, 0, 5000, 15000], n),
            "capital_loss": rng.choice([0, 0, 0, 1500], n),
            "hours_per_week": rng.integers(10, 70, n),
            "native_country": rng.choice(
                ["United-States", "Mexico", "Germany", "India"], n, p=[0.85, 0.05, 0.05, 0.05]
            ),
        }
    )
    # cible corrélée à education_num / hours_per_week pour que l'entraînement soit informatif
    score = 0.15 * df["education_num"] + 0.03 * df["hours_per_week"] + rng.normal(0, 1.5, n)
    df["income"] = np.where(score > score.mean() + 0.6, ">50K", "<=50K")
    return df
