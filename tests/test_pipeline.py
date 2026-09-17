import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synthetic_data import make_synthetic_adult  # noqa: E402

from pipeline import build_pipeline  # noqa: E402
from utils import load_config  # noqa: E402
from validate import DataValidationError, validate  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return load_config(str(Path(__file__).resolve().parents[1] / "configs" / "config.yaml"))


@pytest.fixture(scope="module")
def synthetic_df():
    return make_synthetic_adult(n=1500, seed=1)


def test_pipeline_fits_and_predicts(config, synthetic_df):
    df = synthetic_df.drop(columns=config["features"]["drop"])
    y = (df[config["data"]["target"]] == config["data"]["positive_label"]).astype(int)
    X = df.drop(columns=config["data"]["target"])

    pipe = build_pipeline("logreg", config["features"])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_pipeline_handles_missing_values(config, synthetic_df):
    df = synthetic_df.drop(columns=config["features"]["drop"]).copy()
    df.loc[0, "workclass"] = None
    df.loc[1, "age"] = None
    y = (df[config["data"]["target"]] == config["data"]["positive_label"]).astype(int)
    X = df.drop(columns=config["data"]["target"])

    pipe = build_pipeline("logreg", config["features"])
    pipe.fit(X, y)  # ne doit pas lever malgré les NaN


def test_pipeline_handles_unknown_category(config, synthetic_df):
    df = synthetic_df.drop(columns=config["features"]["drop"])
    y = (df[config["data"]["target"]] == config["data"]["positive_label"]).astype(int)
    X = df.drop(columns=config["data"]["target"])

    pipe = build_pipeline("logreg", config["features"])
    pipe.fit(X.iloc[:-5], y.iloc[:-5])

    unseen = X.iloc[-1:].copy()
    unseen["native_country"] = "Atlantis"  # catégorie jamais vue à l'entraînement
    pipe.predict_proba(unseen)  # ne doit pas lever grâce à handle_unknown="infrequent_if_exist"


def test_unknown_model_type_raises(config):
    with pytest.raises(ValueError):
        build_pipeline("does-not-exist", config["features"])


def test_validate_accepts_healthy_data(config, synthetic_df):
    validate(synthetic_df, config)  # ne doit pas lever


def test_validate_rejects_too_few_rows(config, synthetic_df):
    with pytest.raises(DataValidationError):
        validate(synthetic_df.head(10), config)


def test_validate_rejects_missing_target(config, synthetic_df):
    with pytest.raises(DataValidationError):
        validate(synthetic_df.drop(columns=["income"]), config)


def test_validate_rejects_degenerate_positive_rate(config, synthetic_df):
    degenerate = synthetic_df.copy()
    degenerate["income"] = "<=50K"  # 0% de positifs
    with pytest.raises(DataValidationError):
        validate(degenerate, config)
