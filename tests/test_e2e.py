"""Test bout-en-bout sur données synthétiques et registre MLflow temporaire
(SQLite dans un dossier tmp), pour valider l'enchaînement complet sans
dépendre du vrai dataset UCI ni d'un serveur MLflow partagé."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synthetic_data import make_synthetic_adult  # noqa: E402

from utils import load_config  # noqa: E402


def test_full_cycle(tmp_path, monkeypatch):
    import export
    import promote
    import train

    config = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "config.yaml"))
    config = dict(config)
    config["data"] = dict(config["data"], raw_path=str(tmp_path / "raw.csv"))
    config["mlflow"] = dict(
        config["mlflow"], tracking_uri=f"sqlite:///{tmp_path}/mlflow.db", experiment_name="test-e2e"
    )
    config["paths"] = dict(
        config["paths"],
        artifacts_dir=str(tmp_path / "artifacts"),
        model_path=str(tmp_path / "artifacts" / "model.joblib"),
        model_meta_path=str(tmp_path / "artifacts" / "model_meta.json"),
    )
    config["model"] = dict(config["model"], type="logreg")
    # grille réduite pour que le test reste rapide
    config["model"]["param_grid"] = {"logreg": {"classifier__C": [1.0]}}
    config["model"]["cv_folds"] = 2

    df = make_synthetic_adult(n=1200, seed=3)
    df.to_csv(config["data"]["raw_path"], index=False)

    run_id_1 = train.run(config)
    assert run_id_1

    promoted_1 = promote.promote(config)
    assert promoted_1 is True  # premier challenger : promu par définition

    meta = export.export_champion(config)
    assert str(meta["model_version"]) == "1"
    assert Path(config["paths"]["model_path"]).exists()

    # deuxième run : re-vérifie que promote compare bien contre le champion existant
    run_id_2 = train.run(config)
    assert run_id_2 != run_id_1
    promote.promote(config)  # ne doit pas lever, promu ou non selon le hasard du fit
