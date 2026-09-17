import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_health_without_model_returns_503(tmp_path, monkeypatch):
    import importlib

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text()
    import shutil

    shutil.copy(
        Path(__file__).resolve().parents[1] / "configs" / "config.yaml",
        tmp_path / "configs" / "config.yaml",
    )

    import app as app_module

    importlib.reload(app_module)
    app_module.load_artifacts()

    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    resp = client.get("/health")
    assert resp.status_code == 503
