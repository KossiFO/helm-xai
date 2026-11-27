import json
from fastapi.testclient import TestClient
from helm.ui.app import create_app
from helm.ui.cli import main
from helm.ui.diagnostics import diagnostics


def test_diagnostics_expose_only_shareable_versions():
    data = diagnostics()
    assert set(data) == {"package_version", "python", "system", "dependencies"}
    assert set(data["dependencies"]) == {"numpy", "scipy", "scikit-learn", "shap", "lime", "fastapi"}
    assert all(isinstance(v, str) for v in data["dependencies"].values())


def test_diagnostic_command_does_not_create_history(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HELM_DATA_DIR", str(tmp_path / "unused"))
    assert main(["--diagnostic"]) == 0
    assert json.loads(capsys.readouterr().out) == diagnostics()
    assert not (tmp_path / "unused").exists()


def test_config_supplies_environment_without_exporting_history(tmp_path):
    with TestClient(create_app(tmp_path / "test.sqlite3")) as client:
        config = client.get("/api/config").json()
        assert config["environment"] == diagnostics()
        assert "history" not in config and "text" not in config
