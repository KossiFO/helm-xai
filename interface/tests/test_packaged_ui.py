import importlib

import pytest
from fastapi.testclient import TestClient
from helm.ui.app import create_app, default_data_path
from helm.ui.cli import main


def test_runtime_lives_outside_package_and_survives_recreation(tmp_path, monkeypatch):
    monkeypatch.setenv("HELM_DATA_DIR", str(tmp_path / "personal"))
    expected = tmp_path / "personal/helm.sqlite3"
    assert default_data_path() == expected
    with TestClient(create_app()) as first:
        result = first.post('/api/explanations', json={'text': 'Merci pour cette réponse utile'}).json()
    assert expected.is_file()
    with TestClient(create_app()) as second:
        assert second.get('/api/history').json()[0]['id'] == result['id']
        assert 'source' not in second.get('/api/health').json()


def test_importing_entrypoint_neither_starts_server_nor_creates_database(tmp_path, monkeypatch):
    monkeypatch.setenv("HELM_DATA_DIR", str(tmp_path / "unused"))
    importlib.reload(importlib.import_module('helm.ui.__main__'))
    importlib.reload(importlib.import_module('helm.ui.app'))
    assert not (tmp_path / "unused").exists()


def test_missing_frontend_is_an_explicit_error(tmp_path):
    with TestClient(create_app(tmp_path / 'db.sqlite3', assets_path=tmp_path / 'absent')) as client:
        assert client.get('/').status_code == 503
        assert client.get('/api/health').status_code == 200


@pytest.mark.parametrize('port', ['0', '65536', 'wrong'])
def test_cli_rejects_invalid_port_without_starting(port):
    with pytest.raises(SystemExit) as exc:
        main(['--port', port, '--no-browser'])
    assert exc.value.code == 2


def test_cli_version_does_not_start_server(capsys):
    from helm import __version__
    with pytest.raises(SystemExit) as exc:
        main(['--version'])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
