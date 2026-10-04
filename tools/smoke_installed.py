"""Recette d’une wheel installée, à lancer avec python -I hors editable."""
import importlib.metadata
import json
from pathlib import Path
import re
import tempfile

from fastapi.testclient import TestClient
import helm
from helm.ui.app import create_app


def main():
    version = importlib.metadata.version("helm-xai")
    assert helm.__version__ == version
    direct = importlib.metadata.distribution("helm-xai").read_text("direct_url.json")
    assert not direct or not json.loads(direct).get("dir_info", {}).get("editable"), "Installer une vraie wheel."
    with tempfile.TemporaryDirectory(prefix="helm-wheel-smoke-") as directory:
        database = Path(directory) / "test.sqlite3"
        with TestClient(create_app(database)) as client:
            page = client.get('/')
            assert page.status_code == 200 and 'id="root"' in page.text
            assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', page.text)
            assert len(assets) >= 2
            for asset in assets:
                assert client.get(asset).status_code == 200, asset
            notices = client.get('/THIRD_PARTY_NOTICES.txt')
            assert notices.status_code == 200 and 'Permission is hereby granted' in notices.text
            payload = {'text': 'Merci pour cette réponse utile', 'profile': 'utilisateur_final', 'model': 'demo'}
            response = client.post('/api/explanations', json=payload)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result['package_version'] == version and result['attributions']
            path = '/api/explanations/' + result['id']
            method = next(iter(result['attributions']))
            for rating in (2, 4):
                note = client.post(path + '/feedback', json={'method': method, 'rating': rating})
                assert note.status_code == 200
            assert note.json()['statistics']['utilisateur_final'][method]['count'] == 1
            assert client.get(path + '/export').json()['feedback'] == {method: 4}
            assert client.get('/api/history').json()[0]['id'] == result['id']
        with TestClient(create_app(database)) as reopened:
            assert reopened.get(path).json()['feedback'] == {method: 4}
    print(json.dumps({'version': version, 'module': str(Path(helm.__file__).resolve()),
                      'frontend_assets': assets, 'pipeline': 'demo', 'feedback': 'temporary-only', 'status': 'ok'}))


if __name__ == "__main__":
    main()
