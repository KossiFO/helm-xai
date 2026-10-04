import pytest
from fastapi.testclient import TestClient
from helm.ui.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "test.sqlite3")) as client:
        yield client


@pytest.mark.parametrize("profile,view", [
    ("utilisateur_final", "natural_language"), ("moderateur", "moderator_dashboard"),
    ("expert_technique", "expert_analysis"), ("regulateur", "audit_report"),
])
def test_real_pipeline_and_profile_presentation(client, profile, view):
    response = client.post("/api/explanations", json={
        "text": "Merci pour cette réponse utile", "profile": profile, "model": "demo",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["profile"] == profile
    assert result["presentation"]["format_type"] == view
    assert result["attributions"] and result["evaluation"]
    assert all(len(a["token_scores"]) <= 8 for a in result["attributions"].values())
    assert result["parameters"]["k"] == 3
    assert result["prediction"]["probabilities"][1] < 0.5
    if profile == "moderateur":
        assert result["presentation"]["content"]["toxicity_score"] == result["prediction"]["probabilities"][1]
        assert result["presentation"]["content"]["action"] == "APPROUVER"
    if profile == "utilisateur_final":
        expected = f'{result["prediction"]["confidence"]:.0%}'
        assert expected in result["presentation"]["summary"]
    saved = client.get(f'/api/explanations/{result["id"]}').json()
    assert saved == result
    exported = client.get(f'/api/explanations/{result["id"]}/export')
    assert exported.status_code == 200 and exported.json() == result
    assert 'attachment' in exported.headers['content-disposition']
    assert client.get('/api/history').json()[0]['id'] == result['id']


@pytest.mark.parametrize("payload", [
    {"text": "   "}, {"text": "!!!"}, {"text": "a" * 2001},
    {"text": "bonjour", "profile": "inconnu"},
    {"text": "bonjour", "model": "inconnu"},
])
def test_invalid_inputs_are_rejected(client, payload):
    assert client.post('/api/explanations', json=payload).status_code == 422
    assert client.get('/api/history').json() == []


def test_feedback_is_bound_to_result_and_replaced_in_ucb1(client):
    result = client.post('/api/explanations', json={
        'text': 'Tu es un idiot stupide', 'profile': 'expert_technique', 'model': 'demo',
    }).json()
    path = f'/api/explanations/{result["id"]}/feedback'
    assert client.post(path, json={'method': 'not_computed', 'rating': 5}).status_code == 422
    for rating in [0, 6, 2.5, True]:
        assert client.post(path, json={'method': 'lime', 'rating': rating}).status_code == 422
    for rating in [2, 5]:
        response = client.post(path, json={'method': 'lime', 'rating': rating})
        assert response.status_code == 200
    stats = response.json()['statistics']['expert_technique']['lime']
    assert stats['count'] == 1 and stats['mean_reward'] == 1
    assert client.get(f'/api/explanations/{result["id"]}').json()['feedback']['lime'] == 5
    next_result = client.post('/api/explanations', json={
        'text': result['text'], 'profile': result['profile'], 'model': 'demo',
    }).json()
    lime_selection = next(x for x in next_result['selection'] if x['method_name'] == 'lime')
    assert 'UCB1' in lime_selection['rationale']
    assert client.app.state.service.store.learner('xlmr').get_statistics() == {}
    # Une nouvelle instance relit les mêmes notes, sans état Python partagé.
    with TestClient(create_app(client.app.state.service.store.path)) as reloaded:
        assert reloaded.get(f'/api/explanations/{result["id"]}').json()['feedback']['lime'] == 5


def test_not_found_and_model_loading_failure(client, tmp_path):
    assert client.get('/api/explanations/missing').status_code == 404
    assert client.post('/api/explanations/missing/feedback', json={'method':'lime','rating':4}).status_code == 404
    def unavailable(name):
        raise OSError('local model absent')
    with TestClient(create_app(tmp_path / 'missing.sqlite3', unavailable)) as failed:
        response = failed.post('/api/explanations', json={'text':'bonjour','model':'xlmr'})
        assert response.status_code == 503
        assert 'Démonstration' in response.json()['detail']


def test_previous_package_feedback_remains_saved_but_is_not_reused(client):
    from copy import deepcopy
    from helm import __version__
    result = client.post('/api/explanations', json={
        'text': 'Merci pour cette réponse utile', 'profile': 'expert_technique', 'model': 'demo',
    }).json()
    old = deepcopy(result)
    old['id'] = 'historical-a3'
    old['package_version'] = '0.1.0a3'
    store = client.app.state.service.store
    store.save(old)
    store.rate(old['id'], 'lime', 5)
    assert store.get(old['id'])['feedback']['lime'] == 5
    assert store.learner('demo').get_statistics() == {}
    assert store.learner('demo', '0.1.0a3').get_statistics()['expert_technique']['lime']['count'] == 1
    assert client.get('/api/health').json()['package_version'] == __version__
