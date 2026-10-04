"""Notebook actions must use entered text and invalidate stale predictions."""
import pytest
pytest.importorskip('ipywidgets')
from helm.visualization.text_input import TextInput, prediction_html, validate_comment


def test_input_changes_recompute_and_profile_only_rerenders():
    calls = []
    def calculate(text, explain, progress):
        calls.append((text, explain))
        return text
    panel = TextInput(calculate, lambda result, profile: result + ':' + profile)
    panel.submit(False)
    assert not calls
    assert validate_comment('  texte\n') == '  texte\n'
    panel.text.value = 'Premier texte'
    panel.submit(False)
    assert calls == [('Premier texte', False)]
    panel.profile.value = 'moderateur'
    assert panel.output.value == 'Premier texte:moderateur'
    assert len(calls) == 1
    panel.text.value = 'Deuxième texte'
    assert panel.result is None and panel.output.value == ''
    panel.submit(True)
    assert calls[-1] == ('Deuxième texte', True)


def test_failure_unlocks_and_no_old_result_or_html_injection():
    def fail(*args):
        raise RuntimeError('<script>fail</script>')
    panel = TextInput(fail, lambda *args: 'wrong')
    panel.text.value = 'Test'
    panel.submit(True)
    assert panel.result is None and panel.output.value == ''
    assert not panel.busy and not panel.predict.disabled
    assert '<script>' not in panel.status.value
    html = prediction_html({'text':'<script>1</script>', 'prediction':{'probabilities':[.2,.8], 'label':'toxique'}})
    assert '<script>' not in html and '80.00%' in html
    with pytest.raises(ValueError):
        validate_comment('x'*4001)


def test_profile_render_failure_then_recovery():
    def render(result, profile):
        if profile == 'moderateur':
            raise RuntimeError('render failed')
        return profile
    panel = TextInput(lambda *args: 'saved', render)
    panel.text.value = 'Test'
    panel.submit(True)
    panel.profile.value = 'moderateur'
    assert panel.output.value == '' and 'indisponible' in panel.status.value
    panel.profile.value = 'regulateur'
    assert panel.output.value == 'regulateur'
    assert 'indisponible' not in panel.status.value
