"""Portable controls and evidence-preserving visuals."""
import re
import pytest
from helm.visualization.dashboard import Dashboard, contribution_chart


def test_controls_are_self_contained_and_scoped():
    views={(g,p):f'<h2>{g} {p}</h2>' for g in ['haut','bas'] for p in ['metier','expert']}
    dash=Dashboard('<Test>', {'haut':'Haut','bas':'Bas'}, {'metier':'Métier','expert':'Expert'}, views,'metier')
    first,second=dash.to_html(),dash.to_html()
    assert '<script' not in first and 'ipywidgets' not in first and 'https://' not in first
    assert '&lt;Test&gt;' in first
    assert first.count('type="radio"') == 4 and first.count('class="helm-view"') == 4
    assert first.count(' checked') == 2
    assert set(re.findall(r'id="([^"]+)"',first)).isdisjoint(re.findall(r'id="([^"]+)"',second))
    assert ':checked~' in first


def test_missing_view_refused_and_signed_values_preserved():
    with pytest.raises(ValueError,match='chaque groupe'):
        Dashboard('x',{'g':'G'},{'p':'P'},{},'p').to_html()
    html=contribution_chart([('<script>',.1),('b',-.05)],title='Test',unit='probability')
    assert '+10.00 pts' in html and '-5.00 pts' in html
    assert 'helm-positive' in html and 'helm-negative' in html
    assert '<script>' not in html and '&lt;script&gt;' in html
    with pytest.raises(ValueError,match='finies'):
        contribution_chart([('bad',float('nan'))],title='Test')
