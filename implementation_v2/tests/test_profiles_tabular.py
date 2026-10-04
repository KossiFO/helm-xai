"""Grounded profile reports, synthetic values only."""
from dataclasses import replace
import pandas as pd
import pytest
from helm.tabular.report import LocalExplanation, CohortReport
from helm.tabular.profiles import ProfileReport, profile_plan
from helm.layers.layer2_selection import SelectionLayer
from helm.config import UserProfile, OperationalConstraints
from helm.context import HELMContext


def sample():
    local = LocalExplanation('<dossier>', 1, .6, 1, None, 'shap', 'metier',
        {'x<script>':3, 'y':4}, {'x<script>':.3,'y':-.1},
        quality={'base_value':.4,'additivity_residual':0,'background_sha256':'abc'})
    ranked = pd.DataFrame({'position':[0], 'target_score':[.6], 'prediction':[1]}, index=['<dossier>'])
    return CohortReport(ranked,[local],[],1,'shap','metier')


def test_profile_views_keep_same_evidence_and_escape():
    report=sample()
    outputs={p:ProfileReport(p,{'shap':report}).to_html() for p in ('utilisateur','metier','technique','audit')}
    assert len(set(outputs.values())) == 4
    assert 'Registre de couverture et de provenance' in outputs['audit']
    assert 'Registre de couverture et de provenance' not in outputs['technique']
    assert 'Ce qui soutient' in outputs['metier'] and 'À examiner' in outputs['metier']
    assert 'Facteurs récurrents' not in outputs['utilisateur']
    assert '<script>' not in outputs['metier'] and '&lt;script&gt;' in outputs['metier']
    assert '+30.00 points' in outputs['metier'] and '-10.00 points' in outputs['metier']
    assert report.explanations[0].contributions == {'x<script>':.3,'y':-.1}
    assert report.profile == 'metier'
    assert 'Calculs manquants' in outputs['technique']


def test_profile_plan_and_comparison_coherence():
    assert profile_plan('metier')['methods'] == ('shap',)
    assert profile_plan('technique')['methods'] == ('shap','lime')
    with pytest.raises(ValueError):
        profile_plan('invente')
    report=sample()
    other=replace(report, method='lime',target_class=0)
    with pytest.raises(ValueError,match='même classe'):
        ProfileReport('technique',{'shap':report,'lime':other})


def test_pool_filter_precedes_budget_and_rejects_empty():
    ctx=HELMContext(text='Un texte.',model_name='xlmr',user_profile=UserProfile.MODERATOR)
    ctx.constraints=OperationalConstraints(max_response_time=100)
    result=SelectionLayer(allowed_methods=['shap','lime']).process(ctx)
    assert {m.method_name for m in result.selected_methods} == {'shap','lime'}
    assert any('exclues' in log for log in result.logs)
    with pytest.raises(ValueError):
        SelectionLayer(allowed_methods=[])


def test_different_successful_cohorts_disable_comparison():
    first=sample()
    other=replace(first, method='lime', explanations=[])
    html=ProfileReport('technique', {'shap':first, 'lime':other}).to_html()
    assert 'Comparaison suspendue' in html
    assert 'Intersection des cinq' not in html


def test_text_quality_and_failure_visible_to_end_user():
    from helm.config import Prediction, Attribution, MethodSelection
    from helm.visualization.profile_html import context_html
    ctx=HELMContext(text='<un texte>',model_name='xlmr',user_profile=UserProfile.END_USER)
    ctx.prediction=Prediction('non_toxique', .8, (.8,.2), 'xlmr')
    ctx.selected_methods=[MethodSelection('lime',.5,1,'test'), MethodSelection('shap',.5,1,'test')]
    ctx.attributions={'lime':Attribution('lime',{'mot':.3},metadata={'target_class':0,'local_r2':.2})}
    html=context_html(ctx)
    assert '0.200' in html and 'Explications incomplètes' in html
    assert '&lt;un texte&gt;' in html and '<un texte>' not in html
    assert html.index('Explications incomplètes') < html.index('Choix des méthodes')


def test_shap_class_zero_reference_is_explicit():
    from helm.config import Prediction, Attribution
    from helm.visualization.profile_html import context_html
    ctx=HELMContext(text='Merci.',model_name='xlmr',user_profile=UserProfile.END_USER)
    ctx.prediction=Prediction('non_toxique', .8, (.8,.2), 'xlmr')
    ctx.attributions={'shap':Attribution('shap',{'Merci':.3},base_value=.5,metadata={'target_class':0})}
    html=context_html(ctx)
    assert 'classe 0 (non toxique) : 0.5000' in html
    assert 'leurs signes sont inversés' in html
