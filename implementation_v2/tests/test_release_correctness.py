import numpy as np
import pytest
from helm import HELMPipeline, HELMContext, UserProfile, Prediction
from helm.config import Attribution
from helm.layers import PresentationLayer
from helm.evaluation import evaluate_attribution


def test_scores_of_incompatible_methods_never_averaged():
    ctx = HELMContext('mot autre', user_profile=UserProfile.END_USER)
    ctx.prediction = Prediction('toxique', .8, (.2, .8), 'test')
    ctx.attributions = {
        'shap': Attribution('shap', {'mot': .1}, metadata={'target_class': 1}),
        'lime': Attribution('lime', {'mot': -100}, metadata={'target_class': 1}),
        'counterfactual': Attribution('counterfactual', {'autre': 1e9}, metadata={'target_class': 1})}
    output = PresentationLayer().process(ctx).formatted_output
    assert output.content['by_method']['shap']['positive'] == ['mot']
    assert output.content['by_method']['lime']['negative'] == ['mot']
    assert 'counterfactual' not in output.content['by_method']
    assert output.content['score_fusion'] is False
    assert PresentationLayer()._find_consensus_tokens(ctx.attributions) == []


def test_counterfactual_is_not_scored_as_token_fidelity():
    attr = Attribution('counterfactual', {'mot': 1}, metadata={'scores_by_position':[1], 'target_class':1})
    with pytest.raises(ValueError, match='contrefactuels'):
        evaluate_attribution('mot', attr, lambda _: pytest.fail('No generic metric call'))


def test_metric_target_follows_attribution_explicit_target():
    attr = Attribution('lime', {'merci': .2}, metadata={'scores_by_position':[.2], 'target_class':1})
    def predict(texts):
        return np.array([[.9,.1] if t == 'merci' else [.7,.3] for t in texts])
    result = evaluate_attribution('merci', attr, predict)
    assert result.target_class == 1
    assert result.comprehensiveness == pytest.approx(-.2)


def test_backend_exception_is_reported_not_just_logged():
    class Model:
        def predict_proba(self, texts):
            return np.tile([.1,.9], (len(texts),1))
    class Broken:
        def explain(self, **kwargs):
            raise RuntimeError('deliberate failure')
    pipeline = HELMPipeline(model=Model(), allowed_methods=['lime'], explainers={'lime':Broken()})
    ctx = pipeline.explain('mot', UserProfile.END_USER)
    assert ctx.failed_attributions['lime']['error'] == 'deliberate failure'


@pytest.mark.parametrize('profile', [UserProfile.TECHNICAL_EXPERT, UserProfile.REGULATOR])
def test_rule_and_counterfactual_sections_are_not_token_attributions(profile):
    from helm.layers import ContextualLayer
    ctx = ContextualLayer().process(HELMContext('mot', user_profile=profile))
    ctx.prediction = Prediction('non_toxique', .9, (.9,.1), 'test')
    ctx.attributions = {
        'counterfactual': Attribution('counterfactual', {'mot':999}, metadata={'found':False}),
        'anchors': Attribution('anchors', {'mot':999}, metadata={'anchor_words':['mot'], 'precision':.96}),
        'shap': Attribution('shap', {'mot':.2}, metadata={'target_class':0, 'attribution_protocol':'word_positions_v1'})}
    sections = PresentationLayer().process(ctx).formatted_output.sections
    assert sections[0]['kind'] == 'counterfactual_search' and sections[0]['found'] is False
    assert 'tokens' not in sections[0] and 'tokens' not in sections[1]
    assert sections[1]['rule'] == ['mot']
    assert sections[2]['target_class'] == 0 and sections[2]['units'] == 'probability'


def test_logit_space_does_not_become_probability_direction():
    layer = PresentationLayer()
    attrs = {'integrated_gradients': Attribution('integrated_gradients', {'mot':100}, metadata={'target_class':1, 'target_space':'logit'})}
    assert layer._signals_by_method(attrs) == {}
    assert layer._find_consensus_tokens(attrs) == []


def test_historical_metrics_also_refuse_counterfactual_scores():
    from helm.evaluation import compute_fidelity, compute_sufficiency
    attr = Attribution('counterfactual', {'mot':99})
    for fn in (compute_fidelity, compute_sufficiency):
        with pytest.raises(ValueError, match='non applicable'):
            fn('mot', attr, lambda _: pytest.fail('No prediction for unsupported metric'))


def test_bootstrap_does_not_impute_zero_to_empty_counterfactual():
    from helm.evaluation.fidelity import bootstrap_metric, compute_fidelity
    with pytest.raises(ValueError, match='non applicable'):
        bootstrap_metric('mot', Attribution('counterfactual', {}), lambda _: None, metric_fn=compute_fidelity)
