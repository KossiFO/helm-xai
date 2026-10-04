"""Garde-fous du protocole signé et des identités de méthodes."""
import numpy as np
import pytest
from helm import HELMPipeline, UserProfile
from helm.config import Attribution
from helm.evaluation import evaluate_positions, evaluate_attribution, signed_cs_reward
from helm.explainers import LOOExplainer, IGExplainer, CoTExplainer, CounterfactualExplainer
from helm.layers import SelectionLayer


def test_signed_class_zero_and_real_empty_intervention():
    seen = []
    def predict(texts):
        seen.extend(texts)
        p = {"merci": .8, "le": .95}
        return np.array([[p[t], 1-p[t]] for t in texts])
    result = evaluate_positions("merci", [0], predict)
    assert seen == ["merci", "le", "merci"]
    assert result.target_class == 0
    assert result.comprehensiveness == pytest.approx(-.15)
    assert result.sufficiency == 0
    assert signed_cs_reward(result.comprehensiveness, result.sufficiency) == pytest.approx(.4625)


def test_position_scores_preserve_repeated_words_and_absolute_ranking():
    def predict(texts):
        return np.tile([.2, .8], (len(texts), 1))
    attr = Attribution("test", {"mot": 999}, metadata={"scores_by_position": [.1, -.7, .5]})
    result = evaluate_attribution("mot mot autre", attr, predict, k=1)
    assert result.positions == (1,)
    assert result.removed_text == "mot autre"
    assert result.kept_text == "mot"


@pytest.mark.parametrize("positions", [[], [0, 0], [-1], [2], [True], [.5]])
def test_rejects_invalid_positions(positions):
    with pytest.raises(ValueError):
        evaluate_positions("deux mots", positions, lambda _: None)


def test_invalid_native_attribution_is_not_scored():
    attr = Attribution("integrated_gradients", {"mot": 1}, metadata={"valid": False})
    with pytest.raises(ValueError, match="admissible"):
        evaluate_attribution("mot", attr, lambda _: pytest.fail("Ne doit pas prédire"))


def test_loo_is_replacement_on_predicted_class_with_positions():
    seen = []
    def predict(texts):
        seen.extend(texts)
        return np.array([[.9 if t == "merci  merci" else .7, .1 if t == "merci  merci" else .3] for t in texts])
    attr = LOOExplainer().explain("merci  merci", predict)
    assert seen == ["merci  merci", "le merci", "merci le"]
    assert attr.method_name == "leave_one_out"
    assert attr.metadata["target_class"] == 0
    assert attr.metadata["scores_by_position"] == pytest.approx([.2, .2])


def test_legacy_ig_and_cot_do_not_claim_native_execution(predict_fn):
    attr = IGExplainer().explain("un idiot", predict_fn)
    assert attr.metadata["effective_method"] == "loo_replacement"
    attr = CoTExplainer(model_manager=object()).explain("un idiot", predict_fn)
    assert attr.metadata["mode"] == "perturbation"
    assert attr.metadata["effective_method"] != "chain_of_thought"


def test_cf_non_toxic_confidence_and_no_empty_success():
    def predict(texts):
        return np.array([[.9, .1] if t.strip() else [.1, .9] for t in texts])
    attr = CounterfactualExplainer().explain("merci", predict)
    assert "90%" in attr.metadata["summary"]
    assert attr.metadata["found"] is False
    assert not attr.metadata["counterfactuals"]


def test_historical_fidelity_bonus_disabled_by_default(monkeypatch):
    from helm.context import HELMContext
    from helm.config import OperationalConstraints
    layer = SelectionLayer()
    monkeypatch.setattr(layer, "_adjust_fidelity", lambda *_: pytest.fail("Références historiques utilisées"))
    layer.process(HELMContext(text="merci", user_profile=UserProfile.TECHNICAL_EXPERT,
                              constraints=OperationalConstraints(), model_name="camembert"))


def test_pipeline_signed_metrics_and_native_failure_is_retained(predict_fn):
    class Model:
        predict_proba = staticmethod(predict_fn)
    class Failed:
        def explain(self, **kwargs):
            return Attribution("integrated_gradients", {}, metadata={"valid": False, "delta": 99})
    p = HELMPipeline(model=Model(), explainers={"integrated_gradients": Failed()})
    ctx = p.explain("un idiot fini", UserProfile.TECHNICAL_EXPERT, evaluate=True, num_features=8)
    assert "integrated_gradients" not in ctx.attributions
    assert ctx.failed_attributions["integrated_gradients"]["delta"] == 99
    assert all(m.metadata["protocol"] == "signed_cs_word_positions_v1" for m in ctx.evaluation.values())


@pytest.mark.parametrize("text,scores", [("l'amour revient", {"amour": 10, "revient": .1}), ("mot mot", {"mot": 10})])
def test_ambiguous_lexical_attributions_do_not_claim_position_protocol(text, scores):
    with pytest.raises(ValueError, match="par position"):
        evaluate_attribution(text, Attribution("lime", scores), lambda _: pytest.fail("Ne pas prédire"))


@pytest.mark.parametrize("name", ["lime", "shap"])
def test_native_word_backends_preserve_apostrophes_repetitions_and_target(name):
    from helm.explainers import LIMEExplainer, SHAPExplainer
    def predict(texts):
        p = np.array([.9 if "amour" in t else .6 for t in texts])
        return np.column_stack((p, 1-p))
    text = "l’amour revient l’amour"
    attr = {"lime": LIMEExplainer, "shap": SHAPExplainer}[name]().explain(text, predict, num_features=8)
    assert len(attr.metadata["scores_by_position"]) == 3
    assert attr.metadata["target_class"] == 0
    assert attr.metadata["attribution_protocol"] == "word_positions_v1"
    result = evaluate_attribution(text, attr, predict, k=1)
    assert result.positions[0] in (0, 2)


def test_native_ig_selection_never_claims_legacy_latency():
    from helm.context import HELMContext
    from helm.config import OperationalConstraints
    layer = SelectionLayer(unbudgeted_methods={"integrated_gradients"})
    ctx = layer.process(HELMContext(text="merci", user_profile=UserProfile.TECHNICAL_EXPERT,
                                    constraints=OperationalConstraints()))
    ig = next(m for m in ctx.selected_methods if m.method_name == "integrated_gradients")
    assert ig.estimated_time is None
    assert "hors budget" in ig.rationale


def test_norm_diagnostic_never_enters_directional_consensus():
    from helm.layers import PresentationLayer
    layer = PresentationLayer()
    attrs = {'logit_lens': Attribution('logit_lens', {'merci': 99}, metadata={'score_kind': 'norm_growth_proxy'}),
             'lime': Attribution('lime', {'merci': .2}, metadata={'target_class': 0})}
    assert layer._signals_by_method(attrs)['lime']['negative'] == ['merci']
    assert 'logit_lens' not in layer._signals_by_method(attrs)
    assert layer._find_consensus_tokens({'logit_lens': attrs['logit_lens']}) == []


def test_synthetic_rationale_uses_predicted_class_confidence():
    attr = CoTExplainer().explain("merci beaucoup", lambda texts: np.tile([.9, .1], (len(texts), 1)))
    assert "90%" in attr.metadata["cot_explanation"]
