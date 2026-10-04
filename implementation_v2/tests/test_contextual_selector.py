"""Séparation calibration/décision/récompense et mise à jour du seul bras choisi."""
from dataclasses import replace
import numpy as np
import pytest
from helm import ContextualSelector, HELMPipeline, UserProfile
from helm.config import Attribution


def calibrated():
    return ContextualSelector(["lime", "shap"]).fit_calibration(
        [[1, .8, 1], [2, .9, 0], [3, .7, 1]], [[.4, .6], [.3, .7], [.5, .6]])


def test_requires_calibration_and_freezes_scaler():
    with pytest.raises(ValueError, match="calibration"):
        ContextualSelector(["lime"]).choose([1, .8, 1])
    policy = calibrated()
    mean = policy.mean_.copy()
    choice = policy.choose([9, .6, 0])
    policy.update(choice, .1)
    np.testing.assert_array_equal(policy.mean_, mean)


def test_only_chosen_arm_updated_once_after_authentic_choice():
    policy = calibrated()
    a, b = policy.a_.copy(), policy.b_.copy()
    choice = policy.choose([2, .8, 1])
    np.testing.assert_array_equal(a, policy.a_)
    with pytest.raises(ValueError):
        policy.update(replace(choice, method="unknown"), .9)
    with pytest.raises(ValueError):
        policy.update(choice, 5)
    policy.update(choice, .9)
    other = 1-policy.methods.index(choice.method)
    np.testing.assert_array_equal(a[other], policy.a_[other])
    np.testing.assert_array_equal(b[other], policy.b_[other])
    with pytest.raises(ValueError):
        policy.update(choice, .9)


def test_pipeline_uses_calibrated_pool_but_does_not_learn_implicitly():
    class Model:
        def predict_proba(self, texts):
            return np.tile([.2, .8], (len(texts), 1))
    class Explainer:
        def explain(self, **kwargs):
            return Attribution("shap", {"merci": .2}, metadata={"attribution_protocol": "word_positions_v1", "scores_by_position": [.2]})
    policy = ContextualSelector(["shap"]).fit_calibration([[1, .8, 1]], [[.6]])
    before = policy.a_.copy()
    pipeline = HELMPipeline(model=Model(), contextual_selector=policy, explainers={"shap": Explainer()})
    ctx = pipeline.explain("merci", UserProfile.END_USER, evaluate=True)
    assert list(ctx.attributions) == ["shap"]
    assert ctx.contextual_choice.method == "shap"
    np.testing.assert_array_equal(policy.a_, before)


def test_contextual_ig_refuses_implicit_loo():
    class Model:
        def predict_proba(self, texts):
            return np.tile([.2, .8], (len(texts), 1))
    policy = ContextualSelector(["integrated_gradients"]).fit_calibration([[1, .8, 1]], [[.6]])
    with pytest.raises(ValueError, match="aucun LOO"):
        HELMPipeline(model=Model(), contextual_selector=policy).explain("merci")
