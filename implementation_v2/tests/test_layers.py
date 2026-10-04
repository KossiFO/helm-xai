"""Couches 1 et 2 sans modèle : ordre de sélection figé par l'empreinte du 2026-09-08 (xlmr)."""
import pytest

from helm import HELMContext, UserProfile, DetailLevel
from helm.layers import ContextualLayer, SelectionLayer
from helm.feedback import UCB1Learner

EXPECTED = {
    UserProfile.TECHNICAL_EXPERT: [("shap", 0.3158), ("integrated_gradients", 0.2632),
                                   ("lime", 0.1579), ("logit_lens", 0.1579), ("chain_of_thought", 0.1053)],
    UserProfile.MODERATOR: [("lime", 0.3846), ("counterfactual", 0.3077), ("shap", 0.3077)],
    UserProfile.END_USER: [("counterfactual", 0.4375), ("lime", 0.375), ("shap", 0.1875)],
    UserProfile.REGULATOR: [("shap", 0.3), ("integrated_gradients", 0.2), ("anchors", 0.2),
                            ("counterfactual", 0.15), ("lime", 0.15)],
}


def _run(profile):
    ctx = HELMContext(text="Tu es un idiot fini, retourne dans ton trou",
                      model_name="xlmr", user_profile=profile)
    ctx = ContextualLayer().process(ctx)
    return SelectionLayer(preference_learner=UCB1Learner()).process(ctx)


@pytest.mark.parametrize("profile", list(UserProfile))
def test_selection_order_is_frozen(profile):
    ctx = _run(profile)
    got = [(m.method_name, round(m.priority, 4)) for m in ctx.selected_methods]
    assert got == EXPECTED[profile]


def test_layer1_constraints_follow_profile():
    ctx = _run(UserProfile.REGULATOR)
    assert ctx.constraints.detail_level is DetailLevel.EXHAUSTIVE
    assert ctx.constraints.requires_audit_trail is True
    assert ctx.constraints.num_features == 20


def test_layer1_infers_end_user_when_profile_missing():
    ctx = HELMContext(text="bonjour")
    ctx = ContextualLayer().process(ctx)
    assert ctx.effective_profile is UserProfile.END_USER
