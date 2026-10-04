import pytest
from helm import HELMContext, Prediction, UserProfile
from helm.layers import ContextualLayer, PresentationLayer


@pytest.mark.parametrize("p_toxic,action", [(0.05, "APPROUVER"), (0.6, "RÉVISER"), (0.95, "SUPPRIMER")])
def test_moderator_uses_toxic_probability_not_winning_confidence(p_toxic, action):
    ctx = ContextualLayer().process(HELMContext("bonjour", user_profile=UserProfile.MODERATOR))
    ctx.prediction = Prediction("toxique" if p_toxic > 0.5 else "non_toxique",
                                max(p_toxic, 1-p_toxic), (1-p_toxic, p_toxic), "test")
    result = PresentationLayer().process(ctx).formatted_output
    assert result.content['action'] == action
    assert result.content['toxicity_score'] == p_toxic


def test_end_user_keeps_correct_non_toxic_confidence():
    ctx = ContextualLayer().process(HELMContext("merci", user_profile=UserProfile.END_USER))
    ctx.prediction = Prediction("non_toxique", 0.95, (0.95, 0.05), "test")
    result = PresentationLayer().process(ctx).formatted_output
    assert '95%' in result.summary
