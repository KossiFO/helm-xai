"""UCB1 : formule d'Auer et al. (2002), C=1.5 — valeurs figées par l'empreinte du 2026-09-08."""
import math

import pytest

from helm import UserProfile
from helm.feedback import UCB1Learner


def test_ucb_scores_match_frozen_values():
    u = UCB1Learner()
    for _ in range(10):
        u.record(UserProfile.TECHNICAL_EXPERT, "shap", 5)
        u.record(UserProfile.TECHNICAL_EXPERT, "lime", 2)
    s = u.get_ucb_scores(UserProfile.TECHNICAL_EXPERT, ["shap", "lime", "integrated_gradients"])
    assert s["shap"] == pytest.approx(1.820999245766796)
    assert s["lime"] == pytest.approx(1.070999245766796)
    assert s["integrated_gradients"] == math.inf  # bras jamais tiré ⇒ exploration


def test_reward_normalisation_and_stats():
    u = UCB1Learner()
    u.record(UserProfile.END_USER, "lime", 1)
    u.record(UserProfile.END_USER, "lime", 5)
    st = u.get_statistics()["utilisateur_final"]["lime"]
    assert st == {"count": 2, "mean_reward": 0.5, "total_reward": 1.0}


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "ucb1.json"
    u = UCB1Learner(storage_path=str(path))
    u.record(UserProfile.REGULATOR, "shap", 4)
    u2 = UCB1Learner(storage_path=str(path))
    assert u2.get_statistics() == u.get_statistics()
