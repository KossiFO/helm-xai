"""Matrice de décision : invariants structurels (pas de chiffre expérimental ici)."""
from helm.config import DECISION_MATRIX, DEFAULT_CONSTRAINTS, UserProfile, ContentType

KNOWN_METHODS = {"lime", "shap", "integrated_gradients", "anchors",
                 "counterfactual", "chain_of_thought", "logit_lens"}


def test_every_profile_has_text_candidates():
    for p in UserProfile:
        assert (p, ContentType.TEXT) in DECISION_MATRIX
        assert DEFAULT_CONSTRAINTS[p].num_features > 0


def test_candidates_are_known_methods_with_weights_summing_to_one():
    for key, cands in DECISION_MATRIX.items():
        names = [m for m, _ in cands]
        assert set(names) <= KNOWN_METHODS, key
        assert len(names) == len(set(names)), key
        assert abs(sum(w for _, w in cands) - 1.0) < 1e-9, key
