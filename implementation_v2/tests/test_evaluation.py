"""Fidélité / suffisance (DeYoung et al. 2020) sur un classifieur jouet."""
import pytest

from helm.config import Attribution
from helm.evaluation import compute_fidelity, compute_sufficiency


def _attr():
    return Attribution(method_name="test",
                       token_scores={"idiot": 0.8, "fini": 0.1, "Tu": 0.05, "trou": 0.02})


def test_masking_top_token_drops_toxic_probability(predict_fn):
    fid = compute_fidelity("Tu es un idiot fini", _attr(), predict_fn, k=3)
    assert fid == pytest.approx(0.8)  # 0.9 → 0.1 une fois « idiot » masqué


def test_keeping_top_tokens_preserves_prediction(predict_fn):
    suf = compute_sufficiency("Tu es un idiot fini", _attr(), predict_fn, k=3)
    assert abs(suf) < 1e-9  # « idiot » conservé ⇒ même probabilité
