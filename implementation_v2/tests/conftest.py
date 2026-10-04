"""Fixtures communes — aucun modèle HuggingFace n'est chargé dans la suite."""
import os
import numpy as np
import pytest

os.environ.setdefault("USE_TF", "0")


def keyword_predict_proba(texts):
    """Classifieur jouet déterministe : « idiot » / « crétin » ⇒ toxique.

    Sert à tester fidélité/suffisance sans charger XLM-R.
    """
    out = []
    for t in texts:
        toxic = 0.9 if any(w in t.lower() for w in ("idiot", "crétin")) else 0.1
        out.append([1 - toxic, toxic])
    return np.asarray(out, dtype=float)


@pytest.fixture
def predict_fn():
    return keyword_predict_proba
