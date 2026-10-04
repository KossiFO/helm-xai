"""Les scripts Redaction_These/run_*.py font sys.path.insert(implementation_v2) puis
``from config import UserProfile`` : la façade doit renvoyer LES MÊMES objets que helm.config
(sinon deux enums UserProfile distincts et les comparaisons échouent silencieusement)."""
import os
import sys

IMPL = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def test_config_facade_reexports_same_objects():
    sys.path.insert(0, IMPL)
    try:
        import config
        import helm.config
        assert config.UserProfile is helm.config.UserProfile
        assert config.DECISION_MATRIX is helm.config.DECISION_MATRIX
        assert config.MODEL_REGISTRY is helm.config.MODEL_REGISTRY
        assert config.FIDELITY_REFERENCE is helm.config.FIDELITY_REFERENCE
    finally:
        sys.path.remove(IMPL)


def test_data_facade_reexports_corpus():
    sys.path.insert(0, IMPL)
    try:
        from data.evaluation_corpus import EVALUATION_CORPUS, TOXIC_TEXTS
        from helm.data import EVALUATION_CORPUS as REF
        assert EVALUATION_CORPUS is REF and len(TOXIC_TEXTS) == 15
    finally:
        sys.path.remove(IMPL)
