"""Façade de compatibilité — NE PAS UTILISER dans du nouveau code.

Les scripts historiques (Redaction_These/run_*.py) ajoutent implementation_v2/
au sys.path puis font ``from config import UserProfile``. Ce module ré-exporte
``helm.config`` pour qu'ils tournent inchangés. Il n'est pas inclus dans le
package installable (seul ``helm`` l'est) : un module global ``config`` en
site-packages entrerait en collision avec d'autres projets.
"""
from helm.config import *  # noqa: F401,F403
from helm.config import (  # noqa: F401 — ré-export explicite des noms sans __all__
    DECISION_MATRIX, FIDELITY_REFERENCE, DEFAULT_CONSTRAINTS, MODEL_REGISTRY,
)
