"""HELM — Hybrid Explainability Layered Model.

Framework d'explicabilité adaptative pour la modération de contenus :
trois couches (contexte, sélection, présentation), sept méthodes XAI,
sélection ajustée par bandit UCB1.

Usage minimal ::

    from helm import HELMPipeline, UserProfile
    pipeline = HELMPipeline(model_name="xlmr")
    ctx = pipeline.explain("Tu es un idiot fini", user_profile=UserProfile.MODERATOR)
    print(ctx.formatted_output.summary)
"""

import logging

from .config import (
    UserProfile, ContentType, DetailLevel, ModelBackend,
    OperationalConstraints, Prediction, Attribution, MethodSelection,
    FormattedOutput, EvaluationMetrics, DECISION_MATRIX, MODEL_REGISTRY,
)
from .tabular import TabularHELM
from .context import HELMContext
from .pipeline import HELMPipeline
from .selection import ContextualSelector, ContextualChoice
from .feedback import UCB1Learner, UCB1AutoReward

__version__ = "0.2.0rc5"

# Une bibliothèque ne configure pas le logging de l'application hôte.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "__version__", "TabularHELM",
    "ContextualSelector", "ContextualChoice",
    "HELMPipeline", "HELMContext", "UCB1Learner", "UCB1AutoReward",
    "UserProfile", "ContentType", "DetailLevel", "ModelBackend",
    "OperationalConstraints", "Prediction", "Attribution", "MethodSelection",
    "FormattedOutput", "EvaluationMetrics", "DECISION_MATRIX", "MODEL_REGISTRY",
]
