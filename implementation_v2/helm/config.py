"""
Configuration centrale HELM v2.
Énumérations, dataclasses typées, matrice de décision, scores de référence.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ═══════════════════════════════════════════════════════
# Énumérations
# ═══════════════════════════════════════════════════════

class UserProfile(Enum):
    TECHNICAL_EXPERT = "expert_technique"
    MODERATOR = "moderateur"
    END_USER = "utilisateur_final"
    REGULATOR = "regulateur"


class ContentType(Enum):
    TEXT = "texte"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class DetailLevel(Enum):
    MINIMAL = 1
    STANDARD = 2
    DETAILED = 3
    EXHAUSTIVE = 4


class ModelBackend(Enum):
    """Backend de modèle disponible."""
    TRANSFORMER = "transformer"    # HuggingFace transformers (GPU/CPU)
    LIGHTWEIGHT = "lightweight"    # TF-IDF + LogisticRegression (CPU rapide)


# ═══════════════════════════════════════════════════════
# Dataclasses typées
# ═══════════════════════════════════════════════════════

@dataclass
class OperationalConstraints:
    max_response_time: float = 30.0
    detail_level: DetailLevel = DetailLevel.STANDARD
    requires_audit_trail: bool = False
    language: str = "fr"
    num_features: int = 10


@dataclass(frozen=True)
class Prediction:
    """Résultat de prédiction immutable."""
    label: str
    confidence: float
    probabilities: Tuple[float, ...]
    model_name: str


@dataclass
class Attribution:
    """Attributions token-level d'une méthode XAI."""
    method_name: str
    token_scores: Dict[str, float]        # token → score d'importance
    computation_time: float = 0.0
    base_value: Optional[float] = None    # Valeur de base (SHAP)
    convergence_delta: Optional[float] = None  # Delta convergence (IG)
    metadata: Dict = field(default_factory=dict)

    @property
    def top_tokens(self) -> List[Tuple[str, float]]:
        """Tokens triés par importance décroissante."""
        return sorted(self.token_scores.items(),
                      key=lambda x: abs(x[1]), reverse=True)

    @property
    def positive_tokens(self) -> List[Tuple[str, float]]:
        """Tokens contribuant positivement."""
        return [(t, s) for t, s in self.top_tokens if s > 0]

    @property
    def negative_tokens(self) -> List[Tuple[str, float]]:
        """Tokens contribuant négativement."""
        return [(t, s) for t, s in self.top_tokens if s < 0]


@dataclass
class MethodSelection:
    method_name: str
    priority: float
    estimated_time: Optional[float]
    rationale: str


@dataclass
class FormattedOutput:
    format_type: str
    summary: str
    content: Dict = field(default_factory=dict)
    sections: List[Dict] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Métriques d'évaluation pour une attribution."""
    fidelity: Optional[float] = None          # Comprehensiveness
    sufficiency: Optional[float] = None
    stability: Optional[float] = None         # Spearman correlation
    fidelity_ci: Optional[Tuple[float, float]] = None  # IC 95%
    sufficiency_ci: Optional[Tuple[float, float]] = None
    stability_ci: Optional[Tuple[float, float]] = None
    metadata: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════
# Matrice de décision (manuscrit v7, chap. HELM — tableau profil × méthodes)
# ═══════════════════════════════════════════════════════

DECISION_MATRIX: Dict[Tuple[UserProfile, ContentType], List[Tuple[str, float]]] = {
    (UserProfile.TECHNICAL_EXPERT, ContentType.TEXT): [
        ("shap", 0.30),
        ("integrated_gradients", 0.25),
        ("lime", 0.15),
        ("logit_lens", 0.15),
        ("chain_of_thought", 0.10),
        ("anchors", 0.05),
    ],
    (UserProfile.END_USER, ContentType.TEXT): [
        ("counterfactual", 0.35),
        ("lime", 0.30),
        ("chain_of_thought", 0.20),
        ("shap", 0.15),
    ],
    (UserProfile.MODERATOR, ContentType.TEXT): [
        ("anchors", 0.35),
        ("lime", 0.25),
        ("counterfactual", 0.20),
        ("shap", 0.20),
    ],
    (UserProfile.REGULATOR, ContentType.TEXT): [
        ("shap", 0.30),
        ("integrated_gradients", 0.20),
        ("anchors", 0.20),
        ("counterfactual", 0.15),
        ("lime", 0.15),
    ],
}

# Fallback pour types non-texte
for _profile in UserProfile:
    for _ctype in [ContentType.IMAGE, ContentType.MULTIMODAL]:
        if (_profile, _ctype) not in DECISION_MATRIX:
            DECISION_MATRIX[(_profile, _ctype)] = DECISION_MATRIX.get(
                (_profile, ContentType.TEXT), [("lime", 0.5), ("shap", 0.5)]
            )


# ═══════════════════════════════════════════════════════
# Scores de fidélité de référence — VALEURS HISTORIQUES
# Trio LIME/SHAP/IG : soumission EGC 2026 (non publiée), modèles camembert/
# electra/gpt2. Les quatre autres méthodes sont des ESTIMATIONS non mesurées.
# Utilisé par la couche 2 (étape B) pour pondérer les candidats. À remplacer
# par les valeurs camera-ready XKDD 2026 (décision à acter dans le spec).
# ═══════════════════════════════════════════════════════

FIDELITY_REFERENCE: Dict[Tuple[str, str], float] = {
    # Trio classique (soumission EGC 2026, non publiée)
    ("shap", "camembert"):              0.147,
    ("shap", "electra"):                0.293,
    ("shap", "gpt2"):                   0.249,
    ("lime", "camembert"):              0.067,
    ("lime", "electra"):                0.319,
    ("lime", "gpt2"):                   0.306,
    ("integrated_gradients", "camembert"): 0.185,
    ("integrated_gradients", "electra"):   0.267,
    ("integrated_gradients", "gpt2"):      0.228,
    # Nouvelles méthodes — estimations, jamais mesurées
    ("anchors", "camembert"):           0.132,
    ("anchors", "electra"):             0.285,
    ("anchors", "gpt2"):                0.241,
    ("counterfactual", "camembert"):    0.155,
    ("counterfactual", "electra"):      0.301,
    ("counterfactual", "gpt2"):         0.268,
    ("chain_of_thought", "gpt2"):       0.195,
    ("chain_of_thought", "qwen"):       0.210,
    ("logit_lens", "gpt2"):             0.178,
    ("logit_lens", "qwen"):             0.192,
}


# ═══════════════════════════════════════════════════════
# Contraintes par profil
# ═══════════════════════════════════════════════════════

DEFAULT_CONSTRAINTS: Dict[UserProfile, OperationalConstraints] = {
    UserProfile.TECHNICAL_EXPERT: OperationalConstraints(
        max_response_time=60.0,
        detail_level=DetailLevel.DETAILED,
        requires_audit_trail=False,
        num_features=20,
    ),
    UserProfile.MODERATOR: OperationalConstraints(
        max_response_time=30.0,
        detail_level=DetailLevel.STANDARD,
        requires_audit_trail=False,
        num_features=8,
    ),
    UserProfile.END_USER: OperationalConstraints(
        max_response_time=15.0,
        detail_level=DetailLevel.MINIMAL,
        requires_audit_trail=False,
        num_features=5,
    ),
    UserProfile.REGULATOR: OperationalConstraints(
        max_response_time=120.0,
        detail_level=DetailLevel.EXHAUSTIVE,
        requires_audit_trail=True,
        num_features=20,
    ),
}


# ═══════════════════════════════════════════════════════
# Registre des modèles HuggingFace
# ═══════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, Dict] = {
    "xlmr": {
        "model_id": "textdetox/xlmr-large-toxicity-classifier",
        "task_model_id": "textdetox/xlmr-large-toxicity-classifier",
        "revision": "b9c7c563427c591fc318d91eb592381ae2fbde66",
        "description": "XLM-RoBERTa Large — fine-tuné détection de toxicité, multilingue (encodeur)",
        "type": "encoder",
        "inference_mode": "toxicity_classifier",
        "params": "560M",
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "task_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "description": "Qwen 2.5 0.5B Instruct — LLM open source, classification par prompt (décodeur)",
        "type": "decoder",
        "inference_mode": "generative",
        "params": "494M",
    },
    "camembert": {
        "model_id": "camembert-base",
        "task_model_id": "cmarkea/distilcamembert-base-sentiment",
        "description": "CamemBERT — BERT français, modèle de sentiment (baseline encodeur)",
        "type": "encoder",
        "inference_mode": "sentiment_proxy",
        "params": "68M",
    },
}
