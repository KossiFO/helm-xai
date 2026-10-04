"""
Objet de contexte typé traversant le pipeline HELM.
Remplace le dict non typé de la v1.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid
import time

from helm.config import (
    UserProfile, ContentType, OperationalConstraints, Prediction, Attribution,
    MethodSelection, FormattedOutput, EvaluationMetrics,
)


@dataclass
class HELMContext:
    """
    Objet immuable traversant les 3 couches de HELM.
    Chaque couche enrichit le contexte sans modifier les champs existants.
    """

    # ── Entrées (fournies par l'utilisateur) ──
    text: str
    model_name: str = "camembert"
    user_profile: Optional[UserProfile] = None

    # ── Couche 1 : Caractérisation contextuelle ──
    detected_profile: Optional[UserProfile] = None
    content_type: ContentType = ContentType.TEXT
    constraints: Optional[OperationalConstraints] = None

    # ── Prédiction du modèle ──
    prediction: Optional[Prediction] = None

    # ── Couche 2 : Sélection des méthodes ──
    selected_methods: List[MethodSelection] = field(default_factory=list)

    # ── Résultats XAI ──
    attributions: Dict[str, Attribution] = field(default_factory=dict)

    # ── Couche 3 : Sortie formatée ──
    formatted_output: Optional[FormattedOutput] = None

    # ── Évaluation ──
    evaluation: Dict[str, EvaluationMetrics] = field(default_factory=dict)
    contextual_choice: Optional[object] = None
    failed_attributions: Dict = field(default_factory=dict)

    # ── Métadonnées ──
    explanation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    start_time: float = field(default_factory=time.time)
    total_time: float = 0.0
    logs: List[str] = field(default_factory=list)

    @property
    def effective_profile(self) -> UserProfile:
        """Profil effectif (explicite ou détecté)."""
        return self.user_profile or self.detected_profile or UserProfile.END_USER

    def log(self, message: str) -> None:
        elapsed = time.time() - self.start_time
        self.logs.append(f"[{elapsed:.2f}s] {message}")

    def finalize(self) -> None:
        self.total_time = time.time() - self.start_time
        self.log(f"Pipeline terminé en {self.total_time:.2f}s")
