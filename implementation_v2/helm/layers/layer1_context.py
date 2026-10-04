"""
Couche 1 — Caractérisation Contextuelle.
Détermine le profil utilisateur, le type de contenu et les contraintes.
"""

import logging
import re

from helm.config import (
    UserProfile, ContentType, OperationalConstraints,
    DEFAULT_CONSTRAINTS,
)
from helm.context import HELMContext

logger = logging.getLogger(__name__)


class ContextualLayer:
    """
    Première couche de HELM.
    Analyse le contexte d'entrée pour déterminer :
    - Le profil de l'utilisateur (si non fourni)
    - Le type de contenu (texte, image, multimodal)
    - Les contraintes opérationnelles adaptées au profil
    """

    def process(self, ctx: HELMContext) -> HELMContext:
        # Profil
        if ctx.user_profile is None:
            ctx.detected_profile = self._infer_profile(ctx)
            ctx.log(f"Couche 1 — Profil par défaut (non prédit) : {ctx.detected_profile.value}")
        else:
            ctx.detected_profile = ctx.user_profile
            ctx.log(f"Couche 1 — Profil explicite : {ctx.user_profile.value}")

        # Type de contenu
        ctx.content_type = self._detect_content_type(ctx.text)
        ctx.log(f"Couche 1 — Type contenu : {ctx.content_type.value}")

        # Contraintes
        ctx.constraints = self._determine_constraints(ctx.effective_profile)
        ctx.log(
            f"Couche 1 — Contraintes : temps_max={ctx.constraints.max_response_time}s, "
            f"détail={ctx.constraints.detail_level.name}, "
            f"features={ctx.constraints.num_features}"
        )
        return ctx

    def _infer_profile(self, ctx: HELMContext) -> UserProfile:
        """Infère le profil par défaut (utilisateur final)."""
        return UserProfile.END_USER

    def _detect_content_type(self, text: str) -> ContentType:
        """Détecte le type de contenu à partir du texte."""
        image_patterns = r"(https?://\S+\.(jpg|png|gif|webp)|\[image\]|📷|🖼)"
        if re.search(image_patterns, text, re.IGNORECASE):
            return ContentType.IMAGE
        return ContentType.TEXT

    def _determine_constraints(self, profile: UserProfile) -> OperationalConstraints:
        """Retourne les contraintes adaptées au profil."""
        return DEFAULT_CONSTRAINTS.get(
            profile,
            OperationalConstraints()
        )
