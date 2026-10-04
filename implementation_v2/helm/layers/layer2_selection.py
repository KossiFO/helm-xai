"""
Couche 2 — Sélection Adaptative des Méthodes XAI.
Consulte la matrice de décision, ajuste par fidélité et préférences UCB1.
"""

import logging
from typing import List

from helm.config import (
    UserProfile, ContentType, OperationalConstraints,
    MethodSelection, DECISION_MATRIX, FIDELITY_REFERENCE,
)
from helm.context import HELMContext

logger = logging.getLogger(__name__)


# Heuristiques historiques de budget ; ne prédisent pas la latence IG native.
ESTIMATED_TIMES = {
    "lime": 4.1,
    "shap": 1.3,
    "integrated_gradients": 0.2,
    "anchors": 50.9,
    "counterfactual": 0.7,
    "chain_of_thought": 18.6,
    "logit_lens": 1.4,
}


class SelectionLayer:
    """
    Deuxième couche de HELM.

    Pipeline de sélection en 4 étapes :
    A. Consultation de la matrice de décision → candidats de base
    B. Ajustement par les scores de fidélité du modèle cible
    C. Ajustement par les préférences apprises (UCB1)
    D. Filtrage par budget temporel
    """

    def __init__(self, preference_learner=None, *, use_legacy_fidelity=False, unbudgeted_methods=(), allowed_methods=None):
        self._preference_learner = preference_learner
        self.use_legacy_fidelity = use_legacy_fidelity
        self.unbudgeted_methods = set(unbudgeted_methods)
        self.allowed_methods = None if allowed_methods is None else set(allowed_methods)
        if self.allowed_methods == set():
            raise ValueError("Le pool de méthodes ne peut pas être vide.")

    def process(self, ctx: HELMContext) -> HELMContext:
        profile = ctx.effective_profile
        content_type = ctx.content_type
        constraints = ctx.constraints or OperationalConstraints()

        # A. Candidats
        candidates = self._get_candidates(profile, content_type)
        ctx.log(f"Couche 2 — {len(candidates)} candidats depuis la matrice")

        if self.allowed_methods is not None:
            excluded = [c.method_name for c in candidates if c.method_name not in self.allowed_methods]
            candidates = [c for c in candidates if c.method_name in self.allowed_methods]
            ctx.log("Pool explicite ; méthodes exclues : " + ", ".join(excluded))
            if not candidates:
                raise ValueError("Aucune méthode du pool ne correspond au profil.")

        # B. Fidélité
        if self.use_legacy_fidelity:
            candidates = self._adjust_fidelity(candidates, ctx.model_name)
            ctx.log("Fidélités historiques activées explicitement (proxies/estimations).")

        # C. Préférences UCB1
        candidates = self._adjust_preferences(candidates, profile)

        # D. Budget temporel
        candidates = self._apply_budget(candidates, constraints.max_response_time)

        # Normalisation
        total = sum(c.priority for c in candidates) or 1.0
        for c in candidates:
            c.priority = round(c.priority / total, 4)

        candidates.sort(key=lambda m: m.priority, reverse=True)
        ctx.selected_methods = candidates
        ctx.log(
            "Couche 2 — Méthodes retenues : "
            + ", ".join(f"{m.method_name}({m.priority:.1%})" for m in candidates)
        )
        return ctx

    def _get_candidates(
        self, profile: UserProfile, content_type: ContentType
    ) -> List[MethodSelection]:
        entries = DECISION_MATRIX.get(
            (profile, content_type),
            [("lime", 0.5), ("shap", 0.5)],
        )
        return [
            MethodSelection(
                method_name=name,
                priority=weight,
                estimated_time=None if name in self.unbudgeted_methods else ESTIMATED_TIMES.get(name, 10.0),
                rationale=f"Matrice({profile.value}, poids={weight:.2f})",
            )
            for name, weight in entries
        ]

    def _adjust_fidelity(
        self, candidates: List[MethodSelection], model_name: str
    ) -> List[MethodSelection]:
        max_fidelity = 0.35  # Borne supérieure observée
        for c in candidates:
            fid = FIDELITY_REFERENCE.get((c.method_name, model_name))
            if fid is not None:
                bonus = fid / max_fidelity
                c.priority *= (0.7 + 0.3 * bonus)
                c.rationale += f" → fidélité({model_name})={fid:.3f}"
        return candidates

    def _adjust_preferences(
        self, candidates: List[MethodSelection], profile: UserProfile
    ) -> List[MethodSelection]:
        if self._preference_learner is None:
            return candidates
        weights = self._preference_learner.get_preference_weights(profile)
        for c in candidates:
            w = weights.get(c.method_name, 1.0)
            c.priority *= w
            if abs(w - 1.0) > 0.01:
                c.rationale += f" → UCB1(×{w:.2f})"
        return candidates

    def _apply_budget(
        self, candidates: List[MethodSelection], budget: float
    ) -> List[MethodSelection]:
        candidates.sort(key=lambda m: m.priority, reverse=True)
        selected = []
        remaining = budget
        for c in candidates:
            if c.method_name in self.unbudgeted_methods:
                c.rationale += " → activation native explicite hors budget heuristique ; latence inconnue"
                selected.append(c)
                continue
            if c.estimated_time <= remaining:
                selected.append(c)
                remaining -= c.estimated_time
        if not selected and candidates:
            selected.append(candidates[0])
        return selected
