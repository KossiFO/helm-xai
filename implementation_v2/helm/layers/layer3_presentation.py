"""
Couche 3 — Personnalisation de l'Interface.
Transforme les attributions brutes en sortie adaptée au profil.
"""

import logging
from typing import Dict, List

from helm.config import UserProfile, FormattedOutput, Attribution
from helm.context import HELMContext

logger = logging.getLogger(__name__)


class PresentationLayer:
    """
    Troisième couche de HELM.
    Génère une sortie adaptée au profil utilisateur :
    - Expert technique → métriques détaillées, comparaison inter-méthodes
    - Modérateur → tableau de bord, action recommandée
    - Utilisateur final → explication en langage naturel simple
    - Régulateur → rapport d'audit traçable
    """

    def process(self, ctx: HELMContext) -> HELMContext:
        profile = ctx.effective_profile
        renderer = self._RENDERERS.get(profile, self._render_end_user)
        ctx.formatted_output = renderer(self, ctx)
        ctx.log(f"Couche 3 — Format : {ctx.formatted_output.format_type}")
        return ctx

    # ── Renderers par profil ──

    def _render_expert(self, ctx: HELMContext) -> FormattedOutput:
        sections = [self._method_section(name, attr, ctx.constraints.num_features)
                    for name, attr in ctx.attributions.items()]

        summary_parts = []
        if ctx.prediction:
            summary_parts.append(
                f"Prédiction : {ctx.prediction.label} "
                f"(confiance {ctx.prediction.confidence:.1%})"
            )
        methods_used = [a.metadata.get("display_name", key.upper()) for key, a in ctx.attributions.items()]
        summary_parts.append(
            f"Analysé par {len(methods_used)} méthode(s) : "
            + ", ".join(methods_used)
        )

        # Tokens communs les plus importants
        common_top = self._find_consensus_tokens(ctx.attributions)
        if common_top:
            tokens_str = ", ".join(f"« {t} »" for t in common_top[:5])
            summary_parts.append(f"Mots communs de même direction : {tokens_str}")

        return FormattedOutput(
            format_type="expert_analysis",
            summary=" | ".join(summary_parts),
            content={"consensus_tokens": common_top},
            sections=sections,
        )

    def _render_moderator(self, ctx: HELMContext) -> FormattedOutput:
        toxicity = ctx.prediction.probabilities[1] if ctx.prediction else 0.5
        if toxicity > 0.7:
            action = "SUPPRIMER"
            urgency = "haute"
        elif toxicity > 0.4:
            action = "RÉVISER"
            urgency = "moyenne"
        else:
            action = "APPROUVER"
            urgency = "basse"

        signals = self._signals_by_method(ctx.attributions, k=5)
        triggers = {name: data["positive"] for name, data in signals.items()}

        summary = (
            f"Action recommandée : {action} (urgence {urgency}). "
            f"Score toxicité : {toxicity:.0%}."
        )
        for name, data in signals.items():
            if data["positive"]:
                summary += f" Selon {data['label']}, éléments soutenant la toxicité : {', '.join(data['positive'])}."

        return FormattedOutput(
            format_type="moderator_dashboard",
            summary=summary,
            content={
                "action": action,
                "urgency": urgency,
                "toxicity_score": toxicity,
                "triggers_by_method": triggers,
                "by_method": signals,
                "score_fusion": False,
            },
        )

    def _render_end_user(self, ctx: HELMContext) -> FormattedOutput:
        toxicity = ctx.prediction.probabilities[1] if ctx.prediction else 0.5
        label = ctx.prediction.label if ctx.prediction else "inconnu"

        signals = self._signals_by_method(ctx.attributions, k=3)

        if label == "toxique":
            explanation = (
                f"Ce message a été identifié comme potentiellement toxique "
                f"(confiance : {toxicity:.0%})."
            )

        else:
            explanation = (
                f"Ce message semble acceptable (confiance : {1-toxicity:.0%})."
            )
        direction = 'positive' if label == 'toxique' else 'negative'
        for data in signals.values():
            words = data[direction]
            if words:
                explanation += f" Selon {data['label']}, les éléments allant dans ce sens sont : " + ", ".join(f"« {w} »" for w in words) + "."
        explanation += " Les méthodes sont présentées séparément, sans moyenne de leurs scores."

        return FormattedOutput(
            format_type="natural_language",
            summary=explanation,
            content={"by_method": signals, "score_fusion": False},
        )

    def _render_regulator(self, ctx: HELMContext) -> FormattedOutput:
        sections = []
        for name, attr in ctx.attributions.items():
            section = self._method_section(name, attr, ctx.constraints.num_features)
            if name in ctx.evaluation:
                section['evaluation'] = vars(ctx.evaluation[name])
            sections.append(section)

        summary = (
            f"Rapport d'audit — ID: {ctx.explanation_id} | "
            f"Modèle: {ctx.model_name} | "
            f"Méthodes: {', '.join(a.metadata.get('display_name', key) for key, a in ctx.attributions.items())} | "
            f"Prédiction: {ctx.prediction.label if ctx.prediction else 'N/A'} "
            f"({ctx.prediction.confidence:.1%} confiance)"
        )

        return FormattedOutput(
            format_type="audit_report",
            summary=summary,
            content={
                "explanation_id": ctx.explanation_id,
                "model_name": ctx.model_name,
                "audit_trail": ctx.logs.copy(),
            },
            sections=sections,
        )

    # ── Utilitaires ──

    @staticmethod
    def _method_section(name, attr, k):
        metadata = attr.metadata
        section = {'method': name, 'title': metadata.get('display_name', name.upper()),
                   'backend_metadata': metadata, 'computation_time': attr.computation_time,
                   'target_class': metadata.get('target_class')}
        if name == 'counterfactual':
            section.update(kind='counterfactual_search', found=metadata.get('found', bool(metadata.get('counterfactuals'))),
                           counterfactuals=metadata.get('counterfactuals', []),
                           description='Variantes trouvées dans une recherche bornée ; les poids de suppression ne sont pas des contrefactuels.')
        elif name == 'anchors':
            section.update(kind='rule', rule=metadata.get('anchor_words', []),
                           precision=metadata.get('precision'), coverage=metadata.get('coverage'),
                           threshold=metadata.get('threshold'))
        else:
            space = metadata.get('target_space')
            if space is None and metadata.get('attribution_protocol') == 'word_positions_v1':
                space = 'local_surrogate_coefficient' if name == 'lime' else 'probability'
            section.update(kind='attribution', units=space or metadata.get('score_kind', 'non spécifiées'),
                           tokens=[{'token': t, 'score': score} for t, score in attr.top_tokens[:k]],
                           base_value=attr.base_value)
        return section

    @staticmethod
    def _signals_by_method(attributions: Dict[str, Attribution], k: int = 5) -> Dict:
        """Keep lexical scores within their own method; orient directions toward toxicity."""
        signals = {}
        for name, attr in attributions.items():
            # Rules, counterfactual search and state norms are not token attributions.
            if name not in {'shap', 'lime', 'integrated_gradients', 'leave_one_out'}:
                continue
            if (attr.metadata.get('valid') is False or attr.metadata.get('score_kind') == 'norm_growth_proxy'
                    or attr.metadata.get('target_space') == 'logit'):
                continue
            target = attr.metadata.get('target_class')
            if target not in (0, 1):
                continue  # An unknown direction must not become a claim about toxicity.
            oriented = [(word, -score if target == 0 else score) for word, score in attr.top_tokens[:k]]
            signals[name] = {'label': attr.metadata.get('display_name', name.upper()),
                             'target_class': target,
                             'positive': [w for w, score in oriented if score > 0],
                             'negative': [w for w, score in oriented if score < 0]}
        return signals

    def _find_consensus_tokens(self, attributions: Dict[str, Attribution], k: int = 5) -> List[str]:
        """Lexical intersection with the same direction, never an average of magnitudes."""
        signals = list(self._signals_by_method(attributions, k=10).values())
        if len(signals) < 2:
            return []
        shared = set()
        for direction in ('positive', 'negative'):
            common = set(signals[0][direction])
            for method in signals[1:]:
                common.intersection_update(method[direction])
            shared.update(common)
        return sorted(shared)[:k]

    _RENDERERS = {
        UserProfile.TECHNICAL_EXPERT: _render_expert,
        UserProfile.MODERATOR: _render_moderator,
        UserProfile.END_USER: _render_end_user,
        UserProfile.REGULATOR: _render_regulator,
    }
