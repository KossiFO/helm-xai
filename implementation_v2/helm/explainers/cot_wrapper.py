"""
Wrapper Chain-of-Thought (CoT) pour le framework HELM v2.

Spécifique aux modèles LLM/décodeurs (Qwen, GPT, LLaMA...).
Demande au modèle de produire une **explication en langage naturel**
de son raisonnement avant de donner sa prédiction.

L'explication CoT est ensuite parsée pour en extraire les mots-clés
et produire des scores d'attribution.

Références :
    Wei, J., et al. (2022).
    "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models."
    NeurIPS.

    Kojima, T., et al. (2022).
    "Large Language Models are Zero-Shot Reasoners."
    NeurIPS.
"""

import re
import time
import logging
from typing import Callable, Dict, List, Optional


from helm.config import Attribution
from .base import BaseExplainer

logger = logging.getLogger(__name__)


COT_PROMPT_TEMPLATE = """Analyse le texte suivant et explique étape par étape pourquoi il est toxique ou non-toxique.

Texte : "{text}"

Instructions :
1. Identifie les mots ou expressions clés qui influencent le caractère toxique ou non du texte.
2. Pour chaque mot/expression clé, explique son impact (positif = pousse vers toxique, négatif = pousse vers non-toxique).
3. Donne ta conclusion finale.

Raisonnement étape par étape :"""


class CoTExplainer(BaseExplainer):
    """
    Expliqueur Chain-of-Thought pour LLMs.

    Utilise le modèle LLM lui-même pour générer une explication
    en langage naturel, puis extrait les attributions du raisonnement.

    Parameters
    ----------
    model_manager : object, optional
        ModelManager du LLM (doit avoir generate() et predict_proba()).
        Si None, utilise predict_fn en mode dégradé (sans CoT textuel).
    max_new_tokens : int
        Nombre max de tokens générés pour l'explication.
    temperature : float
        Température de génération (0 = déterministe).
    """

    def __init__(
        self,
        model_manager=None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ):
        self.model_manager = model_manager
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        logger.info(
            "CoTExplainer initialisé (max_new_tokens=%d, temperature=%.1f)",
            self.max_new_tokens, self.temperature,
        )

    @property
    def method_name(self) -> str:
        return "chain_of_thought"

    def explain(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
    ) -> Attribution:
        """
        Génère une explication CoT et en extrait les attributions.

        Si un model_manager est disponible avec generate(), produit
        un raisonnement en langage naturel. Sinon, utilise une
        analyse par perturbation pondérée par la position.
        """
        logger.debug("CoT explain() sur un texte de %d caractères", len(text))
        t0 = time.perf_counter()

        try:
            # Prédiction de référence
            base_probs = predict_fn([text])
            base_toxic_prob = float(base_probs[0, 1])
            base_label = "toxique" if base_toxic_prob > 0.5 else "non_toxique"

            cot_text = None
            if self.model_manager is not None and hasattr(self.model_manager, "generate"):
                cot_text = self._generate_cot(text)

            generated = bool(cot_text)
            if cot_text:
                # Extraire les attributions du raisonnement CoT
                token_scores, key_phrases = self._parse_cot_explanation(
                    text, cot_text, predict_fn, num_features
                )
            else:
                # Mode dégradé : perturbation avec pondération positionnelle
                token_scores = self._fallback_perturbation(
                    text, predict_fn, num_features
                )
                key_phrases = []
                cot_text = self._generate_synthetic_cot(
                    text, base_label, base_toxic_prob if base_label == "toxique" else 1-base_toxic_prob, token_scores
                )

            elapsed = time.perf_counter() - t0
            logger.info(
                "CoT terminé en %.2fs — %d tokens attribués",
                elapsed, len(token_scores),
            )

            return Attribution(
                method_name=self.method_name,
                token_scores=token_scores,
                computation_time=elapsed,
                base_value=base_toxic_prob,
                metadata={
                    "cot_explanation": cot_text,
                    "base_label": base_label,
                    "key_phrases": key_phrases,
                    "mode": "generative" if generated else "perturbation",
                    "effective_method": "generated_rationale" if generated else "loo_position_weighted",
                    "display_name": "Justification générée" if generated else "Perturbations et texte à gabarit",
                    "target_class": 1,
                },
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Erreur CoT après %.2fs : %s", elapsed, exc)
            raise RuntimeError(
                f"Échec du calcul CoT après {elapsed:.2f}s : {exc}"
            ) from exc

    def estimate_time(self, text_length: int) -> float:
        if self.model_manager is not None:
            # Génération LLM : dépend du max_new_tokens
            return self.max_new_tokens * 0.05 + 2.0
        else:
            words_approx = max(1, text_length / 5)
            return (words_approx + 1) * 0.05

    # ── Génération CoT via le LLM ────────────────────────

    def _generate_cot(self, text: str) -> Optional[str]:
        """Génère le raisonnement CoT via le modèle LLM."""
        try:
            prompt = COT_PROMPT_TEMPLATE.format(text=text)

            if hasattr(self.model_manager, "generate"):
                result = self.model_manager.generate(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                return result if isinstance(result, str) else str(result)
            return None
        except Exception as exc:
            logger.warning("Échec de la génération CoT : %s", exc)
            return None

    # ── Parsing du raisonnement CoT ──────────────────────

    def _parse_cot_explanation(
        self,
        original_text: str,
        cot_text: str,
        predict_fn: Callable,
        num_features: int,
    ) -> tuple:
        """
        Parse le raisonnement CoT pour extraire les attributions.

        Stratégie :
        1. Cherche les mots du texte original mentionnés dans le CoT
        2. Pondère par la fréquence de mention et le contexte (positif/négatif)
        3. Valide avec une perturbation rapide
        """
        words = re.findall(r"\w+", original_text.lower())
        unique_words = list(set(words))
        cot_lower = cot_text.lower()

        # Mots mentionnés dans le raisonnement
        mentioned_scores: Dict[str, float] = {}
        key_phrases: List[str] = []

        # Patterns indicateurs de toxicité dans le CoT
        toxic_indicators = [
            r"toxique", r"insult", r"offens", r"agress",
            r"vulg", r"haine", r"méchant", r"négatif",
        ]
        non_toxic_indicators = [
            r"non.toxique", r"positif", r"poli", r"neutre",
            r"respect", r"constructi", r"bienveill",
        ]

        for word in unique_words:
            if len(word) < 3:
                continue

            # Chercher le mot dans le CoT
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            matches = list(pattern.finditer(cot_text))

            if not matches:
                mentioned_scores[word] = 0.0
                continue

            # Analyser le contexte autour de chaque mention
            score = 0.0
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(cot_text), match.end() + 100)
                context = cot_lower[start:end]

                # Score positif si contexte toxique
                for pat in toxic_indicators:
                    if re.search(pat, context):
                        score += 0.3
                # Score négatif si contexte non-toxique
                for pat in non_toxic_indicators:
                    if re.search(pat, context):
                        score -= 0.2

                # Bonus pour fréquence de mention
                score += 0.1

            mentioned_scores[word] = score

            if abs(score) > 0.2:
                key_phrases.append(f"{word} (score CoT: {score:.2f})")

        # Validation par perturbation rapide
        base_probs = predict_fn([original_text])
        base_toxic = float(base_probs[0, 1])

        original_words = re.findall(r"\S+", original_text)
        token_scores: Dict[str, float] = {}

        for i, w in enumerate(original_words):
            clean = w.strip(".,;:!?\"'()[]{}«»").lower()
            if not clean:
                continue

            # Score de perturbation
            remaining = original_words[:i] + original_words[i + 1:]
            if remaining:
                perturbed_probs = predict_fn([" ".join(remaining)])
                perturbation_score = base_toxic - float(perturbed_probs[0, 1])
            else:
                perturbation_score = base_toxic

            # Combiner CoT et perturbation (60% perturbation, 40% CoT)
            cot_score = mentioned_scores.get(clean, 0.0)
            combined = 0.6 * perturbation_score + 0.4 * cot_score

            # Restaurer la casse originale
            original_clean = w.strip(".,;:!?\"'()[]{}«»")
            if original_clean:
                token_scores[original_clean] = (
                    token_scores.get(original_clean, 0.0) + combined
                )

        # Limiter
        if len(token_scores) > num_features:
            sorted_tokens = sorted(
                token_scores.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            token_scores = dict(sorted_tokens[:num_features])

        return token_scores, key_phrases

    # ── Mode dégradé ─────────────────────────────────────

    def _fallback_perturbation(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int,
    ) -> Dict[str, float]:
        """Perturbation simple quand le LLM n'est pas disponible."""
        words = re.findall(r"\S+", text)
        if not words:
            return {}

        base_probs = predict_fn([text])
        base_toxic = float(base_probs[0, 1])

        perturbed_texts = []
        for i in range(len(words)):
            remaining = words[:i] + words[i + 1:]
            perturbed_texts.append(" ".join(remaining) if remaining else "le")

        perturbed_probs = predict_fn(perturbed_texts)

        token_scores: Dict[str, float] = {}
        for i, w in enumerate(words):
            clean = w.strip(".,;:!?\"'()[]{}«»")
            if clean:
                importance = base_toxic - float(perturbed_probs[i, 1])
                token_scores[clean] = token_scores.get(clean, 0.0) + importance

        if len(token_scores) > num_features:
            sorted_tokens = sorted(
                token_scores.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            token_scores = dict(sorted_tokens[:num_features])

        return token_scores

    @staticmethod
    def _generate_synthetic_cot(
        text: str,
        label: str,
        prob: float,
        token_scores: Dict[str, float],
    ) -> str:
        """Génère un raisonnement CoT synthétique à partir des scores."""
        top_positive = sorted(
            [(t, s) for t, s in token_scores.items() if s > 0],
            key=lambda x: x[1], reverse=True,
        )[:3]
        top_negative = sorted(
            [(t, s) for t, s in token_scores.items() if s < 0],
            key=lambda x: x[1],
        )[:2]

        lines = [f"Analyse du texte : \"{text}\"", ""]
        lines.append("Étape 1 : Identification des mots-clés")

        if top_positive:
            words_str = ", ".join(f"'{t}'" for t, _ in top_positive)
            lines.append(
                f"  Les mots {words_str} poussent vers une classification 'toxique'."
            )

        if top_negative:
            words_str = ", ".join(f"'{t}'" for t, _ in top_negative)
            lines.append(
                f"  Les mots {words_str} poussent vers 'non-toxique'."
            )

        lines.append("")
        lines.append("Étape 2 : Conclusion")
        lines.append(
            f"  Le texte est classé '{label}' avec une confiance de {prob:.0%}."
        )

        return "\n".join(lines)
