"""
Wrapper Anchors pour le framework HELM v2.

Implémente la méthode Anchors (Ribeiro et al., 2018) qui produit des
**règles suffisantes** : des conditions minimales sur les mots du texte
qui garantissent la prédiction avec une précision élevée.

Contrairement à LIME/SHAP qui produisent des scores numériques,
Anchors produit des règles IF-THEN lisibles par un humain.

Référence :
    Ribeiro, M. T., Singh, S., & Guestrin, C. (2018).
    "Anchors: High-Precision Model-Agnostic Explanations."
    AAAI Conference on Artificial Intelligence.
"""

import re
import time
import logging
from typing import Callable, List, Dict, Tuple

import numpy as np

from helm.config import Attribution
from .base import BaseExplainer

logger = logging.getLogger(__name__)

try:
    from anchor import anchor_text

    _ANCHOR_AVAILABLE = True
except ImportError:
    _ANCHOR_AVAILABLE = False


class AnchorsExplainer(BaseExplainer):
    """
    Expliqueur Anchors — produit des règles suffisantes.

    Trouve l'ensemble minimal de mots dont la présence garantit
    la prédiction du modèle avec une précision >= threshold.

    Si le paquet ``anchor-exp`` est installé, utilise l'implémentation
    officielle. Sinon, utilise une approximation par beam search interne.

    Parameters
    ----------
    threshold : float
        Précision minimale exigée pour l'ancre (default: 0.95).
    beam_size : int
        Taille du beam search pour la recherche d'ancres.
    num_samples : int
        Nombre d'échantillons perturbés pour estimer la précision.
    use_native : bool ou None
        False force le beam search historique ; True exige anchor-exp.
        None conserve la détection automatique historique.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        beam_size: int = 4,
        num_samples: int = 200,
        use_native=None,
    ):
        self.threshold = threshold
        self.beam_size = beam_size
        self.num_samples = num_samples
        if use_native is True and not _ANCHOR_AVAILABLE:
            raise ImportError("Le mode natif requiert pip install 'helm-xai[anchors]'.")
        self._use_native = _ANCHOR_AVAILABLE if use_native is None else bool(use_native)
        mode = "anchor-exp (natif)" if self._use_native else "beam search (interne)"
        logger.info(
            "AnchorsExplainer initialisé en mode %s (threshold=%.2f, beam_size=%d)",
            mode, self.threshold, self.beam_size,
        )

    @property
    def method_name(self) -> str:
        return "anchors"

    def explain(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
    ) -> Attribution:
        if self._use_native:
            return self._explain_native(text, predict_fn, num_features)
        return self._explain_beam(text, predict_fn, num_features)

    def estimate_time(self, text_length: int) -> float:
        words_approx = max(1, text_length / 5)
        return self.num_samples * 0.01 * (1 + words_approx / 50)

    # ── Mode natif (anchor-exp) ──────────────────────────

    def _explain_native(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int,
    ) -> Attribution:
        logger.debug("Anchors (natif) sur un texte de %d caractères", len(text))
        t0 = time.perf_counter()

        try:
            import spacy

            # Tokenisation française locale ; aucun modèle spaCy à télécharger.
            explainer = anchor_text.AnchorText(
                spacy.blank("fr"),
                class_names=["non_toxique", "toxique"],
                use_unk_distribution=True,
            )

            def _label_fn(texts):
                probas = predict_fn(list(texts))
                return (probas[:, 1] > 0.5).astype(int)

            exp = explainer.explain_instance(
                text, _label_fn,
                threshold=self.threshold,
                batch_size=self.num_samples,
                beam_size=self.beam_size,
            )

            anchor_words = exp.names()
            precision = exp.precision()

            token_scores = self._anchor_words_to_scores(
                text, anchor_words, predict_fn, num_features
            )

            elapsed = time.perf_counter() - t0
            logger.info(
                "Anchors (natif) terminé en %.2fs — %d mots dans l'ancre, précision=%.2f",
                elapsed, len(anchor_words), precision,
            )

            return Attribution(
                method_name=self.method_name,
                token_scores=token_scores,
                computation_time=elapsed,
                metadata={
                    "mode": "native",
                    "effective_method": "anchors",
                    "target_class": int(_label_fn([text])[0]),
                    "coverage": float(exp.coverage()),
                    "perturbation": "anchor-exp UNK distribution",
                    "anchor_words": anchor_words,
                    "anchor_rule": " AND ".join(anchor_words),
                    "precision": precision,
                    "threshold": self.threshold,
                },
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Erreur Anchors (natif) après %.2fs : %s", elapsed, exc)
            raise RuntimeError(
                f"Échec du calcul Anchors (natif) après {elapsed:.2f}s : {exc}"
            ) from exc

    # ── Mode beam search (interne) ───────────────────────

    def _explain_beam(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int,
    ) -> Attribution:
        """
        Approximation par beam search.

        1. Tokenise le texte en mots
        2. Pour chaque candidat (sous-ensemble de mots), estime la précision
           en générant des perturbations où les mots hors candidat sont remplacés
        3. Étend le beam en ajoutant un mot à la fois
        4. S'arrête quand precision >= threshold ou beam épuisé
        """
        logger.debug("Anchors (beam) sur un texte de %d caractères", len(text))
        t0 = time.perf_counter()

        try:
            words = re.findall(r"\S+", text)
            if not words:
                elapsed = time.perf_counter() - t0
                return Attribution(
                    method_name=self.method_name,
                    token_scores={},
                    computation_time=elapsed,
                    metadata={"mode": "beam_search", "anchor_words": [], "anchor_rule": ""},
                )

            # Prédiction de référence
            base_probs = predict_fn([text])
            base_label = int(base_probs[0, 1] > 0.5)

            # Mots neutres pour remplacement
            neutral_words = ["le", "un", "de", "la", "et"]

            def _estimate_precision(anchor_indices: List[int]) -> float:
                """Estime la précision de l'ancre par échantillonnage."""
                if not anchor_indices:
                    return 0.0

                perturbed_texts = []
                rng = np.random.RandomState(42)

                for _ in range(self.num_samples):
                    new_words = []
                    for i, w in enumerate(words):
                        if i in anchor_indices:
                            new_words.append(w)
                        else:
                            new_words.append(rng.choice(neutral_words))
                    perturbed_texts.append(" ".join(new_words))

                probas = predict_fn(perturbed_texts)
                preds = (probas[:, 1] > 0.5).astype(int)
                return float(np.mean(preds == base_label))

            # Phase 1 : scorer chaque mot individuellement
            word_precisions: List[Tuple[int, float]] = []
            for i in range(len(words)):
                prec = _estimate_precision([i])
                word_precisions.append((i, prec))

            word_precisions.sort(key=lambda x: x[1], reverse=True)

            # Phase 2 : beam search
            beam: List[Tuple[List[int], float]] = []
            for idx, prec in word_precisions[: self.beam_size]:
                beam.append(([idx], prec))

            best_anchor: List[int] = beam[0][0] if beam else []
            best_precision = beam[0][1] if beam else 0.0

            max_anchor_size = min(num_features, len(words))
            for size in range(2, max_anchor_size + 1):
                if best_precision >= self.threshold:
                    break

                candidates: List[Tuple[List[int], float]] = []
                for anchor, _ in beam:
                    for idx, _ in word_precisions:
                        if idx not in anchor:
                            new_anchor = sorted(anchor + [idx])
                            prec = _estimate_precision(new_anchor)
                            candidates.append((new_anchor, prec))

                            if prec >= self.threshold:
                                best_anchor = new_anchor
                                best_precision = prec
                                break

                    if best_precision >= self.threshold:
                        break

                if not candidates:
                    break

                candidates.sort(key=lambda x: x[1], reverse=True)
                beam = candidates[: self.beam_size]

                if beam[0][1] > best_precision:
                    best_anchor = beam[0][0]
                    best_precision = beam[0][1]

            # Construire les scores : mots dans l'ancre reçoivent un score
            # proportionnel à leur contribution individuelle
            anchor_words_list = [words[i] for i in best_anchor]
            token_scores = self._anchor_words_to_scores(
                text, anchor_words_list, predict_fn, num_features
            )

            anchor_rule = " AND ".join(
                f"'{words[i]}'" for i in best_anchor
            )

            elapsed = time.perf_counter() - t0
            logger.info(
                "Anchors (beam) terminé en %.2fs — ancre: [%s], précision=%.2f",
                elapsed, anchor_rule, best_precision,
            )

            return Attribution(
                method_name=self.method_name,
                token_scores=token_scores,
                computation_time=elapsed,
                metadata={
                    "mode": "beam_search",
                    "anchor_words": anchor_words_list,
                    "anchor_rule": anchor_rule,
                    "precision": best_precision,
                    "threshold": self.threshold,
                    "beam_size": self.beam_size,
                    "num_samples": self.num_samples,
                },
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Erreur Anchors (beam) après %.2fs : %s", elapsed, exc)
            raise RuntimeError(
                f"Échec du calcul Anchors (beam) après {elapsed:.2f}s : {exc}"
            ) from exc

    # ── Utilitaire commun ────────────────────────────────

    @staticmethod
    def _anchor_words_to_scores(
        text: str,
        anchor_words: List[str],
        predict_fn: Callable,
        num_features: int,
    ) -> Dict[str, float]:
        """
        Convertit les mots de l'ancre en scores d'attribution.

        Les mots dans l'ancre reçoivent un score positif proportionnel
        au changement de prédiction quand ils sont retirés.
        Les mots hors ancre reçoivent un score nul.
        """
        words = re.findall(r"\S+", text)
        base_probs = predict_fn([text])
        base_toxic = float(base_probs[0, 1])

        token_scores: Dict[str, float] = {}

        # Mots dans l'ancre : mesurer leur contribution individuelle
        for anchor_word in anchor_words:
            perturbed_words = [
                "le" if w == anchor_word else w for w in words
            ]
            perturbed_text = " ".join(perturbed_words)
            perturbed_probs = predict_fn([perturbed_text])
            perturbed_toxic = float(perturbed_probs[0, 1])
            importance = base_toxic - perturbed_toxic
            clean = anchor_word.strip(".,;:!?\"'()[]{}«»")
            if clean:
                token_scores[clean] = token_scores.get(clean, 0.0) + importance

        # Mots hors ancre : score nul
        for w in words:
            clean = w.strip(".,;:!?\"'()[]{}«»")
            if clean and clean not in token_scores:
                token_scores[clean] = 0.0

        # Limiter au top num_features
        if len(token_scores) > num_features:
            sorted_tokens = sorted(
                token_scores.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            token_scores = dict(sorted_tokens[:num_features])

        return token_scores
