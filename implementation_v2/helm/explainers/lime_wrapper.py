"""
Wrapper LIME pour le framework HELM v2.

Utilise ``lime.lime_text.LimeTextExplainer`` pour calculer
des attributions réelles au niveau des mots.
"""

import time
import logging
from typing import Callable


from helm.config import Attribution
from .base import BaseExplainer

logger = logging.getLogger(__name__)

try:
    from lime.lime_text import LimeTextExplainer

    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False


class LIMEExplainer(BaseExplainer):
    """
    Expliqueur LIME (Local Interpretable Model-agnostic Explanations).

    Génère des perturbations locales du texte et ajuste un modèle linéaire
    interprétable pour estimer l'importance de chaque mot.

    Parameters
    ----------
    num_samples : int
        Nombre d'échantillons perturbés générés pour chaque explication.
        Valeur réduite (300) pour un fonctionnement rapide sur CPU.
    random_state : int
        Graine aléatoire pour la reproductibilité.
    """

    def __init__(self, num_samples: int = 300, random_state: int = 42, *, word_positions: bool = True):
        if not _LIME_AVAILABLE:
            raise ImportError(
                "Le paquet 'lime' est requis pour LIMEExplainer.\n"
                "Installez-le avec : pip install lime"
            )
        self.word_positions = word_positions
        self.num_samples = num_samples
        self.random_state = random_state
        self._explainer = LimeTextExplainer(
            class_names=["non_toxique", "toxique"],
            random_state=self.random_state,
            split_expression=r"\W+",
        )
        logger.info(
            "LIMEExplainer initialisé (num_samples=%d, random_state=%d)",
            self.num_samples,
            self.random_state,
        )

    # ── Interface BaseExplainer ──────────────────────────

    @property
    def method_name(self) -> str:
        return "lime"

    def explain(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
    ) -> Attribution:
        """
        Calcule les attributions LIME pour le texte donné.

        Parameters
        ----------
        text : str
            Texte à expliquer.
        predict_fn : Callable
            Fonction ``List[str] -> ndarray (n, 2)`` renvoyant les
            probabilités [non_toxique, toxique].
        num_features : int
            Nombre maximal de mots à inclure dans l'explication.

        Returns
        -------
        Attribution
            Scores d'importance LIME par mot.
        """
        if self.word_positions:
            from .word_positions import lime_positions
            return lime_positions(text, predict_fn, num_features, num_samples=self.num_samples, seed=self.random_state)
        logger.debug("LIME explain() sur un texte de %d caractères", len(text))
        t0 = time.perf_counter()

        try:
            explanation = self._explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                num_samples=self.num_samples,
                labels=(1,),  # classe « toxique »
            )

            # Extraction des paires (mot, score) pour la classe toxique (label=1)
            word_scores = dict(explanation.as_list(label=1))

            # Score d'intercept (base value du modèle linéaire local)
            intercept = explanation.intercept.get(1, None)

            elapsed = time.perf_counter() - t0
            logger.info(
                "LIME terminé en %.2fs — %d features extraites",
                elapsed,
                len(word_scores),
            )

            return Attribution(
                method_name=self.method_name,
                token_scores=word_scores,
                computation_time=elapsed,
                base_value=float(intercept) if intercept is not None else None,
                metadata={
                    "num_samples": self.num_samples,
                    "num_features": num_features,
                    "local_prediction": float(explanation.local_pred[0])
                    if hasattr(explanation, "local_pred")
                    and explanation.local_pred is not None
                    else None,
                },
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Erreur LIME après %.2fs : %s", elapsed, exc)
            raise RuntimeError(
                f"Échec du calcul LIME après {elapsed:.2f}s : {exc}"
            ) from exc

    def estimate_time(self, text_length: int) -> float:
        """
        Estimation heuristique du temps de calcul LIME.

        Basée sur le nombre d'échantillons et la longueur du texte.
        """
        # ~0.01s par échantillon pour un texte court, linéaire en longueur
        words_approx = max(1, text_length / 5)
        return self.num_samples * 0.01 * (1 + words_approx / 100)
