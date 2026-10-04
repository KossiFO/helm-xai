"""
Wrapper SHAP pour le framework HELM v2.

Utilise ``shap.Explainer`` avec un masque textuel pour calculer
des valeurs de Shapley réelles au niveau des tokens.
"""

import time
import logging
from typing import Callable

import numpy as np

from helm.config import Attribution
from .base import BaseExplainer

logger = logging.getLogger(__name__)

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


class SHAPExplainer(BaseExplainer):
    """
    Expliqueur SHAP (SHapley Additive exPlanations).

    Calcule les valeurs de Shapley exactes ou approchées pour chaque
    token du texte en utilisant un masque de segmentation textuelle.

    Parameters
    ----------
    max_evals : int
        Nombre maximal d'évaluations du modèle par explication.
        Contrôle le compromis précision / vitesse.
    """

    def __init__(self, max_evals: int = 100, *, word_positions: bool = True):
        if not _SHAP_AVAILABLE:
            raise ImportError(
                "Le paquet 'shap' est requis pour SHAPExplainer.\n"
                "Installez-le avec : pip install shap"
            )
        self.word_positions = word_positions
        self.max_evals = max_evals
        logger.info("SHAPExplainer initialisé (max_evals=%d)", self.max_evals)

    # ── Interface BaseExplainer ──────────────────────────

    @property
    def method_name(self) -> str:
        return "shap"

    def explain(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
    ) -> Attribution:
        """
        Calcule les attributions SHAP pour le texte donné.

        Parameters
        ----------
        text : str
            Texte à expliquer.
        predict_fn : Callable
            Fonction ``List[str] -> ndarray (n, 2)`` renvoyant les
            probabilités [non_toxique, toxique].
        num_features : int
            Nombre maximal de tokens à inclure dans l'explication.

        Returns
        -------
        Attribution
            Valeurs de Shapley par token, avec la valeur de base du modèle.
        """
        if self.word_positions:
            from .word_positions import shap_positions
            return shap_positions(text, predict_fn, num_features, max_evals=self.max_evals)
        logger.debug("SHAP explain() sur un texte de %d caractères", len(text))
        t0 = time.perf_counter()

        try:
            # Wrapper pour garantir que predict_fn reçoit une List[str]
            def _predict_wrapper(texts):
                if isinstance(texts, str):
                    texts = [texts]
                elif isinstance(texts, np.ndarray):
                    texts = texts.tolist()
                if isinstance(texts, list) and len(texts) > 0 and not isinstance(texts[0], str):
                    texts = [str(t) for t in texts]
                return predict_fn(texts)

            # Masque de segmentation sur les frontières non-alphanumériques
            masker = shap.maskers.Text(tokenizer=r"\W+")

            explainer = shap.Explainer(
                _predict_wrapper,
                masker=masker,
                output_names=["non_toxique", "toxique"],
            )

            # Calcul des valeurs de Shapley
            shap_values = explainer(
                [text],
                max_evals=self.max_evals,
            )

            # Extraction des données pour la première (et seule) instance
            values = shap_values.values[0]      # shape: (n_tokens,) ou (n_tokens, 2)
            data_tokens = shap_values.data[0]    # tokens segmentés
            base_values = shap_values.base_values[0]

            # Si multi-classe, prendre la colonne « toxique » (index 1)
            if values.ndim > 1:
                values = values[:, 1]
                base_value = float(base_values[1]) if hasattr(base_values, '__len__') else float(base_values)
            else:
                base_value = float(base_values) if base_values is not None else None

            # Construction du dictionnaire token -> score
            token_scores: dict[str, float] = {}
            for token, score in zip(data_tokens, values):
                token_str = str(token).strip()
                if token_str:
                    # Si un token apparaît plusieurs fois, sommer les contributions
                    token_scores[token_str] = token_scores.get(token_str, 0.0) + float(score)

            # Limiter au top num_features par importance absolue
            if len(token_scores) > num_features:
                sorted_tokens = sorted(
                    token_scores.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                token_scores = dict(sorted_tokens[:num_features])

            elapsed = time.perf_counter() - t0
            logger.info(
                "SHAP terminé en %.2fs — %d tokens attribués",
                elapsed,
                len(token_scores),
            )

            return Attribution(
                method_name=self.method_name,
                token_scores=token_scores,
                computation_time=elapsed,
                base_value=base_value,
                metadata={
                    "max_evals": self.max_evals,
                    "num_tokens_total": len(data_tokens),
                    "algorithm": type(explainer).__name__,
                },
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Erreur SHAP après %.2fs : %s", elapsed, exc)
            raise RuntimeError(
                f"Échec du calcul SHAP après {elapsed:.2f}s : {exc}"
            ) from exc

    def estimate_time(self, text_length: int) -> float:
        """
        Estimation heuristique du temps de calcul SHAP.

        Le coût dépend principalement de ``max_evals`` et du nombre de tokens.
        """
        words_approx = max(1, text_length / 5)
        return self.max_evals * 0.02 * (1 + words_approx / 50)
