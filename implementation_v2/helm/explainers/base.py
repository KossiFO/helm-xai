"""
Classe de base abstraite pour les expliqueurs XAI du framework HELM v2.

Définit l'interface commune que chaque méthode d'explication
(LIME, SHAP, Integrated Gradients) doit implémenter.
"""

from abc import ABC, abstractmethod
from typing import Callable
import logging

from helm.config import Attribution

logger = logging.getLogger(__name__)


class BaseExplainer(ABC):
    """
    Interface abstraite pour tous les expliqueurs HELM v2.

    Chaque sous-classe encapsule une méthode XAI réelle et produit
    un objet ``Attribution`` contenant les scores token-level.
    """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Nom canonique de la méthode (ex. 'lime', 'shap', 'integrated_gradients')."""
        ...

    @abstractmethod
    def explain(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
    ) -> Attribution:
        """
        Calcule les attributions pour un texte donné.

        Parameters
        ----------
        text : str
            Texte d'entrée à expliquer.
        predict_fn : Callable
            Fonction de prédiction du modèle.
            Signature : ``List[str] -> np.ndarray`` de forme ``(n, 2)``.
        num_features : int
            Nombre maximal de tokens/features à retourner.

        Returns
        -------
        Attribution
            Objet contenant les scores d'importance par token.
        """
        ...

    @abstractmethod
    def estimate_time(self, text_length: int) -> float:
        """
        Estime le temps de calcul (en secondes) pour un texte de longueur donnée.

        Parameters
        ----------
        text_length : int
            Nombre de caractères du texte.

        Returns
        -------
        float
            Estimation en secondes.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} method='{self.method_name}'>"
