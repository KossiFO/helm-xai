"""
Métrique de stabilité pour les attributions XAI — HELM v2.

Ce module mesure la robustesse des explications face à de petites
perturbations du texte d'entrée. Une bonne méthode XAI devrait
produire des attributions similaires pour des textes quasi-identiques.

La stabilité est quantifiée par la corrélation de rang de Spearman
entre les attributions du texte original et celles de versions
légèrement perturbées (fautes de frappe, espaces, mots neutres).

Références
----------
- Alvarez-Melis, D. & Jaakkola, T. (2018). *On the Robustness of
  Interpretability Methods*. ICML Workshop.
- Agarwal, C. et al. (2022). *Rethinking Stability for Attribution-based
  Explanations*. arXiv:2203.06877.
"""

import logging
import random
from typing import Callable, List, Optional

import numpy as np
from scipy.stats import spearmanr

from helm.config import Attribution

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Perturbation de texte
# ═══════════════════════════════════════════════════════════════════════════

# Mots neutres français à insérer aléatoirement
_NEUTRAL_WORDS: List[str] = [
    "et", "aussi", "très", "puis", "donc", "bien", "alors",
    "encore", "même", "enfin",
]


def perturb_text(text: str, seed: Optional[int] = None) -> str:
    """
    Crée une perturbation mineure d'un texte français.

    Applique aléatoirement l'une des stratégies suivantes :
      1. Duplication d'une lettre (faute de frappe)
      2. Inversion de deux lettres adjacentes
      3. Ajout ou suppression d'un espace
      4. Insertion d'un mot neutre

    Parameters
    ----------
    text : str
        Texte original à perturber.
    seed : Optional[int]
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    str
        Texte légèrement perturbé.
    """
    if not text or not text.strip():
        return text

    rng = random.Random(seed)
    strategy = rng.choice(["duplicate_letter", "swap_letters",
                           "space_change", "neutral_word"])

    if strategy == "duplicate_letter":
        return _duplicate_letter(text, rng)
    elif strategy == "swap_letters":
        return _swap_adjacent_letters(text, rng)
    elif strategy == "space_change":
        return _change_space(text, rng)
    else:  # neutral_word
        return _insert_neutral_word(text, rng)


def _duplicate_letter(text: str, rng: random.Random) -> str:
    """Duplique une lettre aléatoire dans le texte."""
    # Trouver les positions de lettres alphabétiques
    letter_positions = [i for i, c in enumerate(text) if c.isalpha()]
    if not letter_positions:
        return text
    pos = rng.choice(letter_positions)
    return text[:pos] + text[pos] + text[pos:]


def _swap_adjacent_letters(text: str, rng: random.Random) -> str:
    """Inverse deux lettres adjacentes dans un mot."""
    # Trouver les positions où deux lettres consécutives existent
    swap_positions = [
        i for i in range(len(text) - 1)
        if text[i].isalpha() and text[i + 1].isalpha()
    ]
    if not swap_positions:
        return text
    pos = rng.choice(swap_positions)
    chars = list(text)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def _change_space(text: str, rng: random.Random) -> str:
    """Ajoute ou supprime un espace aléatoirement."""
    # Trouver les positions d'espaces existants
    space_positions = [i for i, c in enumerate(text) if c == " "]

    if space_positions and rng.random() < 0.5:
        # Supprimer un espace
        pos = rng.choice(space_positions)
        return text[:pos] + text[pos + 1:]
    else:
        # Ajouter un espace à un endroit aléatoire (entre des lettres)
        insert_positions = [
            i for i in range(1, len(text))
            if text[i - 1].isalpha() and text[i].isalpha()
        ]
        if not insert_positions:
            return text
        pos = rng.choice(insert_positions)
        return text[:pos] + " " + text[pos:]


def _insert_neutral_word(text: str, rng: random.Random) -> str:
    """Insère un mot neutre à une position aléatoire dans le texte."""
    words = text.split()
    if not words:
        return text
    neutral = rng.choice(_NEUTRAL_WORDS)
    # Insérer à une position aléatoire (pas en début pour garder le sens)
    pos = rng.randint(1, len(words))
    words.insert(pos, neutral)
    return " ".join(words)


# ═══════════════════════════════════════════════════════════════════════════
# Métrique de stabilité
# ═══════════════════════════════════════════════════════════════════════════


def _rank_correlation_on_common_tokens(
    attr_original: Attribution,
    attr_perturbed: Attribution,
) -> Optional[float]:
    """
    Calcule la corrélation de rang de Spearman entre deux attributions
    sur leurs tokens communs.

    Parameters
    ----------
    attr_original : Attribution
        Attribution du texte original.
    attr_perturbed : Attribution
        Attribution du texte perturbé.

    Returns
    -------
    Optional[float]
        Coefficient de Spearman, ou None si le calcul est impossible
        (moins de 3 tokens communs).
    """
    common_tokens = (
        set(attr_original.token_scores.keys())
        & set(attr_perturbed.token_scores.keys())
    )

    if len(common_tokens) < 3:
        logger.debug(
            "Seulement %d tokens communs — corrélation de Spearman "
            "non fiable, ignorée.", len(common_tokens),
        )
        return None

    # Extraire les scores dans le même ordre
    scores_orig = [attr_original.token_scores[t] for t in common_tokens]
    scores_pert = [attr_perturbed.token_scores[t] for t in common_tokens]

    # Vérifier la variance (Spearman requiert de la variabilité)
    if np.std(scores_orig) < 1e-12 or np.std(scores_pert) < 1e-12:
        logger.debug("Variance nulle dans les scores — corrélation = 0.0")
        return 0.0

    corr, p_value = spearmanr(scores_orig, scores_pert)

    # spearmanr peut retourner NaN si les données sont constantes
    if np.isnan(corr):
        return 0.0

    return float(corr)


def compute_stability(
    text: str,
    predict_fn: Callable,
    explainer,
    n_perturbations: int = 5,
    num_features: int = 10,
) -> float:
    """
    Mesure la stabilité d'un expliqueur face à des perturbations mineures.

    Génère ``n_perturbations`` versions légèrement modifiées du texte,
    calcule l'attribution pour chacune, puis mesure la corrélation de
    rang de Spearman entre l'attribution originale et chaque attribution
    perturbée. La stabilité est la moyenne de ces corrélations.

    Parameters
    ----------
    text : str
        Texte original à expliquer.
    predict_fn : Callable
        Fonction de prédiction ``List[str] -> np.ndarray (n, 2)``.
    explainer : BaseExplainer
        Instance de l'expliqueur à évaluer (doit implémenter ``explain()``).
    n_perturbations : int
        Nombre de perturbations à générer.
    num_features : int
        Nombre de features passé à ``explainer.explain()``.

    Returns
    -------
    float
        Score de stabilité dans [-1, 1]. Plus élevé = plus stable.
        Retourne 0.0 si aucune corrélation n'a pu être calculée.
    """
    logger.info(
        "Calcul de stabilité : méthode=%s, n_perturbations=%d",
        explainer.method_name, n_perturbations,
    )

    # Attribution sur le texte original
    attr_original = explainer.explain(text, predict_fn, num_features=num_features)

    correlations: List[float] = []

    for i in range(n_perturbations):
        # Créer une perturbation avec une graine déterministe
        perturbed = perturb_text(text, seed=i)

        # Vérifier que la perturbation est non vide et différente
        if not perturbed.strip():
            logger.debug("Perturbation %d vide, ignorée.", i)
            continue

        logger.debug(
            "Perturbation %d/%d : '%s...' -> '%s...'",
            i + 1, n_perturbations,
            text[:40], perturbed[:40],
        )

        try:
            attr_perturbed = explainer.explain(
                perturbed, predict_fn, num_features=num_features,
            )
        except Exception as exc:
            logger.warning(
                "Erreur lors de l'explication de la perturbation %d : %s",
                i, exc,
            )
            continue

        corr = _rank_correlation_on_common_tokens(attr_original, attr_perturbed)
        if corr is not None:
            correlations.append(corr)
            logger.debug("Perturbation %d : rho=%.4f", i, corr)

    if not correlations:
        logger.warning(
            "Aucune corrélation valide calculée sur %d perturbations. "
            "Stabilité = 0.0", n_perturbations,
        )
        return 0.0

    stability = float(np.mean(correlations))

    logger.info(
        "Stabilité = %.4f (sur %d/%d perturbations valides)",
        stability, len(correlations), n_perturbations,
    )

    return stability
