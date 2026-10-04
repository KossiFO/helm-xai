"""Protocole signé des expériences Codex de septembre 2026.

C = p(classe initiale) - p(texte retiré), S = p(classe initiale) - p(texte gardé).
La classe cible et les interventions sont explicites, sans baseline inventée.
"""
from dataclasses import dataclass, asdict
import numpy as np


def probabilities(predict_fn, texts):
    p = np.asarray(predict_fn(texts), dtype=float)
    if p.ndim != 2 or p.shape != (len(texts), 2) or not np.isfinite(p).all():
        raise ValueError("Prédictions attendues : matrice finie (n, 2).")
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Probabilités hors du simplexe.")
    return p


@dataclass(frozen=True)
class FaithfulnessResult:
    comprehensiveness: float
    sufficiency: float
    target_class: int
    positions: tuple
    removed_text: str
    kept_text: str
    original_probability: float
    removed_probability: float
    kept_probability: float
    protocol: str = "signed_cs_word_positions_v1"
    empty_replacement: str = "le"

    def to_dict(self):
        return asdict(self)


def evaluate_positions(text, positions, predict_fn, *, target_class=None, empty_replacement="le"):
    """Évalue les positions de mots (zéro-indexées) ; cible = classe prédite par défaut."""
    words = text.split()
    positions = tuple(positions)
    if not words or not positions or len(set(positions)) != len(positions):
        raise ValueError("Texte, positions non vides et distinctes requis.")
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(words) for i in positions):
        raise ValueError("Position de mot invalide.")
    if not isinstance(empty_replacement, str) or not empty_replacement.strip():
        raise ValueError("Le remplacement d'un texte vide doit être explicite et non vide.")
    if target_class is not None and (isinstance(target_class, bool) or target_class not in (0, 1)):
        raise ValueError("Classe cible attendue : 0, 1 ou None.")
    selected = set(positions)
    removed = " ".join(w for i, w in enumerate(words) if i not in selected) or empty_replacement
    kept = " ".join(w for i, w in enumerate(words) if i in selected) or empty_replacement
    p = probabilities(predict_fn, [text, removed, kept])
    target = int(np.argmax(p[0])) if target_class is None else int(target_class)
    p0, pr, pk = map(float, p[:, target])
    return FaithfulnessResult(p0-pr, p0-pk, target, tuple(map(int, positions)), removed, kept, p0, pr, pk,
                              empty_replacement=empty_replacement)


def evaluate_attribution(text, attribution, predict_fn, *, k=3, target_class=None):
    """Top-k absolu sur positions ; refuse une attribution sans mot correspondant."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k doit être un entier positif.")
    if attribution.metadata.get("valid") is False:
        raise ValueError("Attribution numériquement non admissible : aucune métrique imputée.")
    if attribution.method_name in {'counterfactual', 'anchors'} or attribution.metadata.get('effective_method') == 'counterfactual_search':
        raise ValueError("C/S de tokens ne mesure pas la validité des règles ou des contrefactuels.")
    if target_class is None:
        target_class = attribution.metadata.get('target_class')
    words = text.split()
    scores = attribution.metadata.get("scores_by_position")
    if scores is None:
        raise ValueError("Le protocole signé exige des scores par position ; dictionnaire lexical ambigu.")
    if len(scores) != len(words):
        raise ValueError("Attributions et positions du texte ne concordent pas.")
    matches = list(enumerate(scores))
    if not matches or not all(np.isfinite(float(score)) for _, score in matches):
        raise ValueError("Aucune attribution finie et alignée avec le texte.")
    positions = [i for i, _ in sorted(matches, key=lambda pair: -abs(pair[1]))[:k]]
    return evaluate_positions(text, positions, predict_fn, target_class=target_class)


def signed_cs_reward(comprehensiveness, sufficiency, *, weight=0.5):
    """Récompense automatique continue, distincte d'une note humaine 1–5."""
    c, s, w = map(float, (comprehensiveness, sufficiency, weight))
    if not np.isfinite([c, s, w]).all() or not -1 <= c <= 1 or not -1 <= s <= 1 or not 0 <= w <= 1:
        raise ValueError("C/S attendus dans [-1,1], poids dans [0,1], valeurs finies.")
    return 0.5 + 0.5 * (w*c - (1-w)*s)
