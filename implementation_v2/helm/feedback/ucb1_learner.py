"""
Apprentissage des préférences par UCB1 (Upper Confidence Bound).

Référence : Auer et al. (2002), "Finite-time Analysis of the
Multiarmed Bandit Problem", Machine Learning, 47(2-3), 235-256.

Chaque méthode XAI est un « bras » du bandit multi-bras.
UCB1 équilibre exploration (méthodes peu testées) et
exploitation (méthodes bien notées).
"""

import math
import json
import os
import logging
from typing import Dict, List, Optional
from collections import defaultdict

from helm.config import UserProfile

logger = logging.getLogger(__name__)


class UCB1Learner:
    """
    score_i = μ_i + C × √(ln(N) / n_i)

    où :
        μ_i = récompense moyenne de la méthode i
        N   = nombre total d'essais
        n_i = nombre d'essais de la méthode i
        C   = paramètre d'exploration (default 1.5)
    """

    def __init__(self, exploration_param: float = 1.5, storage_path: Optional[str] = None):
        self.C = exploration_param
        self.storage_path = storage_path
        # {profile_key: {method: {"total_reward": float, "count": int}}}
        self._data: Dict[str, Dict[str, Dict]] = defaultdict(
            lambda: defaultdict(lambda: {"total_reward": 0.0, "count": 0})
        )
        self._total_count: Dict[str, int] = defaultdict(int)

        if storage_path and os.path.exists(storage_path):
            self._load()

    def record(self, profile: UserProfile, method: str, rating: int) -> None:
        """
        Enregistre un feedback (note 1-5).
        La note est normalisée en récompense [0, 1].
        """
        key = profile.value
        reward = (rating - 1) / 4.0  # 1→0, 5→1
        self._data[key][method]["total_reward"] += reward
        self._data[key][method]["count"] += 1
        self._total_count[key] += 1
        logger.debug(f"UCB1 record: {key}/{method} rating={rating} reward={reward:.2f}")
        if self.storage_path:
            self._save()

    def get_ucb_scores(self, profile: UserProfile, methods: List[str]) -> Dict[str, float]:
        """Calcule le score UCB1 pour chaque méthode."""
        key = profile.value
        N = self._total_count[key]
        scores = {}
        for method in methods:
            stats = self._data[key][method]
            n_i = stats["count"]
            if n_i == 0:
                scores[method] = float("inf")  # Jamais essayé → explorer
            else:
                mu = stats["total_reward"] / n_i
                exploration = self.C * math.sqrt(math.log(N) / n_i)
                scores[method] = mu + exploration
        return scores

    def get_preference_weights(self, profile: UserProfile) -> Dict[str, float]:
        """
        Retourne des poids multiplicatifs pour la couche 2.
        Poids > 1.0 → favorisé, < 1.0 → pénalisé.
        """
        key = profile.value
        N = self._total_count[key]
        if N == 0:
            return {}
        weights = {}
        for method, stats in self._data[key].items():
            n_i = stats["count"]
            if n_i == 0:
                weights[method] = 2.0
            else:
                mu = stats["total_reward"] / n_i
                exploration = self.C * math.sqrt(math.log(N) / n_i)
                ucb = mu + exploration
                weights[method] = max(0.1, min(3.0, ucb * 1.5))
        return weights

    def get_statistics(self) -> Dict[str, Dict[str, Dict]]:
        """Résumé lisible des données."""
        result = {}
        for profile_key, methods in self._data.items():
            result[profile_key] = {}
            for method, stats in methods.items():
                n = stats["count"]
                result[profile_key][method] = {
                    "count": n,
                    "mean_reward": round(stats["total_reward"] / n, 3) if n > 0 else 0,
                    "total_reward": round(stats["total_reward"], 3),
                }
        return result

    def select_arm(self, profile: UserProfile, methods: List[str]) -> str:
        """Sélectionne la méthode à explorer/exploiter (pour simulation)."""
        scores = self.get_ucb_scores(profile, methods)
        return max(scores, key=scores.get)

    def _save(self) -> None:
        data = {
            "preferences": {k: dict(v) for k, v in self._data.items()},
            "total_counts": dict(self._total_count),
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pk, methods in data.get("preferences", {}).items():
            for method, stats in methods.items():
                self._data[pk][method] = stats
        for pk, count in data.get("total_counts", {}).items():
            self._total_count[pk] = count
