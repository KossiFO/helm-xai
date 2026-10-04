"""
Extension du UCB1 Learner avec calcul automatique des recompenses
a partir des metriques XAI (fidelite, suffisance, stabilite).

Motivation :
    Le UCB1Learner de base repose uniquement sur des notes utilisateur
    (ratings 1-5). Cela pose deux problemes :
      1. Le feedback humain est rare et couteux a collecter.
      2. Les notes subjectives varient fortement entre utilisateurs.

    Ce module ajoute un signal de recompense objectif, calcule
    automatiquement a partir des metriques d'evaluation XAI deja
    produites par le pipeline HELM (fidelity, sufficiency, stability).
    Le bandit peut ainsi apprendre meme sans feedback humain (mode
    automatique), ou combiner les deux signaux (mode hybride).

Design des metriques :
    - Fidelite (fidelity) : mesure la correspondance entre l'explication
      et le comportement reel du modele. Valeur dans [0, 1], plus haut
      est mieux. Directement utilisee comme recompense.
    - Suffisance (sufficiency) : mesure la baisse de performance du modele
      quand seules les features de l'explication sont conservees. Une
      valeur proche de 0 indique que l'explication est suffisante.
      Transformee en recompense par 1 - |sufficiency|.
    - Stabilite (stability) : mesure la coherence des explications face
      a de legeres perturbations. Peut etre negative (instabilite).
      Bornee par max(0, stability) pour garantir [0, 1].

Poids par profil :
    Les profils utilisateur HELM ont des priorites differentes.
    Un expert technique valorise surtout la fidelite (le mecanisme
    sous-jacent est-il correctement capture ?). Un regulateur privilegie
    la stabilite (les decisions sont-elles reproductibles ?).
    Les poids par defaut suivent les recommendations de la Section 4.2
    du manuscrit de these.

Reference : Auer et al. (2002), "Finite-time Analysis of the
Multiarmed Bandit Problem", Machine Learning, 47(2-3), 235-256.
"""

import logging
from typing import Dict, Optional, Tuple

from helm.config import UserProfile
from .ucb1_learner import UCB1Learner

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# Poids par profil
# ═══════════════════════════════════════════════════════

# Chaque profil definit l'importance relative de chaque metrique XAI.
# Les poids sont normalises pour sommer a 1.0 (hors "speed" qui est
# un placeholder pour une metrique de temps de calcul future).
#
# Expert technique : priorise la fidelite car il a besoin de comprendre
#   le mecanisme reel du modele pour le deboguer ou l'ameliorer.
# Moderateur : equilibre entre les metriques, avec un bonus de vitesse
#   car il doit traiter de nombreux cas rapidement.
# Utilisateur final : priorise fidelite + suffisance car il veut une
#   explication fiable et auto-suffisante (pas besoin de contexte externe).
# Regulateur : priorise la stabilite car la reproductibilite des
#   explications est cruciale pour la conformite reglementaire.

PROFILE_METRIC_WEIGHTS: Dict[str, Dict[str, float]] = {
    UserProfile.TECHNICAL_EXPERT.value: {
        "fidelity": 0.5,
        "sufficiency": 0.25,
        "stability": 0.25,
    },
    UserProfile.MODERATOR.value: {
        "fidelity": 0.3,
        "sufficiency": 0.3,
        "stability": 0.2,
        "speed": 0.2,
    },
    UserProfile.END_USER.value: {
        "fidelity": 0.4,
        "sufficiency": 0.4,
        "stability": 0.2,
    },
    UserProfile.REGULATOR.value: {
        "fidelity": 0.2,
        "sufficiency": 0.2,
        "stability": 0.6,
    },
}


def _default_weights_for_profile(profile: UserProfile) -> Dict[str, float]:
    """
    Retourne les poids metriques pour un profil donne.
    Si le profil n'est pas enregistre, retourne des poids uniformes.
    """
    key = profile.value
    if key in PROFILE_METRIC_WEIGHTS:
        return PROFILE_METRIC_WEIGHTS[key]
    return {"fidelity": 1 / 3, "sufficiency": 1 / 3, "stability": 1 / 3}


# ═══════════════════════════════════════════════════════
# UCB1 avec recompense automatique
# ═══════════════════════════════════════════════════════


class UCB1AutoReward(UCB1Learner):
    """
    Extension du UCB1Learner qui calcule des recompenses a partir
    des metriques XAI, en plus du systeme de notes utilisateur existant.

    Trois modes de fonctionnement :
      - Manuel (record) : identique au UCB1Learner de base, note 1-5.
      - Automatique (record_metrics) : recompense calculee a partir
        des metriques fidelite/suffisance/stabilite.
      - Hybride (record_hybrid) : combinaison ponderee des deux signaux.

    Le mode hybride est recommande quand du feedback humain est
    disponible, car il combine l'objectivite des metriques avec
    les preferences subjectives de l'utilisateur.

    Parameters
    ----------
    exploration_param : float
        Parametre C de UCB1. Controle le compromis exploration/exploitation.
        Valeur par defaut 1.5 (heritee de UCB1Learner).
    alpha : float
        Poids du signal humain dans le mode hybride.
        alpha=1.0 → uniquement notes utilisateur (mode manuel pur).
        alpha=0.0 → uniquement metriques (mode automatique pur).
        Valeur par defaut 0.5 (poids egal).
    use_profile_weights : bool
        Si True, utilise les poids specifiques au profil au lieu de
        poids uniformes pour les metriques. Recommande en production.
    storage_path : str, optional
        Chemin du fichier JSON de persistence (herite de UCB1Learner).
    """

    def __init__(
        self,
        exploration_param: float = 1.5,
        alpha: float = 0.5,
        use_profile_weights: bool = True,
        storage_path: Optional[str] = None,
    ):
        super().__init__(exploration_param=exploration_param, storage_path=storage_path)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha doit etre dans [0, 1], recu: {alpha}")
        self.alpha = alpha
        self.use_profile_weights = use_profile_weights

    # ───────────────────────────────────────────────────
    # Calcul de la recompense a partir des metriques
    # ───────────────────────────────────────────────────

    @staticmethod
    def compute_metric_reward(
        fidelity: float,
        sufficiency: float,
        stability: float,
        weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        speed: Optional[float] = None,
        speed_weight: float = 0.0,
    ) -> float:
        """
        Calcule une recompense composite a partir des metriques XAI.

        Formule :
            reward = w_f * fidelity
                   + w_s * (1 - |sufficiency|)
                   + w_st * max(0, stability)
                   [+ w_sp * speed]   (si fourni)

        Le resultat est borne a [0, 1].

        Parameters
        ----------
        fidelity : float
            Fidelite de l'explication. Attendu dans [0, 1].
            Plus la valeur est elevee, plus l'explication reflete
            fidelement le comportement du modele.
        sufficiency : float
            Suffisance de l'explication. Attendu dans [-1, 1].
            Une valeur proche de 0 signifie que les features selectionnees
            par l'explication sont suffisantes pour reproduire la decision.
            Transformee en recompense par 1 - |sufficiency|.
        stability : float
            Stabilite de l'explication face aux perturbations.
            Peut etre negative si l'explication est instable.
            Bornee a max(0, stability) pour eviter les recompenses negatives.
        weights : tuple of 3 floats
            Poids (w_fidelity, w_sufficiency, w_stability).
            Doivent sommer a 1.0 (ou a 1.0 - speed_weight si speed est fourni).
        speed : float, optional
            Metrique de vitesse normalisee dans [0, 1].
            Utilisee principalement pour le profil Moderateur.
        speed_weight : float
            Poids de la metrique speed. Ignore si speed est None.

        Returns
        -------
        float
            Recompense normalisee dans [0, 1].
        """
        w_f, w_s, w_st = weights

        # Transformations des metriques brutes en recompenses individuelles
        r_fidelity = max(0.0, min(1.0, fidelity))
        r_sufficiency = 1.0 - min(1.0, abs(sufficiency))
        r_stability = max(0.0, min(1.0, stability))

        reward = w_f * r_fidelity + w_s * r_sufficiency + w_st * r_stability

        # Composante optionnelle de vitesse (profil Moderateur)
        if speed is not None and speed_weight > 0:
            r_speed = max(0.0, min(1.0, speed))
            reward += speed_weight * r_speed

        # Borne de securite
        return max(0.0, min(1.0, reward))

    # ───────────────────────────────────────────────────
    # Mode automatique : metriques uniquement
    # ───────────────────────────────────────────────────

    def record_metrics(
        self,
        profile: UserProfile,
        method: str,
        fidelity: float,
        sufficiency: float,
        stability: float,
        weights: Optional[Tuple[float, float, float]] = None,
        speed: Optional[float] = None,
    ) -> float:
        """
        Enregistre une recompense calculee automatiquement a partir
        des metriques XAI. Aucun feedback humain n'est necessaire.

        Si use_profile_weights est actif et que weights n'est pas
        fourni explicitement, les poids specifiques au profil sont
        utilises (voir PROFILE_METRIC_WEIGHTS).

        Parameters
        ----------
        profile : UserProfile
            Profil de l'utilisateur courant.
        method : str
            Nom de la methode XAI (ex: "lime", "shap", "ig").
        fidelity : float
            Fidelite de l'explication dans [0, 1].
        sufficiency : float
            Suffisance de l'explication dans [-1, 1].
        stability : float
            Stabilite de l'explication, peut etre negative.
        weights : tuple of 3 floats, optional
            Poids (fidelity, sufficiency, stability). Si None, les poids
            du profil sont utilises (ou 1/3 uniforme si non disponible).
        speed : float, optional
            Metrique de vitesse normalisee [0, 1].

        Returns
        -------
        float
            La recompense calculee (pour inspection/debug).
        """
        # Resolution des poids
        speed_weight = 0.0
        if weights is not None:
            effective_weights = weights
        elif self.use_profile_weights:
            profile_w = _default_weights_for_profile(profile)
            effective_weights = (
                profile_w.get("fidelity", 1 / 3),
                profile_w.get("sufficiency", 1 / 3),
                profile_w.get("stability", 1 / 3),
            )
            speed_weight = profile_w.get("speed", 0.0)
        else:
            effective_weights = (1 / 3, 1 / 3, 1 / 3)

        reward = self.compute_metric_reward(
            fidelity=fidelity,
            sufficiency=sufficiency,
            stability=stability,
            weights=effective_weights,
            speed=speed,
            speed_weight=speed_weight,
        )

        # Enregistrement dans les structures de donnees heritees
        key = profile.value
        self._data[key][method]["total_reward"] += reward
        self._data[key][method]["count"] += 1
        self._total_count[key] += 1

        logger.debug(
            f"UCB1-auto record: {key}/{method} "
            f"fid={fidelity:.3f} suf={sufficiency:.3f} stab={stability:.3f} "
            f"→ reward={reward:.3f}"
        )

        if self.storage_path:
            self._save()

        return reward

    # ───────────────────────────────────────────────────
    # Mode hybride : note utilisateur + metriques
    # ───────────────────────────────────────────────────

    def record_hybrid(
        self,
        profile: UserProfile,
        method: str,
        rating: int,
        fidelity: float,
        sufficiency: float,
        stability: float,
        alpha: Optional[float] = None,
        weights: Optional[Tuple[float, float, float]] = None,
        speed: Optional[float] = None,
    ) -> float:
        """
        Enregistre une recompense hybride combinant la note utilisateur
        et les metriques XAI.

        Formule :
            reward_hybrid = alpha * reward_rating + (1 - alpha) * reward_metrics

        ou :
            reward_rating = (rating - 1) / 4   (normalisation 1-5 → 0-1)
            reward_metrics = compute_metric_reward(...)

        Pourquoi un mode hybride ?
            Les metriques XAI sont objectives mais ne capturent pas
            les preferences subjectives (clarte, lisibilite, utilite
            percue). Le feedback humain capture ces aspects mais est
            bruite et rare. La combinaison permet de beneficier des
            deux signaux.

        Parameters
        ----------
        profile : UserProfile
            Profil de l'utilisateur courant.
        method : str
            Nom de la methode XAI.
        rating : int
            Note utilisateur de 1 (mauvais) a 5 (excellent).
        fidelity, sufficiency, stability : float
            Metriques XAI (memes conventions que record_metrics).
        alpha : float, optional
            Poids du signal humain. Si None, utilise self.alpha.
            alpha=1.0 → pur rating, alpha=0.0 → pures metriques.
        weights : tuple, optional
            Poids metriques. Si None, utilise les poids du profil.
        speed : float, optional
            Metrique de vitesse normalisee [0, 1].

        Returns
        -------
        float
            La recompense hybride calculee.
        """
        if not 1 <= rating <= 5:
            raise ValueError(f"rating doit etre dans [1, 5], recu: {rating}")

        a = alpha if alpha is not None else self.alpha

        # Signal humain : normalisation identique au UCB1Learner de base
        reward_rating = (rating - 1) / 4.0

        # Signal metrique
        speed_weight = 0.0
        if weights is not None:
            effective_weights = weights
        elif self.use_profile_weights:
            profile_w = _default_weights_for_profile(profile)
            effective_weights = (
                profile_w.get("fidelity", 1 / 3),
                profile_w.get("sufficiency", 1 / 3),
                profile_w.get("stability", 1 / 3),
            )
            speed_weight = profile_w.get("speed", 0.0)
        else:
            effective_weights = (1 / 3, 1 / 3, 1 / 3)

        reward_metrics = self.compute_metric_reward(
            fidelity=fidelity,
            sufficiency=sufficiency,
            stability=stability,
            weights=effective_weights,
            speed=speed,
            speed_weight=speed_weight,
        )

        # Combinaison hybride
        reward = a * reward_rating + (1 - a) * reward_metrics
        reward = max(0.0, min(1.0, reward))

        # Enregistrement
        key = profile.value
        self._data[key][method]["total_reward"] += reward
        self._data[key][method]["count"] += 1
        self._total_count[key] += 1

        logger.debug(
            f"UCB1-hybrid record: {key}/{method} "
            f"rating={rating} (r={reward_rating:.2f}) "
            f"metrics=(r={reward_metrics:.3f}) "
            f"alpha={a:.2f} → reward={reward:.3f}"
        )

        if self.storage_path:
            self._save()

        return reward

    # ───────────────────────────────────────────────────
    # Utilitaires
    # ───────────────────────────────────────────────────

    def get_profile_weights(self, profile: UserProfile) -> Dict[str, float]:
        """
        Retourne les poids metriques actuellement configures pour un profil.
        Utile pour l'inspection et le debug.
        """
        return _default_weights_for_profile(profile)

    def get_statistics(self) -> Dict[str, Dict[str, Dict]]:
        """
        Etend les statistiques de base avec les poids de profil actifs.
        """
        stats = super().get_statistics()
        stats["_profile_weights"] = {
            profile.value: _default_weights_for_profile(profile)
            for profile in UserProfile
        }
        stats["_config"] = {
            "alpha": self.alpha,
            "use_profile_weights": self.use_profile_weights,
            "exploration_param": self.C,
        }
        return stats
