"""LinUCB disjoint expérimental, porté du calcul Codex du 10 septembre.

Normalisation apprise sur calibration seule. Récompense du seul bras choisi
après la décision ; aucune préférence humaine artificielle ni adaptation implicite.
"""
from dataclasses import dataclass
import math
import secrets
import numpy as np


@dataclass(frozen=True)
class ContextualChoice:
    decision_id: str
    method: str
    predicted_reward: float
    exploration_bonus: float


class ContextualSelector:
    def __init__(self, methods, *, ridge=10.0, alpha=0.5, model_name="xlmr"):
        self.methods = tuple(methods)
        if not self.methods or len(set(self.methods)) != len(self.methods) or not all(isinstance(m, str) and m for m in self.methods):
            raise ValueError("Méthodes non vides et distinctes requises.")
        if not math.isfinite(ridge) or ridge <= 0 or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("Régularisation positive, exploration non négative et finies requises.")
        self.ridge, self.alpha, self.model_name = ridge, alpha, model_name
        self._pending = {}
        self._fitted = False

    @staticmethod
    def context(text, confidence, predicted_class):
        return np.array([math.log1p(len(text.split())), confidence, predicted_class], dtype=float)

    @staticmethod
    def _validate(raw):
        raw = np.asarray(raw, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw) or not np.isfinite(raw).all():
            raise ValueError("Contexte fini attendu : log1p(longueur), confiance, classe.")
        if (raw[:, 0] < 0).any() or (raw[:, 1] < 0).any() or (raw[:, 1] > 1).any() or not np.isin(raw[:, 2], [0, 1]).all():
            raise ValueError("Contexte hors domaine.")
        return raw

    def _transform(self, raw):
        raw = self._validate(raw)
        return np.column_stack((np.ones(len(raw)), (raw[:, :2]-self.mean_)/self.scale_, raw[:, 2]))

    def fit_calibration(self, contexts, rewards):
        """Ajustement explicite sur calibration (lignes × méthodes dans l'ordre déclaré)."""
        raw = self._validate(contexts)
        y = np.asarray(rewards, dtype=float)
        if y.shape != (len(raw), len(self.methods)) or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
            raise ValueError("Matrice de récompenses de calibration invalide.")
        if self._pending:
            raise ValueError("Décisions en attente : terminer ou annuler avant recalibration.")
        self.mean_ = raw[:, :2].mean(0)
        sd = raw[:, :2].std(0)
        self.scale_ = np.where(sd > 1e-12, sd, 1.)
        x = self._transform(raw)
        self.a_ = np.repeat((self.ridge*np.eye(4)+x.T@x)[None, :, :], len(self.methods), axis=0)
        self.b_ = (x.T@(y-.5)).T
        self._fitted = True
        return self

    def choose(self, context, *, allowed_methods=None):
        if not self._fitted:
            raise ValueError("Une calibration explicite est requise avant tout choix.")
        v = self._transform(np.asarray(context, dtype=float).reshape(1, -1))[0]
        allowed = set(self.methods if allowed_methods is None else allowed_methods)
        indices = [i for i, method in enumerate(self.methods) if method in allowed]
        if not indices:
            raise ValueError("Aucune méthode de la politique n'est disponible pour ce profil.")
        z = np.linalg.solve(self.a_, np.broadcast_to(v, (len(self.methods), 4))[..., None])[..., 0]
        means = .5 + np.einsum("ij,ij->i", z, self.b_)
        bonuses = self.alpha*np.sqrt(np.maximum(z@v, 0))
        arm = max(indices, key=lambda i: means[i]+bonuses[i])
        choice = ContextualChoice(secrets.token_hex(16), self.methods[arm], float(means[arm]), float(bonuses[arm]))
        self._pending[choice.decision_id] = (choice, v.copy(), arm)
        return choice

    def update(self, choice, reward):
        """Mise à jour explicite, une fois, de la méthode effectivement choisie."""
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Récompense continue finie dans [0,1] requise.")
        pending = self._pending.get(choice.decision_id)
        if pending is None or pending[0] != choice:
            raise ValueError("Décision inconnue, altérée ou déjà mise à jour.")
        _, v, arm = self._pending.pop(choice.decision_id)
        self.a_[arm] += np.outer(v, v)
        self.b_[arm] += (reward-.5)*v

    def discard(self, choice):
        pending = self._pending.get(choice.decision_id)
        if pending is None or pending[0] != choice:
            raise ValueError("Décision inconnue.")
        del self._pending[choice.decision_id]
