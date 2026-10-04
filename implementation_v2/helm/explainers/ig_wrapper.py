"""Compatibilité historique : IG sur logit avec modèle, sinon LOO par remplacement.

LOO est une méthode distincte, pas une approximation numérique d'IG.
Le protocole natif vérifié de septembre est dans VerifiedIGExplainer.
"""

import re
import time
import logging
import warnings
from typing import Callable, List


from helm.config import Attribution
from .base import BaseExplainer

logger = logging.getLogger(__name__)

# ── Détection optionnelle de captum & torch ──────────

try:
    import torch
    import captum  # noqa: F401 — sonde de disponibilité
    from captum.attr import LayerIntegratedGradients

    _CAPTUM_AVAILABLE = True
except ImportError:
    _CAPTUM_AVAILABLE = False

_NEUTRAL_WORD = "le"  # Mot neutre français pour le remplacement LOO


class IGExplainer(BaseExplainer):
    """
    Façade historique Integrated Gradients / Leave-One-Out.

    Si ``captum`` est installé et qu'un modèle PyTorch avec couche
    d'embeddings est fourni, les gradients intégrés sont calculés
    par intégration numérique via ``LayerIntegratedGradients``.

    Sinon, la méthode distincte leave-one-out est utilisée :
    chaque mot est remplacé par un mot neutre (« le ») et la
    variation de la probabilité « toxique » est mesurée.

    Parameters
    ----------
    n_steps : int
        Nombre de pas d'interpolation pour Integrated Gradients.
        Ignoré en mode leave-one-out.
    model : object, optional
        Modèle PyTorch (nn.Module) pour le mode captum.
    embedding_layer : object, optional
        Couche d'embeddings du modèle pour ``LayerIntegratedGradients``.
    tokenizer : object, optional
        Tokenizer compatible HuggingFace pour le mode captum.
    """

    def __init__(
        self,
        n_steps: int = 50,
        model=None,
        embedding_layer=None,
        tokenizer=None,
    ):
        self.n_steps = n_steps
        self.model = model
        self.embedding_layer = embedding_layer
        self.tokenizer = tokenizer

        # Déterminer le mode d'exécution
        self._use_captum = (
            _CAPTUM_AVAILABLE
            and model is not None
            and embedding_layer is not None
            and tokenizer is not None
        )

        mode = "captum (LayerIntegratedGradients)" if self._use_captum else "leave-one-out"
        logger.info(
            "IGExplainer initialisé en mode %s (n_steps=%d)", mode, self.n_steps
        )

    # ── Interface BaseExplainer ──────────────────────────

    @property
    def method_name(self) -> str:
        return "integrated_gradients"

    def explain(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
    ) -> Attribution:
        """
        Calcule les attributions Integrated Gradients ou LOO pour le texte.

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
            Scores d'importance par token, avec ``convergence_delta``
            si disponible (mode captum).
        """
        if self._use_captum:
            return self._explain_captum(text, predict_fn, num_features)
        else:
            warnings.warn("IGExplainer sans modèle exécute LOO par remplacement, pas IG. "
                          "Utiliser VerifiedIGExplainer pour le protocole natif.", DeprecationWarning, stacklevel=2)
            return self._explain_loo(text, predict_fn, num_features)

    def estimate_time(self, text_length: int) -> float:
        """
        Estimation heuristique du temps de calcul.

        En mode captum, le coût dépend de ``n_steps``.
        En mode LOO, le coût est proportionnel au nombre de mots.
        """
        words_approx = max(1, text_length / 5)
        if self._use_captum:
            return self.n_steps * 0.02 * (1 + words_approx / 50)
        else:
            # LOO : une évaluation par mot + une évaluation de base
            return (words_approx + 1) * 0.05

    # ── Mode Captum (Integrated Gradients) ─────────

    def _explain_captum(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int,
    ) -> Attribution:
        """
        Calcul des gradients intégrés via Captum.

        Utilise ``LayerIntegratedGradients`` sur la couche d'embeddings
        du modèle PyTorch.
        """
        logger.debug("IG captum sur un texte de %d caractères", len(text))
        t0 = time.perf_counter()

        try:
            device = next(self.model.parameters()).device
            self.model.eval()

            # Tokenisation
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Baseline : séquence de tokens [PAD]
            pad_token_id = self.tokenizer.pad_token_id or 0
            baseline_ids = torch.full_like(input_ids, pad_token_id)

            # Fonction forward pour captum : renvoie le logit de la classe toxique
            def forward_fn(input_ids_batch, attention_mask_batch):
                # Captum a besoin du graphe de gradients à travers le modèle.
                outputs = self.model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                # Classe toxique = index 1
                return logits[:, 1]

            # LayerIntegratedGradients sur la couche d'embeddings
            lig = LayerIntegratedGradients(forward_fn, self.embedding_layer)
            attributions, convergence_delta = lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                n_steps=self.n_steps,
                return_convergence_delta=True,
            )

            # Agréger les attributions par token (somme sur la dimension embedding)
            attr_scores = attributions.sum(dim=-1).squeeze(0)  # shape: (seq_len,)
            attr_scores = attr_scores.detach().cpu().numpy()
            conv_delta = float(convergence_delta.mean().item())

            # Associer les scores aux tokens
            tokens = self.tokenizer.convert_ids_to_tokens(
                input_ids.squeeze(0).tolist()
            )
            token_scores = {}
            for token, score in zip(tokens, attr_scores):
                # Ignorer les tokens spéciaux
                if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                    continue
                # Nettoyer les préfixes de sous-mots (e.g. "##", "▁")
                clean = token.lstrip("▁").replace("##", "")
                if clean:
                    token_scores[clean] = token_scores.get(clean, 0.0) + float(score)

            # Limiter au top num_features
            if len(token_scores) > num_features:
                sorted_tokens = sorted(
                    token_scores.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                token_scores = dict(sorted_tokens[:num_features])

            elapsed = time.perf_counter() - t0
            logger.info(
                "IG (captum) terminé en %.2fs — delta=%.4f, %d tokens",
                elapsed,
                conv_delta,
                len(token_scores),
            )

            return Attribution(
                method_name=self.method_name,
                token_scores=token_scores,
                computation_time=elapsed,
                convergence_delta=conv_delta,
                metadata={
                    "mode": "captum",
                    "effective_method": "integrated_gradients_legacy",
                    "display_name": "IG historique (logit toxique)",
                    "target_class": 1, "target_space": "logit",
                    "n_steps": self.n_steps,
                    "num_tokens_total": len(tokens),
                },
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Erreur IG (captum) après %.2fs : %s", elapsed, exc)
            raise RuntimeError(
                f"Échec du calcul IG (captum) après {elapsed:.2f}s : {exc}"
            ) from exc

    # ── Mode Leave-One-Out (approximation sans gradients) ─

    def _explain_loo(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int,
    ) -> Attribution:
        """
        Approximation leave-one-out de l'importance des mots.

        Pour chaque mot du texte, on le remplace par un mot neutre
        (« le ») et on mesure la variation de la probabilité
        « toxique ». Le score d'un mot est la différence entre la
        prédiction originale et la prédiction perturbée.

        Cette méthode ne nécessite ni gradients ni accès aux embeddings.
        """
        logger.debug("IG (LOO) sur un texte de %d caractères", len(text))
        t0 = time.perf_counter()

        try:
            # Segmenter le texte en mots avec leurs positions
            word_pattern = re.compile(r"\S+")
            matches = list(word_pattern.finditer(text))

            if not matches:
                elapsed = time.perf_counter() - t0
                return Attribution(
                    method_name=self.method_name,
                    token_scores={},
                    computation_time=elapsed,
                    metadata={"mode": "leave_one_out", "n_words": 0},
                )

            words = [m.group() for m in matches]

            # Prédiction de référence sur le texte original
            base_probs = predict_fn([text])  # shape: (1, 2)
            base_toxic_prob = float(base_probs[0, 1])

            # Pour chaque mot, remplacer par le mot neutre et mesurer le changement
            perturbed_texts: List[str] = []
            for i, match in enumerate(matches):
                start, end = match.start(), match.end()
                perturbed = text[:start] + _NEUTRAL_WORD + text[end:]
                perturbed_texts.append(perturbed)

            # Évaluation par lot pour l'efficacité
            perturbed_probs = predict_fn(perturbed_texts)  # shape: (n_words, 2)

            # Score = baisse de la probabilité toxique quand le mot est retiré
            # Un score positif signifie que le mot contribue positivement à la toxicité
            token_scores: dict[str, float] = {}
            for i, word in enumerate(words):
                perturbed_toxic_prob = float(perturbed_probs[i, 1])
                importance = base_toxic_prob - perturbed_toxic_prob
                # Si un mot apparaît plusieurs fois, sommer les contributions
                clean_word = word.strip(".,;:!?\"'()[]{}«»")
                if clean_word:
                    token_scores[clean_word] = (
                        token_scores.get(clean_word, 0.0) + importance
                    )

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
                "IG (LOO) terminé en %.2fs — %d mots analysés, %d retenus",
                elapsed,
                len(words),
                len(token_scores),
            )

            return Attribution(
                method_name=self.method_name,
                token_scores=token_scores,
                computation_time=elapsed,
                base_value=base_toxic_prob,
                metadata={
                    "mode": "leave_one_out",
                    "effective_method": "loo_replacement",
                    "display_name": "LOO (remplacement par le)",
                    "target_class": 1,
                    "scores_by_position": (base_toxic_prob-perturbed_probs[:, 1]).tolist(),
                    "n_words": len(words),
                    "base_toxic_probability": base_toxic_prob,
                    "neutral_word": _NEUTRAL_WORD,
                },
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Erreur IG (LOO) après %.2fs : %s", elapsed, exc)
            raise RuntimeError(
                f"Échec du calcul IG (LOO) après {elapsed:.2f}s : {exc}"
            ) from exc
