"""
Wrapper Logit Lens pour le framework HELM v2.

Spécifique aux modèles LLM/décodeurs (Qwen, GPT, LLaMA...).
Inspecte les représentations internes du modèle **couche par couche**
en projetant les états cachés intermédiaires dans l'espace du vocabulaire.

Cela permet de voir comment la « compréhension » du modèle évolue
à travers les couches du Transformer.

Références :
    Nostalgebraist (2020).
    "Interpreting GPT: The Logit Lens." Blog post.

    Belrose, N., et al. (2023).
    "Eliciting Latent Predictions from Transformers with the Tuned Lens."
    NeurIPS.
"""

import re
import time
import logging
from typing import Callable, Dict, List, Optional, Tuple


from helm.config import Attribution
from .base import BaseExplainer

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class LogitLensExplainer(BaseExplainer):
    """
    Expliqueur Logit Lens pour LLMs — inspection couche par couche.

    Projette les états cachés intermédiaires de chaque couche du Transformer
    via la tête LM (lm_head) pour voir comment la prédiction évolue.

    Si l'accès aux couches internes n'est pas disponible, utilise un mode
    dégradé basé sur l'attention (si disponible) ou la perturbation.

    Parameters
    ----------
    model_manager : object, optional
        ModelManager du LLM avec accès au modèle PyTorch interne.
    layers_to_inspect : list of int, optional
        Indices des couches à inspecter. Si None, échantillonne uniformément.
    num_layers_sample : int
        Nombre de couches à échantillonner si layers_to_inspect est None.
    """

    def __init__(
        self,
        model_manager=None,
        layers_to_inspect: Optional[List[int]] = None,
        num_layers_sample: int = 8,
    ):
        self.model_manager = model_manager
        self.layers_to_inspect = layers_to_inspect
        self.num_layers_sample = num_layers_sample
        logger.info(
            "LogitLensExplainer initialisé (num_layers_sample=%d)",
            self.num_layers_sample,
        )

    @property
    def method_name(self) -> str:
        return "logit_lens"

    def explain(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
    ) -> Attribution:
        logger.debug("LogitLens explain() sur un texte de %d caractères", len(text))
        t0 = time.perf_counter()

        try:
            can_inspect = (
                _TORCH_AVAILABLE
                and self.model_manager is not None
                and hasattr(self.model_manager, "_model")
                and self.model_manager._model is not None
            )

            if can_inspect:
                result = self._explain_logit_lens(text, predict_fn, num_features)
            else:
                result = self._explain_attention_fallback(text, predict_fn, num_features)

            elapsed = time.perf_counter() - t0
            result.computation_time = elapsed
            logger.info(
                "LogitLens terminé en %.2fs — %d tokens attribués",
                elapsed, len(result.token_scores),
            )
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Erreur LogitLens après %.2fs : %s", elapsed, exc)
            raise RuntimeError(
                f"Échec du calcul LogitLens après {elapsed:.2f}s : {exc}"
            ) from exc

    def estimate_time(self, text_length: int) -> float:
        if self.model_manager is not None and _TORCH_AVAILABLE:
            return self.num_layers_sample * 0.5 + 2.0
        words_approx = max(1, text_length / 5)
        return (words_approx + 1) * 0.05

    # ── Mode Logit Lens (accès aux couches internes) ─────

    def _explain_logit_lens(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int,
    ) -> Attribution:
        """
        Inspection des couches internes via projection lm_head.

        Pour chaque couche intermédiaire :
        1. Récupère les hidden states
        2. Les projette via lm_head pour obtenir les logits
        3. Regarde les tokens qui émergent comme importants

        L'attribution finale mesure à quelle couche chaque token
        du texte original commence à influencer la prédiction.
        """
        model = self.model_manager._model
        tokenizer = self.model_manager._tokenizer

        # Tokenisation
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)

        # Forward pass avec output_hidden_states
        with torch.no_grad():
            outputs = model(
                **{k: v.to(device) for k, v in inputs.items()},
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states  # tuple de (1, seq_len, hidden_dim)
        num_layers = len(hidden_states)

        # Sélectionner les couches à inspecter
        if self.layers_to_inspect:
            layer_indices = [i for i in self.layers_to_inspect if i < num_layers]
        else:
            step = max(1, num_layers // self.num_layers_sample)
            layer_indices = list(range(0, num_layers, step))
            if (num_layers - 1) not in layer_indices:
                layer_indices.append(num_layers - 1)

        # Récupérer la tête LM pour projection
        lm_head = None
        if hasattr(model, "lm_head"):
            lm_head = model.lm_head
        elif hasattr(model, "cls"):
            lm_head = model.cls

        # Tokens du texte
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

        # Analyser chaque couche
        layer_predictions: Dict[int, List[Tuple[str, float]]] = {}

        if lm_head is not None:
            for layer_idx in layer_indices:
                hidden = hidden_states[layer_idx]  # (1, seq_len, hidden_dim)

                # Projeter via lm_head
                with torch.no_grad():
                    logits = lm_head(hidden)  # (1, seq_len, vocab_size)

                # Prendre le dernier token (prédiction next-token)
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)

                # Top tokens prédits à cette couche
                top_k = min(20, probs.size(0))
                top_probs, top_ids = torch.topk(probs, top_k)
                top_tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())

                layer_predictions[layer_idx] = [
                    (t, float(p)) for t, p in zip(top_tokens, top_probs)
                ]

        # Calculer les attributions basées sur la variation entre couches
        # Pour chaque token du texte, mesurer quand il devient "important"
        base_probs = predict_fn([text])
        base_toxic = float(base_probs[0, 1])

        # Diagnostic par sous-token, sans orientation toxique et sans faux alignement mot/sous-token.
        subtoken_diagnostics = []
        for i, token in enumerate(tokens):
            norms = [float(torch.norm(hidden_states[layer][0, i, :]).item()) for layer in layer_indices]
            if len(norms) >= 2:
                subtoken_diagnostics.append({"position": i, "token": token,
                                            "norm_growth": (norms[-1]-norms[0])/(abs(norms[0])+1e-8)})

        return Attribution(
            method_name=self.method_name,
            token_scores={},
            computation_time=0.0,
            base_value=base_toxic,
            metadata={
                "mode": "logit_lens",
                "subtoken_diagnostics": subtoken_diagnostics,
                "effective_method": "hidden_state_norm_growth",
                "display_name": "Variation de norme des états cachés",
                "score_kind": "norm_growth_proxy",
                "num_layers_total": num_layers,
                "layers_inspected": layer_indices,
                "layer_predictions": {
                    k: v[:5] for k, v in layer_predictions.items()
                },
            },
        )

    # ── Mode dégradé (perturbation) ──────────────────────

    def _explain_attention_fallback(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int,
    ) -> Attribution:
        """
        Mode dégradé quand l'accès aux couches internes n'est pas possible.
        Utilise une perturbation simple avec pondération positionnelle.
        """
        words = re.findall(r"\S+", text)
        if not words:
            return Attribution(
                method_name=self.method_name,
                token_scores={},
                computation_time=0.0,
                metadata={"mode": "perturbation_fallback", "effective_method": "loo_deletion_proxy",
                      "display_name": "Perturbations par suppression", "target_class": 1},
            )

        base_probs = predict_fn([text])
        base_toxic = float(base_probs[0, 1])

        perturbed_texts = []
        for i in range(len(words)):
            remaining = words[:i] + words[i + 1:]
            perturbed_texts.append(" ".join(remaining) if remaining else "le")

        perturbed_probs = predict_fn(perturbed_texts)

        token_scores: Dict[str, float] = {}
        for i, w in enumerate(words):
            clean = w.strip(".,;:!?\"'()[]{}«»")
            if clean:
                importance = base_toxic - float(perturbed_probs[i, 1])
                token_scores[clean] = token_scores.get(clean, 0.0) + importance

        if len(token_scores) > num_features:
            sorted_tokens = sorted(
                token_scores.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            token_scores = dict(sorted_tokens[:num_features])

        return Attribution(
            method_name=self.method_name,
            token_scores=token_scores,
            computation_time=0.0,
            base_value=base_toxic,
            metadata={"mode": "perturbation_fallback", "effective_method": "loo_deletion_proxy",
                      "display_name": "Perturbations par suppression", "target_class": 1},
        )
