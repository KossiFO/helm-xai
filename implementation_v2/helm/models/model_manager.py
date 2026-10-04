"""
Gestionnaire de modèles NLP pour le framework HELM v2.

Trois modèles HuggingFace supportés :
  1. **XLM-RoBERTa Large** (textdetox/xlmr-large-toxicity-classifier)
     — Encodeur fine-tuné toxicité, state-of-the-art multilingue
  2. **Qwen 2.5-0.5B-Instruct** — LLM génératif, classification par prompt
     few-shot (approche innovante, montre limites des petits LLM)
  3. **CamemBERT** (cmarkea/distilcamembert-base-sentiment)
     — Encodeur français, baseline sentiment → toxicité

Chaque modèle expose la même interface `predict_proba(texts) -> ndarray (n, 2)`.
"""

import logging
import re
import time
from typing import List, Optional

import numpy as np

from helm.config import ModelBackend, MODEL_REGISTRY, Prediction

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Prompt few-shot pour Qwen (classification de toxicité)
# ═══════════════════════════════════════════════════════════════════════

QWEN_FEW_SHOT_SYSTEM = """Tu es un détecteur de toxicité pour les réseaux sociaux français.
Tu dois classer chaque texte comme « toxique » ou « non_toxique ».

Un texte toxique contient des insultes, du harcèlement, des menaces, de la haine, du mépris ou de la discrimination.
Un texte non toxique peut exprimer un désaccord, une critique, ou une émotion négative SANS être abusif.

Exemples :
Texte : « Va te faire foutre espèce de con » → toxique
Texte : « Je ne suis pas d'accord avec cette analyse » → non_toxique
Texte : « T'es qu'un déchet, crève » → toxique
Texte : « Belle journée, merci pour le partage ! » → non_toxique
Texte : « Ferme ta gueule sale merde » → toxique
Texte : « Je trouve ce point de vue discutable » → non_toxique
Texte : « Bande de crétins vous comprenez rien » → toxique
Texte : « Pourrais-tu expliquer ce point ? » → non_toxique"""


class ModelManager:
    """
    Gestionnaire unifié de modèles HuggingFace pour la détection de toxicité.

    Les 3 modèles partagent la même interface `predict_proba()` pour
    permettre la comparaison directe des méthodes XAI sur chaque architecture.
    """

    _LABEL_MAP = {0: "non_toxique", 1: "toxique"}

    def __init__(self, model_name: str = "xlmr", backend: str = "transformer") -> None:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Modèle inconnu : « {model_name} ». "
                f"Modèles disponibles : {list(MODEL_REGISTRY.keys())}"
            )

        self.model_name: str = model_name
        self._requested_backend: str = backend
        self.backend: Optional[ModelBackend] = None
        self.model_info: dict = MODEL_REGISTRY[model_name]

        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._is_loaded: bool = False

        # Qwen-specific
        self._token_ids_toxique = None
        self._token_ids_non = None

        logger.info("ModelManager créé — modèle=%s (%s)", model_name, self.model_info["description"])

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # ─── Chargement ───────────────────────────────────────────────────

    def load(self) -> "ModelManager":
        """Charge le modèle HuggingFace. Tous les modèles nécessitent transformers+torch."""
        if self._is_loaded:
            logger.warning("Modèle déjà chargé, rechargement ignoré.")
            return self

        try:
            import transformers  # noqa: F401 — sonde de disponibilité
            import torch  # noqa: F401 — sonde de disponibilité
        except ImportError:
            raise ImportError(
                "Les bibliothèques 'transformers' et 'torch' sont requises. "
                "Installez-les : pip install transformers torch"
            )

        inference_mode = self.model_info["inference_mode"]
        task_model_id = self.model_info.get("task_model_id", self.model_info["model_id"])

        logger.info("Chargement de %s (%s)...", self.model_name, task_model_id)
        t0 = time.time()

        if inference_mode == "toxicity_classifier":
            self._load_xlmr(task_model_id)
        elif inference_mode == "generative":
            self._load_qwen(task_model_id)
        elif inference_mode == "sentiment_proxy":
            self._load_camembert(task_model_id)
        else:
            raise ValueError(f"Mode d'inférence inconnu : {inference_mode}")

        self.backend = ModelBackend.TRANSFORMER
        self._is_loaded = True
        logger.info("Modèle %s chargé en %.1fs", self.model_name, time.time() - t0)
        return self

    # ── Loaders spécifiques ────────────────────────────────────────────

    def _load_xlmr(self, model_id: str) -> None:
        """XLM-RoBERTa Large fine-tuné pour la toxicité (binary: toxic/not_toxic)."""
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline as hf_pipeline,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self.model_info.get("revision"))
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=self.model_info.get("revision"))
        self._pipeline = hf_pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            top_k=None,
            truncation=True,
            max_length=512,
        )

    def _load_qwen(self, model_id: str) -> None:
        """Qwen 2.5-0.5B-Instruct — classification par prompt few-shot + logits."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Compat transformers : l'argument s'appelle `dtype` (versions récentes)
        # ou `torch_dtype` (< 4.56) — on tente l'un puis l'autre.
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, dtype=torch.float32,
            )
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.float32,
            )
        self._model.eval()

        # Pré-calculer les token IDs pour les premiers tokens de "toxique" et "non"
        self._token_ids_toxique = self._tokenizer.encode("toxique", add_special_tokens=False)[0]
        self._token_ids_non = self._tokenizer.encode("non", add_special_tokens=False)[0]
        logger.info(
            "Qwen tokens — toxique=%d, non=%d",
            self._token_ids_toxique, self._token_ids_non,
        )

    def _load_camembert(self, model_id: str) -> None:
        """CamemBERT sentiment (5 étoiles) → proxy toxicité."""
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline as hf_pipeline,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self.model_info.get("revision"))
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=self.model_info.get("revision"))
        self._pipeline = hf_pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            top_k=None,
            truncation=True,
            max_length=512,
        )

    # ─── Inférence ────────────────────────────────────────────────────

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Probabilités de classification.

        Returns
        -------
        np.ndarray shape (n, 2) — [P(non_toxique), P(toxique)]
        """
        self._ensure_loaded()

        inference_mode = self.model_info["inference_mode"]
        if inference_mode == "toxicity_classifier":
            return self._predict_proba_xlmr(texts)
        elif inference_mode == "generative":
            return self._predict_proba_qwen(texts)
        elif inference_mode == "sentiment_proxy":
            return self._predict_proba_camembert(texts)
        else:
            raise ValueError(f"Mode inconnu : {inference_mode}")

    def _predict_proba_xlmr(self, texts: List[str]) -> np.ndarray:
        """
        XLM-RoBERTa toxicity classifier.
        Labels du modèle : 'toxic' et 'not_toxic' (ou 'LABEL_0'/'LABEL_1').
        """
        results = self._pipeline(texts)
        probas = np.zeros((len(texts), 2), dtype=np.float64)

        for i, scores_list in enumerate(results):
            score_map = {item["label"]: item["score"] for item in scores_list}
            # Le modèle textdetox a les labels 'toxic' et 'not_toxic'
            p_toxic = score_map.get("toxic", score_map.get("LABEL_1", 0.5))
            probas[i] = [1.0 - p_toxic, p_toxic]

        return probas

    def _predict_proba_qwen(self, texts: List[str]) -> np.ndarray:
        """
        Classification par prompt few-shot avec Qwen.

        Approche : on envoie le prompt few-shot, on récupère les logits
        du dernier token, et on compare P(toxique) vs P(non).
        """
        import torch

        probas = np.zeros((len(texts), 2), dtype=np.float64)

        for i, text in enumerate(texts):
            # Construire le prompt few-shot via chat template
            messages = [
                {"role": "system", "content": QWEN_FEW_SHOT_SYSTEM},
                {"role": "user", "content": f'Texte : « {text} » →'},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0, -1, :]

            # Extraire les logits pour "toxique" et "non"
            logit_tox = logits[self._token_ids_toxique].item()
            logit_non = logits[self._token_ids_non].item()

            # Softmax sur ces deux logits
            max_l = max(logit_tox, logit_non)
            exp_tox = np.exp(logit_tox - max_l)
            exp_non = np.exp(logit_non - max_l)
            total = exp_tox + exp_non

            probas[i, 0] = exp_non / total  # P(non_toxique)
            probas[i, 1] = exp_tox / total  # P(toxique)

        return probas

    def _predict_proba_camembert(self, texts: List[str]) -> np.ndarray:
        """
        CamemBERT sentiment (5 étoiles) → proxy toxicité.
        Agrégation : étoiles 1-2 → toxique, 3-5 → non_toxique.
        C'est une baseline volontairement imparfaite pour montrer
        l'importance d'un modèle spécialisé.
        """
        results = self._pipeline(texts)
        probas = np.zeros((len(texts), 2), dtype=np.float64)

        for i, scores_list in enumerate(results):
            score_map = {item["label"]: item["score"] for item in scores_list}
            p_toxic = 0.0
            p_non_toxic = 0.0
            for label, score in score_map.items():
                star = self._extract_star_number(label)
                if star is not None and star <= 2:
                    p_toxic += score
                else:
                    p_non_toxic += score
            total = p_toxic + p_non_toxic
            if total > 0:
                probas[i] = [p_non_toxic / total, p_toxic / total]
            else:
                probas[i] = [0.5, 0.5]

        return probas

    @staticmethod
    def _extract_star_number(label: str) -> Optional[int]:
        match = re.search(r"(\d+)", label)
        return int(match.group(1)) if match else None

    # ─── Méthodes utilitaires ─────────────────────────────────────────

    def predict(self, texts: List[str]) -> List[str]:
        probas = self.predict_proba(texts)
        indices = np.argmax(probas, axis=1)
        return [self._LABEL_MAP[idx] for idx in indices]

    def predict_single(self, text: str) -> Prediction:
        probas = self.predict_proba([text])[0]
        label_idx = int(np.argmax(probas))
        return Prediction(
            label=self._LABEL_MAP[label_idx],
            confidence=float(probas[label_idx]),
            probabilities=tuple(probas.tolist()),
            model_name=self.model_name,
        )

    def tokenize(self, text: str) -> List[str]:
        self._ensure_loaded()
        if self._tokenizer is not None:
            encoding = self._tokenizer(text, add_special_tokens=False)
            return self._tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        return re.findall(r'\w+|[^\w\s]', text)

    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            raise RuntimeError(
                "Le modèle n'est pas chargé. Appelez load() d'abord."
            )

    def get_model_info(self) -> dict:
        info = dict(self.model_info)
        info["backend"] = self.backend.value if self.backend else None
        info["is_loaded"] = self._is_loaded
        return info

    def __repr__(self) -> str:
        status = "chargé" if self._is_loaded else "non chargé"
        return f"ModelManager(model={self.model_name!r}, {status})"
