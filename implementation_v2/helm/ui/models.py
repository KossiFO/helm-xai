from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from helm.config import MODEL_REGISTRY
from helm.models import ModelManager
import os
import importlib.util
try:
    from huggingface_hub.constants import HF_HUB_CACHE
except ImportError:
    HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"))


class DemoModel:
    """Classifieur pédagogique entraîné localement sur 12 phrases fabriquées."""
    model_name = "demo"

    def load(self):
        self.estimator = make_pipeline(
            TfidfVectorizer(), LogisticRegression(C=3, random_state=42),
        )
        self.estimator.fit([
            "merci pour cette réponse utile",
            "belle journée merci pour le partage",
            "je respecte ton avis sur cet article",
            "cette discussion est intéressante",
            "bonjour et bienvenue dans cette discussion",
            "je ne partage pas cet avis",
            "tu es un idiot stupide",
            "quel crétin ce commentaire est débile",
            "ferme ta gueule idiot",
            "ce stupide imbécile est un crétin",
            "vous êtes des idiots et des imbéciles",
            "ton commentaire est débile et stupide",
        ], [0] * 6 + [1] * 6)
        return self

    def predict_proba(self, texts):
        return self.estimator.predict_proba(texts)


def model_options():
    options = [{"id": "demo", "label": "Démonstration locale", "available": True,
                "description": "TF-IDF + régression logistique · 12 phrases fabriquées"}]
    for key, label in [("xlmr", "XLM-R · toxicité"), ("qwen", "Qwen 2.5 · génération"),
                       ("camembert", "CamemBERT · proxy de sentiment")]:
        model_id = MODEL_REGISTRY[key]["task_model_id"]
        snapshots = Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--")) / "snapshots"
        text_installed = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))
        cached = text_installed and snapshots.exists() and any(snapshots.iterdir())
        options.append({"id": key, "label": label, "available": cached,
                        "description": "Cache local détecté" if cached else "Installer l’extra [text] et télécharger le modèle" if not text_installed else "Absent du cache local"})
    return options


def create_model(name):
    if name == "demo":
        return DemoModel().load()
    return ModelManager(model_name=name).load()
