import math
from dataclasses import asdict
from enum import Enum
from threading import Lock
import numpy as np
from helm import HELMPipeline, __version__
from helm.explainers import AnchorsExplainer
from .models import create_model
from .store import Store, now


PROFILES = [
    {"id": "utilisateur_final", "label": "Utilisateur", "description": "Comprendre simplement", "view": "natural_language"},
    {"id": "moderateur", "label": "Modérateur", "description": "Examiner un contenu", "view": "moderator_dashboard"},
    {"id": "expert_technique", "label": "Expert technique", "description": "Comparer les méthodes", "view": "expert_analysis"},
    {"id": "regulateur", "label": "Régulateur", "description": "Retracer la décision", "view": "audit_report"},
]


def portable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ExplanationService:
    def __init__(self, path, model_factory=create_model):
        self.store = Store(path)
        self.model_factory = model_factory
        self.models = {}
        self.lock = Lock()

    def explain(self, request):
        # Les prédicteurs et LIME ne sont pas partagés entre calculs concurrents.
        with self.lock:
            if request.model not in self.models:
                self.models[request.model] = self.model_factory(request.model)
            model = self.models[request.model]
            pipeline = HELMPipeline(
                model_name=request.model, model=model,
                preference_learner=self.store.learner(request.model),
                explainers={"anchors": AnchorsExplainer(use_native=False)},
            )
            ctx = pipeline.explain(request.text, request.profile, evaluate=True,
                                   num_features=8, evaluation_k=3)
            if not ctx.attributions:
                raise RuntimeError("Aucune méthode n'a pu expliquer ce texte. Essayez un autre texte.")
            result = portable({
                "id": ctx.explanation_id, "created_at": now(),
                "text": ctx.text, "profile": ctx.effective_profile.value,
                "model": request.model, "package_version": __version__,
                "prediction": asdict(ctx.prediction),
                "presentation": asdict(ctx.formatted_output),
                "selection": [asdict(m) for m in ctx.selected_methods],
                "attributions": {k: asdict(v) for k, v in ctx.attributions.items()},
                "evaluation": {k: asdict(v) for k, v in ctx.evaluation.items()},
                "failed_attributions": ctx.failed_attributions,
                "parameters": {"seed": 42, "num_features": 8, "k": 3,
                               "metric_target": "classe cible de chaque attribution", "evaluation_protocol": "signed_cs", "anchors_mode": "beam_search"},
                "duration": ctx.total_time, "logs": ctx.logs, "feedback": {},
                "warnings": [line for line in ctx.logs if "Erreur XAI" in line or "échouée" in line],
            })
            self.store.save(result)
            return result

    def feedback(self, explanation_id, request):
        with self.lock:
            result = self.store.rate(explanation_id, request.method, request.rating)
            stats = self.store.learner(result["model"], package_version=result["package_version"]).get_statistics()
            return {"saved": True, "rating": request.rating, "method": request.method,
                    "profile": result["profile"], "statistics": stats}
