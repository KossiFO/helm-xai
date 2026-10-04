"""Fabrique de l’interface locale installable, sans écriture à l’import."""
import os
from pathlib import Path

# Avant tout import de transformers : modèles locaux uniquement.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from helm import __version__
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from .models import create_model, model_options
from .schemas import ExplainRequest, FeedbackRequest
from .service import ExplanationService, PROFILES

BASE = Path(__file__).resolve().parent


def default_data_path():
    from platformdirs import user_data_path
    directory = Path(os.environ["HELM_DATA_DIR"]).expanduser() if os.environ.get("HELM_DATA_DIR") else user_data_path("helm-xai", appauthor=False)
    return directory / "helm.sqlite3"


def create_app(data_path=None, model_factory=create_model, *, assets_path=None, source_label=None):
    app = FastAPI(title="HELM · Interface Codex", version=__version__, docs_url="/api/docs")
    service = ExplanationService(data_path if data_path is not None else default_data_path(), model_factory)
    app.state.service = service

    @app.get("/api/health")
    def health():
        result = {"status": "ok", "edition": "codex", "package_version": __version__}
        if source_label is not None:
            result["source"] = str(source_label)
        return result

    @app.get("/api/config")
    def config():
        return {"profiles": PROFILES, "models": model_options(), "edition": "Codex"}

    @app.post("/api/explanations")
    def explain(request: ExplainRequest):
        try:
            return service.explain(request)
        except (OSError, ImportError) as exc:
            raise HTTPException(503, "Le modèle local n'est pas disponible. Choisissez Démonstration locale.") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/history")
    def history():
        return service.store.history()

    @app.get("/api/explanations/{explanation_id}")
    def explanation(explanation_id: str):
        try:
            return service.store.get(explanation_id)
        except KeyError as exc:
            raise HTTPException(404, "Analyse introuvable.") from exc

    @app.get("/api/explanations/{explanation_id}/export")
    def export(explanation_id: str):
        result = explanation(explanation_id)
        return JSONResponse(result, headers={
            "Content-Disposition": f'attachment; filename="helm-{result["id"]}.json"',
        })

    @app.post("/api/explanations/{explanation_id}/feedback")
    def feedback(explanation_id: str, request: FeedbackRequest):
        try:
            return service.feedback(explanation_id, request)
        except KeyError as exc:
            raise HTTPException(404, "Analyse introuvable.") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    dist = Path(assets_path) if assets_path is not None else BASE / "static"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="interface")
    else:
        @app.get("/")
        def missing_interface():
            return JSONResponse({"detail": "Interface absente de cette installation. Installez la wheel HELM distribuée avec ses fichiers d’interface."}, status_code=503)
    return app
