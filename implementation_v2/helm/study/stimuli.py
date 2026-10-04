"""Calcul direct d'une attribution choisie par le protocole, sans UCB1 ni fusion.

Les exports complets sont destinés aux chercheurs. Le rendu participant exclut
la méthode, le profil, les métadonnées et la référence humaine. La référence du
modèle est déclarée par l'appelant : son identité n'est pas vérifiée à distance.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from helm import __version__
from helm.evaluation.protocol import probabilities


@dataclass(frozen=True)
class FixedStudyConfig:
    method: str
    model_reference: str
    num_features: int = 8
    shap_max_evals: int = 100

    def __post_init__(self):
        if self.method not in {"shap", "leave_one_out"}:
            raise ValueError("L'étude figée accepte shap ou leave_one_out.")
        if not isinstance(self.model_reference, str) or not self.model_reference.strip():
            raise ValueError("Référence du modèle requise (version ou empreinte vérifiée par l'appelant).")
        for value in (self.num_features, self.shap_max_evals):
            if type(value) is not int or value < 1:
                raise ValueError("Budgets entiers positifs requis.")


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _verify(record):
    payload = {k: v for k, v in record.items() if k != "sha256"}
    if _digest(payload) != record.get("sha256"):
        raise ValueError("Empreinte du stimulus incohérente.")


def prepare_stimulus(text, predict_fn, config: FixedStudyConfig, *, case_id, reference_label=None):
    """Retourne un stimulus de recherche, succès ou échec explicite, sans repli.

    Scores par position pour conserver les mots répétés. Les scores sont orientés
    vers la classe initialement prédite, puis normalisés par le maximum absolu
    dans CE stimulus pour le rendu. Les valeurs brutes restent dans l'audit.
    Cette normalisation visuelle ne rend pas les unités des méthodes équivalentes.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Texte non vide requis.")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("Identifiant de cas requis.")
    if reference_label is not None and (type(reference_label) is not int or reference_label not in (0, 1)):
        raise ValueError("Référence humaine attendue : 0, 1 ou None.")
    record = {
        "schema": "helm_fixed_stimulus_v1", "package_version": __version__,
        "case_id": case_id, "text": text, "config": asdict(config),
        "reference_label": reference_label, "learning_enabled": False,
        "status": "failed", "prediction": None, "display": None, "attribution": None,
        "error": None,
    }
    record["config_sha256"] = _digest(record["config"])
    calls, evaluated_texts = 0, 0

    def counted(texts):
        nonlocal calls, evaluated_texts
        calls += 1
        evaluated_texts += len(texts)
        return probabilities(predict_fn, texts)

    start = time.perf_counter()
    try:
        p = probabilities(counted, [text])[0]
        target = int(np.argmax(p))
        record["prediction"] = {"class": target, "probabilities": p.tolist()}
        if config.method == "shap":
            from helm.explainers.shap_wrapper import SHAPExplainer
            explainer = SHAPExplainer(max_evals=config.shap_max_evals, word_positions=True)
        else:
            from helm.explainers.loo_wrapper import LOOExplainer
            explainer = LOOExplainer()
        attr = explainer.explain(text, counted, num_features=config.num_features)
        effective = attr.metadata.get("effective_method", attr.method_name)
        if effective != config.method or attr.metadata.get("valid") is False:
            raise ValueError("Méthode exécutée différente ou attribution non admissible.")
        if attr.metadata.get("target_class") != target:
            raise ValueError("Classe cible incohérente avec la prédiction initiale.")
        words = text.split()
        scores = np.asarray(attr.metadata.get("scores_by_position"), dtype=float)
        if scores.shape != (len(words),) or not np.isfinite(scores).all():
            raise ValueError("Attributions finies par position requises.")
        selected = set(sorted(range(len(words)), key=lambda i: -abs(scores[i]))[:config.num_features])
        scale = float(np.max(np.abs(scores)))
        record["display"] = {
            "normalization": "max_abs_within_stimulus", "target_class": target,
            "legend": "Positif : contribution vers la classe affichée ; négatif : contribution opposée.",
            "words": [{"position": i, "text": word, "selected": i in selected,
                       "score": float(scores[i] / scale) if scale else 0.0}
                      for i, word in enumerate(words)],
        }
        record["attribution"] = asdict(attr)
        record["status"] = "ok"
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["cost"] = {"seconds": time.perf_counter()-start,
                      "predict_calls": calls, "evaluated_texts": evaluated_texts}
    # JSON strict : aucun NaN silencieux dans les fichiers de protocole.
    record = json.loads(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                  default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist()))
    record["sha256"] = _digest(record)
    return record


def participant_payload(record):
    """Vue aveugle, à associer à un identifiant opaque dans l'outil de passation.

    Pas de référence humaine, nom de méthode, profil, coût ni identifiant de cas.
    L'appelant ne doit pas servir le JSON de recherche aux participants.
    """
    _verify(record)
    if record["status"] != "ok":
        raise ValueError("Un stimulus échoué ne peut pas être présenté comme une explication.")
    return json.loads(json.dumps({"text": record["text"], "prediction": record["prediction"],
                                 "display": record["display"]}, ensure_ascii=False))


def export_stimulus(record, path):
    """Écrit un nouveau fichier ; refuse l'écrasement et les traces altérées."""
    _verify(record)
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
