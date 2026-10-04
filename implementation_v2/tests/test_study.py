import json
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from helm.study import (FixedStudyConfig, prepare_stimulus, participant_payload,
                        export_stimulus, balanced_control_accuracy, confidence_brier)


def toy_predict(texts):
    p = np.array([0.8 if "idiot" in text.split() else 0.2 for text in texts])
    return np.column_stack([1-p, p])


@pytest.mark.parametrize("method", ["leave_one_out", "shap"])
def test_real_explainers_isolated_and_repeated_words_preserved(method, tmp_path):
    config = FixedStudyConfig(method, "synthetic-rule-v1")
    with pytest.raises(FrozenInstanceError):
        config.method = "lime"
    record = prepare_stimulus("idiot merci merci", toy_predict, config,
                              case_id="synthetic-1", reference_label=1)
    assert record["status"] == "ok", record["error"]
    assert record["attribution"]["method_name"] == method
    assert record["learning_enabled"] is False
    assert [w["text"] for w in record["display"]["words"]] == ["idiot", "merci", "merci"]
    assert record["display"]["words"][0]["score"] > 0
    assert record["cost"]["predict_calls"] > 1
    public = participant_payload(record)
    assert set(public) == {"text", "prediction", "display"}
    assert "reference_label" not in json.dumps(public)
    public["display"]["words"][0]["score"] = 999
    assert record["display"]["words"][0]["score"] != 999
    path = tmp_path / "case.json"
    export_stimulus(record, path)
    assert json.loads(path.read_text()) == record
    with pytest.raises(FileExistsError):
        export_stimulus(record, path)
    record["text"] = "altéré"
    with pytest.raises(ValueError, match="Empreinte"):
        export_stimulus(record, tmp_path / "tampered.json")


def test_predictor_failure_is_visible_without_fallback():
    def broken(texts):
        raise RuntimeError("modèle indisponible")
    record = prepare_stimulus("bonjour", broken, FixedStudyConfig("shap", "missing"), case_id="x")
    assert record["status"] == "failed"
    assert record["attribution"] is None
    assert record["error"]["message"] == "modèle indisponible"
    with pytest.raises(ValueError):
        participant_payload(record)


@pytest.mark.parametrize("field", ["text", "prediction", "display"])
def test_modified_stimulus_never_reaches_participant(field):
    record = prepare_stimulus("idiot merci", toy_predict,
                              FixedStudyConfig("leave_one_out", "toy"), case_id="x")
    if field == "text":
        record["text"] = "autre texte"
    elif field == "prediction":
        record["prediction"]["class"] = 0
    else:
        record["display"]["words"][0]["score"] = -99
    with pytest.raises(ValueError, match="Empreinte"):
        participant_payload(record)


def test_shap_rejects_invalid_probabilities_on_perturbations():
    text = "idiot merci"
    def malformed(texts):
        return np.array([[.2, .8] if value == text else [.8, .8] for value in texts])
    record = prepare_stimulus(text, malformed, FixedStudyConfig("shap", "toy"), case_id="x")
    assert record["status"] == "failed"
    assert record["attribution"] is None
    assert "simplexe" in record["error"]["message"]


def test_unbalanced_cases_do_not_reward_rejecting_everything():
    truth = [True]*9+[False]
    assert balanced_control_accuracy(truth, [False]*10) == 0.5
    assert balanced_control_accuracy(truth, truth) == 1
    with pytest.raises(ValueError):
        balanced_control_accuracy([True], [True])
    with pytest.raises(ValueError):
        balanced_control_accuracy([True, False], [1, 0])


def test_confidence_is_distinct_from_correctness():
    assert confidence_brier([True, False], [1., 0.]) == 0
    assert confidence_brier([True, False], [.5, .5]) == .25
    with pytest.raises(ValueError):
        confidence_brier([True], [float("nan")])
