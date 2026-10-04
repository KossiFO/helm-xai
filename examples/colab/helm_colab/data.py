"""Prepare a functional benchmark, never a production model."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .mixed_preprocessing import (
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    UNKNOWN,
    fit_schema,
    normalize,
    transform,
)

BLOCKED = {
    "score_supervise", "score_nonsupervise", "pertinence",
    "statut_dossier", "statut_dossier2", "bc_fraude_gmf", "bc_fraude_maaf",
    "numero_sinistre", "gainsDossier", "nature_decision", "cclu_dos",
}


def prepare(csv_path, seed, cutoff):
    """Use all rows, a chronological holdout and train-only imputation."""
    path = Path(csv_path)
    frame = pd.read_csv(path, low_memory=False)
    target = frame["bc_fraude_gmf"]
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError("The target must contain only non-missing binary labels.")
    dates = pd.to_datetime(frame["date_survenance"], errors="raise")
    if dates.isna().any() or frame["numero_sinistre"].duplicated().any():
        raise ValueError("Invalid dates or duplicated claims require investigation.")
    assert not BLOCKED.intersection(FEATURES)
    raw = normalize(frame)
    if np.isinf(raw[NUMERIC].to_numpy()).any():
        raise ValueError("Infinite input values are not supported.")
    train = dates < pd.Timestamp(cutoff)
    if not train.any() or train.all():
        raise ValueError("The chronological split is empty.")
    imputer = fit_schema(raw.loc[train])
    x_train = transform(raw.loc[train], imputer).reset_index(drop=True)
    x_test = transform(raw.loc[~train], imputer).reset_index(drop=True)
    y_train = target.loc[train].to_numpy(dtype=int)
    y_test = target.loc[~train].to_numpy(dtype=int)
    for labels in [y_train, y_test]:
        assert set(np.unique(labels)) == {0, 1}
    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=10,
        random_state=seed, n_jobs=1,
    )
    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    model = Pipeline([("preprocessing", preprocessing), ("forest", model)])
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    predicted = (scores >= 0.5).astype(int)
    background = x_train.sample(n=min(128, len(x_train)), random_state=seed)
    audit = {
        "source_name": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_bytes": path.stat().st_size,
        "rows": len(frame), "columns": len(frame.columns),
        "positive_labels": int(target.sum()),
        "alternative_positive_labels": int(frame["bc_fraude_maaf"].sum()),
        "date_min": str(dates.min().date()), "date_max": str(dates.max().date()),
        "cutoff_test_start": cutoff, "seed": seed,
        "train_rows": len(x_train), "test_rows": len(x_test),
        "train_positive_labels": int(y_train.sum()),
        "test_positive_labels": int(y_test.sum()),
        "features": FEATURES,
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
        "encoded_feature_count": int(model.named_steps["forest"].n_features_in_),
        "category_schema_source": "training_only",
        "categorical_audit": {
            name: {
                "training_modalities": len(imputer["categories"][name]),
                "background_modalities_observed": int(background[name].nunique()),
                "missing_rows": int(frame[name].isna().sum()),
                "unknown_test_rows": int((x_test[name] == UNKNOWN).sum()),
                "modalities": imputer["categories"][name],
            } for name in CATEGORICAL
        },
        "excluded_all_other_columns": [c for c in frame if c not in FEATURES],
        "missing_values_imputed": {c: int(frame[c].isna().sum()) for c in FEATURES},
        "background_rows": len(background),
        "model": {k: model.named_steps["forest"].get_params()[k] for k in [
            "n_estimators", "max_depth", "min_samples_leaf", "random_state", "n_jobs"
        ]},
        "metrics_sample_holdout": {
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "average_precision": float(average_precision_score(y_test, scores)),
            "positive_rate": float(y_test.mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
            "predicted_positive_count": int(predicted.sum()),
            "precision_at_0_5": (float(precision_score(y_test, predicted))
                                 if predicted.sum() else None),
            "recall_at_0_5": float(recall_score(y_test, predicted, zero_division=0)),
            "confusion_matrix_true_rows_predicted_columns": confusion_matrix(
                y_test, predicted, labels=[0, 1]
            ).tolist(),
        },
    }
    cases = choose_cases(x_test, y_test, scores, frame.loc[~train, FEATURES].reset_index(drop=True))
    for case in cases:
        pos = case["test_position"]
        case["unknown_features"] = [name for name in CATEGORICAL if x_test.iloc[pos][name] == UNKNOWN]
    return model, imputer, background, cases, audit


def choose_cases(features, labels, scores, raw):
    """Select three score positions per predicted class without using labels."""
    chosen = []
    observed_classes = np.unique((scores >= 0.5).astype(int)).tolist()
    quantiles = [("bas", 0.0), ("médian", 0.5), ("haut", 1.0)]
    if len(observed_classes) == 1:
        quantiles = [(f"quantile {q:.0%}", q) for q in [0, 0.25, 0.5, 0.75, 0.9, 1]]
    for cls in observed_classes:
        positions = np.flatnonzero((scores >= 0.5).astype(int) == cls)
        positions = positions[np.argsort(scores[positions], kind="stable")]
        for label, quantile in quantiles:
            if not len(positions):
                continue
            pos = int(positions[round(quantile * (len(positions) - 1))])
            if any(item["test_position"] == pos for item in chosen):
                continue
            chosen.append({
                "case_id": f"Cas {len(chosen) + 1:02d}",
                "test_position": pos,
                "selection": f"Score {label} parmi les prédictions de classe {cls}",
                "target_recorded": int(labels[pos]), "predicted_class": cls,
                "positive_score": float(scores[pos]),
                "instance": features.iloc[pos].to_dict(),
                "imputed_features": raw.columns[raw.iloc[pos].isna()].tolist(),
            })
    return chosen
