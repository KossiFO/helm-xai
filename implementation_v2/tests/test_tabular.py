"""Synthetic software checks; no real-world data or scientific validation claim."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from helm import TabularHELM
from helm.tabular.adapter import MixedCodec


@pytest.fixture
def mixed():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"age": rng.uniform(18, 85, 100), "montant": rng.uniform(50, 5000, 100),
                      "contrat": np.tile(["auto", "habitation", "rare", "autre"], 25)})
    y = ((X.age < 35) & X.contrat.isin(["auto", "rare"])).astype(int)
    prep = ColumnTransformer([("quant", SimpleImputer(), ["age", "montant"]),
                              ("qual", OneHotEncoder(handle_unknown="ignore"), ["contrat"])])
    model = make_pipeline(prep, RandomForestClassifier(n_estimators=12, max_depth=4, random_state=42))
    model.fit(X, y)
    return model, X, y


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_mixed_original_columns_and_target_not_decision(mixed, method):
    model, X, _ = mixed
    before = pickle.dumps(model)
    saved = X.copy(deep=True)
    h = TabularHELM(model, X, target_class=1, background_size=12)
    row = X.loc[model.predict(X) == 0].iloc[[0]]
    e = h.explain(row, method=method, budget=128)
    from helm import __version__
    assert e.quality["helm_version"] == __version__
    assert e.prediction == 0 and e.target_class == 1
    assert e.target_score == pytest.approx(model.predict_proba(row)[0, 1])
    assert set(e.contributions) == set(X.columns)
    if method == "shap":
        assert e.quality["base_value"] == pytest.approx(model.predict_proba(h.background)[:, 1].mean())
        assert sum(e.contributions.values()) + e.quality["base_value"] == pytest.approx(e.target_score)
    else:
        assert "contrat=" in e.conditions["contrat"]
        assert any(value in e.conditions["contrat"] for value in X.contrat.unique())
        assert np.isfinite(e.quality["local_r2"])
    assert pickle.dumps(model) == before
    pd.testing.assert_frame_equal(X, saved)


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_multiclass_string_target_first_column(mixed, method):
    model, X, _ = mixed
    y = np.where(X.age < 35, "A_risque", np.where(X.age > 60, "C", "B"))
    model.fit(X, y)
    h = TabularHELM(model, X, target_class="A_risque", background_size=10)
    e = h.explain(X.iloc[[0]], method=method, budget=128)
    assert h.adapter.target_index == 0
    assert e.target_score == pytest.approx(model.predict_proba(X.iloc[[0]])[0, 0])
    assert e.prediction != e.target_class
    if method == "shap":
        assert sum(e.contributions.values()) + e.quality["base_value"] == pytest.approx(e.target_score)


def test_ranking_filters_alignment_and_empty(mixed):
    model, X, y = mixed
    h = TabularHELM(model, X, target_class=1)
    top = h.rank(X, k=5, labels=y)
    bottom = h.rank(X, k=10, order="lowest")
    assert len(top) == 5 and len(bottom) == 10
    assert top.target_score.is_monotonic_decreasing and bottom.target_score.is_monotonic_increasing
    assert h.rank(X, group="predicted_target").prediction.eq(1).all()
    assert h.rank(X, group="observed_target", labels=y).observed_label.eq(1).all()
    assert h.rank(X, group="observed_other", labels=y).observed_label.eq(0).all()
    empty = h.explain_top(X.loc[y == 0], group="observed_target", labels=y[y == 0])
    assert empty.ranking.empty and empty.summary().empty and not empty.errors
    assert "0 sélectionnés" in empty.to_html()
    assert h.rank(X.iloc[:0]).empty
    with pytest.raises(ValueError, match="index"):
        h.rank(X, labels=y.iloc[::-1])
    with pytest.raises(ValueError, match="nécessite"):
        h.rank(X, group="observed_target")
    with pytest.raises(ValueError, match="uniques"):
        h.rank(pd.concat([X.iloc[:2], X.iloc[:2]]))


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_rare_and_unseen_categories(mixed, method):
    model, X, _ = mixed
    h = TabularHELM(model, X, target_class=1, background_size=2)
    row = X.iloc[[0]].copy()
    row["contrat"] = "jamais_vu"
    e = h.explain(row, method=method, budget=128)
    assert e.values["contrat"] == "jamais_vu"
    assert e.target_score == pytest.approx(model.predict_proba(row)[0, 1])


def test_missing_and_category_dtype_roundtrip(mixed):
    model, X, _ = mixed
    X = X.copy()
    X["contrat"] = X.contrat.astype("category")
    h = TabularHELM(model, X, target_class=1)
    codec = MixedCodec(h.adapter, X.iloc[[0]])
    rebuilt = codec.decode(codec.encode(X))
    assert rebuilt.contrat.dtype == X.contrat.dtype
    np.testing.assert_allclose(model.predict_proba(rebuilt), model.predict_proba(X))
    row = X.iloc[[0]].copy()
    row["age"] = np.nan
    assert h.explain(row, method="shap").quality["additivity_residual"] == pytest.approx(0)
    with pytest.raises(ValueError, match="valeurs manquantes"):
        h.explain(row, method="lime")


def test_partial_cohort_failure_and_html_escaping(mixed):
    model, X, _ = mixed
    h = TabularHELM(model, X, target_class=1)
    rows = X.iloc[:2].copy()
    rows.index = ["<script>alert(1)</script>", "second"]
    rows.loc["second", "age"] = np.nan
    report = h.explain_top(rows, method="lime", k=2)
    assert len(report.explanations) == 1 and len(report.errors) == 1
    assert report.summary().n.eq(1).all()
    html = report.to_html()
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "1 explications réussies et 1 échecs" in html


def test_shap_restores_random_state(mixed):
    model, X, _ = mixed
    h = TabularHELM(model, X, target_class=1, background_size=5)
    state = np.random.get_state()
    h.explain(X.iloc[[0]])
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.parametrize("kwargs", [{"target_class": 9}, {"target_class": 1, "categorical_features": ["absent"]},
                                    {"target_class": 1, "background_size": 0}])
def test_validation(mixed, kwargs):
    model, X, _ = mixed
    with pytest.raises(ValueError):
        TabularHELM(model, X, **kwargs)


def test_background_preserves_distinct_missing_categories():
    X = pd.DataFrame({"cat": pd.Series(["a", None, np.nan, "b"] * 10, dtype=object)})
    y = [0, 1, 0, 0] * 10
    model = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                          RandomForestClassifier(n_estimators=20, random_state=42)).fit(X, y)
    h = TabularHELM(model, X, target_class=1, background_size=40)
    e = h.explain(X.iloc[[0]], method="shap")
    assert e.quality["base_value"] == pytest.approx(model.predict_proba(h.background)[:, 1].mean())
    assert len(e.quality["background_sha256"]) == 64
    assert e.quality["backend_version"]
    assert e.quality["background_sha256"] in e.to_html()
    assert "backend_version" in e.to_html()
    changed = X.copy()
    changed["cat"] = changed.cat.map(lambda v: np.nan if v is None else v)
    altered = TabularHELM(model, changed, target_class=1, background_size=40)
    assert altered.background_fingerprint != h.background_fingerprint
    assert e.quality["base_value"] == pytest.approx(0.25)
    assert e.contributions["cat"] == pytest.approx(-0.25)


def test_native_profile_api_selects_and_keeps_predictions(mixed):
    model, X, y = mixed
    engine = TabularHELM(model, X, target_class=1, background_size=10)
    business = engine.explain_profile(X, profile='metier', k=2, labels=y, budgets={'shap':128})
    expert = engine.explain_profile(X, profile='technique', k=2, labels=y, budgets={'shap':128,'lime':128})
    assert set(business.reports) == {'shap'}
    assert set(expert.reports) == {'shap','lime'}
    for report in expert.reports.values():
        assert len(report.explanations) == 2 and not report.errors
        pd.testing.assert_frame_equal(report.ranking, business.reports['shap'].ranking)
    assert [e.contributions for e in business.reports['shap'].explanations] == [e.contributions for e in expert.reports['shap'].explanations]
