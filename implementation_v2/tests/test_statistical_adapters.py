"""Guard probability semantics, schema and fitted-model eligibility."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from helm.tabular import BinaryProbabilityAdapter


def test_class_order_threshold_and_schema():
    X = pd.DataFrame({'a': [0., 1., 2.], 'b': [3., 4., 5.]})
    prep = StandardScaler().fit(X)
    model = SimpleNamespace(prob=lambda Z: np.array([.2, .6, .9]))
    adapter = BinaryProbabilityAdapter(model, probability_method='prob', transformer=prep,
                                       classes=('non', 'oui'), threshold=.7)
    assert adapter.predict(X).tolist() == ['non', 'non', 'oui']
    np.testing.assert_allclose(adapter.predict_proba(X)[:, 0], [.8, .4, .1])
    with pytest.raises(ValueError, match='Colonnes'):
        adapter.predict(X[['b', 'a']])


@pytest.mark.parametrize('probabilities', [[1.01, .1], [-.01, .1], [np.nan, .1], [np.inf, .1],
                                         [[.1, .9], [.2, .8]], [.5]])
def test_invalid_probability_not_silently_repaired(probabilities):
    model = SimpleNamespace(prob=lambda X: probabilities)
    with pytest.raises(ValueError, match='probabilité'):
        BinaryProbabilityAdapter(model, probability_method='prob').predict_proba(np.ones((2, 1)))


@pytest.mark.parametrize('labels', [(0, 0), (0,), (0, 1, 2), (None, 'yes')])
def test_invalid_classes(labels):
    with pytest.raises(ValueError, match='étiquettes'):
        BinaryProbabilityAdapter(SimpleNamespace(prob=lambda X: X), probability_method='prob', classes=labels)


def test_glm_rejects_regression_formula_and_offset():
    sm = pytest.importorskip('statsmodels.api')
    X = pd.DataFrame({'x': np.linspace(-1, 1, 30)})
    y = np.tile([0, 1, 0], 10)
    with pytest.raises(TypeError, match='Binomial'):
        BinaryProbabilityAdapter.from_glm(sm.GLM(y, sm.add_constant(X)).fit())
    result = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial(), offset=np.ones(30)).fit()
    with pytest.raises(ValueError, match='offset'):
        BinaryProbabilityAdapter.from_glm(result)
    formula = sm.GLM.from_formula('y ~ x', X.assign(y=y), family=sm.families.Binomial()).fit()
    with pytest.raises(ValueError, match='formule'):
        BinaryProbabilityAdapter.from_glm(formula)


def test_gam_rejects_unfitted_and_wrong_family():
    gam = pytest.importorskip('pygam')
    for model in (gam.LogisticGAM(), gam.LinearGAM()):
        with pytest.raises(TypeError, match='LogisticGAM'):
            BinaryProbabilityAdapter.from_gam(model)


@pytest.mark.parametrize('intercept', [True, False])
def test_glm_preserves_probabilities_for_design_and_event_class(intercept):
    sm = pytest.importorskip('statsmodels.api')
    X = pd.DataFrame({'x': np.linspace(-1, 1, 30)})
    design = sm.add_constant(X, has_constant='add') if intercept else X
    result = sm.GLM(np.tile([0, 1, 0], 10), design, family=sm.families.Binomial()).fit()
    adapter = BinaryProbabilityAdapter.from_glm(result, add_intercept=intercept, classes=('other', 'event'))
    np.testing.assert_allclose(adapter.predict_proba(X)[:, 1], result.predict(design), atol=1e-12)
    assert set(adapter.predict(X)) <= {'other', 'event'}


def test_mixed_label_types_are_preserved():
    model = SimpleNamespace(prob=lambda X: np.array([.2, .9]))
    adapter = BinaryProbabilityAdapter(model, probability_method='prob', classes=(0, 'event'))
    assert adapter.predict(np.ones((2, 1))).tolist() == [0, 'event']
    assert isinstance(adapter.classes_[0], int)
