"""Reproducible mixed synthetic data; no business dataset is required."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from helm.tabular import BinaryProbabilityAdapter


def sample():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({'mesure_a': rng.normal(size=320), 'mesure_b': rng.normal(size=320),
                      'categorie': rng.choice(['A', 'B'], 320)})
    logit = 1.2 * X.mesure_a - .8 * X.mesure_b + .7 * (X.categorie == 'B')
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))), index=X.index)
    return train_test_split(X, y, test_size=.25, random_state=42, stratify=y)


def fit_classifier(name, X_train, y_train):
    """Training belongs to this example, not to HELM's explanation engine.

    Returns (classifier accepting original columns, independent probability
    callable for verifying that adaptation preserves the trained prediction).
    """
    prep = ColumnTransformer([
        ('numeric', StandardScaler(), ['mesure_a', 'mesure_b']),
        ('category', OneHotEncoder(drop='first', handle_unknown='error', sparse_output=False), ['categorie']),
    ])
    if name in ('xgboost', 'lightgbm'):
        if name == 'xgboost':
            from xgboost import XGBClassifier
            estimator = XGBClassifier(n_estimators=25, max_depth=3, n_jobs=1, random_state=42)
        else:
            from lightgbm import LGBMClassifier
            estimator = LGBMClassifier(n_estimators=25, max_depth=3, n_jobs=1, random_state=42, verbosity=-1)
        pipeline = make_pipeline(prep, estimator).fit(X_train, y_train)
        return pipeline, lambda X: estimator.predict_proba(prep.transform(X))[:, 1]
    Z = prep.fit_transform(X_train)
    if name == 'glm':
        import statsmodels.api as sm
        result = sm.GLM(y_train.to_numpy(), sm.add_constant(Z, has_constant='add'),
                        family=sm.families.Binomial()).fit()
        classifier = BinaryProbabilityAdapter.from_glm(result, transformer=prep)
        return classifier, lambda X: result.predict(sm.add_constant(prep.transform(X), has_constant='add'))
    if name == 'gam':
        from pygam import LogisticGAM, s, f
        model = LogisticGAM(s(0, n_splines=6) + s(1, n_splines=6) + f(2), max_iter=100).fit(Z, y_train)
        classifier = BinaryProbabilityAdapter.from_gam(model, transformer=prep)
        return classifier, lambda X: model.predict_proba(prep.transform(X))
    raise ValueError('Modèle attendu : xgboost, lightgbm, glm ou gam.')
