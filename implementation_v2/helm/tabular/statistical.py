"""Prediction-only bridges for fitted binary statistical models."""

import numpy as np
import pandas as pd


class BinaryProbabilityAdapter:
    """Expose predict/predict_proba/classes_ without fitting or changing a model.

    Use from_glm or from_gam. The fitted transformer maps original columns to
    the numeric training design. classes[1] is the event encoded as 1 at fit.
    threshold defines this adapter's decision, not an original model decision.
    """

    def __init__(self, model, *, probability_method, transformer=None,
                 classes=(0, 1), threshold=0.5, add_intercept=False):
        labels = np.asarray(classes, dtype=object)
        if labels.shape != (2,) or pd.isna(labels).any() or labels[0] == labels[1]:
            raise ValueError("classes doit contenir deux étiquettes distinctes non manquantes.")
        if isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold doit être strictement entre 0 et 1.")
        if not callable(getattr(model, probability_method, None)):
            raise TypeError("Le modèle ne fournit pas la méthode de probabilité demandée.")
        if transformer is not None and not callable(getattr(transformer, 'transform', None)):
            raise TypeError("transformer doit être un prétraitement déjà ajusté avec transform.")
        self.model = model
        self.probability_method = probability_method
        self.transformer = transformer
        self.classes_ = labels.copy()
        self.threshold = float(threshold)
        self.add_intercept = add_intercept
        if transformer is not None and hasattr(transformer, 'feature_names_in_'):
            self.feature_names_in_ = transformer.feature_names_in_.copy()

    @classmethod
    def from_glm(cls, result, *, transformer=None, classes=(0, 1),
                 threshold=0.5, add_intercept=True):
        """Wrap fitted statsmodels GLM(Binomial) with a numeric design matrix.

        add_intercept=True prepends ones, matching sm.add_constant(...,
        has_constant='add') at training. Formulas, offsets and exposure are
        rejected: their design would require a separate explicit adapter.
        """
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.genmod.families import Binomial
        fitted_model = getattr(result, 'model', None)
        if not isinstance(fitted_model, GLM) or not isinstance(fitted_model.family, Binomial):
            raise TypeError("Un résultat ajusté GLM de famille Binomial est requis.")
        if not hasattr(result, 'params'):
            raise TypeError("Le GLM doit être déjà ajusté.")
        if any(getattr(fitted_model, name, None) is not None for name in ('formula', 'offset', 'exposure')):
            raise ValueError("GLM avec formule, offset ou exposure : adaptateur spécifique requis.")
        return cls(result, probability_method='predict', transformer=transformer,
                   classes=classes, threshold=threshold, add_intercept=add_intercept)

    @classmethod
    def from_gam(cls, model, *, transformer=None, classes=(0, 1), threshold=0.5):
        """Wrap fitted pyGAM LogisticGAM; other GAM families are not classifiers."""
        from pygam import LogisticGAM
        if not isinstance(model, LogisticGAM) or not hasattr(model, 'coef_'):
            raise TypeError("Un LogisticGAM pyGAM déjà ajusté est requis.")
        return cls(model, probability_method='predict_proba', transformer=transformer,
                   classes=classes, threshold=threshold)

    def predict_proba(self, X):
        if hasattr(self, 'feature_names_in_'):
            if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
                raise ValueError("Colonnes originales requises dans l’ordre de l’entraînement.")
        design = self.transformer.transform(X) if self.transformer is not None else X
        if hasattr(design, 'toarray'):
            raise ValueError("Prétraitement dense requis (OneHotEncoder sparse_output=False).")
        if self.add_intercept:
            design = np.asarray(design)
            if design.ndim != 2:
                raise ValueError("Matrice de conception bidimensionnelle requise.")
            design = np.column_stack([np.ones(len(design)), design])
        p = np.asarray(getattr(self.model, self.probability_method)(design), dtype=float)
        if p.shape != (len(X),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Une probabilité finie entre 0 et 1 par ligne est requise.")
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= self.threshold).astype(int)]
