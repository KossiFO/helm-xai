"""Validate fitted probabilistic classifiers and preserve original columns."""

import numpy as np
import pandas as pd


class ClassifierAdapter:
    def __init__(self, model, reference, target_class, categorical_features=None):
        if not isinstance(reference, pd.DataFrame) or reference.empty:
            raise ValueError("Le fond doit être un DataFrame d’apprentissage non vide.")
        if not reference.columns.is_unique or not all(isinstance(c, str) for c in reference.columns):
            raise ValueError("Les noms de variables doivent être des chaînes uniques.")
        if not all(hasattr(model, name) for name in ("predict_proba", "predict", "classes_")):
            raise TypeError("Un classifieur entraîné avec predict, predict_proba et classes_ est requis.")
        classes = np.asarray(model.classes_)
        if classes.ndim != 1 or len(classes) < 2:
            raise ValueError("Classification mono-sortie avec au moins deux classes requise.")
        self.classes = classes.tolist()
        if target_class not in self.classes:
            raise ValueError(f"Classe d’intérêt absente : {target_class!r} ; classes : {self.classes}")
        self.target_class, self.target_index = target_class, self.classes.index(target_class)
        self.model, self.reference = model, reference.copy(deep=True)
        self.columns = reference.columns.tolist()
        if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != self.columns:
            raise ValueError("Les colonnes du fond doivent suivre le schéma d’entrée du modèle.")
        inferred = [c for c in self.columns if not pd.api.types.is_numeric_dtype(reference[c]) or pd.api.types.is_bool_dtype(reference[c])]
        self.categorical = list(inferred if categorical_features is None else categorical_features)
        if len(set(self.categorical)) != len(self.categorical) or not set(self.categorical) <= set(self.columns):
            raise ValueError("Liste de variables qualitatives invalide.")
        for col in self.columns:
            if pd.api.types.is_datetime64_any_dtype(reference[col]):
                raise ValueError("Transformer explicitement les dates avant l’explication.")
            if col not in self.categorical:
                values = reference[col].to_numpy(dtype=float, na_value=np.nan)
                if np.isinf(values).any():
                    raise ValueError("Valeurs infinies non prises en charge.")
        self.probabilities(reference.iloc[:2])

    def frame(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or set(X.columns) != set(self.columns):
            raise ValueError("Fournir un DataFrame avec exactement les variables du modèle.")
        return X.loc[:, self.columns].copy(deep=True)

    def probabilities(self, X):
        frame = self.frame(X)
        p = np.asarray(self.model.predict_proba(frame), dtype=float)
        if p.shape != (len(frame), len(self.classes)) or not np.isfinite(p).all():
            raise ValueError("Matrice de probabilités invalide.")
        if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-7):
            raise ValueError("Les probabilités doivent être normalisées dans [0, 1].")
        return p

    def decisions(self, X):
        result = np.asarray(self.model.predict(self.frame(X)))
        if result.shape != (len(X),) or any(v not in self.classes for v in result):
            raise ValueError("Prédictions de classes invalides.")
        return result


class MixedCodec:
    """Use codes only inside explainers; call the fitted model on original values."""
    def __init__(self, adapter, instance):
        self.adapter = adapter
        self.vocab = {}
        for name in adapter.categorical:
            values = pd.concat([adapter.reference[name].astype(object), instance[name].astype(object)])
            unique = {}
            for value in values:
                unique.setdefault(self._key(value), value)
            self.vocab[name] = list(unique.values())
        self.codes = {name: {self._key(v): i for i, v in enumerate(values)}
                      for name, values in self.vocab.items()}

    @staticmethod
    def _key(value):
        if value is None:
            return ("none",)
        if value is pd.NA:
            return ("pd.NA",)
        if pd.isna(value):
            return ("nan",)
        return ("value", value)

    def encode(self, frame):
        result = np.empty((len(frame), len(self.adapter.columns)), dtype=float)
        for i, name in enumerate(self.adapter.columns):
            if name in self.vocab:
                values = frame[name].astype(object)
                result[:, i] = [self.codes[name][self._key(v)] for v in values]
            else:
                result[:, i] = frame[name].to_numpy(dtype=float, na_value=np.nan)
        return result

    def decode(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        frame = pd.DataFrame(index=range(len(matrix)))
        for i, name in enumerate(self.adapter.columns):
            if name in self.vocab:
                codes = matrix[:, i]
                if not np.isfinite(codes).all() or not np.allclose(codes, np.round(codes)):
                    raise ValueError("Perturbation catégorielle non valide.")
                if (codes < 0).any() or (codes >= len(self.vocab[name])).any():
                    raise ValueError("Code catégoriel hors vocabulaire.")
                values = [self.vocab[name][int(v)] for v in codes]
                frame[name] = pd.Series(values, dtype=object)
                dtype = self.adapter.reference[name].dtype
                if isinstance(dtype, pd.CategoricalDtype):
                    categories = list(dtype.categories)
                    categories.extend(v for v in self.vocab[name] if not pd.isna(v) and v not in categories)
                    frame[name] = pd.Categorical(frame[name], categories=categories, ordered=dtype.ordered)
                elif isinstance(dtype, pd.StringDtype) or pd.api.types.is_bool_dtype(dtype):
                    frame[name] = frame[name].astype(dtype)
            else:
                frame[name] = matrix[:, i]
        return frame[self.adapter.columns]

    def predict(self, matrix):
        return self.adapter.probabilities(self.decode(matrix))
