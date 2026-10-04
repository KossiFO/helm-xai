"""Official SHAP/LIME backends on a caller-selected output class."""

from contextlib import contextmanager
from threading import RLock
import warnings

import numpy as np

from .adapter import MixedCodec

_RANDOM_LOCK = RLock()


@contextmanager
def temporary_seed(seed):
    # Kernel SHAP uses NumPy's legacy RNG. Restore it after this serial call.
    with _RANDOM_LOCK:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)


def explain(adapter, instance, background, method, seed, budget):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        weights, quality, conditions = _compute(adapter, instance, background, method, seed, budget)
    quality["warnings"] = ([quality["perturbation_distribution"]] if method == "lime" else []) + list(dict.fromkeys(str(w.message) for w in captured
                                           if not issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning))))
    return weights, quality, conditions


def _compute(adapter, instance, background, method, seed, budget):
    codec = MixedCodec(adapter, instance)
    bg, row = codec.encode(background), codec.encode(instance)
    target = adapter.target_index
    expected = adapter.probabilities(instance)[0, target]
    # Check the representation round-trip before generating perturbations.
    np.testing.assert_allclose(codec.predict(row), adapter.probabilities(instance), atol=1e-10, rtol=0)
    np.testing.assert_allclose(codec.predict(bg), adapter.probabilities(background), atol=1e-10, rtol=0)
    if method == "shap":
        import shap
        with temporary_seed(seed):
            engine = shap.KernelExplainer(lambda X: codec.predict(X)[:, target], bg, link="identity")
            weights = np.asarray(engine.shap_values(row, nsamples=budget, l1_reg=0, silent=True)).reshape(-1)
        base = float(engine.expected_value)
        residual = float(expected - base - weights.sum())
        if abs(residual) > 1e-6:
            raise ValueError("L’identité additive SHAP n’est pas satisfaite.")
        quality = {"base_value": base, "additivity_residual": residual, "units": "probability",
                   "backend": "shap.KernelExplainer", "feature_dependence": "marginal background replacement"}
        conditions = {}
    elif method == "lime":
        from lime.lime_tabular import LimeTabularExplainer
        if not np.isfinite(bg).all() or not np.isfinite(row).all():
            raise ValueError("LIME nécessite des entrées numériques sans valeurs manquantes ; SHAP peut être essayé avec un pipeline d’imputation.")
        cat = [adapter.columns.index(c) for c in adapter.categorical]
        engine = LimeTabularExplainer(bg, feature_names=adapter.columns, categorical_features=cat,
                                     categorical_names={adapter.columns.index(c): [str(v) for v in codec.vocab[c]] for c in adapter.categorical},
                                     class_names=[str(c) for c in adapter.classes], mode="classification",
                                     discretize_continuous=True, random_state=seed)
        exp = engine.explain_instance(row[0], codec.predict, labels=(target,),
                                      num_features=len(adapter.columns), num_samples=budget)
        weights = np.zeros(len(adapter.columns))
        for i, weight in exp.local_exp[target]:
            weights[i] = weight
        # Preserve ordered conditions independently from coefficient values.
        conditions = {adapter.columns[i]: exp.domain_mapper.map_exp_ids([(i, 0.0)])[0][0]
                      for i in range(len(adapter.columns))}
        quality = {"local_r2": float(exp.score), "units": "local_surrogate_weight",
                   "backend": "lime.lime_tabular.LimeTabularExplainer", "base_value": None,
                   "perturbation_reference_rows": len(background),
                   "perturbation_distribution": "Quartiles et fréquences catégorielles estimés sur le fond fourni, pas sur tout l’entraînement.",
                   "warnings": []}
    else:
        raise ValueError("Méthode tabulaire prise en charge : shap ou lime.")
    if weights.shape != (len(adapter.columns),) or not np.isfinite(weights).all():
        raise ValueError("Contributions invalides.")
    return dict(zip(adapter.columns, weights.tolist())), quality, conditions
