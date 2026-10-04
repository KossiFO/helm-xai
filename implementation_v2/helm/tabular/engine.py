"""Native tabular classification entry point. No text-domain selection policy."""

from numbers import Integral
from hashlib import sha256
from importlib.metadata import version
import json

import numpy as np
import pandas as pd

from .adapter import ClassifierAdapter
from .explainers import explain as run_explainer
from .report import CohortReport, LocalExplanation


class TabularHELM:
    """Explain a fitted probabilistic classifier on original, mixed columns.

    reference: training/background DataFrame (never fitted by HELM).
    target_class: label in model.classes_, independent from model.predict.
    explain_profile proposes methods by an explicit, untrained profile policy.
    """

    def __init__(self, model, reference, *, target_class, categorical_features=None,
                 background_size=50, random_state=42):
        if not isinstance(background_size, Integral) or background_size < 2:
            raise ValueError("background_size doit être un entier >= 2.")
        if not isinstance(random_state, Integral) or not 0 <= random_state < 2**32:
            raise ValueError("random_state doit être un entier entre 0 et 2**32 - 1.")
        self.adapter = ClassifierAdapter(model, reference, target_class, categorical_features)
        self.random_state = int(random_state)
        self.background = reference.sample(min(background_size, len(reference)), random_state=random_state).copy(deep=True)
        payload = {"schema": [(c, repr(reference[c].dtype)) for c in reference],
                   "index": [repr(i) for i in self.background.index],
                   "values": [[(type(v).__name__, repr(v)) for v in row]
                              for row in self.background.itertuples(index=False, name=None)]}
        self.background_fingerprint = sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()

    @property
    def capabilities(self):
        return {"task": "probabilistic_classification", "methods": ["shap", "lime"],
                "features": self.adapter.columns.copy(), "categorical_features": self.adapter.categorical.copy(),
                "target_class": self.adapter.target_class, "adaptive_selection": False}

    def rank(self, X, *, k=10, order="highest", group="all", labels=None):
        """Rank by P(target_class). Labels are optional observed annotations.

        Series labels must exactly match X.index, including order. Arrays are
        positional. group never changes the class whose score is ranked.
        """
        if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
            raise ValueError("k doit être un entier positif.")
        if order not in ("highest", "lowest"):
            raise ValueError("order doit être highest ou lowest.")
        if group not in ("all", "predicted_target", "observed_target", "observed_other"):
            raise ValueError("Groupe inconnu.")
        frame = self.adapter.frame(X)
        if not frame.index.is_unique:
            raise ValueError("Les identifiants de lignes doivent être uniques.")
        observed = None
        if labels is not None:
            if isinstance(labels, pd.Series) and not labels.index.equals(frame.index):
                raise ValueError("Les étiquettes doivent suivre exactement l’index des données.")
            observed = np.asarray(labels)
            if observed.shape != (len(frame),):
                raise ValueError("Une étiquette par ligne est requise.")
            if any(pd.isna(v) or v not in self.adapter.classes for v in observed):
                raise ValueError("Étiquettes absentes ou étrangères aux classes du modèle.")
        if group.startswith("observed_") and observed is None:
            raise ValueError("Ce groupe nécessite les étiquettes observées.")
        scores = self.adapter.probabilities(frame)[:, self.adapter.target_index] if len(frame) else []
        decisions = self.adapter.decisions(frame) if len(frame) else []
        table = pd.DataFrame({"position": np.arange(len(frame)), "target_score": scores,
                              "prediction": decisions}, index=frame.index)
        if observed is not None:
            table["observed_label"] = observed
        if group == "predicted_target":
            table = table[table.prediction == self.adapter.target_class]
        elif group == "observed_target":
            table = table[table.observed_label == self.adapter.target_class]
        elif group == "observed_other":
            table = table[table.observed_label != self.adapter.target_class]
        eligible = len(table)
        table = table.sort_values("target_score", ascending=(order == "lowest"), kind="stable").head(k)
        table.attrs.update(eligible_count=eligible, input_count=len(frame), group=group, order=order)
        return table

    def explain(self, X, *, method="shap", profile="metier", budget=256, observed_label=None):
        """Explain exactly one row; failures are explicit, without fallback."""
        from helm import __version__
        self._options(method, profile, budget)
        frame = self.adapter.frame(X)
        if len(frame) != 1:
            raise ValueError("explain attend exactement une ligne ; utiliser explain_top pour un groupe.")
        if observed_label is not None and observed_label not in self.adapter.classes:
            raise ValueError("Étiquette observée étrangère aux classes du modèle.")
        weights, quality, conditions = run_explainer(self.adapter, frame, self.background,
                                                    method, self.random_state, int(budget))
        quality.update(background_size=len(self.background), random_state=self.random_state, budget=int(budget),
                       background_sha256=self.background_fingerprint,
                       backend_version=version(method), sklearn_version=version("scikit-learn"),
                       helm_version=__version__, pandas_version=pd.__version__, numpy_version=np.__version__)
        return LocalExplanation(row_id=frame.index[0], target_class=self.adapter.target_class,
                                target_score=float(self.adapter.probabilities(frame)[0, self.adapter.target_index]),
                                prediction=self.adapter.decisions(frame)[0], observed_label=observed_label,
                                method=method, profile=profile, values=frame.iloc[0].to_dict(),
                                contributions=weights, conditions=conditions, quality=quality)

    def explain_top(self, X, *, k=10, order="highest", group="all", labels=None,
                    method="shap", profile="metier", budget=256):
        """Explain selected rows and summarize successful explanations only."""
        self._options(method, profile, budget)
        ranked = self.rank(X, k=k, order=order, group=group, labels=labels)
        results, errors = [], []
        for row_id, row in ranked.iterrows():
            try:
                results.append(self.explain(X.iloc[[int(row.position)]], method=method, profile=profile,
                                            budget=budget, observed_label=(ranked.loc[row_id, "observed_label"]
                                                                         if "observed_label" in ranked else None)))
            except (ValueError, TypeError, RuntimeError, AssertionError, np.linalg.LinAlgError) as exc:
                errors.append({"row_id": row_id, "error": f"{type(exc).__name__}: {exc}"})
        return CohortReport(ranked, results, errors, self.adapter.target_class, method, profile)

    def explain_profile(self, X, *, profile="metier", k=10, order="highest", group="all", labels=None, budgets=None):
        """Select methods by a documented design rule; no learned-policy claim."""
        from .profiles import profile_plan, ProfileReport
        plan = profile_plan(profile)
        budgets = {"shap": 512, "lime": 2048, **(budgets or {})}
        reports = {method: self.explain_top(X, k=k, order=order, group=group, labels=labels,
                   profile=profile, method=method, budget=budgets[method]) for method in plan['methods']}
        return ProfileReport(profile, reports)

    def dashboard(self, X, *, k=10, labels=None, group="all", orders=("highest", "lowest"), budgets=None):
        """Compute once per method/group, then offer all profiles without a kernel callback."""
        from .profiles import profile_dashboard
        if not orders or len(set(orders)) != len(orders) or any(o not in ('highest', 'lowest') for o in orders):
            raise ValueError('orders : highest et/ou lowest, sans doublon.')
        budgets = {"shap":512, "lime":2048, **(budgets or {})}
        reports = {order:{method:self.explain_top(X,k=k,order=order,group=group,labels=labels,method=method,budget=budgets[method])
                          for method in ('shap','lime')} for order in orders}
        labels_display = {o:f"{k} scores {'élevés' if o == 'highest' else 'faibles'}" for o in orders}
        return profile_dashboard(reports, group_labels=labels_display)

    def _options(self, method, profile, budget):
        if method not in ("shap", "lime"):
            raise ValueError("Méthode tabulaire prise en charge : shap ou lime.")
        if profile not in ("metier", "technique", "utilisateur", "audit"):
            raise ValueError("Profil : metier, technique, utilisateur ou audit.")
        minimum = max(50, 2 * len(self.adapter.columns) + 1)
        if not isinstance(budget, Integral) or isinstance(budget, bool) or budget < minimum:
            raise ValueError(f"budget doit être un entier >= {minimum}.")
