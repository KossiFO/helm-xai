"""Self-contained, escaped French reports; no external service required."""

from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import pandas as pd


def _text(value):
    return escape(str(value))


def _page(body):
    from helm.visualization.dashboard import STYLE
    return '<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HELM — Explications</title><style>' + STYLE + '</style><body><div class="helm-dashboard"><main class="helm-panels">' + body + '</main></div></body></html>'


class HTMLReport:
    def to_html(self, path=None):
        """Return HTML and optionally save it. Values are embedded in the file."""
        result = _page(self._body())
        if path is not None:
            Path(path).write_text(result, encoding="utf-8")
        return result

    def _repr_html_(self):
        return self.to_html()


@dataclass
class LocalExplanation(HTMLReport):
    row_id: object
    target_class: object
    target_score: float
    prediction: object
    observed_label: object
    method: str
    profile: str
    values: dict
    contributions: dict
    conditions: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)

    def table(self):
        data = [{"variable": name, "valeur": self.values[name], "contribution": weight,
                 "condition_locale": self.conditions.get(name, "")}
                for name, weight in self.contributions.items()]
        return pd.DataFrame(data).sort_values("contribution", key=abs, ascending=False, kind="stable")

    def _body(self):
        text = f'<section><h3>Dossier {_text(self.row_id)}</h3><p>Score pour la classe {_text(self.target_class)} : <b>{self.target_score:.4f}</b> · Décision du modèle : {_text(self.prediction)}'
        if self.observed_label is not None:
            text += f' · Étiquette observée : {_text(self.observed_label)}'
        from helm.visualization.dashboard import contribution_chart, score_card
        text += '</p>' + score_card(f'Score de classe {self.target_class}', self.target_score)
        text += contribution_chart(self.contributions.items(), title=f'{self.method.upper()} — classe {self.target_class}', unit='probability' if self.method == 'shap' else 'coefficient')
        text += '<p>Une contribution positive soutient la classe expliquée ; une contribution négative s’y oppose.</p>'
        if self.profile in ('metier', 'utilisateur'):
            from .profiles import narrative
            text += narrative(self, limit=3 if self.profile == 'utilisateur' else 5)
            if self.profile == 'metier':
                text += '<p><b>À examiner :</b> vérifier les valeurs du dossier et les rapprocher des pièces disponibles. Une contribution décrit le modèle ; elle ne prouve pas une fraude et ne justifie pas à elle seule une décision.</p>'
            text += '<details><summary>Valeurs détaillées et qualité du calcul</summary>'
        table = self.table()
        if self.profile in ("metier", "utilisateur"):
            table = table.head(8)
        if not table["condition_locale"].any():
            table = table.drop(columns="condition_locale")
        table = table.rename(columns={"variable":"Variable", "valeur":"Valeur", "contribution":"Contribution", "condition_locale":"Condition locale"})
        text += '<div class="table">' + table.to_html(index=False, escape=True,
            formatters={"Contribution":lambda v: f'{v:+.4f}'}, float_format=lambda v: f'{v:.4f}'.rstrip('0').rstrip('.')) + '</div>'
        if self.method == "shap":
            text += f'<p>SHAP : score de référence {self.quality["base_value"]:.4f} + somme des contributions = score expliqué (résidu {self.quality["additivity_residual"]:.2g}). Les 8 premières variables seulement sont affichées en profil métier.</p>'
        else:
            text += f'<p>LIME : approximation locale, R² = {self.quality["local_r2"]:.3f}. Les coefficients ne décomposent pas exactement le score du modèle ; un R² faible limite leur interprétation.</p>'
        if self.quality.get("warnings"):
            text += '<p><b>Avertissements de calcul :</b> ' + _text(" ; ".join(self.quality["warnings"])) + '</p>'
        provenance = self.quality if self.profile in ("technique", "audit") else {
            k: v for k, v in self.quality.items() if k.endswith("version") or k == "background_sha256"}
        text += '<details><summary>Traçabilité du calcul</summary><pre>' + _text(provenance) + '</pre></details>'
        if self.profile in ('metier', 'utilisateur'):
            text += '</details>'
        return text + '</section>'


@dataclass
class CohortReport(HTMLReport):
    ranking: pd.DataFrame
    explanations: list
    errors: list
    target_class: object
    method: str
    profile: str

    def summary(self):
        """Descriptive aggregation, not a causal or population fraud profile."""
        if not self.explanations:
            return pd.DataFrame(columns=["variable", "importance_moyenne_absolue", "contribution_moyenne", "nombre_contributions_positives", "n"])
        weights = pd.DataFrame([e.contributions for e in self.explanations])
        return pd.DataFrame({"variable": weights.columns,
                             "importance_moyenne_absolue": weights.abs().mean().values,
                             "contribution_moyenne": weights.mean().values,
                             "nombre_contributions_positives": (weights > 0).sum().values,
                             "n": len(weights)}).sort_values("importance_moyenne_absolue", ascending=False, kind="stable")

    def _body(self):
        attrs = self.ranking.attrs
        text = ''
        text += f'<p>Classe expliquée : <b>{_text(self.target_class)}</b> · Méthode : {_text(self.method.upper())} · Profil : {_text(self.profile)}</p>'
        groups = {"all": "tous les dossiers", "predicted_target": "prédits dans la classe expliquée",
                  "observed_target": "étiquetés dans la classe expliquée", "observed_other": "étiquetés dans les autres classes"}
        order = "plus élevés" if attrs.get("order") == "highest" else "plus bas"
        text += f'<p>Scores {order} parmi les dossiers {_text(groups.get(attrs.get("group"), ""))}. {len(self.ranking)} sélectionnés sur {attrs.get("eligible_count", 0)} admissibles ; {len(self.explanations)} explications réussies et {len(self.errors)} échecs.</p>'
        text += '<p>Le score est celui du modèle, sans garantie de calibration. Une prédiction risquée n’établit pas une fraude.</p>'
        text += '<h2>Dossiers sélectionnés</h2><div class="table">' + self.ranking.drop(columns="position").to_html(escape=True, float_format=lambda v: f'{v:.4f}') + '</div>'
        if self.explanations and self.profile != 'utilisateur':
            text += '<h2>Facteurs récurrents dans ce groupe</h2><p>Calculés uniquement sur les explications réussies. Cette sélection de scores extrêmes ne représente pas toute la population ; les contributions décrivent le modèle et ne sont pas des causes de fraude.</p>'
            if self.profile == 'metier':
                text += '<ul>'
                for row in self.summary().head(5).itertuples():
                    direction = 'soutient en moyenne' if row.contribution_moyenne > 0 else 'réduit en moyenne' if row.contribution_moyenne < 0 else 'a une contribution moyenne nulle pour'
                    text += f'<li><b>{_text(row.variable.replace("_", " "))}</b> {direction} le score de ce groupe ; contribution positive dans {row.nombre_contributions_positives}/{row.n} dossiers expliqués.</li>'
                text += '</ul><p>Ces tendances de groupe peuvent différer du cas individuel. Ouvrir un dossier ci-dessous pour examiner ses facteurs.</p>'
            else:
                text += '<div class="table">' + self.summary().to_html(index=False, escape=True, float_format=lambda v: f'{v:.4f}') + '</div>'
        if self.errors:
            text += '<h2>Échecs à examiner</h2>' + pd.DataFrame(self.errors).to_html(index=False, escape=True)
        text += '<h2>Explications individuelles</h2>' + ''.join('<details><summary>Dossier ' + _text(e.row_id) + f' — score {e.target_score:.4f}</summary>' + e._body() + '</details>' for e in self.explanations)
        return text + '<small>Prototype tabulaire expérimental. Les perturbations peuvent combiner des valeurs rarement observées. La fidélité et la stabilité doivent être évaluées séparément. La proposition de méthodes repose sur des règles explicites par profil ; aucune politique apprise n’est calibrée pour ce domaine.</small>'
