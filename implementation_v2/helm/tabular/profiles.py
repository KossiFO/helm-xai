"""Explicit, versioned tabular presentation policies; not learned preferences."""
from dataclasses import dataclass, replace
import json
from html import escape
import pandas as pd
from .report import HTMLReport

POLICY_VERSION = "tabular_profiles_v1"
PROFILES = {
    "utilisateur": {"title": "Comprendre une prédiction", "methods": ("shap",), "purpose": "Comprendre les principaux éléments qui soutiennent ou réduisent le score."},
    "metier": {"title": "Examiner les dossiers", "methods": ("shap",), "purpose": "Repérer les facteurs récurrents puis examiner chaque dossier dans son contexte."},
    "technique": {"title": "Comparer les explications", "methods": ("shap", "lime"), "purpose": "Comparer deux approximations et examiner leurs contrôles numériques, sans fusionner leurs poids."},
    "audit": {"title": "Auditer les calculs", "methods": ("shap", "lime"), "purpose": "Vérifier la classe expliquée, la sélection des dossiers, les échecs et la provenance."},
}


def profile_plan(profile):
    if profile not in PROFILES:
        raise ValueError("Profil attendu : " + ", ".join(PROFILES))
    return {**PROFILES[profile], "profile": profile, "policy_version": POLICY_VERSION,
            "learned": False, "rationale": "SHAP pour une décomposition additive ; SHAP et LIME pour comparer deux méthodes. Règle de conception à évaluer auprès des utilisateurs."}


def narrative(local, limit=3):
    """Grounded prose; no causal, fraud or counterfactual assertions."""
    parts = []
    for sign, title in ((1, "Ce qui soutient le score"), (-1, "Ce qui réduit le score")):
        rows = local.table()
        rows = rows[rows.contribution * sign > 0].head(limit)
        items = []
        for row in rows.itertuples():
            amount = f"{row.contribution * 100:+.2f} points de score" if local.method == "shap" else f"coefficient local {row.contribution:+.4f}"
            condition = f" — condition : {row.condition_locale}" if local.method == "lime" else ""
            items.append(f"<li><b>{escape(str(row.variable).replace('_', ' '))}</b> = {escape(str(row.valeur))}{escape(condition)} : {amount}.</li>")
        parts.append(f"<h4>{title}</h4><ul>" + ("".join(items) or "<li>Aucun facteur de ce signe dans cette explication.</li>") + "</ul>")
    if local.method == "shap":
        shown = local.table().head(0)
        for sign in (1, -1):
            rows = local.table()
            shown = pd.concat([shown, rows[rows.contribution * sign > 0].head(limit)])
        remainder = sum(local.contributions.values()) - shown.contribution.sum()
        parts.append(f"<p>Référence : {local.quality['base_value']:.2%}. Autres contributions non détaillées : {remainder*100:+.2f} points. Score final : {local.target_score:.2%}.</p>")
    else:
        parts.append(f"<p>LIME reproduit localement le modèle avec un R² de {local.quality['local_r2']:.3f}. Ces coefficients ne s’additionnent pas pour reconstruire le score.</p>")
    return "".join(parts)


@dataclass
class ProfileReport(HTMLReport):
    profile: str
    reports: dict

    def __post_init__(self):
        profile_plan(self.profile)
        if not self.reports:
            raise ValueError("Aucun calcul disponible.")
        first = next(iter(self.reports.values()))
        for name, report in self.reports.items():
            if name != report.method or report.target_class != first.target_class or not report.ranking.equals(first.ranking):
                raise ValueError("Les méthodes doivent expliquer la même classe et les mêmes dossiers.")

    @property
    def plan(self):
        return profile_plan(self.profile)

    def _body(self):
        plan = self.plan
        text = f"<h2>{plan['title']}</h2><p>{plan['purpose']}</p>"
        text += "<p><b>Méthodes proposées pour ce profil :</b> " + ", ".join(m.upper() for m in plan['methods']) + "</p>"
        text += f"<details><summary>Pourquoi ces méthodes ?</summary><p>{plan['rationale']}</p><p>Politique {POLICY_VERSION} ; aucune préférence apprise.</p></details>"
        missing = [m for m in plan['methods'] if m not in self.reports]
        if missing:
            text += "<p><b>Calculs manquants :</b> " + ", ".join(missing) + "</p>"
        for method in plan['methods']:
            if method in self.reports:
                report = self.reports[method]
                view = replace(report, profile=self.profile, explanations=[replace(e, profile=self.profile) for e in report.explanations])
                text += view._body()
        if self.profile == 'audit':
            records = {}
            for method, report in self.reports.items():
                records[method] = {'target_class':report.target_class,'selection':report.ranking.attrs,
                    'requested':len(report.ranking),'successful':len(report.explanations),'errors':report.errors,
                    'background_sha256':sorted({e.quality.get('background_sha256', 'inconnu') for e in report.explanations}),
                    'parameters_and_quality':[{'row_id':e.row_id, 'quality':e.quality} for e in report.explanations]}
            text += '<h2>Registre de couverture et de provenance</h2><p>Vérifier les dossiers sélectionnés, les erreurs, le fond de référence et les paramètres avant toute interprétation.</p><pre>' + escape(json.dumps(records, ensure_ascii=False, indent=2, default=str)) + '</pre>'
        if self.profile == 'technique' and all(m in self.reports for m in ('shap','lime')):
            groups = [{e.row_id for e in self.reports[m].explanations} for m in ('shap', 'lime')]
            if groups[0] != groups[1] or not groups[0]:
                return text + '<p><b>Comparaison suspendue :</b> les deux méthodes ne disposent pas des mêmes dossiers expliqués avec succès.</p>'
            text += f'<p>Comparaison sur les mêmes {len(groups[0])} dossiers expliqués avec succès.</p>'
            a, b = (self.reports[m].summary().head(5).variable.tolist() for m in ('shap','lime'))
            common = sorted(set(a) & set(b))
            text += '<h2>Comparaison des facteurs dominants</h2><p>Intersection des cinq premiers facteurs moyens absolus : ' + escape(', '.join(common) or 'aucune') + f' ({len(common)}/5 au maximum). Ce recouvrement descriptif ne mesure pas la fidélité.</p>'
        return text


def profile_dashboard(reports_by_group, *, title="Comprendre les scores", group_labels=None):
    from helm.visualization.dashboard import Dashboard
    labels = group_labels or {g:g for g in reports_by_group}
    if set(labels) != set(reports_by_group):
        raise ValueError('Les libellés doivent correspondre aux groupes.')
    profiles = {'utilisateur':'Utilisateur','metier':'Métier','technique':'Technique','audit':'Audit'}
    views = {(group,profile):ProfileReport(profile,reports)._body() for group,reports in reports_by_group.items() for profile in profiles}
    return Dashboard(title,labels,profiles,views,'metier', 'Explorez les facteurs du modèle, selon votre besoin et votre profil.')
