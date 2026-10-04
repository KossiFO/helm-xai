"""French, profile-specific HTML backed by a real HELMContext."""
import json
from html import escape
from helm.config import UserProfile
from .dashboard import contribution_chart, score_card
from .method_coverage import METHOD_LABELS, coverage_html, method_status, is_verified_ig


def context_html(ctx, *, available=None, failures=None):
    p = ctx.effective_profile
    titles = {UserProfile.END_USER:'Comprendre le classement du commentaire', UserProfile.MODERATOR:'Examiner le commentaire avant de décider', UserProfile.TECHNICAL_EXPERT:'Comparer les méthodes sur ce commentaire', UserProfile.REGULATOR:'Auditer cette explication'}
    score = ctx.prediction.probabilities[1]
    text = f"<h2>{titles[p]}</h2><blockquote>{escape(ctx.text)}</blockquote>"
    text += score_card('Score de toxicité', score)
    text += f"<p>Classe prédite : {escape(ctx.prediction.label)} · Seuil : 50 %.</p>"
    text += '<p>La prédiction porte sur ce commentaire, pas sur son auteur. Le contexte peut modifier son interprétation.</p>'
    if p == UserProfile.MODERATOR:
        text += '<h2>Points à examiner</h2><p>Relire les passages signalés dans leur contexte, distinguer une attaque d’une citation ou d’un désaccord, puis appliquer les règles de modération. Aucune suppression automatique n’est effectuée.</p>'
    missing = [m.method_name for m in ctx.selected_methods if m.method_name not in ctx.attributions]
    if missing:
        text += '<p><b>Explications incomplètes — méthodes en échec :</b> ' + escape(', '.join(missing)) + '</p>'
    available = ctx.attributions if available is None else available
    failures = ctx.failed_attributions if failures is None else failures
    selected = [m.method_name for m in ctx.selected_methods]
    text += coverage_html(available, failures, selected)
    order = list(dict.fromkeys(selected + list(METHOD_LABELS)))
    for name in order:
        if name not in METHOD_LABELS:
            continue
        attr = available.get(name)
        status = method_status(name, attr, failures)
        if attr is None or attr.metadata.get('valid') is False:
            text += '<details><summary>' + METHOD_LABELS[name] + ' — ' + escape(status) + '</summary><pre>' + escape(json.dumps(failures.get(name, 'Aucun résultat disponible dans cette recette.'), ensure_ascii=False, indent=2, default=str)) + '</pre></details>'
            continue
        if name == 'integrated_gradients' and not is_verified_ig(attr):
            text += '<details><summary>Integrated Gradients (IG) — ' + escape(status) + '</summary><p>Enregistrement issu d’un backend historique. LOO est une méthode distincte d’IG. Cette vue ne présente pas ces valeurs comme un calcul IG natif vérifié.</p><pre>' + escape(json.dumps(attr.metadata, ensure_ascii=False, indent=2, default=str)) + '</pre></details>'
            continue
        supplementary = name not in selected
        if supplementary:
            text += '<details><summary>' + METHOD_LABELS[name] + ' — complément consultable</summary>'
        target = attr.metadata.get('target_class', 1)
        text += '<h2>' + escape(METHOD_LABELS[name]) + '</h2>'
        if name == 'lime':
            r2 = attr.metadata.get('local_r2')
            text += '<p>LIME est une approximation locale du modèle. Ses coefficients ne sont pas des variations mesurées par suppression et ne décomposent pas exactement le score.</p>'
            text += '<p>Qualité locale R² : ' + (f'{r2:.3f}' if isinstance(r2, (int, float)) else 'non disponible') + '. Une faible valeur limite l’interprétation des facteurs.</p>'
        elif name == 'shap':
            text += '<p>Les contributions SHAP sont exprimées sur l’échelle de probabilité de la classe expliquée ; ce ne sont pas des effets causaux.</p>'
            if attr.base_value is not None:
                label = 'non toxique' if target == 0 else 'toxique'
                text += f'<p>Score de référence SHAP pour la classe {target} ({label}) : {attr.base_value:.4f}.</p>'
        if name == 'integrated_gradients':
            text += '<p>IG natif : intégration des gradients de la probabilité de la classe expliquée depuis une référence PAD lexicale. Les contrôles numériques doivent être satisfaits ; aucun remplacement par LOO.</p>'
            text += f'<p>Classe expliquée : {target} ; score de référence : {attr.base_value:.4f} ; résidu de convergence : {attr.convergence_delta:.3g}.</p>'
        if name == 'anchors':
            meta = attr.metadata
            words = meta.get('anchor_words', [])
            text += '<p>Règle locale pour maintenir la prédiction de classe ' + str(target) + ' sous les perturbations testées :</p>'
            text += '<blockquote>' + (escape(' ET '.join(words)) if words else 'Règle vide : aucun mot exigé par cette recherche.') + '</blockquote>'
            precision = meta.get('precision')
            coverage = meta.get('coverage')
            text += '<p>Précision estimée de la règle : ' + (f'{precision:.1%}' if isinstance(precision,(float,int)) else 'indisponible') + f" ; seuil demandé : {meta.get('threshold',.95):.0%}."
            text += ' Couverture estimée : ' + (f'{coverage:.1%}' if isinstance(coverage,(float,int)) else 'indisponible') + '.</p>'
            if coverage == 0:
                text += '<p class="helm-note">Couverture estimée nulle : cette règle ne permet pas de conclure à une explication représentative. La précision seule ne suffit pas à juger sa qualité.</p>'
            text += '<p>' + escape(status) + '. Ces proportions concernent les perturbations d’Anchors, pas la population des commentaires ; ni garantie universelle ni relation causale.</p>'
        elif name == 'counterfactual':
            examples = attr.metadata.get('counterfactuals', [])
            text += '<h3>Variantes testées qui changent le classement</h3>'
            if examples:
                for example in examples[:3]:
                    text += '<blockquote>' + escape(example['counterfactual_text']) + '</blockquote>'
                    text += f"<p>Après cette modification : score de toxicité {example['new_prob']:.2%} (avant : {example['original_prob']:.2%}).</p>"
                text += '<p>Inversion observée pour les variantes affichées ; ni correction garantie du langage ni preuve causale.</p>'
            else:
                text += '<p>Aucune inversion trouvée dans les modifications explorées.</p>'
        else:
            limit = 3 if p == UserProfile.END_USER else 5 if p == UserProfile.MODERATOR else 8
            # Orient prose toward toxicity, while preserving raw numbers and target in expert/audit views.
            oriented = [(word, -weight if target == 0 else weight) for word, weight in attr.top_tokens]
            chart_values = oriented if p in (UserProfile.END_USER, UserProfile.MODERATOR) else attr.top_tokens
            chart_target = 1 if p in (UserProfile.END_USER, UserProfile.MODERATOR) else target
            text += contribution_chart(chart_values, title=f'{name.upper()} — classe {chart_target}', unit='probability' if name in ('shap','integrated_gradients') else 'coefficient', limit=limit)
            text += f'<p>Jusqu’à {limit} contributions dominantes sont affichées ; elles ne représentent pas nécessairement la somme complète des contributions.</p>'
            if p in (UserProfile.END_USER, UserProfile.MODERATOR):
                text += '<p>Les directions ci-dessous sont exprimées par rapport à la toxicité ; si la méthode explique la classe non toxique, leurs signes sont inversés pour cette lecture.</p>'
                for sign, heading in ((1, 'Éléments qui soutiennent le score de toxicité'),(-1,'Éléments qui s’opposent au score de toxicité')):
                    words = [escape(word) for word, weight in oriented if weight*sign > 0][:limit]
                    text += f'<h3>{heading}</h3><p>' + (', '.join('« '+w+' »' for w in words) or 'Aucun élément de ce signe.') + '</p>'
            else:
                text += f'<p>Classe expliquée par cette attribution : {target}. Les poids de méthodes différentes ne sont pas fusionnés.</p><pre>' + escape(json.dumps(dict(attr.top_tokens[:limit]), ensure_ascii=False, indent=2)) + '</pre>'
        if p in (UserProfile.TECHNICAL_EXPERT, UserProfile.REGULATOR):
            text += '<details><summary>Qualité et paramètres</summary><pre>' + escape(json.dumps({'metadata':attr.metadata, 'evaluation':vars(ctx.evaluation[name]) if name in ctx.evaluation else None}, ensure_ascii=False, indent=2, default=str)) + '</pre></details>'
        if supplementary:
            text += '</details>'
    text += '<details><summary>Choix des méthodes et traçabilité</summary><pre>' + escape(json.dumps({'profile':p.value,'model':ctx.model_name,'selected':[vars(m) for m in ctx.selected_methods], 'successful':list(ctx.attributions),'failed':ctx.failed_attributions,'logs':ctx.logs}, ensure_ascii=False, indent=2, default=str)) + '</pre></details>'
    text += '<p>Test fonctionnel : politique par profil issue de HELM, couverture et résultats du pool indiqués ci-dessus. Aucune préférence utilisateur apprise pendant ce test ; temps de sélection heuristiques, pas des garanties de latence.</p>'
    return text


def toxicity_dashboard(records):
    """Render saved HELM records without running the model again."""
    from helm.context import HELMContext
    from helm.config import Prediction, Attribution, EvaluationMetrics, MethodSelection
    from .dashboard import Dashboard
    groups, views = {}, {}
    profiles = {'utilisateur_final':'Utilisateur','moderateur':'Modérateur','expert_technique':'Expert','regulateur':'Audit'}
    for record in records:
        group = str(record['comment_id'])
        groups[group] = f"Commentaire {record['comment_id'] + 1}"
        ctx = HELMContext(text=record['text'],model_name=record['prediction']['model_name'],user_profile=UserProfile(record['profile']))
        ctx.prediction = Prediction(**record['prediction'])
        ctx.attributions = {k:Attribution(**v) for k,v in record['attributions'].items()}
        ctx.evaluation = {k:EvaluationMetrics(**v) for k,v in record['evaluation'].items()}
        ctx.selected_methods = ([MethodSelection(**item) for item in record['selection_details']]
                                if 'selection_details' in record else
                                [MethodSelection(name,None,None,'Priorité, durée estimée et motif indisponibles dans cet ancien enregistrement.') for name in record['selected_methods']])
        ctx.failed_attributions = record.get('failed_attributions', {'archive': 'Détails des échecs non enregistrés dans cette ancienne archive.'})
        ctx.logs = record['logs']
        if (group, record['profile']) in views:
            raise ValueError('Enregistrement dupliqué pour ce commentaire et ce profil.')
        available = {k:Attribution(**v) for k,v in record.get('computed_attributions',record['attributions']).items()}
        views[(group,record['profile'])] = context_html(ctx, available=available, failures=record.get('computation_failures',ctx.failed_attributions))
    profiles = {key: label for key, label in profiles.items() if any(p == key for _, p in views)}
    if not profiles:
        raise ValueError('Aucune explication à afficher.')
    default_profile = 'utilisateur_final' if 'utilisateur_final' in profiles else next(iter(profiles))
    return Dashboard('Comprendre un commentaire',groups,profiles,views,default_profile, 'Toxicité, facteurs de décision et explications adaptées au profil.')
