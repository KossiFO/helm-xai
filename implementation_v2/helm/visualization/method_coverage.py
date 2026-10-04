"""Explicit distinction between calculation coverage and profile selection."""
from html import escape

METHOD_LABELS = {'shap':'SHAP', 'lime':'LIME', 'integrated_gradients':'Integrated Gradients (IG)',
                 'anchors':'Anchors', 'counterfactual':'Contrefactuels'}


def is_verified_ig(attr):
    return (attr.metadata.get('effective_method') == 'integrated_gradients'
            and attr.metadata.get('mode') == 'verified_native'
            and attr.metadata.get('target_space') == 'probability'
            and attr.metadata.get('valid') is True)


def method_status(name, attribution, failures):
    if name in failures:
        failure = failures[name]
        if isinstance(failure, dict) and failure.get('metadata', {}).get('valid') is False:
            return 'Échec des contrôles numériques'
        return 'Échec du calcul'
    if attribution is None:
        return 'Non calculée'
    if attribution.metadata.get('valid') is False:
        return 'Échec des contrôles numériques'
    if name == 'integrated_gradients' and not is_verified_ig(attribution):
        return 'Backend différent du protocole IG natif'
    if name == 'counterfactual' and not attribution.metadata.get('counterfactuals'):
        return 'Calculée — aucune inversion trouvée'
    if name == 'anchors':
        meta = attribution.metadata
        if meta.get('precision', 0) < meta.get('threshold', .95):
            return 'Calculée — seuil de précision non atteint'
        if not meta.get('anchor_words'):
            return 'Calculée — règle vide'
    return 'Calculée'


def coverage_html(available, failures, selected):
    text = '<h3>Les cinq méthodes — état du calcul</h3><div class="table"><table><thead><tr><th>Méthode</th><th>État</th><th>Pour ce profil</th></tr></thead><tbody>'
    for name, label in METHOD_LABELS.items():
        status = method_status(name, available.get(name), failures)
        choice = 'Retenue par HELM' if name in selected else 'Complément consultable' if name in available or name in failures else 'Hors sélection'
        text += f'<tr><td>{label}</td><td>{escape(status)}</td><td>{choice}</td></tr>'
    return text + '</tbody></table></div><p>Les états distinguent le calcul et la sélection par profil. Consulter un complément ne modifie pas la sélection de HELM.</p>'
