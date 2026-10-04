"""Small notebook input, with caller-provided prediction and explanation backends."""
from html import escape

PROFILES = [('Utilisateur', 'utilisateur_final'), ('Modérateur', 'moderateur'),
            ('Expert', 'expert_technique'), ('Audit', 'regulateur')]


def validate_comment(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Saisissez un commentaire avant de lancer le calcul.')
    if len(text) > 4000:
        raise ValueError('Limite de cette démonstration : 4 000 caractères. Raccourcissez le commentaire.')
    return text


def prediction_html(payload):
    score = payload['prediction']['probabilities'][1]
    label = payload['prediction']['label']
    color = '#b45309' if score >= .5 else '#0369a1'
    return (f'<div style="background:#f0f6ff;color:#183153;padding:18px;border-radius:12px">'
            f'<blockquote>{escape(payload["text"])}</blockquote>'
            f'<b>Score de toxicité : <span style="color:{color}">{score:.2%}</span></b>'
            f'<p>Classe prédite : {escape(label)} · Seuil : 50 %.</p>'
            '<p>La prédiction porte sur le commentaire, pas sur son auteur.</p></div>')


class TextInput:
    """Callbacks: calculate(text, explain, progress) -> result; render(result, profile) -> HTML."""

    def __init__(self, calculate, render):
        import ipywidgets as w
        self.calculate, self.render = calculate, render
        self.result = None
        self.busy = False
        self.text = w.Textarea(placeholder='Écrivez votre propre commentaire ici…',
                               layout=w.Layout(width='100%', height='110px'))
        self.profile = w.Dropdown(options=PROFILES, description='Profil :')
        self.predict = w.Button(description='Prédire', button_style='info', icon='play')
        self.explain = w.Button(description='Expliquer', button_style='primary', icon='search')
        self.status = w.HTML('<p>Saisissez un texte, puis cliquez sur Prédire ou Expliquer.</p>')
        self.output = w.HTML()
        self.box = w.VBox([w.HTML('<h2>HELM — votre commentaire</h2><p>Prédire calcule le score. Expliquer calcule aussi les cinq méthodes ; cela peut prendre plusieurs minutes. Une session Python connectée est nécessaire.</p>'),
                           self.text, self.profile, w.HBox([self.predict, self.explain]), self.status, self.output])
        self.predict.on_click(lambda _: self.submit(False))
        self.explain.on_click(lambda _: self.submit(True))
        self.text.observe(self._changed, names='value')
        self.profile.observe(self._profile_changed, names='value')

    def _changed(self, change):
        self.result = None
        self.output.value = ''
        self.status.value = '<p>Texte modifié : lancez un nouveau calcul.</p>'

    def _profile_changed(self, change):
        if self.result is not None and not self.busy:
            self.output.value = ''
            try:
                self.output.value = self.render(self.result, self.profile.value)
                self._progress('Profil affiché ; résultats conservés sans recalcul.')
            except Exception as error:
                self._progress('Affichage indisponible pour ce profil : ' + str(error))

    def _progress(self, message):
        self.status.value = '<p>' + escape(message) + '</p>'

    def submit(self, explain):
        if self.busy:
            return
        try:
            text = validate_comment(self.text.value)
        except ValueError as error:
            self._progress(str(error))
            return
        self.busy = True
        self.result = None
        self.output.value = ''
        for control in (self.text, self.profile, self.predict, self.explain):
            control.disabled = True
        try:
            self._progress('Chargement du modèle et calcul dans Colab…')
            self.result = self.calculate(text, explain, self._progress)
            self.output.value = self.render(self.result, self.profile.value)
            self._progress('Calcul terminé. Vous pouvez changer de profil ou saisir un autre commentaire.')
        except Exception as error:
            self.result = None
            self._progress('Calcul interrompu : ' + str(error))
        finally:
            self.busy = False
            for control in (self.text, self.profile, self.predict, self.explain):
                control.disabled = False

    def show(self):
        from IPython.display import display
        display(self.box)
        return self
