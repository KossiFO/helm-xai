# HELM — explications adaptées au profil

HELM (`helm-xai`) est une bibliothèque Python d’explicabilité pour la classification
de textes, en particulier la modération de contenus. Elle propose des explications
adaptées aux profils utilisateur, modérateur, expert technique et régulateur.
Une extension expérimentale prend en charge les classifieurs tabulaires.

## Installation

Python **3.10 à 3.12**. Pour installer la bibliothèque et son interface locale :

```bash
python -m pip install "helm-xai[ui]==0.2.0rc4"
helm-ui
```

Le navigateur ouvre `http://127.0.0.1:8767/`. Choisir un profil, saisir un commentaire
et cliquer sur **Expliquer**. La wheel inclut l’interface ; Node.js n’est pas requis.
Pour un autre port : `helm-ui --port 8771`. Arrêter le serveur avec Ctrl+C.

Le modèle de démonstration est entraîné sur **12 phrases synthétiques**. Il permet
d’essayer le parcours, mais ne constitue pas un détecteur de toxicité validé.
L’interface utilise les poids des modèles de texte déjà présents dans le cache
local ; elle ne les télécharge pas.

## Utilisation Python

```python
from helm import HELMPipeline, UserProfile
from helm.ui.models import DemoModel

pipeline = HELMPipeline(model_name="demo", model=DemoModel().load())
resultat = pipeline.explain(
    "Merci pour cette réponse utile",
    user_profile=UserProfile.END_USER,
    evaluate=True, num_features=8, evaluation_k=3,
)
print(resultat.prediction)
print(resultat.formatted_output.summary)
```

Le profil désigne le destinataire de l’explication ; il est choisi explicitement.
HELM ne déduit pas le profil de la personne à partir de son commentaire.

## Fonctionnalités

- Sélection et présentation d’explications par profil, avec backends effectivement exécutés identifiés.
- Attributions, règles locales et recherches de contrefactuels, selon les méthodes et dépendances disponibles.
- Interface locale avec historique, export JSON et avis liés à chaque explication individuelle.
- Extension `TabularHELM` pour les modèles scikit-learn avec `predict_proba`, variables numériques et catégorielles, classement des scores et sorties HTML par profil.

Les analyses restent dans le dossier de données utilisateur `helm-xai` de la
machine qui lance le serveur. `helm-ui --data-dir ./mes-analyses` permet de le changer.

## Dépendances optionnelles

| Extra | Usage |
|---|---|
| `ui` | Interface locale |
| `text` | Modèles de texte avec PyTorch et Transformers |
| `ig` | Integrated Gradients via Captum |
| `anchors` | Backend Anchors natif |
| `notebook` | Widgets Jupyter |
| `dev` | Outils de test et de construction |

Pour un modèle de texte et ces explicateurs :
`python -m pip install "helm-xai[text,ig,anchors,notebook]==0.2.0rc4"`.
Les poids des modèles sont à obtenir séparément, selon leurs licences.

## Documentation et exemples

- [API et fonctionnement](https://github.com/KossiFO/helm-xai/blob/codex/publication/implementation_v2/README.md)
- [Exemple tabulaire synthétique](https://github.com/KossiFO/helm-xai/blob/codex/publication/examples/tabular_usage.py)
- [Développement et tests](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/DEVELOPPEMENT.md)
- [Notes de version](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/RELEASE_0.2.0rc4_v1_2026-10-05.md)

## Périmètre et licence

**Version candidate de recherche.** Les tests logiciels vérifient les parcours et
les calculs sur des cas contrôlés ; ils ne constituent pas une validation humaine.
L’extension tabulaire est expérimentale et la compatibilité Databricks reste à
vérifier sur le Runtime cible. L’interface locale n’est pas un service d’étude hébergé.

Les explications décrivent le comportement du modèle, sans établir de causalité.
Les scores de méthodes différentes ne sont pas moyennés. Les métriques de fidélité
ne mesurent pas la satisfaction des utilisateurs.

Code sous [licence MIT](https://github.com/KossiFO/helm-xai/blob/codex/publication/LICENSE).
Les bibliothèques et modèles tiers conservent leurs licences respectives.
