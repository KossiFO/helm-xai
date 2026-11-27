# HELM — explications adaptées au profil

HELM (`helm-xai`) est une bibliothèque Python d’explicabilité pour la classification
de textes, en particulier la modération de contenus. Elle propose des explications
adaptées aux profils utilisateur, modérateur, expert technique et régulateur.
Une extension expérimentale explique les prédictions de modèles entraînés sur
des tableaux de variables numériques et catégorielles, notamment **Random Forest, XGBoost et LightGBM**, ainsi que les **GLM binomiaux
et GAM logistiques** avec un adaptateur fourni par HELM.

## Premier essai et retour utilisateur

**Commencez par le [guide testeur](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/TESTER_HELM.md)** : installation isolée sous Windows/macOS/Linux,
exemples à essayer, résultats attendus et procédure pour transmettre un retour.
Cet essai s’adresse aux personnes qui utilisent Python. Il ne constitue pas une étude scientifique.

Python **3.10 à 3.13** (pas 3.14). Dans un environnement virtuel :

```bash
python -m pip install "helm-xai[ui]==0.2.0rc7"
python -m helm.ui
```

Le navigateur ouvre `http://127.0.0.1:8767/`. Garder **Démonstration locale**, choisir
**Remerciement**, puis cliquer sur **Expliquer**. Passer ensuite au profil **Expert technique**
pour explorer une autre présentation. Ni GPU, ni compte, ni clé d’API, ni Node.js ne sont nécessaires.
Laisser le terminal ouvert ; arrêter avec Ctrl+C. Pour un autre port : `python -m helm.ui --port 8771`.

Le modèle de démonstration est entraîné sur **12 phrases synthétiques**. Il permet
d’essayer le parcours, mais ne constitue pas un détecteur de toxicité validé.
Les autres modèles utilisent des poids déjà présents dans le cache local ;
la démonstration n’a pas besoin de les télécharger.

### Transmettre un retour

L’onglet **Tester et faire un retour** permet de rédiger et télécharger un rapport
avec les versions utiles et les notes de la dernière analyse affichée, sans ajouter
le texte analysé. Relisez-le, puis partagez-le :

- [GitHub Issues](https://github.com/KossiFO/helm-xai/issues/new/choose) : problème ou retour d’essai (public, compte GitHub requis).
- Courriel à **follyametepkossi@gmail.com**, objet « Retour de test HELM » (aucun compte GitHub nécessaire).

**Les notes saisies dans l’interface restent locales et ne nous sont pas envoyées.**
Elles peuvent influencer les sélections suivantes sur votre poste. Les retours
sur le logiciel ne démontrent pas l’efficacité de l’adaptation aux utilisateurs.
L’export complet d’une analyse contient le texte saisi : ne publiez pas de données confidentielles.
Pour un diagnostic de versions : `python -m helm.ui --diagnostic`.

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
- Extension `TabularHELM` : explications de prédictions individuelles, classement des scores élevés/faibles et sorties HTML par profil pour des données tabulaires.

Les analyses restent dans le dossier de données utilisateur `helm-xai` de la
machine qui lance le serveur. `helm-ui --data-dir ./mes-analyses` permet de le changer.

## Modèles tabulaires : compatibilité et périmètre

HELM **explique un modèle déjà entraîné** ; il ne l’entraîne pas à votre place.

| Modèle | Prise en charge dans cette version |
|---|---|
| **Random Forest** (`RandomForestClassifier`) | Testé sur des données synthétiques, y compris avec des variables numériques et catégorielles dans une Pipeline scikit-learn. |
| **XGBoost** (`XGBClassifier`) et **LightGBM** (`LGBMClassifier`) | Testés en classification binaire sur données synthétiques mixtes, dans une Pipeline scikit-learn. |
| **GLM binomial** (statsmodels) | Testé avec `BinaryProbabilityAdapter.from_glm`, fourni dans HELM. |
| **GAM logistique** (`pyGAM.LogisticGAM`) | Testé avec `BinaryProbabilityAdapter.from_gam`, fourni dans HELM. |
| Autres classifieurs compatibles scikit-learn | À vérifier individuellement ; aucune compatibilité universelle n’est revendiquée. |

Le modèle ou la Pipeline doit fournir `predict`, `predict_proba` et `classes_`,
et accepter les DataFrames pandas transmis avec le même schéma et le même
prétraitement qu’à l’entraînement. La classe à expliquer est choisie explicitement.
Les modèles de régression, les objets XGBoost `Booster` bruts et les modèles
Spark ML ne sont pas directement couverts par cet adaptateur.

Les tests couvrent les probabilités, les classements, SHAP/LIME et les quatre
profils. [Exemples et conditions précises](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/COMPATIBILITE_MODELES_v2_2026-10-12.md).

## Dépendances optionnelles

| Extra | Usage |
|---|---|
| `ui` | Interface locale |
| `text` | Modèles de texte avec PyTorch et Transformers |
| `ig` | Integrated Gradients via Captum |
| `anchors` | Backend Anchors natif |
| `notebook` | Widgets Jupyter |
| `boosting` | XGBoost et LightGBM |
| `statistics` | GLM binomial statsmodels et GAM logistique pyGAM |
| `dev` | Outils de test et de construction |

Pour un modèle de texte et ces explicateurs :
`python -m pip install "helm-xai[text,ig,anchors,notebook]==0.2.0rc7"`.
Les poids des modèles sont à obtenir séparément, selon leurs licences.

## Documentation et exemples

- [API et fonctionnement](https://github.com/KossiFO/helm-xai/blob/codex/publication/implementation_v2/README.md)
- [Exemple tabulaire synthétique](https://github.com/KossiFO/helm-xai/blob/codex/publication/examples/tabular_usage.py)
- [Installation et recette Databricks (à valider sur votre Runtime)](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/Databricks_HELM_v2_2026-10-12.md)
- [Développement et tests](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/DEVELOPPEMENT.md)
- [Notes de version](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/RELEASE_0.2.0rc7_v1_2026-11-27.md)

Les documents versionnés du 14 septembre décrivent les anciennes versions alpha ;
les instructions ci-dessus et la note rc7 s’appliquent à cette distribution.

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
