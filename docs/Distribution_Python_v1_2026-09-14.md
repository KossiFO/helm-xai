# HELM — distribution Python publique visée

L’objectif est une bibliothèque publique `helm-xai`, importée avec `import helm`, distribuée sur PyPI. GitHub héberge les sources et les exemples ; PyPI distribuera les versions installables. **La version 0.2.0a3 est préparée en wheel et archive source, mais n’est pas publiée sur PyPI.**

## Usage disponible dans la wheel

Après installation de la wheel, sur Python 3.10–3.12 :

```python
from helm import TabularHELM

helm = TabularHELM(model, X_reference, target_class=1)
vue = helm.dashboard(X_test, k=5)
vue                         # Jupyter / Colab
vue.to_html("explications.html")  # fichier autonome, profils cliquables
```

`model` est un classifieur scikit-learn déjà entraîné, éventuellement une Pipeline, avec `predict_proba`. Les colonnes de `X_reference` et `X_test` doivent correspondre à ses entrées. Les variables numériques et catégorielles sont acceptées. La classe expliquée est explicite ; un score élevé ne signifie pas nécessairement une prédiction 1.

Le tableau de bord contient quatre profils, les scores élevés et faibles, ainsi que les explications SHAP/LIME adaptées au profil. Les onglets HTML ne nécessitent ni JavaScript, ni serveur web, ni rappel au noyau Python. Les calculs SHAP/LIME restent nécessaires pour de nouvelles données. L’extension tabulaire est expérimentale ; elle ne transforme pas les contributions en causes de fraude.

## Dépendances selon l’usage

- Installation de base : ML tabulaire, SHAP, LIME, tableaux de bord HTML.
- Extra `[text]` : Torch et Transformers pour les modèles de texte.
- Extra `[ui]` : serveur local `helm-ui` et interface embarquée.
- Extra `[ig]` : Captum ; extra `[viz]` : graphiques matplotlib/seaborn.
- Extra `[all]` : ensemble des composants optionnels.

L’interface incluse ne constitue pas un hébergement public : un service en ligne avec calcul nécessite son propre serveur. La wheel est portable au niveau Python ; ses dépendances et les modèles doivent aussi être accessibles sur l’environnement cible. Databricks dispose d’un exemple séparé, non encore exécuté sur un cluster.

## Dernières étapes avant publication publique

1. Confirmer les droits de diffusion et choisir une licence explicite, puis modifier `DROITS.md`, ajouter `LICENSE` et retirer le marqueur `Private :: Do Not Upload`.
2. Configurer le projet PyPI et son éditeur de confiance GitHub Actions (Trusted Publishing), sans secret dans le code.
3. Vérifier les distributions avec `twine check`, les installer hors sources, puis essayer une publication TestPyPI et une installation propre.
4. Publier une préversion, avec ses limites et des exemples publics synthétiques. Une version alpha n’est pas une validation scientifique ni une garantie tous modèles/tous environnements.

La commande future visée est `pip install helm-xai`. Elle ne doit pas être présentée comme fonctionnelle avant la publication effective. Le nom observé libre ne constitue pas une réservation.

Références : [guide officiel de packaging](https://packaging.python.org/en/latest/tutorials/packaging-projects/), [outils et Trusted Publishing](https://packaging.python.org/en/latest/guides/tool-recommendations/), [bibliothèques notebook Databricks](https://docs.databricks.com/aws/en/libraries/notebooks-python-libraries).
