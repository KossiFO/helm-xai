# HELM sur Databricks — version 0.2.0rc6

Le package accepte Python 3.10 à 3.13 : NumPy 1.x sous 3.10–3.12 et
NumPy 2.x sous 3.13. Ces versions doivent être compatibles avec les dépendances
et les règles d’installation du Runtime choisi. Cette recette n’a pas été
exécutée sur Databricks et ne constitue pas une certification de Runtime.

Dans un notebook disposant d’un accès PyPI :

```python
%pip install helm-xai==0.2.0rc6
```

Dans une cellule distincte :

```python
dbutils.library.restartPython()
```

Puis, pour un classifieur déjà entraîné acceptant les DataFrames pandas :

```python
from helm import TabularHELM
explicateur = TabularHELM(modele, X_reference, target_class=1)
displayHTML(explicateur.dashboard(X_test, k=5).to_html())
```

La recette synthétique complète est dans
`examples/databricks/HELM_Databricks_v2_2026-10-12.py`.
Les Spark DataFrames et les modèles Spark ML ne sont pas directement pris en charge.
Si PyPI est inaccessible, installer la wheel depuis un emplacement autorisé,
avec un dépôt interne de dépendances compatible avec le Runtime.

Les guides et notebooks v1 du 14 septembre décrivent l’ancienne alpha a3 ;
leurs limites Python et instructions de publication sont historiques.
