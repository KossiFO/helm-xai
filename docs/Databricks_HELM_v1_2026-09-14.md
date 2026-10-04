# HELM sur Databricks : distribution et périmètre

HELM est un package Python installable. La version 0.2.0a3 est distribuée en wheel `helm_xai-0.2.0a3-py3-none-any.whl`. GitHub conserve les sources ; l’environnement Databricks peut installer la wheel depuis un fichier Workspace ou un Volume autorisé, sans cloner GitHub.

Le noyau tabulaire requiert Python 3.10–3.12, NumPy <2, pandas, scikit-learn, SHAP et LIME. Torch et Transformers sont désormais dans l’extra `[text]` pour la toxicité. Les dépendances de votre Runtime doivent être vérifiées avant installation. La wheel seule n’embarque pas ces dépendances ni les poids des modèles ; pour un environnement sans Internet, préparer un dépôt interne ou un wheelhouse correspondant au Runtime/OS/architecture choisi.

Exemple d’installation dans un notebook :

```python
%pip install /Workspace/Shared/helm/helm_xai-0.2.0a3-py3-none-any.whl
```

Puis, dans une cellule distincte :

```python
dbutils.library.restartPython()
```

Après installation, avec un modèle/pipeline scikit-learn déjà entraîné et des DataFrames pandas :

```python
from helm import TabularHELM
helm = TabularHELM(modele, X_reference, target_class=1)
displayHTML(helm.dashboard(X_test, k=10).to_html())
```

Le dashboard HTML embarque toutes ses vues et ses styles, sans CDN ni widget Colab. Les profils et groupes se changent dans le navigateur, sans recalcul. Un export contient les valeurs individuelles affichées : le conserver dans un emplacement adapté à ces données.

La recette `examples/databricks/HELM_Databricks_v1_2026-09-14.py` commence par un diagnostic et utilise des données artificielles. Aucun accès à votre Workspace n’a été utilisé : la compatibilité réelle reste à vérifier sur votre Runtime, son mode d’accès, les autorisations d’installation et les dépendances présentes. Le support actuel cible les modèles scikit-learn/predict_proba sur pandas ; il ne signifie pas prise en charge directe de modèles Spark ML ou de calculs distribués Spark.

Références officielles consultées le 14/09/2026 : [installation par notebook](https://docs.databricks.com/aws/en/libraries/notebooks-python-libraries), [fichiers Workspace](https://docs.databricks.com/aws/en/libraries/workspace-files-libraries), [rendu displayHTML](https://docs.databricks.com/aws/en/notebooks/notebook-media). Les possibilités exactes dépendent du Runtime et des politiques de l’organisation.
