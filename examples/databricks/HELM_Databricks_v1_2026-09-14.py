# Databricks notebook source
# MAGIC %md
# MAGIC # HELM — installer la wheel et expliquer un modèle tabulaire
# MAGIC Recette à exécuter dans votre Workspace. Ce fichier n’atteste pas encore une exécution sur Databricks.
# MAGIC La wheel doit être placée dans un chemin Workspace ou Volume autorisé. Aucun accès GitHub n’est nécessaire après son dépôt.

# COMMAND ----------
import sys
import os
import platform
from importlib.metadata import version, PackageNotFoundError

print({'python':sys.version, 'platform':platform.platform(), 'runtime':os.environ.get('DATABRICKS_RUNTIME_VERSION','non communiqué')})
for package in ['numpy','pandas','scikit-learn','shap','lime','helm-xai']:
    try:
        print(package, version(package))
    except PackageNotFoundError:
        print(package, 'non installé')
assert (3,10) <= sys.version_info[:2] < (3,13), 'Cette wheel déclare Python 3.10 à 3.12 ; sélectionner un runtime compatible.'

# COMMAND ----------
# MAGIC %md
# MAGIC ## Installation
# MAGIC Remplacer le chemin ci-dessous par l’emplacement autorisé de la wheel. Vérifier les dépendances proposées avant une installation sur un environnement métier. Le noyau tabulaire ne demande pas Torch/Transformers ; l’extra `[text]` les ajoute pour la toxicité. Un accès au dépôt de dépendances autorisé reste nécessaire, ou un wheelhouse préparé pour le runtime cible.

# COMMAND ----------
# MAGIC %pip install /Workspace/Shared/helm/helm_xai-0.2.0a3-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Vérification fonctionnelle avec données artificielles
# MAGIC Ce test ne charge aucun dossier réel et ne valide pas le modèle métier.

# COMMAND ----------
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from helm import TabularHELM

rng = np.random.RandomState(42)
X = pd.DataFrame({'montant':rng.uniform(50,5000,200), 'contrat':rng.choice(['auto','habitation'],200)})
y = ((X.montant > 2800) & (X.contrat == 'auto')).astype(int)
preparation = ColumnTransformer([('qual',OneHotEncoder(handle_unknown='ignore'),['contrat']),('quant','passthrough',['montant'])])
modele = make_pipeline(preparation,RandomForestClassifier(n_estimators=25,max_depth=4,random_state=42))
modele.fit(X.iloc[:150],y.iloc[:150])

# COMMAND ----------
helm = TabularHELM(modele,X.iloc[:150],target_class=1,background_size=20)
tableau = helm.dashboard(X.iloc[150:],k=5,labels=y.iloc[150:])
displayHTML(tableau.to_html())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Votre modèle
# MAGIC Remplacer le modèle et les DataFrames artificiels par votre pipeline scikit-learn déjà entraîné, un fond d’entraînement et un lot de test au format pandas. HELM n’entraîne pas votre modèle. Les catégories restent dans leur espace d’origine. Les Spark DataFrames et modèles natifs Spark ML ne sont pas pris en charge directement par cet adaptateur ; sélectionner un lot de taille maîtrisée avant conversion, sans collecte implicite de toute la table Spark.
