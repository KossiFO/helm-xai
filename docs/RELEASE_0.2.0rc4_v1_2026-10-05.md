# Version 0.2.0rc4

La présentation publique est recentrée sur l’installation, les fonctionnalités,
les exemples synthétiques et les limites. Les documents de travail internes et
les recettes historiques spécifiques à un projet sont retirés de cette distribution.
Les liens de documentation sont absolus pour fonctionner depuis PyPI.

Le moteur de calcul est inchangé. Le test des catégories manquantes impose un
dtype objet pour conserver la distinction entre None et NaN, que l’inférence
automatique de pandas 3 normalise avant l’appel à HELM.

Cette version reste une candidate de recherche. La validation humaine, les
modèles réels et les environnements Databricks nécessitent leurs propres évaluations.
