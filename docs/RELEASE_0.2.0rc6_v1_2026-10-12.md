# Version 0.2.0rc6

HELM accepte désormais Python 3.13 avec NumPy 2. Python 3.10 à 3.12
conserve NumPy 1 afin de préserver les environnements existants. Python 3.14
n’est pas couvert par cette version.

La recette couvre l’interface locale, SHAP, LIME, les backends natifs Integrated
Gradients et Anchors, ainsi que XGBoost, LightGBM, GLM binomial et GAM logistique
sur des données synthétiques. Les workflows CI ajoutent Python 3.13 aux contrôles
de bibliothèque, d’interface et de modèles tabulaires.

Les versions contraintes de recette sont décrites dans
`requirements/constraints-py312.txt` et `requirements/constraints-py313.txt`.
Les tests de compatibilité logicielle ne constituent pas une validation de la
qualité d’un modèle de toxicité ni une étude utilisateur.
