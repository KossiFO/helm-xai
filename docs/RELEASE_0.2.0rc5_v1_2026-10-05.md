# Version 0.2.0rc5

L’extension tabulaire ajoute `BinaryProbabilityAdapter.from_glm` et `from_gam`
pour expliquer les probabilités de GLM binomiaux statsmodels et GAM logistiques
pyGAM déjà ajustés. Le prétraitement entraîné est conservé ; le seuil de décision
et les libellés des deux classes sont explicites.

Les extras `boosting` et `statistics` installent les bibliothèques correspondantes.
Une recette reproductible teste XGBoost, LightGBM, GLM et GAM sur des données
synthétiques mixtes, avec SHAP/LIME et les quatre profils. Elle est ajoutée à la CI.
Voir [le guide](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/COMPATIBILITE_MODELES_v1_2026-10-05.md).

Le moteur de toxicité et l’interface restent inchangés. Cette version n’ajoute
pas la régression, les GLM avec formule/offset/exposition, les GAM non logistiques,
les Booster bruts ou Spark ML. Les tests logiciels ne valent pas validation
humaine ou métier ; Databricks reste à vérifier sur un Runtime cible.
