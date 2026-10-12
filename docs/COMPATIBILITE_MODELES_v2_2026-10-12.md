# Modèles tabulaires : utilisation et vérification

HELM explique les prédictions d’un modèle entraîné. Les exemples ci-dessous
conservent le prétraitement ajusté sur l’entraînement. Les explications portent
sur les colonnes originales, numériques et catégorielles.

## XGBoost et LightGBM

Installer `helm-xai[boosting]==0.2.0rc6`, puis transmettre à `TabularHELM` une
Pipeline scikit-learn déjà ajustée, terminée par `XGBClassifier` ou
`LGBMClassifier` :

```python
from helm.tabular import TabularHELM

helm = TabularHELM(pipeline_entrainee, X_train, target_class=1)
rapport = helm.explain_profile(X_test, profile="metier", k=5,
                               group="predicted_target")
rapport  # sortie HTML colorée dans un notebook
```

Les autres profils sont `utilisateur`, `technique` et `audit`.
`helm.dashboard(X_test, k=5)` propose les profils et scores élevés/faibles.
Les noms de colonnes, leur ordre et le prétraitement doivent correspondre
à l’entraînement. Le `Booster` brut et Spark ML nécessitent un autre adaptateur.

## GLM binomial et GAM logistique

Installer `helm-xai[statistics]==0.2.0rc6`. L’adaptateur est fourni par HELM :

```python
from helm.tabular import BinaryProbabilityAdapter, TabularHELM

# result_glm : résultat de statsmodels.GLM(..., family=Binomial()).fit()
modele = BinaryProbabilityAdapter.from_glm(result_glm, transformer=pretraitement)

# Ou gam : pyGAM.LogisticGAM déjà ajusté
modele = BinaryProbabilityAdapter.from_gam(gam, transformer=pretraitement)

helm = TabularHELM(modele, X_train, target_class=1)
rapport = helm.explain_profile(X_test, profile="audit", k=5)
```

Choisir une seule des deux affectations de `modele`. `pretraitement` est le
transformeur **déjà ajusté** qui produit la matrice numérique d’entraînement ;
HELM appelle uniquement `transform`, jamais `fit`. Utiliser une sortie dense,
par exemple `OneHotEncoder(sparse_output=False)`.

Pour GLM, la valeur par défaut `add_intercept=True` ajoute une colonne de 1 en
première position, comme `sm.add_constant(Z, has_constant="add")`. Si cette
colonne était absente ou est déjà produite par le transformeur, préciser
`add_intercept=False`. Les GLM avec formule, offset ou exposition sont refusés.
Les GAM non logistiques et les GLM d’autres familles ne sont pas couverts.

`classes=(0, 1)` correspond au codage d’entraînement : le deuxième élément est
l’événement dont le modèle renvoie la probabilité. Pour d’autres libellés,
indiquer par exemple `classes=("autre", "cible")` et `target_class="cible"`.
Le seuil `threshold=0.5` détermine la décision de l’adaptateur ; le modifier ne
change pas les probabilités expliquées. Les valeurs exactement au seuil sont
classées dans la deuxième classe.

## Recette reproductible

Depuis le dépôt, après installation des extras `boosting,statistics` :

```bash
python examples/verify_classifiers.py --output /tmp/helm-models
```

La préparation est séparée dans `examples/classifier_models.py`.
La recette utilise 320 observations synthétiques (240 entraînement, 80 test),
deux variables numériques et une catégorielle, graine 42. Elle contrôle les
probabilités contre le modèle initial, les cinq scores les plus hauts et bas,
le filtrage des prédictions 1, les explications SHAP/LIME d’un cas prédit 1,
les quatre profils et la conservation des probabilités après explication.
Le JSON conserve les versions, le résidu additif SHAP et le R² local LIME.

Les attributions utilisent `shap.KernelExplainer` et `LimeTabularExplainer`,
pas TreeSHAP ni une décomposition des termes GAM. L’égalité additive SHAP est
un contrôle numérique ; le R² local LIME est une mesure de l’approximation.
Ces tests établissent une compatibilité logicielle sur ces cas synthétiques,
sans validation métier, causale, humaine ou Databricks.

Contrats tiers : [prédiction GLM statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.predict.html)
et [API pyGAM](https://pygam.readthedocs.io/en/latest/api/).

Recettes du 12 octobre 2026 sous Python 3.12 / NumPy 1 et Python 3.13 / NumPy 2 :
les quatre modèles passent les contrôles décrits ci-dessus. Les résultats peuvent
varier selon les versions des bibliothèques et les perturbations. La réussite
des contrôles ne signifie pas que LIME reproduit parfaitement le modèle.
