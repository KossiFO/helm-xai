# Extension tabulaire

`TabularHELM` explique un classifieur déjà entraîné avec `predict_proba` sur des
DataFrames pandas. La classe cible est explicite. Les variables numériques et
catégorielles sont acceptées, avec un prétraitement cohérent avec celui du modèle.

`rank` classe les scores. `explain_top` explique un groupe sélectionné ;
`explain_profile` et `dashboard` proposent des rendus par profil. Les groupes de
scores extrêmes ne représentent pas toute la population. Les contributions ne
sont pas des causes, ni une preuve qu’une observation appartient à la classe risquée.

Exemple synthétique : `python examples/tabular_usage.py`.
