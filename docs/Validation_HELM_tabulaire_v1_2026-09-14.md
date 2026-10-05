# Validation logicielle HELM tabulaire — 14 septembre 2026

Version `0.2.0a1`, branche Codex isolée, non publiée. Cette recette utilise des données synthétiques et ne constitue ni une expérience sur données réelles ni une validation utilisateur.

## Résultats vérifiés

- **134 tests passent** dans la suite complète (14 tests tabulaires, 120 tests préexistants), Python 3.12, scikit-learn 1.4.2, SHAP 0.49.1. Les quatre avertissements globaux concernent des dépréciations SHAP/Starlette.
- Les 14 tests tabulaires sont rejoués après les dernières modifications de traçabilité du fond.
- Ruff sur le module tabulaire, ses tests et les exemples : aucune erreur.
- Contrôles : classe expliquée indépendante de la décision, classe textuelle/multiclasse, contributions dans les variables originales, catégories rares/inconnues, données manquantes, filtres et groupes vides, alignement des étiquettes, absence de mutation, restauration du générateur aléatoire SHAP, échec partiel de cohorte et échappement HTML.
- La relecture adversariale a détecté puis vérifié la correction d’une confusion entre `None` et `NaN`. Le cas de régression exige une base SHAP de **0,25** et une contribution de **−0,25** ; la reconstruction du fond et de la ligne doit conserver les probabilités à `1e-10` près.

## Distribution et démonstration

La wheel et l’archive source ont été construites. La wheel a été installée par pip dans un dossier temporaire hors des sources, sans réinstaller les dépendances de l’environnement de recette. L’import vérifié vient de `/private/tmp/helm-tabular-installed-final/helm/__init__.py`, version `0.2.0a1`. Ce contrôle n’est pas une résolution de dépendances dans un environnement vierge. Les fichiers d’interface compilés sont repris à l’identique de la release a5 puisque leurs sources ne sont pas modifiées. Le nouveau module ne remplace pas le parcours texte de l’interface locale.

Le script `examples/tabular_usage.py` entraîne un Random Forest de démonstration avec 2 variables quantitatives et 2 qualitatives, sur 140 observations artificielles, puis sélectionne les 5 plus hauts scores parmi 60 observations test. Les 5 explications SHAP ont réussi depuis la wheel installée ; les 5 scores les plus faibles ont également été classés. La recette FastAPI existante valide la page d’accueil, ses fichiers statiques, une explication texte pédagogique, le feedback et sa persistance dans une base temporaire. Ces effectifs décrivent une recette logicielle, pas une performance métier.

## Portée

Les tests n’établissent pas une compatibilité universelle avec tous les estimateurs, ni la fidélité ou l’utilité métier des explications. L’additivité SHAP est un contrôle arithmétique ; LIME fournit un R² local et ses avertissements numériques. Les catégories peuvent générer des combinaisons peu plausibles lors des perturbations. Les deux profils règlent seulement la présentation ; aucune politique profil/méthode n’a été calibrée sur le tabulaire.

Le prochain essai sur données réelles doit être réalisé dans Colab, avec données mixtes, score de classe 1, prédiction et étiquette observée séparés, groupes à forts/faibles scores et analyse des erreurs de décision.

## Relecture

Audit `qa-these` : ACCEPTÉ pour livraison comme prototype expérimental, avec correction revérifiée du défaut None/NaN. Les remarques mineures ont été traitées : README distinguant a5 publiée et a1 expérimentale ; empreinte et versions présentes dans l’objet et dans les rapports HTML des deux profils. Le test d’export métier vérifie leur présence.
