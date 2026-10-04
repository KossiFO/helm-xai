# Vérification HELM 0.1.0a5 — 10 septembre 2026

Recette de l’auteur sur macOS ARM64 et Python3.12 :

- 115 tests Python réussis, couvrant moteur, API, persistance et CLI ;
- 8 tests JavaScript réussis ;
- analyse Ruff réussie ;
- interface React compilée avec ses notices de composants tiers ;
- installation réelle de la wheel avec `ui` et `httpx` dans un environnement vierge,
  distinct des sources, sous les contraintes de recette ; `pip check` sans conflit ;
- recette de la wheel réussie : page et assets, licences, explication pédagogique,
  note remplacée sans double comptage, export et persistance après réouverture ;
- contrôle `twine check` réussi sur wheel et archive source.

La suite a produit quatre avertissements de dépréciation provenant de bibliothèques
externes (SHAP et Starlette/httpx). Aucun échec de test n’a été observé.
Ces vérifications utilisent des données fabriquées. Elles ne remplacent ni la
campagne expérimentale complète ni une évaluation par des utilisateurs.

La CI GitHub est fournie ; aucune exécution distante n’est attestée tant que le
dépôt et le workflow n’ont pas été publiés. La recette documentée vise Python3.12 ;
les versions Python3.10/3.11 déclarées compatibles ne sont pas certifiées par cette
exécution. Aucun déploiement Internet de l’interface n’est inclus.
