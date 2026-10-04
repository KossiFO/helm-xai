# Explications par profil — HELM 0.2.0a2

L’API `helm.explain_profile(X, profile="metier", k=10)` choisit les méthodes selon une règle explicite et produit un rapport HTML autonome. Elle réutilise le modèle fourni ; elle ne l’entraîne pas.

| Profil tabulaire | Méthodes proposées | Contenu |
|---|---|---|
| utilisateur | SHAP | Facteurs positifs/négatifs, score de référence, autres contributions |
| metier | SHAP | Synthèse du groupe, facteurs individuels, éléments à examiner |
| technique | SHAP, LIME | Attributions, qualité, comparaison de facteurs sur les mêmes dossiers réussis |
| audit | SHAP, LIME | Registre de couverture, erreurs, classe, fond et paramètres |

La politique `tabular_profiles_v1` est une règle de conception à évaluer auprès des utilisateurs. Elle n’est pas apprise. Les poids des méthodes ne sont pas fusionnés ; une contribution décrit le modèle et ne prouve pas la fraude. Les notebooks peuvent précalculer les deux méthodes puis présenter leur sous-ensemble selon le profil sans recalcul ni changement des contributions.

Les cellules d’usage restent courtes ; installation, préparation et audit sont dans `examples/colab/helm_colab`. Les notebooks MAAF v2 et toxicité v1 téléchargent une révision GitHub précise. Le dépôt étant privé, l’URL temporaire d’archive doit être renouvelée après expiration ; aucun jeton de compte n’est stocké dans le notebook.

Le test toxicité utilise le pipeline HELM et sa matrice historique sur un pool explicite de méthodes réellement exécutables : SHAP, LIME, contrefactuels. Ce pool est filtré avant le budget. Les heuristiques de temps ne constituent pas des garanties. Les préférences restent vierges ; aucune note humaine n’est simulée. Les vues utilisateur/modérateur affichent aussi la qualité LIME. Les résultats sur deux commentaires de démonstration ne constituent pas un benchmark ni une étude humaine.
