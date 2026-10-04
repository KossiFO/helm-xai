# Notebooks historiques et modules Colab

Les notebooks datés du 14 septembre sont conservés comme recettes historiques :
ils pointent vers des révisions de l’ancien dépôt privé. Ils ne constituent pas
encore un parcours public autonome pour la rc3. Pour installer le package actuel,
utiliser la wheel du dépôt public `KossiFO/helm-xai`, décrite dans le README racine.
Aucune donnée MAAF ni sortie exécutée n’est incluse dans cette copie publique.

---

# Test HELM sur MAAF dans Google Colab

Le notebook télécharge le code depuis une révision figée de la branche GitHub `codex/helm-tabulaire` ; aucune donnée MAAF n’est publiée dans ce dépôt. Le dépôt est privé : le premier téléchargement utilise un lien GitHub temporaire limité à l’archive de cette révision, saisi sans affichage. Il faut renouveler ce lien pour une nouvelle session après expiration ; aucun jeton de compte GitHub n’est envoyé à Colab.

Les modules `helm_colab/data.py` et `mixed_preprocessing.py` reprennent la préparation du test ProfileXAI précédent : 17 variables mixtes, coupure temporelle au 1er mars 2025, graine du modèle 43561735 et cible `bc_fraude_gmf`. Le modèle de démonstration est réentraîné dans Colab. Les colonnes de scores, Sentinelle et la cible sont exclues des prédicteurs. L’imputation et le vocabulaire sont appris sur l’entraînement. Les valeurs expliquées sont les valeurs préparées, les manquants quantitatifs étant imputés en amont.

HELM est installé depuis les sources GitHub dans un environnement Python 3.12 réservé au tabulaire, avec ses dépendances numériques figées. Les dépendances NLP ne sont pas installées dans cette recette ; le parcours texte n’y est pas testé. SHAP et LIME expliquent explicitement la classe 1. Le fond est un tirage aléatoire de 50 lignes d’entraînement, graine 42, différent du fond de couverture ProfileXAI : ne pas comparer directement les poids comme si le protocole était identique.

Les rapports sont descriptifs du modèle et du groupe sélectionné. Aucun seuil n’est ajusté ; aucun résultat de fidélité, de calibration ou d’utilité utilisateur ne découle du seul succès des calculs. Les fichiers individuels et modèles restent dans Colab ; seuls les constats agrégés et le code sont versionnés.

## Notebooks par profil

- [MAAF v2](HELM_MAAF_profils_v2_2026-09-14.ipynb) : quatre vues, restitution en français et méthodes proposées selon une règle explicite.
- [Toxicité v1](HELM_toxicite_profils_v1_2026-09-14.ipynb) : test séparé du modèle XLM-R fine-tuné toxicité, deux commentaires de démonstration, quatre profils HELM, pool SHAP/LIME/contrefactuels.

Le module natif `helm.tabular.profiles` définit les profils tabulaires ; `helm.visualization.profile_html` restitue les contextes du pipeline texte. Les helpers du notebook orchestrent ces APIs. Les notes utilisateurs et l’apprentissage de leurs préférences ne sont pas exercés dans ces recettes.

## Commentaire libre (a6)
Après récupération des sources GitHub et `toxicite.install()`, appeler
`toxicite.interactive()`. Saisir le texte puis **Prédire** (score seul) ou
**Expliquer** (cinq méthodes, puis présentation selon le profil). Le noyau Colab
doit rester connecté. Les runs libres sont distincts de la recette enregistrée,
et changer de profil relit les résultats sans recalcul ML. Les textes de plus de
512 tokens sont refusés pour éviter une explication sur texte tronqué.

La saisie utilise `helm.visualization.text_input.TextInput` et l’extra `[notebook]`.
Le backend XLM-R propre à cette démonstration se trouve dans les modules
`toxicity_live.py` et `toxicity.py`. La fusion historique du résumé API reste une
limite connue ; les vues HTML présentent séparément les méthodes.
