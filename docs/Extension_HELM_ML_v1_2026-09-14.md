# HELM : expliquer une classe risquée et les dossiers qui la composent

Version de travail du 14 septembre 2026 — extension Codex isolée, `0.2.0a1`, non publiée.

## Le besoin détermine l’extension

Un analyste veut comprendre pourquoi un dossier obtient un score élevé et quels facteurs reviennent dans les dossiers signalés. HELM explique déjà des modèles de ML sur du texte ; l’extension porte ici sur **les données tabulaires mixtes, la classe d’intérêt et les groupes de dossiers**. Le pipeline historique construit encore des labels « toxique / non toxique » et représente les attributions par mots (`helm/pipeline.py`, `helm/config.py`).

Trois éléments sont distincts : le score de la classe d’intérêt, la décision du modèle et l’étiquette observée. Les 10 scores les plus élevés peuvent tous être inférieurs au seuil de décision. HELM peut les expliquer pour la classe 1 sans les déclarer prédits 1. Cette séparation suit le contrat des classifieurs : les probabilités et la règle de décision sont deux étapes différentes ([scikit-learn](https://scikit-learn.org/stable/modules/classification_threshold.html)).

## Première extension native livrée

Le module `helm.tabular` travaille avec un classifieur déjà entraîné et ses variables originales. Il appelle directement SHAP ou LIME ; ProfileXAI, Gemini et un serveur FastAPI externe ne sont pas nécessaires.

| Fonction | Comportement de cette version | Limite explicite |
|---|---|---|
| Modèle | Classifieur mono-sortie avec `classes_`, `predict_proba`, `predict` ; pipeline scikit-learn conservé | Compatibilité vérifiée sur Random Forest et prétraitement mixte ; aucun support universel revendiqué |
| Variables | Quantitatives et qualitatives ; contributions par variable originale, avant one-hot | Dates à transformer explicitement ; pas de texte libre, image ou attribut composite |
| Classe d’intérêt | Étiquette explicite, binaire ou multiclasse, y compris une chaîne ; indépendante de la prédiction | Pas de régression, multilabel ou score d’anomalie dans ce jalon |
| Sélection | Top/bottom 5 ou 10 ; tous les dossiers, prédits cible, étiquetés cible ou autres classes | Les étiquettes doivent être disponibles et correctement alignées pour les filtres observés |
| Explication | SHAP Kernel sur le score de la classe cible ; LIME local pour cette même classe | Méthodes par perturbation ; combinaisons de valeurs parfois peu plausibles et coût croissant avec le nombre de variables |
| Synthèse | Contributions moyennes signées, importance moyenne absolue et fréquence des signes positifs | Descriptif du groupe sélectionné, uniquement sur les succès ; ni causalité ni profil représentatif de tous les fraudeurs |
| Présentation | Rapport HTML autonome et affichage notebook ; profils `metier` (8 variables locales) et `technique` (détail et paramètres) | Ces présentations restent à valider auprès des utilisateurs ; pas de sélection adaptative transférée du texte |
| Interface | Rapport produit par la bibliothèque, sans service distant | L’interface React et le service d’étude de la release a5 ne proposent pas encore ce parcours tabulaire |

### Utilisation avec votre modèle déjà entraîné

```python
from helm import TabularHELM

helm = TabularHELM(model, X_train, target_class=1)
rapport = helm.explain_top(X_test, k=10, labels=y_test, method="shap")
rapport.to_html("scores_risques.html")
```

`X_train` et `X_test` sont des DataFrames des variables d’entrée originales, sans cible. Le modèle inclut ses transformations, par exemple imputation et one-hot dans un `Pipeline`. `y_test` est facultatif ; une Series doit avoir exactement l’index de `X_test`. Pour des catégories codées par des nombres, fournir `categorical_features=[...]` explicitement.

```python
# Les décisions positives du modèle, toujours expliquées pour la classe 1.
positifs = helm.explain_top(X_test, k=10, group="predicted_target")
# Les scores les plus bas comme groupe de contraste.
bas = helm.explain_top(X_test, k=5, order="lowest")
# Les cas réellement étiquetés 1, indépendamment de la décision du modèle.
connus = helm.explain_top(X_test, k=10, group="observed_target", labels=y_test)
```

Le profil change le niveau de présentation, pas le calcul des contributions. Le rapport expose les erreurs ; aucune explication de remplacement n’est inventée. Les rapports conservent l’empreinte du fond et les versions des logiciels ; pour rejouer une explication, il faut également conserver le modèle et les données de référence dans leur environnement autorisé. Il contient les valeurs des dossiers : le conserver dans l’environnement autorisé pour ces données.

## Ce qu’il faut encore établir pour parler de « profils de fraude »

Le rang d’un dossier exprime un score du modèle. Une moyenne de contributions sur les top 10 répond à « qu’est-ce qui soutient ces scores ? ». Elle ne suffit pas à décrire les fraudes observées. L’étape suivante doit comparer, sur des données tenues à l’écart de l’apprentissage, vrais positifs, faux positifs, faux négatifs et vrais négatifs, avec des effectifs et des distributions des variables. Plusieurs sous-profils peuvent coexister et s’annuler dans une moyenne.

Pour MAAF, les scores issus d’un échantillon surreprésentant la fraude doivent être examinés sous l’angle de la calibration et de la prévalence de production. Les seuils se choisissent sur validation, selon le coût des erreurs et la capacité d’investigation, puis se figent avant l’évaluation. Le top k est un outil de priorisation ; il ne remplace pas cette évaluation. **Aucun nouveau résultat MAAF n’est revendiqué ici : ce jalon est testé sur des données synthétiques. Le prochain essai MAAF doit être exécuté dans Colab, conformément à la demande.**

La fidélité indique dans quelle mesure une explication rend compte du comportement du modèle. L’additivité SHAP est un contrôle arithmétique, pas une preuve de fidélité globale ni de causalité. LIME expose le R² de son approximation locale. Il faudra aussi étudier la stabilité entre graines/fonds, les dépendances entre variables et la sensibilité aux perturbations.

## Architecture de généralisation

Conserver trois couches, avec un contrat explicite à chaque frontière :

1. **Contexte** : tâche, modalités, schéma des variables, sortie à expliquer, profil et contraintes.
2. **Sélection** : filtrer les méthodes selon les capacités du modèle, puis appliquer une politique validée pour ce domaine. Ce prototype choisit SHAP ou LIME explicitement ; il n’exécute pas la matrice toxicité/UCB1.
3. **Présentation** : une explication structurée conserve classe, score, décision, unité des contributions, empreinte du fond de référence, versions des logiciels et contrôles de qualité ; une vue adapte le détail au lecteur.

La prochaine factorisation pourra faire partager ce contrat au texte et au tabulaire. Pour ce jalon, `TabularHELM` reste un point d’entrée séparé afin de préserver le pipeline des expériences publiées. Un backend Tree SHAP pourra accélérer les modèles d’arbres, à condition de vérifier la sortie expliquée et les hypothèses de dépendance ([documentation SHAP](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)).

## Ordre recommandé des extensions de recherche

Les directions du manuscrit v7 (§ perspectives, lignes 685–693) sont cohérentes, mais ne sont ni de même coût ni au même niveau de validation. Les priorités ci-dessous sont une proposition de travail, pas un résultat expérimental.

| Direction | Apport pour HELM | Travail concret et critère de réussite | Priorité |
|---|---|---|---|
| **Classification tabulaire et classe risquée** | Répond directement au besoin métier de scoring ; variables mixtes et groupes | Valider le prototype sur MAAF dans Colab ; comparer classes, erreurs de décision, fidélité/stabilité et temps ; intégrer ensuite le parcours à l’interface | **1 — commencé** |
| **Étude utilisateur élargie** | Vérifier si l’adaptation au profil améliore réellement les décisions | Pilote puis calcul de puissance ; critère principal de qualité de décision, décision horodatée avant questionnaire, présentation aveugle, ordre contrebalancé ; analyser la dépendance entre réponses d’un participant | **2 — après stabilisation du parcours** |
| **Autres domaines textuels** | Tester la transférabilité du principe de sélection et de présentation | Adapter labels, références et contraintes métier pour sentiments, désinformation ou documents ; recalibrer et évaluer chaque domaine séparément | **3 — réutilisation partielle du pipeline texte** |
| **Modèles plus grands / GPU** | Mesurer la faisabilité et le rapport coût/qualité des explications | Fixer tâches, modèles/versions et budgets ; mesurer latence, mémoire, taux d’échec et fidélité. Le GPU n’est pas en soi une validation scientifique | **4 — après contrat modèle stable** |
| **Encodeur-décodeur** | Étendre aux tâches produisant une séquence, telle une justification ou un résumé | Définir la sortie à expliquer (classe, token, séquence), adapter les attributions ; distinguer texte plausible et explication fidèle | **5 — chantier génération distinct** |
| **Multimodal** | Expliquer conjointement texte et image, et leurs interactions | Définir masquages valides et sortie cible ; contrôler les contributions propres à chaque modalité et leurs interactions | **6 — chantier exploratoire** |

Pour l’étude, `N > 100` peut être une cible de recrutement, mais ne dispense pas d’un calcul de puissance tenant compte de l’effet recherché, de la variabilité et du plan expérimental. Les évaluations d’explications isolées attribuent les notes à ces explications ; une note de présentation fusionnée porte sur la configuration complète. Comparaison figée et apprentissage adaptatif doivent constituer des conditions distinctes.

T5/mBART n’expliquent pas automatiquement leur propre décision parce qu’ils génèrent du texte. La distinction entre plausibilité et fidélité est établie dans la littérature ([Jacovi et Goldberg, 2020](https://aclanthology.org/2020.acl-main.386/)) ; des justifications peuvent omettre des facteurs qui ont influencé la réponse ([Turpin et al., 2023](https://arxiv.org/abs/2305.04388)). Mistral 7B, LLaMA 3.1, CamemBERT-large, GPT-4V et LLaVA restent des exemples de familles/versions cités dans le manuscrit ; éviter de les qualifier automatiquement d’« architectures de pointe » en 2026. Les interfaces de classification et de génération multimodale répondent à des contrats différents ([classification Transformers](https://huggingface.co/docs/transformers/en/tasks/sequence_classification), [image-texte vers texte](https://huggingface.co/docs/transformers/en/tasks/image_text_to_text)).

## Proposition de perspective à intégrer lors d’une prochaine révision

> Une première extension de HELM vise la classification tabulaire à variables mixtes, avec une classe d’intérêt indépendante de la décision du modèle et une analyse des explications par groupes de dossiers. Sa validation devra distinguer les facteurs associés aux scores du modèle des caractéristiques des cas réellement observés. Une étude utilisateur dimensionnée selon un critère principal de qualité de décision permettra ensuite d’évaluer l’intérêt de l’adaptation aux profils. La généralisation à d’autres domaines textuels et à des modèles plus grands nécessitera une nouvelle validation des politiques de sélection et des coûts de calcul. Les architectures génératives et multimodales constituent des perspectives distinctes : elles exigent de définir la sortie à expliquer et d’évaluer la fidélité des explications, au-delà de leur seule plausibilité.

Le manuscrit, les résultats publiés, la copie Claude et la release GitHub a5 restent inchangés.
