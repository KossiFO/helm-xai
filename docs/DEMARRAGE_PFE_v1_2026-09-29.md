# HELM — Point de départ pour les expérimentations PFE

Version locale 0.2.0a7, préparée depuis `45d24b3` (alpha 6). Ce module prépare des stimuli pour une étude ; ce n’est pas une plateforme de collecte humaine complète.

## Ce qui est disponible

- Bibliothèque HELM, interface locale et exemples notebook de l’alpha 6.
- Nouveau module `helm.study` : calcul direct d’une seule méthode choisie par le protocole, sans UCB1, accès aux notes antérieures ni fusion d’attributions.
- SHAP positionnel et occlusion individuelle LOO : le backend LOO **remplace chaque mot par « le »**, en repartant du texte original. Ce n’est pas une suppression pure. La substitution et la classe cible sont conservées dans les métadonnées.
- Scores par position conservant les répétitions ; même convention de rendu normalisée à l’intérieur du stimulus. Les valeurs brutes restent disponibles, sans prétendre à des unités équivalentes entre méthodes.
- Exports de recherche avec configuration, version, empreinte SHA256, prédiction, coûts et échecs explicites. Aucun repli automatique en cas d’échec ; aucun écrasement d’un export existant.
- Vue destinée au participant, sans méthode, profil, référence humaine ni identifiant de cas. L’outil de passation doit utiliser des identifiants opaques et ne jamais servir les exports complets.
- Exactitude équilibrée du contrôle et score de Brier descriptif. Brier n’est pas une mesure pure de calibration ; ajouter des courbes confiance/exactitude et une analyse par participant/message.

## Recette de prise en main

Depuis les sources remises par l’encadrement, avec Python 3.12 :

```sh
python -m pip install -e .
python examples/pfe_fixed_stimuli.py --output ./sorties_pfe_demo
```

Le script utilise une règle synthétique locale. Il produit quatre explications (deux textes × deux méthodes), aucun résultat humain. Les deux textes servent à vérifier l’outil ; ils ne constituent pas un corpus expérimental. La configuration `FixedStudyConfig` est immuable et choisit la méthode explicitement. La référence du modèle reste déclarative : l’encadrement doit vérifier et figer le modèle réellement transmis.

Pour XLM-R, installer l’extra `[text]` et passer `model.predict_proba` au même point d’entrée. Les calculs sur modèle réel sont à exécuter dans l’environnement prévu (par exemple Colab) après vérification de ses versions et de sa révision ; aucune nouvelle recette XLM-R n’est revendiquée pour cette alpha.

## Répartition du travail étudiant

**PFE 1 :** choisir une méthode et garder ses sorties identiques, concevoir deux rendus contrôlés, préparer sondage/profil/questionnaire, randomiser la présentation, conduire et analyser les tests.

**PFE 2 :** produire les deux méthodes à présentation constante, constituer une cohorte de développement et une cohorte de test distinctes, figer la règle par profil et la référence uniforme avant la collecte finale. Conserver séparément préférence et réussite objective.

La collecte, le consentement, l’affectation aléatoire, le contrôle d’accès et les analyses statistiques de l’étude restent à développer et valider. `helm-ui` conserve son fonctionnement de démonstration avec retours UCB1 ; il ne devient pas un instrument contrôlé grâce au seul ajout de `helm.study`.

## Limites à conserver dans l’annexe

Le résumé de certaines vues historiques du pipeline combine encore des scores inter-méthodes ; ce chemin n’est pas utilisé par `helm.study`. La nouvelle voie ne corrige pas l’ensemble de l’API historique. Les profils d’affichage prédéfinis ne prouvent pas une adaptation utile ni apprise. La réussite de la recette logicielle ne démontre aucune efficacité humaine. Le package reste une préversion privée ; aucune publication PyPI n’est annoncée.

Référence comparative possible : [ProfileXAI, article v1](https://arxiv.org/abs/2510.22998v1) et [dépôt public](https://github.com/MysticDeepAI/Herramienta-de-Explicabilidad). Son adaptation narrative est une piste de comparaison documentaire, pas une intégration déjà compatible ni un substitut aux tests utilisateurs prévus.
