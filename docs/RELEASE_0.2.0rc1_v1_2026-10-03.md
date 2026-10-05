# HELM 0.2.0rc1 — candidate de distribution

Cette version réunit le package texte/tabulaire, l’interface locale, la saisie libre Colab et les stimuli figés `helm.study`. Elle reste privée ; aucune étude humaine ni publication PyPI n’est revendiquée.

## Corrections

- Les résumés de l’API présentent les mots séparément pour chaque méthode. Aucune moyenne SHAP/LIME/contrefactuels ; le consensus est une intersection lexicale de même signe entre au moins deux méthodes, sans score fusionné.
- Les recherches de contrefactuels et règles Anchors n’obtiennent plus de métrique C/S calculée sur des poids auxiliaires. Les variantes et leurs inversions, ou précision/couverture des règles, restent disponibles. La classe de chaque attribution détermine sa métrique ; les deltas de suppression contrefactuels restent explicitement orientés vers la toxicité, distincts de la cible de recherche.
- Toute exception de backend apparaît dans les échecs du résultat API. Les contrôles numériques IG restent obligatoires lorsqu’IG vérifié est demandé.
- LIME tabulaire indique le nombre de lignes du fond utilisé pour estimer quartiles et fréquences ; cela ne démontre ni la qualité du modèle ni l’absence de fuite métier.
- La démonstration `[ui]` fonctionne sans installation des modèles texte. Les modèles en cache sont indisponibles si leurs dépendances texte ne sont pas installées.
- La recette Colab future mesure les méthodes sans cache partagé de prédictions et signale la réutilisation des résultats entre profils. Les anciennes recettes restent inchangées.
- Les notebooks, exemples et guides sont inclus dans les archives source. L’interface compilée est embarquée ; le build refuse une wheel dépourvue d’interface. Le manifeste de sources est régénéré et vérifiable.

## Installation

Python 3.12 est l’environnement de recette. Installer la wheel de cette release :

```bash
python -m pip install "./helm_xai-0.2.0rc1-py3-none-any.whl[ui]"
helm-ui
```

`[text,ig,anchors,notebook]` ajoute les dépendances des calculs texte et des widgets. `[all]` regroupe les extras d’usage. Aucun poids ni donnée sur données réelles n’est inclus. Le notebook de saisie libre téléchargé depuis une ancienne version conserve sa révision d’origine : il n’est pas mis à jour silencieusement par cette release.

Pour reconstruire depuis un clone, utiliser `python tools/build_release.py` après installation de Node.js/npm et de l’extra `[dev]`. Une wheel publiée ne nécessite pas Node. Le sdist contient l’interface compilée pour permettre sa reconstruction Python seule.

## Portée

Le profil est choisi explicitement ou reçoit une valeur par défaut ; il n’est pas prédit à partir du commentaire. Le code UCB1 et le sélecteur contextuel existent, mais leur utilité humaine n’est pas établie par la recette logicielle. `helm.study` prépare des stimuli figés ; il ne fournit pas une étude humaine complète (consentement, recrutement, passation et analyse restent à organiser).

Certaines méthodes historiques sont des proxys explicitement étiquetés (LOO, perturbations, variation de norme). L’interface démo ne doit pas être citée comme exécution native de toutes les méthodes de la thèse. L’extension tabulaire et Databricks ne constituent pas une validation métier ; aucun test Databricks réel n’est revendiqué.

Les droits de diffusion ouverte et le choix de licence restent à confirmer par le titulaire des droits. Le marqueur privé reste actif.

L’API `HELMBenchmark` et les fonctions lexicales historiques restent destinées à la reproduction des anciens protocoles (classe toxique fixe et baseline historique explicites). Elles ne remplacent pas `evaluate_attribution`, utilisé par le pipeline actuel. Les règles et contrefactuels y sont également refusés ; les métadonnées de `run_single` portent `historical_only=True`.
