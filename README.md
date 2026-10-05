> **Version candidate 0.2.0rc3 — diffusion publique sous licence MIT.** Installation et interface vérifiées ; validation humaine distincte ; [publiée sur PyPI](https://pypi.org/project/helm-xai/0.2.0rc3/).
>
> [État, corrections et limites de cette version](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/RELEASE_0.2.0rc3_v1_2026-10-04.md).

>
> Point de départ des PFE : [calculs figés sans apprentissage et guide de prise en main](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/DEMARRAGE_PFE_v1_2026-09-29.md). Le module `helm.study` est inclus dans cette candidate.
> Les anciens notebooks conservent leur révision ; aucune expérience antérieure n’est remplacée.
> `TabularHELM.explain_profile` propose des méthodes et une restitution par profil
> (règles explicites, sans politique apprise). [Notice des profils](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/Profils_HELM_v1_2026-09-14.md).
> [Notebooks Colab : MAAF et toxicité séparés](https://github.com/KossiFO/helm-xai/blob/codex/publication/examples/colab/README.md).

# HELM — explications adaptées au profil

HELM est le package de recherche de Kossi Folly pour explorer des explications
adaptées à quatre profils : utilisateur, modérateur, expert technique et régulateur.
La version **0.2.0rc3** réunit la bibliothèque Python et une interface locale épurée.

## Installer et ouvrir

Prérequis : **Python 3.12**. Installer la distribution publiée sur PyPI :

```sh
python -m pip install "helm-xai[ui]==0.2.0rc3"
helm-ui
```

Le navigateur s’ouvre sur **http://127.0.0.1:8767/**. Choisir un profil, écrire un
message et cliquer **Expliquer**. Le modèle de démonstration fonctionne sans poids
à télécharger. Fermer le serveur avec Ctrl+C. Pour choisir un autre port :
`helm-ui --port 8771`. Node.js n’est pas nécessaire pour utiliser la wheel.

La démonstration utilise un classifieur entraîné sur **12 phrases fabriquées** :
elles servent à essayer le parcours, pas à valider la détection de toxicité.
Les modèles XLM-R, Qwen et CamemBERT deviennent disponibles lorsque leurs poids
sont déjà dans le cache local. L’interface ne télécharge pas les poids.

## Ce que contient cette livraison

- Une bibliothèque `helm` : pipeline, explications, évaluation, sélection par profil
  et UCB1 ; sélecteur contextuel LinUCB expérimental en option.
- Une interface : quatre profils, mots importants, changement de méthode, détails
  et traces repliés, historique et export JSON.
- Des avis de 1 à 5 liés à l’explication individuelle affichée. Les avis restent
  séparés par modèle, profil et version du package.

Les analyses et les notes sont enregistrées sur l’ordinateur qui lance HELM, dans
le dossier de données utilisateur `helm-xai`. `helm-ui --data-dir ./mes-analyses`
permet de choisir ce dossier. Aucun historique personnel, résultat expérimental,
poids de modèle ou donnée de participant n’est livré dans le package.

## Utiliser la bibliothèque Python

```python
from helm import HELMPipeline, UserProfile
from helm.ui.models import DemoModel

pipeline = HELMPipeline(model_name="demo", model=DemoModel().load())
resultat = pipeline.explain(
    "Merci pour cette réponse utile",
    user_profile=UserProfile.END_USER,
    evaluate=True, num_features=8, evaluation_k=3,
)
print(resultat.prediction)
print(resultat.attributions.keys())
```

[Guide scientifique et API](https://github.com/KossiFO/helm-xai/blob/codex/publication/implementation_v2/README.md) ·
[Installation et développement](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/DEVELOPPEMENT.md) ·
[Note de version](https://github.com/KossiFO/helm-xai/blob/codex/publication/docs/RELEASE_0.2.0rc3_v1_2026-10-04.md).

## Portée

Candidate de recherche, dérivée de l’édition Codex isolée. Les identités effectives des
méthodes sont affichées : le remplacement LOO reste distinct d’IG natif vérifié,
qui s’active explicitement dans l’API Python. Les métriques signées C/S ciblent
la classe prédite ; elles ne mesurent pas la satisfaction humaine.

Cette interface est un démonstrateur local. Le protocole d’étude distante à
l’aveugle est un projet distinct ; l’hébergement du code sur GitHub ne crée pas
un lien d’étude utilisable par les participants. Tests logiciels et recette ne
constituent pas une validation humaine du framework.

Le code HELM est distribué sous [licence MIT](https://github.com/KossiFO/helm-xai/blob/codex/publication/LICENSE) ; voir [DROITS.md](https://github.com/KossiFO/helm-xai/blob/codex/publication/DROITS.md).
La candidate est disponible sur [PyPI](https://pypi.org/project/helm-xai/0.2.0rc3/) et dans les [releases GitHub](https://github.com/KossiFO/helm-xai/releases/tag/v0.2.0rc3).
