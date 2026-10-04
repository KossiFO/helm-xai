# HELM 0.2.0rc2 — diffusion publique sous licence MIT

Cette candidate rend la bibliothèque Python HELM et son interface accessibles
publiquement. Elle conserve les corrections et le fonctionnement de la rc1.
La toxicité est le domaine central ; l’explication tabulaire est une extension.

## Installation depuis GitHub

Python 3.12 recommandé. Télécharger la wheel ci-dessous puis installer :

```sh
python -m pip install "./helm_xai-0.2.0rc2-py3-none-any.whl[ui]"
helm-ui
```

L’extra `[text,ig,anchors,notebook]` ajoute les dépendances texte et notebook ;
`[all]` les extras d’usage. Les modèles sont à obtenir séparément.
L’interface de démonstration utilise des phrases fabriquées ; elle ne valide
pas les performances d’un détecteur de toxicité. Les profils sont choisis,
ils ne sont pas prédits à partir du commentaire.

## Diffusion et validation

La licence MIT et les droits de publication ont été confirmés par le titulaire.
Aucune donnée MAAF, sortie privée de notebook ni réponse de participant n’est
incluse. Les anciennes archives restent identiques.

La rc1 a passé 175 tests Python et 8 tests frontend, avec installation UI
sous Python 3.10/3.11/3.12. La rc2 modifie la licence, les métadonnées et la
préparation de publication ; la validation de chaque commit est disponible
dans GitHub Actions. Aucun nouvel essai Databricks ou Colab n’est revendiqué.

La publication PyPI nécessite encore la configuration du compte du titulaire.
La commande `pip install helm-xai` sans URL n’est pas annoncée comme disponible.
Les instructions et le workflow de publication figurent dans
`docs/PUBLICATION_PYPI_v1_2026-10-03.md`.
