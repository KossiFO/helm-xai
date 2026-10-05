# HELM 0.2.0rc3 — distribution publique MIT

Le dépôt `KossiFO/helm-xai` distribue une copie propre du package actuel, sans
historique Git privé ni données sur données réelles. La licence MIT a été choisie par le titulaire.
Le moteur et l’interface sont ceux de la candidate rc2 ; seuls version,
métadonnées et documentation de distribution sont adaptés.

## Installer

```sh
python -m pip install "helm-xai[ui] @ https://github.com/KossiFO/helm-xai/releases/download/v0.2.0rc3/helm_xai-0.2.0rc3-py3-none-any.whl"
helm-ui
```

Python 3.12 recommandé. L’extra `ui` lance une interface sur le poste local ;
`text,ig,anchors,notebook` ajoute les dépendances des calculs texte et des notebooks.
Les poids des modèles restent à obtenir séparément. La démonstration utilise un
classifieur entraîné sur 12 phrases fabriquées ; elle ne valide pas un détecteur.

## Portée

La toxicité reste le domaine central, avec choix explicite du profil ; le
tabulaire est une extension. Les corrections rc1 ont passé 175 tests Python
et 8 tests frontend. Les recettes GitHub Actions de chaque commit sont liées
au code concerné. Pas de nouvel essai Databricks ou Colab dans cette livraison.
Les anciens notebooks gardent les liens vers leurs révisions privées et ne
sont pas annoncés comme exécutables sans adaptation depuis ce nouveau dépôt.

Cette diffusion GitHub ne signifie pas encore publication PyPI. Le compte
PyPI doit terminer son activation 2FA, puis autoriser `publish-pypi.yml` du
dépôt `KossiFO/helm-xai`, environnement `pypi`, pour le projet `helm-xai`.
Aucun résultat privé, réponse de participant ou poids de modèle n’est distribué.
