# Publication PyPI — configuration du dépôt public

Mise à jour du 4 octobre 2026 : dépôt public `KossiFO/helm-xai`.

Le code est publié sous MIT avec l’autorisation du titulaire. Les distributions
GitHub sont installables. La publication PyPI attend la création du compte et
la configuration de l’éditeur de confiance ; aucun jeton ne doit être ajouté
au code, aux notebooks ou aux conversations.

## Configuration du compte du titulaire

Après création du compte PyPI, vérification de l’e-mail et activation de la
double authentification, ouvrir les paramètres Publishing et ajouter un
éditeur GitHub en attente :

| Champ | Valeur |
|---|---|
| Projet PyPI | `helm-xai` |
| Propriétaire GitHub | `KossiFO` |
| Dépôt | `helm-xai` |
| Workflow | `publish-pypi.yml` |
| Environnement | `pypi` |

Le nom n’est réservé qu’au premier envoi réussi. Le workflow est manuel :
lancer « Publier les distributions vérifiées sur PyPI » avec le tag validé
`v0.2.0rc3`. Il contrôle les SHA256, la version et la licence, puis publie les
mêmes fichiers que la release GitHub. La création du compte et sa connexion
ne déclenchent aucun envoi à elles seules.

## Vérification après publication

Vérifier la page du projet PyPI et installer dans un environnement vierge
`helm-xai[ui]==0.2.0rc3`, puis lancer `helm-ui`. Tant que cet envoi n’est pas
réussi, utiliser la wheel GitHub ; ne pas présenter `pip install helm-xai`
comme disponible sur PyPI.

Références : [création du projet par éditeur de confiance](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/),
[publication officielle GitHub Actions](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
