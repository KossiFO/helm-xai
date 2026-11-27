# Tester HELM et transmettre un retour

Cet essai s’adresse aux personnes qui utilisent Python. Il sert à améliorer
l’installation, le fonctionnement et la compréhension de l’interface.
Ce n’est pas encore l’étude scientifique sur l’adaptation des explications aux utilisateurs.

## 1. Installer dans un environnement séparé

Utilisez Python **3.10 à 3.13** (3.12 ou 3.13 conseillé pour cet essai).
Python 3.14 n’est pas pris en charge par cette distribution.
Une connexion Internet est nécessaire pour télécharger les dépendances.
Ni GPU, ni compte, ni clé d’API, ni Node.js ne sont nécessaires pour la démonstration.

### macOS ou Linux

```bash
python3 --version
python3 -m venv .venv-helm
.venv-helm/bin/python -m pip install --upgrade pip
.venv-helm/bin/python -m pip install "helm-xai[ui]==0.2.0rc7"
.venv-helm/bin/python -m helm.ui
```

### Windows (PowerShell ou invite de commandes)

```powershell
py -3.13 --version
py -3.13 -m venv .venv-helm
.venv-helm\Scripts\python.exe -m pip install --upgrade pip
.venv-helm\Scripts\python.exe -m pip install "helm-xai[ui]==0.2.0rc7"
.venv-helm\Scripts\python.exe -m helm.ui
```

Si vous avez Python 3.12, remplacez `-3.13` par `-3.12`.
Ces commandes utilisent directement le Python de l’environnement : pas de modification
de la politique d’exécution PowerShell ni d’activation à effectuer.

Gardez le terminal ouvert. Le navigateur s’ouvre sur <http://127.0.0.1:8767/>.
Si ce n’est pas le cas, ouvrez vous-même cette adresse après l’affichage de « Uvicorn running ».
Le premier démarrage peut prendre plus de temps pour préparer les caches.
Arrêtez le serveur avec **Ctrl+C** dans le terminal.

## 2. Suivre un parcours court

1. Gardez **Démonstration locale** et le profil **Utilisateur**.
2. Cliquez sur **Remerciement**, puis **Expliquer**. Une prédiction, des méthodes
   et un résultat doivent apparaître. Le résultat attendu est l’affichage, pas un score précis.
3. Choisissez **LIME** ou **SHAP** dans la liste des méthodes. Examinez les mots mis en avant.
4. Passez au profil **Expert technique**, avec le même texte. Notez ce qui change
   et ce qui reste difficile à comprendre. La liste des méthodes peut aussi changer :
   cette comparaison libre ne mesure pas à elle seule l’effet du profil.
5. Attribuez une note de 1 à 5 à l’explication affichée. **La note reste sur votre ordinateur**
   et peut influencer les sélections suivantes. Elle n’est pas envoyée à l’auteur.
6. Ouvrez **Historique**, puis retrouvez votre analyse et la note enregistrée.
7. Ouvrez **Tester et faire un retour** pour préparer votre rapport et le partager.

Le modèle de démonstration est entraîné sur **12 phrases fabriquées**. Il permet
de tester le parcours sans télécharger de modèle externe. Ce n’est pas un détecteur
de toxicité validé. Ne l’utilisez pas pour modérer réellement des personnes.
Les autres modèles restent indisponibles si leurs dépendances et leurs poids
ne sont pas présents sur votre ordinateur : cela n’empêche pas la démonstration.

## 3. Faire parvenir votre retour

Le bouton **Télécharger mon retour** produit un fichier texte à relire. Il contient
les versions du logiciel, votre texte de retour et, si une analyse est affichée,
son modèle, son profil, ses méthodes et vos notes. Il n’inclut pas automatiquement
le commentaire analysé, les journaux, les chemins locaux ou l’historique complet.
Le texte de retour n’est pas sauvegardé dans HELM : téléchargez-le avant de quitter la page.

- **Avec un compte GitHub** : [ouvrir une issue](https://github.com/KossiFO/helm-xai/issues/new/choose),
  choisir « Problème d’installation ou de fonctionnement » ou « Retour sur un essai »,
  puis coller le rapport relu. L’issue est publique.
- **Sans compte GitHub** : envoyer le rapport à **follyametepkossi@gmail.com**,
  objet « Retour de test HELM ». Le lien de l’interface ouvre votre messagerie
  si elle est configurée ; il n’ajoute pas de pièce jointe et n’envoie rien.

L’export **Exporter l’analyse (avec le texte saisi)** est différent : il contient
le commentaire analysé et des détails de calcul. Ne le publiez pas si vous avez
utilisé des données personnelles ou confidentielles. Préférez les exemples fournis.

Si l’installation échoue avant l’ouverture de l’interface, utilisez directement
le lien GitHub ou le courriel. Indiquez : système, version Python, commande exécutée,
résultat attendu et message d’erreur (après retrait des informations privées).

## Dépannage

| Problème | Action |
|---|---|
| `python3` ou `py` introuvable | Installer Python depuis <https://www.python.org/downloads/>, puis rouvrir le terminal. |
| `No matching distribution found` | Vérifier la version Python et l’accès à PyPI ; utiliser la version épinglée ci-dessus. |
| Téléchargement interrompu | Relancer l’installation ; si le réseau est lent, ajouter `--timeout 120`. Ne pas désactiver TLS. |
| `helm-ui` introuvable | Utiliser `python -m helm.ui` avec le Python de l’environnement, comme ci-dessus. |
| Port déjà occupé | Ajouter `--port 8771` à la commande de lancement et ouvrir l’adresse affichée. |
| Aucune fenêtre de navigateur | Garder le terminal ouvert et ouvrir l’URL locale affichée. |
| Modèles XLM-R/Qwen/CamemBERT indisponibles | Garder **Démonstration locale** pour ce premier essai. |
| Besoin des versions | Lancer le même Python avec `-m helm.ui --diagnostic`. |

L’historique et les notes sont stockés localement. Pour un nouvel essai indépendant,
lancez HELM avec `--data-dir ./helm-essai-separe`. Aucune donnée n’est transmise
automatiquement à l’auteur. Il n’existe pas, dans cette version, de plateforme d’étude hébergée.
