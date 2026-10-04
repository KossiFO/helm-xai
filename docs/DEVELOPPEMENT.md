# Installation depuis les sources et reproduction technique

Environnement de recette : Python 3.12 et Node.js 22 ou version compatible Vite7.
La wheel inclut le JavaScript compilé ; Node.js intervient uniquement pour
reconstruire l’interface depuis les sources.

```sh
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell : .venv\Scripts\Activate.ps1
python -m pip install build
python tools/build_release.py
python -m pip install -c requirements/constraints-py312.txt "./dist/helm_xai-0.2.0rc3-py3-none-any.whl[ui,dev,ig]"
python -m pytest
npm test --prefix interface/frontend
python -m ruff check implementation_v2/helm interface/tests tools
```

Le script construit l’interface, copie ses fichiers dans `helm/ui/static`, puis
produit la wheel et l’archive source dans `dist/`. Il refuse d’écraser une version
existante. Pour un développement après compilation, utiliser `pip install -e ".[ui]"`.
Les extras `ig` et `anchors` ajoutent respectivement Captum et Anchors officiel.
Sans extra optionnel, les tests concernés sont ignorés explicitement.

## Vérifier la distribution hors des sources

Créer un second environnement, installer la wheel avec l’extra `ui` et `httpx` (dépendance de recette), puis lancer
`python -I tools/smoke_installed.py` avec le Python de ce second environnement.
Le script vérifie le résultat pédagogique, les fichiers d’interface, l’historique
et le feedback dans un dossier temporaire ; il ne touche pas aux données locales.
La vérification d’installation doit utiliser une vraie wheel, pas un editable.

`helm-ui --no-browser --port 8771 --data-dir ./test-local` lance le service local
sans ouvrir le navigateur. L’API est consultable sur `/api/docs`.
L’interface écoute exclusivement sur `127.0.0.1` et reste sans authentification :
elle est prévue pour un poste local, pas pour une exposition Internet.

## Dépendances et reproductibilité

`requirements/constraints-py312.txt` fixe les versions principales utilisées pour
la recette ; ce n’est pas un verrouillage exhaustif des dépendances transitives.
Le frontend dispose d’un `package-lock.json`. Le workflow GitHub exécute tests,
construction et installation de la wheel sur Ubuntu/Python3.12 lorsqu’il est activé
sur le dépôt distant ; sa présence ne prouve pas qu’une exécution distante a eu lieu.

Les paramètres scientifiques restent explicites : graine42, huit caractéristiques,
k=3. Les tests utilisent des données fabriquées et ne reproduisent pas la campagne
expérimentale complète. Pour celle-ci, conserver code, poids et environnement de
la révision ayant produit les résultats.
