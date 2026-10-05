# Développement et tests

Utiliser Python 3.12 et Node.js 22 pour construire les sources. L’installation
de la wheel ne nécessite pas Node.js.

```bash
python -m pip install ".[text,ui,ig,anchors,notebook,dev]"
USE_TF=0 HF_HUB_OFFLINE=1 python -m pytest
npm ci --prefix interface/frontend
npm test --prefix interface/frontend
python tools/build_release.py
python -m twine check dist/*.whl dist/*.tar.gz
```

Pour une recette hors sources, installer la wheel dans un environnement virtuel
vierge puis lancer `python -I tools/smoke_installed.py`. Le test utilise un modèle
synthétique et une base temporaire. Il ne mesure pas la qualité d’un modèle réel.
