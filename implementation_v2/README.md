# HELM — Hybrid Explainability Layered Model

Package Python `helm-xai` (import : `helm`). Framework d'explicabilité adaptative
pour la modération de contenus : trois couches (contexte → sélection →
présentation), backends XAI identifiés, sélection par profil et UCB1 ; LinUCB expérimental en option.

> Thèse de Kossi Folly — CY Cergy Paris Université (lab ETIS),
> direction Maria Malek. Article : *HELM: A Hybrid Explainable Layered Model for
> Adaptive Post-Hoc Explanation Selection in Textual Toxicity Classification*,
> workshop XKDD, ECML-PKDD 2026.

## Livraison installable 0.2.0rc4

La wheel embarque maintenant l’interface épurée. Installer la wheel avec l’extra
`[ui]`, puis lancer `helm-ui` ; voir le [guide de livraison](../README.md).
Le moteur décrit dans le protocole a4 ci-dessous est conservé.

## Installation

Depuis la racine du dépôt (le `pyproject.toml` y est) :

```bash
pip install -e .              # noyau tabulaire et visualisations HTML
pip install -e ".[text]"      # modèles de texte : Torch et Transformers
pip install -e ".[all,dev]"   # + captum, anchors officiel, matplotlib, tests
```

Contraintes d'environnement : Python ≥ 3.10 et < 3.13 (recette : 3.12), `USE_TF=0` dans l'environnement
avant tout import (sinon transformers tente de charger TensorFlow), `protobuf < 5`.
Les modèles HuggingFace se téléchargent au premier chargement.

## Utilisation

```python
from helm import HELMPipeline, UserProfile

pipeline = HELMPipeline(model_name="xlmr")          # xlmr | qwen | camembert
ctx = pipeline.explain(
    "Tu es un idiot fini, retourne dans ton trou",
    user_profile=UserProfile.MODERATOR,
    evaluate=True,                                   # fidélité + suffisance
)
print(ctx.prediction.label, ctx.prediction.confidence)
print([m.method_name for m in ctx.selected_methods])  # couche 2
print(ctx.formatted_output.summary)                    # couche 3
print(ctx.evaluation)                                  # métriques par méthode

pipeline.submit_feedback(UserProfile.MODERATOR, "lime", rating=4)  # → UCB1
```

Les explicateurs s'utilisent aussi seuls, comme dans les scripts d'expériences :

```python
from helm.models import ModelManager
from helm.explainers import LIMEExplainer
from helm.evaluation import evaluate_attribution

model = ModelManager("xlmr").load()
text = "Tu es un idiot fini, retourne dans ton trou"
attr = LIMEExplainer(random_state=42).explain(text, model.predict_proba, num_features=8)
evaluate_attribution(text, attr, model.predict_proba, k=3)
```

## Structure

```
implementation_v2/
├── helm/                      # le package installé
│   ├── config.py              # enums, dataclasses, matrice de décision, registre des modèles
│   ├── context.py             # HELMContext : objet typé traversant les couches
│   ├── pipeline.py            # HELMPipeline : orchestrateur
│   ├── layers/                # couche 1 contexte, 2 sélection, 3 présentation
│   ├── models/                # ModelManager : xlmr, qwen, camembert (predict_proba commun)
│   ├── explainers/            # LIME, SHAP, IG, Anchors, Counterfactual, CoT, Logit Lens
│   ├── evaluation/            # fidélité, suffisance, stabilité, benchmark (Wilcoxon, bootstrap)
│   ├── feedback/              # UCB1Learner (notes 1-5), UCB1AutoReward (métriques)
│   ├── visualization/         # graphiques (extra `viz`)
│   └── data/                  # corpus de démonstration, 30 phrases
├── tests/                     # pytest, sans chargement de modèle
├── config.py, data/           # façades de compatibilité pour Redaction_These/run_*.py — hors wheel
├── scripts/run_demo.py        # démonstration complète
└── APPROCHE.md                # justification des choix (pourquoi, comment, références)
```

## Protocole récent — version 0.1.0a4

LIME et SHAP utilisent désormais les positions de mots séparés par espaces,
y compris répétitions et apostrophes, sur la classe initialement prédite.
Le pipeline évalue **C/S signés, k=3** ; il conserve les positions et textes
intervenus dans les métadonnées. Si une intervention est vide, il évalue « le ».
Une attribution sans scores positionnels est refusée par ce protocole.
Les scores lexicaux sont une vue agrégée pour l'affichage, pas la source des métriques.
Les C/S d'un contrefactuel portent sur ses scores de suppression, pas sur la
qualité de la modification trouvée. Les règles Anchors et les diagnostics internes
ne reçoivent aucune mesure positionnelle imputée.

```python
from helm.explainers import LOOExplainer, VerifiedIGExplainer
from helm.evaluation import evaluate_attribution, signed_cs_reward

# LOO par remplacement, distinct d'IG :
attr = LOOExplainer().explain(text, model.predict_proba, num_features=8)
metrics = evaluate_attribution(text, attr, model.predict_proba, k=3)
automatic_reward = signed_cs_reward(metrics.comprehensiveness, metrics.sufficiency)

# IG natif XLM-R : deux quadratures, contrôles internes, accord des vecteurs et top-3.
ig = VerifiedIGExplainer(model._model, model._tokenizer)
attr = ig.explain(text, model.predict_proba, num_features=8)
if attr.metadata["valid"]:
    metrics = evaluate_attribution(text, attr, model.predict_proba, k=3)
```

Le pipeline active ce dernier avec `ig_mode="verified"`. Cette option inclut IG
hors du budget temporel heuristique du profil ; sa latence reste inconnue
(`estimated_time=None`). Le défaut `ig_mode="legacy_loo"` conserve le remplacement
historique rapide, affiché **LOO (remplacement par le)**. La clé historique
`integrated_gradients` reste compatible ; `metadata.effective_method` identifie
l'exécution réelle. Aucun échec d'IG natif ne déclenche LOO : ses diagnostics sont
conservés dans `ctx.failed_attributions`, sans explication ni métrique imputées.
Le protocole IG natif vise XLM-R, pas tous les transformeurs.

Le checkpoint XLM-R est figé à `b9c7c563427c591fc318d91eb592381ae2fbde66`.
Le réglage expérimental est graine 42, 8 features et k=3 ; passer explicitement
`num_features=8` au pipeline, dont les contraintes de présentation dépendent du profil.
Cette mise à jour ne réexécute pas la campagne de 493 textes ni les expériences humaines.

## Sélection contextuelle expérimentale

`ContextualSelector` expose LinUCB disjoint, normalisé sur la **calibration seule** :
log1p(nombre de mots), confiance dans la classe prédite, classe prédite.
La récompense automatique est `0.5 + 0.5*(w*C - (1-w)*S)`, w=0.5 par défaut.
Elle est distincte d'une note humaine. Aucune moyenne publiée n'est chargée comme
une préférence par profil. Le sélecteur exige une calibration fournie explicitement.

```python
from helm import ContextualSelector, HELMPipeline

# contexts_calibration : n×3 ; rewards_calibration : n×3, ordre LIME/SHAP/IG.
policy = ContextualSelector(["lime", "shap", "integrated_gradients"], ridge=10, alpha=0.5)
policy.fit_calibration(contexts_calibration, rewards_calibration)
pipeline = HELMPipeline(model_name="xlmr", ig_mode="verified", contextual_selector=policy)
ctx = pipeline.explain(text, evaluate=True, num_features=8, evaluation_k=3)
choice = ctx.contextual_choice
if choice is not None and choice.method in ctx.evaluation:
    m = ctx.evaluation[choice.method]
    policy.update(choice, signed_cs_reward(m.fidelity, m.sufficiency))
```

Une calibration doit conserver le checkpoint, les backends, leurs paramètres et le protocole
de récompense utilisés pour la produire ; elle ne se transfère pas silencieusement à
d’autres réglages ou profils.

Le pool calibré remplace la matrice de sélection dans ce mode opt-in ; la
présentation reste adaptée au profil. Le choix précède l'observation de la récompense.
Une seule mise à jour est autorisée par décision, pour le seul bras choisi.
Utiliser `policy.discard(choice)` pour abandonner une décision sans récompense.
Le sélecteur est en mémoire : cette API ne persiste pas une politique apprise.
Les résultats récents ne montrent pas un gain uniforme selon le pool de méthodes.
Les fidélités historiques, dont plusieurs étaient estimées ou issues de proxies,
ne pondèrent plus la matrice par défaut.

## Compatibilité historique

`compute_fidelity` et `compute_sufficiency` conservent leurs conventions historiques
(classe toxique, k=5 par défaut). `evaluation_protocol="legacy"` les active dans le
pipeline ; préciser aussi `evaluation_k=5` pour ce réglage ancien.
`LIMEExplainer(word_positions=False)` et `SHAPExplainer(word_positions=False)`
conservent la segmentation lexicale historique. L'ancien IG sur logit reste dans
`IGExplainer(model=..., embedding_layer=..., tokenizer=...)`. Il est distinct du
nouveau `VerifiedIGExplainer` sur probabilité. `use_legacy_fidelity=True` réactive
explicitement l'ancienne pondération de sélection.

Anchors officiel et recherche interne restent sélectionnables avec `use_native`.
Les scores de croissance de norme ne sont plus assimilés à des contributions
toxiques : ils sont conservés par sous-token dans les diagnostics exportés.
Les justifications à gabarit sont identifiées comme telles, même si un gestionnaire
de modèle a été fourni sans génération effective.

Les façades `config.py` et `data/` restent importables par les scripts historiques.
Pour reproduire un ancien résultat à l'identique, conserver son environnement et
sa révision de code : la compatibilité d'import n'est pas une identité numérique.
Voir [la note de version](RELEASE_0.1.0a4_v1_2026-09-10.md) pour les preuves et limites.

## Vérification

```bash
USE_TF=0 python -m pytest      # les tests natifs sont ignorés sans leurs extras
ruff check implementation_v2
python tools/build_release.py # helm/ et interface compilée incluse
```

## Extension tabulaire expérimentale (0.2.0a1)

`from helm import TabularHELM` permet d’expliquer un classifieur déjà entraîné
sur des variables mixtes, de choisir la classe à expliquer et de produire
un rapport HTML sur les 5 ou 10 scores extrêmes. Les vues métier et technique
conservent la même explication. La sélection adaptative du pipeline texte
n’est pas transférée à ce domaine.

La [notice](../docs/Extension_HELM_ML_v1_2026-09-14.md) précise le contrat,
les limites et la feuille de route. Exemple synthétique exécutable :
`python examples/tabular_usage.py`. L’interface `helm-ui` conserve son parcours
texte ; le rapport tabulaire est disponible depuis la bibliothèque et le notebook.

### Profils tabulaires (0.2.0a2)

```python
rapport = helm.explain_profile(X_test, profile="metier", k=10)
rapport.to_html("rapport.html")
```

Profils : `utilisateur`, `metier`, `technique`, `audit`. Les deux premiers proposent SHAP ; les deux autres SHAP et LIME. La restitution métier explique les facteurs en phrases ; la vue technique compare les méthodes sans fusionner leurs poids ; la vue audit expose un registre de couverture et de provenance. Il s’agit de règles explicites à évaluer, sans préférence apprise. [Notice](../docs/Profils_HELM_v1_2026-09-14.md).
