# HELM 0.1.0a4 — résultats récents intégrés

Version Codex isolée, 10 septembre 2026. Aucun changement dans Claude, le dépôt
principal, le lot distant v1 ou les réponses aux questionnaires.

## Changements utiles

- **Identité des méthodes** : LOO est nommé comme tel ; une justification à gabarit
  ne se déclare plus générée. Les variations de normes restent des diagnostics
  de sous-tokens, sans contribution toxique ni consensus directionnel.
- **IG natif XLM-R** : port du protocole de campagne avec PAD lexical, attention
  et positions fixes, cible probabilité, tolérances 0,025/0,0125 et quadratures
  16/32 puis 24/48. Contrôles internes, accord L1 et top-3 ; échec conservé sans LOO.
- **LIME/SHAP positionnels** : paramètres de la campagne pour les valeurs par
  défaut (graine 42, LIME 300 perturbations, SHAP 100 évaluations, batch 4).
  Répétitions préservées ; vue lexicale séparée des scores complets.
- **C/S signés** : classe prédite, positions absolues top-k, k=3, « le » si vide.
  Les métriques ne sont plus calculées sur un dictionnaire ambigu.
- **LinUCB** : calibration explicite, échelle figée sur calibration, choix avant
  récompense, mise à jour unique du bras choisi. Option expérimentale en mémoire.
- **Contrefactuels** : confiance de la classe correctement affichée, textes vides
  exclus, classement par nombre de modifications parmi les candidats explorés.
- **Démonstrateur local** : libellés corrigés, anciennes analyses conservées,
  notes a3 séparées des notes a4. Lot d'étude distant v1 inchangé.

## Ce qui motive ces changements

Les fichiers récupérés de la campagne comprennent 493 textes et 1 972 résultats ;
IG est admissible sur 491 textes. Les cas 225 et483 restent exclus de la cohorte
comparative commune aux quatre méthodes. Ces résultats motivent des contrôles
numériques, pas une préférence humaine attribuée automatiquement à une méthode.

L'expérience contextuelle utilise 199 textes de calibration et 292 d'évaluation.
Son gain dépend du pool : un résultat favorable avec LIME/SHAP/IG ne se généralise
pas lorsque LOO est ajouté. LinUCB reste donc une option, sans recalibration
silencieuse des quatre profils.

Le pilote natif Anchors/CF v2 porte sur 5 mêmes textes : quatre règles non vides et
deux contrefactuels non vides inversant la classe. Il ne constitue pas la validation
complète des sept méthodes ni une mesure d'utilité humaine.

Sources en lecture dans `These_redaction` :
`Redaction_These/resultats/campagne493_brut_2026-09-10/`,
`codex/IG_natif_HELM_v1_2026-09-09_code/`,
`codex/Colab_HELM_v1_2026-09-09_code/`,
`Redaction_These/resultats/natifs_AC_pilot5_v2_2026-09-10_code/`.

## Vérification de cette livraison

100 tests Python et 8 tests JavaScript ont réussi ; analyse statique et compilation
React réussies. Les quatre backends ont été exécutés sur le checkpoint XLM-R
local, dont IG avec deux contrôles numériques réussis sur le texte de recette.

Les preuves techniques et le verdict indépendant sont conservés dans
`qa/package_resultats_recents_codex_v1_2026-09-10/`. La recette XLM-R est bornée
sur un texte fabriqué ; elle ne répète pas les 1 972 calculs et ne valide pas
l'étude humaine. Les paramètres d'IG natif restent coûteux et sa latence n'est
pas estimée par l'ancien coût de LOO. Le package est une version alpha privée.
