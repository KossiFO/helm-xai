"""
Corpus d'évaluation — 30 phrases françaises annotées (démonstration).
Échantillon construit à la main, inspiré du corpus data.gouv.fr des premières
expériences. Les résultats publiés (XKDD 2026) utilisent le jeu
textdetox/multilingual_toxicity_dataset, pas ce corpus.
"""

EVALUATION_CORPUS = [
    # ── Toxiques (15) ──
    {"text": "Tu es vraiment un idiot fini, retourne dans ton trou",
     "label": "toxique", "category": "insulte"},
    {"text": "Les gens comme toi devraient disparaître de cette planète",
     "label": "toxique", "category": "menace"},
    {"text": "C'est du n'importe quoi ton commentaire, ferme-la un peu",
     "label": "toxique", "category": "agressivité"},
    {"text": "Bande de crétins vous comprenez rien à la politique",
     "label": "toxique", "category": "insulte"},
    {"text": "Ce mec est un vrai déchet humain sans aucune valeur",
     "label": "toxique", "category": "déshumanisation"},
    {"text": "Allez tous vous faire voir espèces de parasites",
     "label": "toxique", "category": "insulte"},
    {"text": "T'as vraiment rien dans le crâne pour sortir des trucs pareils",
     "label": "toxique", "category": "insulte"},
    {"text": "On devrait interdire les gens aussi stupides que toi de voter",
     "label": "toxique", "category": "discrimination"},
    {"text": "Quel commentaire débile, va te coucher au lieu de polluer le débat",
     "label": "toxique", "category": "agressivité"},
    {"text": "C'est la pire chose que j'ai lue, quel imbécile",
     "label": "toxique", "category": "insulte"},
    {"text": "Je hais ces gens-là, ils méritent le pire",
     "label": "toxique", "category": "haine"},
    {"text": "Encore un abruti qui donne son avis sans savoir de quoi il parle",
     "label": "toxique", "category": "insulte"},
    {"text": "Vous êtes tous des moutons manipulés et pathétiques",
     "label": "toxique", "category": "mépris"},
    {"text": "Cette personne devrait avoir honte d'exister",
     "label": "toxique", "category": "déshumanisation"},
    {"text": "Fermez vos gueules bande de connards incultes",
     "label": "toxique", "category": "insulte"},

    # ── Non toxiques (15) ──
    {"text": "Je ne suis pas d'accord avec cette analyse mais je respecte le point de vue",
     "label": "non_toxique", "category": "désaccord_poli"},
    {"text": "Merci pour cet article très intéressant et bien documenté",
     "label": "non_toxique", "category": "appréciation"},
    {"text": "La situation économique est préoccupante mais il y a des solutions",
     "label": "non_toxique", "category": "opinion_nuancée"},
    {"text": "Excellent travail de la part de l'équipe de recherche",
     "label": "non_toxique", "category": "appréciation"},
    {"text": "Ce débat est important et mérite une discussion approfondie",
     "label": "non_toxique", "category": "encouragement"},
    {"text": "Je pense qu'on devrait considérer d'autres perspectives également",
     "label": "non_toxique", "category": "suggestion"},
    {"text": "Les résultats de cette étude sont prometteurs pour l'avenir",
     "label": "non_toxique", "category": "observation"},
    {"text": "Bonjour à tous, j'espère que vous allez bien aujourd'hui",
     "label": "non_toxique", "category": "salutation"},
    {"text": "Cette politique a des avantages et des inconvénients qu'il faut peser",
     "label": "non_toxique", "category": "opinion_nuancée"},
    {"text": "Bonne fête nationale à tous les Français",
     "label": "non_toxique", "category": "célébration"},
    {"text": "Le rapport montre une amélioration significative des indicateurs",
     "label": "non_toxique", "category": "observation"},
    {"text": "Pouvez-vous expliquer davantage votre raisonnement s'il vous plaît",
     "label": "non_toxique", "category": "question"},
    {"text": "C'est un sujet complexe qui nécessite plus de recherche",
     "label": "non_toxique", "category": "observation"},
    {"text": "Bravo pour cette initiative citoyenne vraiment inspirante",
     "label": "non_toxique", "category": "appréciation"},
    {"text": "Le gouvernement devrait investir davantage dans l'éducation",
     "label": "non_toxique", "category": "suggestion"},
]

# Textes seuls pour usage rapide
TOXIC_TEXTS = [item["text"] for item in EVALUATION_CORPUS if item["label"] == "toxique"]
NON_TOXIC_TEXTS = [item["text"] for item in EVALUATION_CORPUS if item["label"] == "non_toxique"]
ALL_TEXTS = [item["text"] for item in EVALUATION_CORPUS]
ALL_LABELS = [item["label"] for item in EVALUATION_CORPUS]
