"""
Métriques de fidélité pour les attributions XAI — HELM v2.

Ce module implémente les métriques *comprehensiveness* (fidélité) et
*sufficiency* telles que définies par DeYoung et al. (2020) et
adaptées au cadre HELM pour la détection de toxicité en français.

Les métriques reposent sur la perturbation réelle des tokens : on
masque ou isole les tokens les plus importants selon l'attribution,
puis on mesure la variation de la probabilité prédite par le modèle.

Références
----------
- DeYoung, J. et al. (2020). *ERASER: A Benchmark to Evaluate
  Rationalized NLP Models*. ACL 2020.
- Atanasova, P. et al. (2020). *A Diagnostic Study of Explainability
  Techniques for Text Classification*. EMNLP 2020.
"""

import logging
import re
from typing import Callable, List, Tuple

import numpy as np

from helm.config import Attribution

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Utilitaires de masquage
# ═══════════════════════════════════════════════════════════════════════════


def _tokenize_simple(text: str) -> List[str]:
    """
    Découpage simple d'un texte en tokens (mots + ponctuation).

    Utilisé pour reconstruire le texte après masquage. Le découpage
    est cohérent avec celui de LIME (split sur les non-alphanumériques).

    Parameters
    ----------
    text : str
        Texte à découper.

    Returns
    -------
    List[str]
        Liste de tokens.
    """
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def _mask_tokens(text: str, tokens_to_mask: List[str], mask_token: str = "") -> str:
    """
    Masque les occurrences des tokens spécifiés dans le texte.

    Chaque occurrence d'un token est remplacée par ``mask_token``.
    La comparaison est insensible à la casse pour gérer les variations
    typographiques courantes en français.

    Parameters
    ----------
    text : str
        Texte original.
    tokens_to_mask : List[str]
        Tokens à masquer.
    mask_token : str
        Chaîne de remplacement (par défaut chaîne vide).

    Returns
    -------
    str
        Texte avec les tokens masqués.
    """
    tokens = _tokenize_simple(text)
    mask_set = {t.lower() for t in tokens_to_mask}

    result_tokens = []
    for tok in tokens:
        if tok.lower() in mask_set:
            if mask_token:
                result_tokens.append(mask_token)
            # sinon on omet simplement le token
        else:
            result_tokens.append(tok)

    return " ".join(result_tokens)


def _keep_only_tokens(text: str, tokens_to_keep: List[str]) -> str:
    """
    Conserve uniquement les tokens spécifiés, masque le reste.

    Parameters
    ----------
    text : str
        Texte original.
    tokens_to_keep : List[str]
        Tokens à conserver.

    Returns
    -------
    str
        Texte ne contenant que les tokens spécifiés.
    """
    tokens = _tokenize_simple(text)
    keep_set = {t.lower() for t in tokens_to_keep}

    result_tokens = [tok for tok in tokens if tok.lower() in keep_set]
    return " ".join(result_tokens)


# ═══════════════════════════════════════════════════════════════════════════
# Métriques principales
# ═══════════════════════════════════════════════════════════════════════════


def compute_fidelity(
    text: str,
    attribution: Attribution,
    predict_fn: Callable,
    k: int = 5,
) -> float:
    """
    Calcule la fidélité (*comprehensiveness*) d'une attribution.

    La fidélité mesure à quel point les tokens identifiés comme
    importants influencent réellement la prédiction du modèle. On
    retire les top-k tokens et on observe la chute de probabilité.

    Formule : fidelity = P_original(toxique) - P_masqué(toxique)

    Un score élevé indique que les tokens sélectionnés sont
    effectivement déterminants pour la prédiction.

    Parameters
    ----------
    text : str
        Texte original à évaluer.
    attribution : Attribution
        Objet contenant les scores d'importance par token.
    predict_fn : Callable
        Fonction de prédiction ``List[str] -> np.ndarray (n, 2)``.
        Colonne 0 = P(non_toxique), colonne 1 = P(toxique).
    k : int
        Nombre de tokens les plus importants à masquer.

    Returns
    -------
    float
        Score de fidélité dans [-1, 1]. Plus élevé = meilleure fidélité.
    """
    if (attribution.method_name in {'counterfactual', 'anchors'}
            or attribution.metadata.get('effective_method') == 'counterfactual_search'
            or attribution.metadata.get('score_kind') == 'norm_growth_proxy'):
        raise ValueError('Métrique lexicale non applicable aux règles, contrefactuels ou normes cachées.')
    # Récupérer les top-k tokens par importance absolue
    top_k_tokens = [token for token, _score in attribution.top_tokens[:k]]

    if not top_k_tokens:
        logger.warning(
            "Aucun token disponible dans l'attribution pour compute_fidelity."
        )
        return 0.0

    # Prédiction sur le texte original
    proba_original = predict_fn([text])
    p_original_toxic = float(proba_original[0, 1])

    # Masquer les top-k tokens
    masked_text = _mask_tokens(text, top_k_tokens)

    # Gérer le cas où le texte masqué est vide
    if not masked_text.strip():
        logger.debug(
            "Texte entièrement masqué après retrait de %d tokens.", k
        )
        # Si tout est masqué, la prédiction tombe à la baseline (~0.5)
        p_masked_toxic = 0.5
    else:
        proba_masked = predict_fn([masked_text])
        p_masked_toxic = float(proba_masked[0, 1])

    fidelity = p_original_toxic - p_masked_toxic

    logger.debug(
        "Fidélité : P_orig=%.4f, P_masqué=%.4f, fidelity=%.4f "
        "(k=%d, tokens=%s)",
        p_original_toxic, p_masked_toxic, fidelity, k, top_k_tokens,
    )

    return fidelity


def compute_sufficiency(
    text: str,
    attribution: Attribution,
    predict_fn: Callable,
    k: int = 5,
) -> float:
    """
    Calcule la suffisance d'une attribution.

    La suffisance mesure si les top-k tokens suffisent à eux seuls
    à maintenir la prédiction du modèle. On garde uniquement les
    top-k tokens et on mesure la perte de probabilité.

    Formule : sufficiency = P_original(toxique) - P_topk_seul(toxique)

    Un score bas indique que les tokens sélectionnés sont suffisants
    pour reproduire la prédiction originale.

    Parameters
    ----------
    text : str
        Texte original à évaluer.
    attribution : Attribution
        Objet contenant les scores d'importance par token.
    predict_fn : Callable
        Fonction de prédiction ``List[str] -> np.ndarray (n, 2)``.
    k : int
        Nombre de tokens les plus importants à conserver.

    Returns
    -------
    float
        Score de suffisance dans [-1, 1]. Plus bas = meilleure suffisance.
    """
    if (attribution.method_name in {'counterfactual', 'anchors'}
            or attribution.metadata.get('effective_method') == 'counterfactual_search'
            or attribution.metadata.get('score_kind') == 'norm_growth_proxy'):
        raise ValueError('Métrique lexicale non applicable aux règles, contrefactuels ou normes cachées.')
    # Récupérer les top-k tokens par importance absolue
    top_k_tokens = [token for token, _score in attribution.top_tokens[:k]]

    if not top_k_tokens:
        logger.warning(
            "Aucun token disponible dans l'attribution pour compute_sufficiency."
        )
        return 0.0

    # Prédiction sur le texte original
    proba_original = predict_fn([text])
    p_original_toxic = float(proba_original[0, 1])

    # Créer un texte ne contenant que les top-k tokens
    topk_only_text = _keep_only_tokens(text, top_k_tokens)

    # Gérer le cas où le texte résultant est vide (tokens non trouvés)
    if not topk_only_text.strip():
        logger.debug(
            "Aucun token conservé trouvé dans le texte original (k=%d).", k
        )
        p_topk_toxic = 0.5
    else:
        proba_topk = predict_fn([topk_only_text])
        p_topk_toxic = float(proba_topk[0, 1])

    sufficiency = p_original_toxic - p_topk_toxic

    logger.debug(
        "Suffisance : P_orig=%.4f, P_topk=%.4f, sufficiency=%.4f "
        "(k=%d, tokens=%s)",
        p_original_toxic, p_topk_toxic, sufficiency, k, top_k_tokens,
    )

    return sufficiency


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap pour intervalles de confiance
# ═══════════════════════════════════════════════════════════════════════════


def bootstrap_metric(
    text: str,
    attribution: Attribution,
    predict_fn: Callable,
    metric_fn: Callable,
    n_bootstrap: int = 100,
    k: int = 5,
) -> Tuple[float, Tuple[float, float]]:
    """
    Calcule une métrique avec rééchantillonnage bootstrap pour
    estimer un intervalle de confiance à 95 %.

    Le bootstrap est effectué en faisant varier la valeur de k
    de 1 à ``num_features`` (nombre total de tokens dans l'attribution).
    Pour chaque itération bootstrap, un sous-ensemble de valeurs de k
    est tiré avec remplacement, et la métrique est calculée pour chacune.

    Parameters
    ----------
    text : str
        Texte original.
    attribution : Attribution
        Attributions token-level.
    predict_fn : Callable
        Fonction de prédiction ``List[str] -> np.ndarray (n, 2)``.
    metric_fn : Callable
        Fonction de métrique à appliquer (``compute_fidelity`` ou
        ``compute_sufficiency``). Signature :
        ``(text, attribution, predict_fn, k) -> float``.
    n_bootstrap : int
        Nombre d'itérations bootstrap.
    k : int
        Valeur nominale de k (utilisée pour la moyenne, mais le
        bootstrap explore différentes valeurs).

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        ``(mean_metric, (ci_lower, ci_upper))`` — Métrique moyenne et
        intervalle de confiance à 95 % (percentiles 2.5 et 97.5).
    """

    if (attribution.method_name in {'counterfactual', 'anchors'}
            or attribution.metadata.get('effective_method') == 'counterfactual_search'
            or attribution.metadata.get('score_kind') == 'norm_growth_proxy'):
        raise ValueError('Métrique lexicale non applicable aux règles, contrefactuels ou normes cachées.')
    num_features = len(attribution.token_scores)

    if num_features == 0:
        logger.warning("Attribution vide, bootstrap impossible.")
        return 0.0, (0.0, 0.0)

    # Calculer la métrique pour chaque valeur de k de 1 à num_features
    max_k = min(k, num_features)
    k_values = list(range(1, num_features + 1))

    # Pré-calculer les métriques pour chaque k afin d'éviter
    # d'appeler le modèle de manière redondante
    metric_by_k = {}
    for ki in k_values:
        metric_by_k[ki] = metric_fn(text, attribution, predict_fn, k=ki)

    # Bootstrap : rééchantillonnage des valeurs de k
    rng = np.random.default_rng(seed=42)
    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        # Tirer un échantillon avec remplacement parmi les valeurs de k
        sampled_ks = rng.choice(k_values, size=len(k_values), replace=True)
        sampled_metrics = [metric_by_k[ki] for ki in sampled_ks]
        bootstrap_means[b] = np.mean(sampled_metrics)

    # Statistiques
    mean_metric = float(metric_by_k[max_k])
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))

    logger.info(
        "Bootstrap (%d itérations) : mean=%.4f, IC95=[%.4f, %.4f]",
        n_bootstrap, mean_metric, ci_lower, ci_upper,
    )

    return mean_metric, (ci_lower, ci_upper)
