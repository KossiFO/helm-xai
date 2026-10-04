"""
Banc d'essai HELM v2 pour l'évaluation comparative des méthodes XAI.

Ce module orchestre l'exécution systématique des expliqueurs sur un corpus
de textes, calcule les métriques de fidélité, suffisance et stabilité,
et fournit des outils d'analyse statistique (tests de Wilcoxon, tableaux
récapitulatifs) au format de l'article XKDD 2026.

Architecture
------------
``HELMBenchmark`` encapsule un ``ModelManager`` et un dictionnaire
d'expliqueurs. Il produit un ``pd.DataFrame`` contenant les métriques
brutes pour chaque couple (texte, méthode), ainsi que des résumés
statistiques directement utilisables dans l'article.
"""

import logging
import time
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from helm.config import EvaluationMetrics
from .fidelity import compute_fidelity, compute_sufficiency, bootstrap_metric
from .stability import compute_stability

logger = logging.getLogger(__name__)


class HELMBenchmark:
    """
    Banc d'essai pour l'évaluation comparative des méthodes XAI.

    Exécute les expliqueurs sur un corpus de textes, calcule les métriques
    réelles (fidélité, suffisance, stabilité) et fournit les tests
    statistiques nécessaires à la comparaison rigoureuse des méthodes.

    Parameters
    ----------
    model_manager : ModelManager
        Gestionnaire de modèle chargé et prêt pour l'inférence.
    explainers : Dict[str, BaseExplainer]
        Dictionnaire {nom_méthode: instance_expliqueur}.

    Attributes
    ----------
    predict_fn : Callable
        Fonction de prédiction extraite du ``model_manager``, compatible
        avec la signature ``List[str] -> np.ndarray (n, 2)``.
    """

    def __init__(self, model_manager, explainers: Dict) -> None:
        self.model_manager = model_manager
        self.explainers = explainers

        # Vérifier que le modèle est chargé
        if not model_manager.is_loaded:
            raise RuntimeError(
                "Le ModelManager doit être chargé (load()) avant de créer "
                "un HELMBenchmark."
            )

        # Fonction de prédiction compatible avec les expliqueurs
        self.predict_fn = model_manager.predict_proba

        logger.info(
            "HELMBenchmark initialisé — modèle=%s, méthodes=%s",
            model_manager.model_name,
            list(explainers.keys()),
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Évaluation unitaire
    # ═══════════════════════════════════════════════════════════════════════

    def run_single(
        self,
        text: str,
        method_name: str,
        k: int = 5,
        n_bootstrap: int = 100,
        n_perturbations: int = 5,
        num_features: int = 10,
    ) -> Dict:
        """
        Exécute un expliqueur sur un texte et calcule toutes les métriques.

        Parameters
        ----------
        text : str
            Texte à expliquer et évaluer.
        method_name : str
            Nom de la méthode XAI (clé dans ``self.explainers``).
        k : int
            Nombre de top-k tokens pour fidélité/suffisance.
        n_bootstrap : int
            Nombre d'itérations bootstrap pour les IC.
        n_perturbations : int
            Nombre de perturbations pour la stabilité.
        num_features : int
            Nombre de features demandées à l'expliqueur.

        Returns
        -------
        Dict
            Dictionnaire contenant les métriques, l'attribution et les
            métadonnées d'exécution.

        Raises
        ------
        KeyError
            Si ``method_name`` n'est pas dans les expliqueurs disponibles.
        """
        if method_name not in self.explainers:
            raise KeyError(
                f"Méthode inconnue : « {method_name} ». "
                f"Méthodes disponibles : {list(self.explainers.keys())}"
            )

        explainer = self.explainers[method_name]
        logger.info(
            "run_single : méthode=%s, texte='%s...' (%d car.)",
            method_name, text[:50], len(text),
        )

        t0 = time.perf_counter()

        # ── Étape 1 : Calcul de l'attribution ──────────────────────────
        attribution = explainer.explain(
            text, self.predict_fn, num_features=num_features,
        )

        # ── Étape 2 : Fidélité avec bootstrap ─────────────────────────
        fidelity_val, fidelity_ci = bootstrap_metric(
            text, attribution, self.predict_fn,
            metric_fn=compute_fidelity,
            n_bootstrap=n_bootstrap, k=k,
        )

        # ── Étape 3 : Suffisance avec bootstrap ───────────────────────
        sufficiency_val, sufficiency_ci = bootstrap_metric(
            text, attribution, self.predict_fn,
            metric_fn=compute_sufficiency,
            n_bootstrap=n_bootstrap, k=k,
        )

        # ── Étape 4 : Stabilité ───────────────────────────────────────
        stability_val = compute_stability(
            text, self.predict_fn, explainer,
            n_perturbations=n_perturbations,
            num_features=num_features,
        )

        elapsed = time.perf_counter() - t0

        # ── Assembler les résultats ────────────────────────────────────
        metrics = EvaluationMetrics(
            fidelity=fidelity_val,
            sufficiency=sufficiency_val,
            stability=stability_val,
            metadata={"protocol": "legacy_lexical_toxic_class", "historical_only": True, "empty_baseline": 0.5},
            fidelity_ci=fidelity_ci,
            sufficiency_ci=sufficiency_ci,
        )

        result = {
            "text": text,
            "method": method_name,
            "fidelity": fidelity_val,
            "sufficiency": sufficiency_val,
            "stability": stability_val,
            "fidelity_ci_low": fidelity_ci[0],
            "fidelity_ci_high": fidelity_ci[1],
            "sufficiency_ci_low": sufficiency_ci[0],
            "sufficiency_ci_high": sufficiency_ci[1],
            "attribution": attribution,
            "metrics": metrics,
            "computation_time": elapsed,
            "top_tokens": attribution.top_tokens[:k],
        }

        logger.info(
            "run_single terminé en %.2fs — fidelity=%.4f, "
            "sufficiency=%.4f, stability=%.4f",
            elapsed, fidelity_val, sufficiency_val, stability_val,
        )

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Évaluation sur corpus
    # ═══════════════════════════════════════════════════════════════════════

    def run_corpus(
        self,
        texts: List[str],
        methods: Optional[List[str]] = None,
        k: int = 5,
        n_bootstrap: int = 100,
        n_perturbations: int = 5,
        num_features: int = 10,
    ) -> pd.DataFrame:
        """
        Exécute toutes les méthodes sur tous les textes du corpus.

        Parameters
        ----------
        texts : List[str]
            Corpus de textes à évaluer.
        methods : Optional[List[str]]
            Liste des méthodes à exécuter. Si None, utilise toutes les
            méthodes disponibles dans ``self.explainers``.
        k : int
            Nombre de top-k tokens pour fidélité/suffisance.
        n_bootstrap : int
            Nombre d'itérations bootstrap.
        n_perturbations : int
            Nombre de perturbations pour la stabilité.
        num_features : int
            Nombre de features pour les expliqueurs.

        Returns
        -------
        pd.DataFrame
            DataFrame avec colonnes : text, method, fidelity, sufficiency,
            stability, fidelity_ci_low, fidelity_ci_high, sufficiency_ci_low,
            sufficiency_ci_high, computation_time.
        """
        if methods is None:
            methods = list(self.explainers.keys())

        # Vérifier que toutes les méthodes demandées existent
        unknown = set(methods) - set(self.explainers.keys())
        if unknown:
            raise KeyError(
                f"Méthodes inconnues : {unknown}. "
                f"Disponibles : {list(self.explainers.keys())}"
            )

        total = len(texts) * len(methods)
        logger.info(
            "run_corpus : %d textes × %d méthodes = %d évaluations",
            len(texts), len(methods), total,
        )

        rows: List[Dict] = []
        t0_global = time.perf_counter()

        for i, text in enumerate(texts):
            for j, method in enumerate(methods):
                idx = i * len(methods) + j + 1
                logger.info(
                    "[%d/%d] texte %d, méthode=%s",
                    idx, total, i + 1, method,
                )

                try:
                    result = self.run_single(
                        text, method, k=k,
                        n_bootstrap=n_bootstrap,
                        n_perturbations=n_perturbations,
                        num_features=num_features,
                    )

                    rows.append({
                        "text": result["text"],
                        "method": result["method"],
                        "fidelity": result["fidelity"],
                        "sufficiency": result["sufficiency"],
                        "stability": result["stability"],
                        "fidelity_ci_low": result["fidelity_ci_low"],
                        "fidelity_ci_high": result["fidelity_ci_high"],
                        "sufficiency_ci_low": result["sufficiency_ci_low"],
                        "sufficiency_ci_high": result["sufficiency_ci_high"],
                        "computation_time": result["computation_time"],
                    })

                except Exception as exc:
                    logger.error(
                        "Erreur pour texte %d, méthode=%s : %s",
                        i + 1, method, exc,
                    )
                    rows.append({
                        "text": text,
                        "method": method,
                        "fidelity": np.nan,
                        "sufficiency": np.nan,
                        "stability": np.nan,
                        "fidelity_ci_low": np.nan,
                        "fidelity_ci_high": np.nan,
                        "sufficiency_ci_low": np.nan,
                        "sufficiency_ci_high": np.nan,
                        "computation_time": np.nan,
                    })

        elapsed_total = time.perf_counter() - t0_global
        df = pd.DataFrame(rows)

        logger.info(
            "run_corpus terminé en %.1fs — %d résultats (%d erreurs)",
            elapsed_total,
            len(df),
            int(df["fidelity"].isna().sum()),
        )

        return df

    # ═══════════════════════════════════════════════════════════════════════
    # Analyse statistique
    # ═══════════════════════════════════════════════════════════════════════

    def compare_methods(self, df: pd.DataFrame) -> Dict:
        """
        Effectue des tests de Wilcoxon signés entre toutes les paires
        de méthodes pour chaque métrique.

        Le test de Wilcoxon est un test non-paramétrique approprié pour
        comparer des distributions appariées (chaque texte sert de bloc).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame produit par ``run_corpus()``.

        Returns
        -------
        Dict
            Dictionnaire structuré :
            ``{metric: {(method_a, method_b): {"statistic": W, "p_value": p,
            "significant": bool}}}`` pour chaque métrique et chaque paire
            de méthodes. Significativité à alpha=0.05.
        """
        methods = df["method"].unique().tolist()
        metrics = ["fidelity", "sufficiency", "stability"]

        results: Dict = {}

        for metric in metrics:
            results[metric] = {}

            for method_a, method_b in combinations(methods, 2):
                # Extraire les valeurs appariées (même texte)
                scores_a = df.loc[
                    df["method"] == method_a, ["text", metric]
                ].set_index("text")[metric]

                scores_b = df.loc[
                    df["method"] == method_b, ["text", metric]
                ].set_index("text")[metric]

                # Aligner sur les textes communs
                common_texts = scores_a.index.intersection(scores_b.index)

                if len(common_texts) < 6:
                    logger.warning(
                        "Trop peu de textes communs (%d) pour le test de "
                        "Wilcoxon entre %s et %s sur %s.",
                        len(common_texts), method_a, method_b, metric,
                    )
                    results[metric][(method_a, method_b)] = {
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "significant": False,
                        "n_samples": len(common_texts),
                    }
                    continue

                vals_a = scores_a.loc[common_texts].values.astype(np.float64)
                vals_b = scores_b.loc[common_texts].values.astype(np.float64)

                # Supprimer les paires avec NaN
                valid_mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
                vals_a = vals_a[valid_mask]
                vals_b = vals_b[valid_mask]

                # Vérifier que les différences ne sont pas toutes nulles
                diffs = vals_a - vals_b
                if np.all(np.abs(diffs) < 1e-12):
                    logger.info(
                        "Différences nulles entre %s et %s sur %s "
                        "— pas de test nécessaire.",
                        method_a, method_b, metric,
                    )
                    results[metric][(method_a, method_b)] = {
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "significant": False,
                        "n_samples": len(vals_a),
                    }
                    continue

                if len(vals_a) < 6:
                    logger.warning(
                        "Trop peu de paires valides (%d) après filtrage NaN.",
                        len(vals_a),
                    )
                    results[metric][(method_a, method_b)] = {
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "significant": False,
                        "n_samples": len(vals_a),
                    }
                    continue

                try:
                    stat, p_value = wilcoxon(vals_a, vals_b)
                    significant = p_value < 0.05

                    results[metric][(method_a, method_b)] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "significant": significant,
                        "n_samples": len(vals_a),
                    }

                    logger.info(
                        "Wilcoxon %s vs %s [%s] : W=%.1f, p=%.4f %s",
                        method_a, method_b, metric, stat, p_value,
                        "(significatif)" if significant else "(non significatif)",
                    )

                except ValueError as exc:
                    logger.warning(
                        "Test de Wilcoxon échoué pour %s vs %s [%s] : %s",
                        method_a, method_b, metric, exc,
                    )
                    results[metric][(method_a, method_b)] = {
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "significant": False,
                        "n_samples": len(vals_a),
                    }

        return results

    def summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère un tableau récapitulatif des métriques par méthode.

        Produit un DataFrame au format « Moyenne +/- Ecart-type » pour
        chaque métrique et chaque méthode, au format du tableau de
        résultats de l'article XKDD 2026.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame produit par ``run_corpus()``.

        Returns
        -------
        pd.DataFrame
            Tableau indexé par méthode avec les colonnes :
            fidelity_mean, fidelity_std, fidelity_formatted,
            sufficiency_mean, sufficiency_std, sufficiency_formatted,
            stability_mean, stability_std, stability_formatted,
            computation_time_mean.
        """
        metrics = ["fidelity", "sufficiency", "stability"]
        methods = df["method"].unique().tolist()

        summary_rows: List[Dict] = []

        for method in methods:
            method_df = df[df["method"] == method]
            row: Dict = {"method": method}

            for metric in metrics:
                values = method_df[metric].dropna().values

                if len(values) == 0:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_std"] = np.nan
                    row[f"{metric}_formatted"] = "N/A"
                    continue

                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val
                row[f"{metric}_formatted"] = f"{mean_val:.3f} ± {std_val:.3f}"

            # Temps de calcul moyen
            time_values = method_df["computation_time"].dropna().values
            row["computation_time_mean"] = (
                float(np.mean(time_values)) if len(time_values) > 0 else np.nan
            )

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows).set_index("method")

        logger.info(
            "Tableau récapitulatif généré pour %d méthodes :\n%s",
            len(methods), summary_df.to_string(),
        )

        return summary_df
