"""
Wrapper Counterfactual Explanations pour le framework HELM v2.

Trouve la **modification minimale** du texte qui inverse la décision
du modèle. Produit des explications du type :
"Votre commentaire a été classé toxique à cause du mot 'idiot'.
 Sans ce mot, il aurait été classé non-toxique."

Référence :
    Wachter, S., Mittelstadt, B., & Russell, C. (2017).
    "Counterfactual Explanations without Opening the Black Box."
    Harvard Journal of Law & Technology.

    Wu, T., Ribeiro, M. T., Heer, J., & Weld, D. S. (2021).
    "Polyjuice: Generating Counterfactuals for Explaining,
    Evaluating, and Improving Models." ACL.
"""

import re
import time
import logging
from typing import Callable, Dict, List, Optional


from helm.config import Attribution
from .base import BaseExplainer

logger = logging.getLogger(__name__)


class CounterfactualExplainer(BaseExplainer):
    """
    Expliqueur par contrefactuels — trouve le changement minimal
    qui inverse la prédiction.

    Stratégie :
    1. **Suppression** : retire un mot à la fois, vérifie si la prédiction s'inverse
    2. **Remplacement** : remplace un mot par un mot neutre
    3. **Combinaisons** : retire des paires de mots si aucun mot seul ne suffit

    L'explication finale indique quels mots changer/retirer pour inverser la décision.

    Parameters
    ----------
    neutral_replacements : list of str
        Mots neutres utilisés pour le remplacement.
    max_combination_size : int
        Taille maximale des combinaisons de mots à tester.
    """

    def __init__(
        self,
        neutral_replacements: Optional[List[str]] = None,
        max_combination_size: int = 3,
    ):
        self.neutral_replacements = neutral_replacements or [
            "le", "un", "cela", "chose", "personne",
        ]
        self.max_combination_size = max_combination_size
        logger.info(
            "CounterfactualExplainer initialisé (max_combo=%d)",
            self.max_combination_size,
        )

    @property
    def method_name(self) -> str:
        return "counterfactual"

    def explain(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
    ) -> Attribution:
        """
        Trouve les contrefactuels minimaux pour le texte donné.

        Returns
        -------
        Attribution
            token_scores : importance de chaque mot (mesurée par le changement
            de prédiction quand le mot est retiré/remplacé).
            metadata contient les contrefactuels trouvés.
        """
        logger.debug("Counterfactual explain() sur un texte de %d caractères", len(text))
        t0 = time.perf_counter()

        try:
            words = re.findall(r"\S+", text)
            if not words:
                elapsed = time.perf_counter() - t0
                return Attribution(
                    method_name=self.method_name,
                    token_scores={},
                    computation_time=elapsed,
                    metadata={"counterfactuals": []},
                )

            # Prédiction de référence
            base_probs = predict_fn([text])
            base_toxic_prob = float(base_probs[0, 1])
            base_label = "toxique" if base_toxic_prob > 0.5 else "non_toxique"
            target_label = "non_toxique" if base_label == "toxique" else "toxique"

            # Phase 1 : Suppression simple (un mot à la fois)
            deletion_results: List[Dict] = []
            deletion_texts: List[str] = []
            for i in range(len(words)):
                remaining = words[:i] + words[i + 1 :]
                deletion_texts.append(" ".join(remaining) if remaining else "")

            if deletion_texts:
                del_probs = predict_fn(deletion_texts)
                for i, word in enumerate(words):
                    new_toxic_prob = float(del_probs[i, 1])
                    new_label = "toxique" if new_toxic_prob > 0.5 else "non_toxique"
                    flipped = new_label != base_label and bool(deletion_texts[i].strip())
                    deletion_results.append({
                        "word": word,
                        "index": i,
                        "action": "suppression",
                        "original_prob": base_toxic_prob,
                        "new_prob": new_toxic_prob,
                        "delta": base_toxic_prob - new_toxic_prob,
                        "flipped": flipped,
                        "counterfactual_text": deletion_texts[i],
                    })

            # Phase 2 : Remplacement par mots neutres
            replacement_results: List[Dict] = []
            for neutral in self.neutral_replacements[:2]:
                replace_texts = []
                for i in range(len(words)):
                    new_words = words.copy()
                    new_words[i] = neutral
                    replace_texts.append(" ".join(new_words))

                if replace_texts:
                    rep_probs = predict_fn(replace_texts)
                    for i, word in enumerate(words):
                        new_toxic_prob = float(rep_probs[i, 1])
                        new_label = "toxique" if new_toxic_prob > 0.5 else "non_toxique"
                        flipped = new_label != base_label
                        replacement_results.append({
                            "word": word,
                            "index": i,
                            "action": f"remplacement par '{neutral}'",
                            "original_prob": base_toxic_prob,
                            "new_prob": new_toxic_prob,
                            "delta": base_toxic_prob - new_toxic_prob,
                            "flipped": flipped,
                            "counterfactual_text": replace_texts[i],
                        })

            # Phase 3 : Combinaisons (si aucun mot seul n'inverse)
            single_flips = [r for r in deletion_results if r["flipped"]]
            combo_results: List[Dict] = []

            if not single_flips and self.max_combination_size >= 2:
                # Trier par delta décroissant pour prioriser les meilleurs candidats
                sorted_by_delta = sorted(
                    deletion_results,
                    key=lambda r: abs(r["delta"]),
                    reverse=True,
                )
                top_indices = [r["index"] for r in sorted_by_delta[:6]]

                for i in range(len(top_indices)):
                    for j in range(i + 1, len(top_indices)):
                        idx_i, idx_j = top_indices[i], top_indices[j]
                        remaining = [
                            w for k, w in enumerate(words)
                            if k not in (idx_i, idx_j)
                        ]
                        combo_text = " ".join(remaining) if remaining else ""
                        combo_probs = predict_fn([combo_text])
                        new_toxic_prob = float(combo_probs[0, 1])
                        new_label = "toxique" if new_toxic_prob > 0.5 else "non_toxique"
                        flipped = new_label != base_label and bool(combo_text.strip())

                        combo_results.append({
                            "words": [words[idx_i], words[idx_j]],
                            "indices": [idx_i, idx_j],
                            "action": "suppression combinée",
                            "original_prob": base_toxic_prob,
                            "new_prob": new_toxic_prob,
                            "delta": base_toxic_prob - new_toxic_prob,
                            "flipped": flipped,
                            "counterfactual_text": combo_text,
                        })

                        if flipped:
                            break
                    if any(r["flipped"] for r in combo_results):
                        break

            # Construire les scores d'attribution
            token_scores: Dict[str, float] = {}
            for r in deletion_results:
                clean = r["word"].strip(".,;:!?\"'()[]{}«»")
                if clean:
                    token_scores[clean] = token_scores.get(clean, 0.0) + r["delta"]

            # Limiter au top num_features
            if len(token_scores) > num_features:
                sorted_tokens = sorted(
                    token_scores.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                token_scores = dict(sorted_tokens[:num_features])

            # Collecter les contrefactuels trouvés
            all_results = deletion_results + replacement_results + combo_results
            counterfactuals = [r for r in all_results if r["flipped"] and r["counterfactual_text"].strip()]
            for candidate in counterfactuals:
                candidate["edit_count"] = len(candidate.get("indices", [candidate.get("index")]))
            counterfactuals.sort(key=lambda r: (r["edit_count"], abs(r["delta"])))

            # Générer un résumé en langage naturel
            summary = self._generate_summary(
                base_label, target_label, counterfactuals,
                base_toxic_prob if base_label == "toxique" else 1-base_toxic_prob
            )

            elapsed = time.perf_counter() - t0
            logger.info(
                "Counterfactual terminé en %.2fs — %d contrefactuels trouvés",
                elapsed, len(counterfactuals),
            )

            return Attribution(
                method_name=self.method_name,
                token_scores=token_scores,
                computation_time=elapsed,
                base_value=base_toxic_prob,
                metadata={
                    "base_label": base_label,
                    "effective_method": "counterfactual_search",
                    "display_name": "Contrefactuels (recherche bornée)",
                    "minimality": "within_searched_candidates_only",
                    "found": bool(counterfactuals),
                    "score_kind": "deletion_probability_drop",
                    "scores_by_position": [r["delta"] for r in deletion_results],
                    "target_class": 1,  # deltas above explicitly describe P(toxicity), not the predicted-class target
                    "predicted_class": int(base_label == "toxique"),
                    "counterfactual_target_class": int(target_label == "toxique"),
                    "generic_cs_applicable": False,
                    "target_label": target_label,
                    "counterfactuals": counterfactuals[:5],
                    "summary": summary,
                    "num_single_flips": len(single_flips),
                    "num_combo_flips": len([r for r in combo_results if r.get("flipped")]),
                },
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Erreur Counterfactual après %.2fs : %s", elapsed, exc)
            raise RuntimeError(
                f"Échec du calcul Counterfactual après {elapsed:.2f}s : {exc}"
            ) from exc

    def estimate_time(self, text_length: int) -> float:
        words_approx = max(1, text_length / 5)
        # Suppressions + remplacements + quelques combos
        n_evals = words_approx * (1 + len(self.neutral_replacements[:2])) + 15
        return n_evals * 0.05

    @staticmethod
    def _generate_summary(
        base_label: str,
        target_label: str,
        counterfactuals: List[Dict],
        base_prob: float,
    ) -> str:
        """Génère un résumé en langage naturel des contrefactuels."""
        if not counterfactuals:
            return (
                f"Le texte est classé '{base_label}' (confiance : {base_prob:.0%}). "
                f"Aucune inversion n'a été trouvée parmi les modifications explorées."
            )

        best = counterfactuals[0]
        if "word" in best:
            word = best["word"]
            action = best["action"]
            new_prob = best["new_prob"]
            return (
                f"Le texte est classé '{base_label}' (confiance : {base_prob:.0%}). "
                f"En appliquant '{action}' sur le mot '{word}', "
                f"la prédiction passerait à '{target_label}' "
                f"(P(toxique) = {new_prob:.0%})."
            )
        elif "words" in best:
            words_str = "' et '".join(best["words"])
            new_prob = best["new_prob"]
            return (
                f"Le texte est classé '{base_label}' (confiance : {base_prob:.0%}). "
                f"En retirant les mots '{words_str}', "
                f"la prédiction passerait à '{target_label}' "
                f"(P(toxique) = {new_prob:.0%})."
            )

        return f"Le texte est classé '{base_label}' (confiance : {base_prob:.0%})."
