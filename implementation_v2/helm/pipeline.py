"""
Pipeline principal HELM v2.
Orchestre les 3 couches + exécution des méthodes XAI.
"""

import logging
from typing import Optional, Dict
from dataclasses import replace

from helm.config import UserProfile, Prediction
from helm.context import HELMContext
from helm.layers import ContextualLayer, SelectionLayer, PresentationLayer
from helm.models import ModelManager
from helm.feedback.ucb1_learner import UCB1Learner

logger = logging.getLogger(__name__)


class HELMPipeline:
    """
    Orchestrateur principal du framework HELM.

    Usage :
        pipeline = HELMPipeline(model_name="camembert")
        ctx = pipeline.explain("Ce texte est toxique", user_profile=UserProfile.MODERATOR)
        print(ctx.formatted_output.summary)
    """

    def __init__(
        self,
        model_name: str = "xlmr",
        backend: str = "transformer",
        preference_learner: Optional[UCB1Learner] = None,
        *,
        model=None,
        explainers: Optional[Dict] = None,
        ig_mode: str = "legacy_loo",
        evaluation_protocol: str = "signed_cs",
        use_legacy_fidelity: bool = False,
        contextual_selector=None,
        allowed_methods=None,
    ):
        if ig_mode not in {"legacy_loo", "verified"}:
            raise ValueError("ig_mode attendu : legacy_loo ou verified.")
        if evaluation_protocol not in {"signed_cs", "legacy"}:
            raise ValueError("Protocole attendu : signed_cs ou legacy.")
        if contextual_selector is not None and contextual_selector.model_name != model_name:
            raise ValueError("Le modèle doit correspondre à la calibration contextuelle.")
        self.ig_mode, self.evaluation_protocol = ig_mode, evaluation_protocol
        self.contextual_selector = contextual_selector
        if contextual_selector is not None and allowed_methods is not None:
            raise ValueError("Le sélecteur contextuel définit déjà son pool de méthodes.")
        self.model_name = model_name
        self._preference_learner = preference_learner or UCB1Learner()

        # Chargement du modèle
        logger.info(f"Initialisation HELM v2 — modèle={model_name}, backend={backend}")
        self.model = model if model is not None else ModelManager(model_name=model_name, backend=backend)
        if model is None:
            self.model.load()

        # Couches
        self.layer1 = ContextualLayer()
        self.layer2 = SelectionLayer(preference_learner=self._preference_learner,
                                     use_legacy_fidelity=use_legacy_fidelity, allowed_methods=allowed_methods,
                                     unbudgeted_methods={"integrated_gradients"} if ig_mode == "verified" else ())
        self.layer3 = PresentationLayer()

        # Explainers (chargés à la demande)
        self._explainers: Dict = dict(explainers or {})

    def _get_explainer(self, method_name: str):
        """Charge un explainer à la demande."""
        if method_name not in self._explainers:
            if method_name == "lime":
                from helm.explainers import LIMEExplainer
                self._explainers[method_name] = LIMEExplainer()
            elif method_name == "shap":
                from helm.explainers import SHAPExplainer
                self._explainers[method_name] = SHAPExplainer()
            elif method_name == "integrated_gradients":
                from helm.explainers import IGExplainer, VerifiedIGExplainer
                if self.ig_mode == "verified":
                    self._explainers[method_name] = VerifiedIGExplainer(
                        self.model._model, self.model._tokenizer)
                else:
                    self._explainers[method_name] = IGExplainer()
            elif method_name == "leave_one_out":
                from helm.explainers import LOOExplainer
                self._explainers[method_name] = LOOExplainer()
            elif method_name == "anchors":
                from helm.explainers import AnchorsExplainer
                self._explainers[method_name] = AnchorsExplainer()
            elif method_name == "counterfactual":
                from helm.explainers import CounterfactualExplainer
                self._explainers[method_name] = CounterfactualExplainer()
            elif method_name == "chain_of_thought":
                from helm.explainers import CoTExplainer
                self._explainers[method_name] = CoTExplainer()
            elif method_name == "logit_lens":
                from helm.explainers import LogitLensExplainer
                self._explainers[method_name] = LogitLensExplainer()
            else:
                raise ValueError(f"Méthode inconnue : {method_name}")
        return self._explainers[method_name]

    def explain(
        self,
        text: str,
        user_profile: Optional[UserProfile] = None,
        evaluate: bool = False,
        *,
        num_features: Optional[int] = None,
        evaluation_k: int = 3,
    ) -> HELMContext:
        """
        Pipeline complet : Texte → Explication adaptée.

        Args:
            text: Texte à expliquer.
            user_profile: Profil utilisateur (ou détection auto).
            evaluate: Si True, calcule fidélité et suffisance.
            num_features: Nombre de mots à expliquer ; None conserve les contraintes du profil.
            evaluation_k: Nombre de mots pour les métriques ; défaut 3 (protocole signé depuis a4).

        Returns:
            HELMContext contenant tous les résultats.
        """
        if num_features is not None and num_features < 1:
            raise ValueError("num_features doit être positif.")
        if evaluation_k < 1:
            raise ValueError("evaluation_k doit être positif.")
        ctx = HELMContext(
            text=text,
            model_name=self.model_name,
            user_profile=user_profile,
        )
        ctx.log(f"Début pipeline — texte='{text[:50]}...' profil={user_profile}")

        # ── Couche 1 : Caractérisation ──
        ctx = self.layer1.process(ctx)
        if num_features is not None:
            ctx.constraints = replace(ctx.constraints, num_features=num_features)
            ctx.log(f"Paramètres explicites — features={num_features}, évaluation k={evaluation_k}")

        # ── Prédiction ──
        probas = self.model.predict_proba([text])[0]
        label = "toxique" if probas[1] > 0.5 else "non_toxique"
        ctx.prediction = Prediction(
            label=label,
            confidence=float(probas[1]) if label == "toxique" else float(probas[0]),
            probabilities=tuple(float(p) for p in probas),
            model_name=self.model_name,
        )
        ctx.log(f"Prédiction : {label} ({ctx.prediction.confidence:.1%})")

        # ── Couche 2 : Sélection ──
        ctx = self.layer2.process(ctx)
        if self.contextual_selector is not None:
            from helm.config import MethodSelection
            policy = self.contextual_selector
            # Politique opt-in : son pool calibré remplace la matrice heuristique.
            if "integrated_gradients" in policy.methods and self.ig_mode != "verified" and "integrated_gradients" not in self._explainers:
                raise ValueError("LinUCB calibré avec IG exige ig_mode='verified', aucun LOO implicite.")
            raw = policy.context(text, ctx.prediction.confidence, int(probas[1] > probas[0]))
            ctx.contextual_choice = policy.choose(raw)
            ctx.selected_methods = [MethodSelection(
                method_name=ctx.contextual_choice.method, priority=1., estimated_time=None,
                rationale="LinUCB expérimental ; pool de calibration, latence non estimée")]
            ctx.log("Sélection LinUCB expérimentale ; aucune mise à jour automatique ni fusion.")

        # ── Exécution des méthodes XAI ──
        for method_sel in ctx.selected_methods:
            try:
                explainer = self._get_explainer(method_sel.method_name)
                attribution = explainer.explain(
                    text=text,
                    predict_fn=self.model.predict_proba,
                    num_features=ctx.constraints.num_features,
                )
                if attribution.metadata.get("valid") is False:
                    ctx.failed_attributions[method_sel.method_name] = attribution.metadata
                    raise ValueError("Attribution non admissible selon les contrôles numériques.")
                if ctx.contextual_choice is not None:
                    effective = attribution.metadata.get("effective_method", attribution.method_name)
                    if effective in {"lime", "shap"} and attribution.metadata.get("attribution_protocol") != "word_positions_v1":
                        raise ValueError("Le bras calibré exige le backend positionnel.")
                    if effective != method_sel.method_name:
                        raise ValueError(f"Méthode exécutée {effective} différente du bras calibré.")
                ctx.attributions[method_sel.method_name] = attribution
                ctx.log(
                    f"XAI {method_sel.method_name} — "
                    f"{len(attribution.token_scores)} tokens, "
                    f"{attribution.computation_time:.2f}s"
                )
            except Exception as e:
                ctx.failed_attributions.setdefault(method_sel.method_name, {"error": str(e), "type": type(e).__name__})
                ctx.log(f"Erreur XAI {method_sel.method_name} : {e}")
                logger.warning(f"Explainer {method_sel.method_name} a échoué : {e}")

        if ctx.contextual_choice is not None and not ctx.attributions:
            self.contextual_selector.discard(ctx.contextual_choice)
            ctx.contextual_choice = None
        # ── Évaluation optionnelle ──
        if evaluate and ctx.attributions:
            self._evaluate(ctx, k=evaluation_k)

        # ── Couche 3 : Présentation ──
        ctx = self.layer3.process(ctx)

        ctx.finalize()
        return ctx

    def _evaluate(self, ctx: HELMContext, k: int = 3) -> None:
        from helm.config import EvaluationMetrics
        from helm.evaluation import evaluate_attribution, compute_fidelity, compute_sufficiency
        for method_name, attr in ctx.attributions.items():
            if method_name in {'counterfactual', 'anchors'} or attr.metadata.get('score_kind') == 'norm_growth_proxy':
                ctx.log(f"Évaluation {method_name} — C/S non applicable ; consulter les mesures propres à la méthode.")
                continue
            try:
                if self.evaluation_protocol == "signed_cs":
                    result = evaluate_attribution(ctx.text, attr, self.model.predict_proba, k=k)
                    fid, suf = result.comprehensiveness, result.sufficiency
                    metadata = result.to_dict()
                else:
                    fid = compute_fidelity(ctx.text, attr, self.model.predict_proba, k=k)
                    suf = compute_sufficiency(ctx.text, attr, self.model.predict_proba, k=k)
                    metadata = {"protocol": "legacy_toxic_class", "target_class": 1}
                ctx.evaluation[method_name] = EvaluationMetrics(fidelity=fid, sufficiency=suf, metadata=metadata)
                ctx.log(f"Évaluation {method_name} — C={fid:.3f}, S={suf:.3f} ({self.evaluation_protocol})")
            except Exception as exc:
                ctx.log(f"Évaluation échouée {method_name} : {exc}")

    def submit_feedback(self, profile: UserProfile, method: str, rating: int) -> None:
        """Enregistre un feedback utilisateur pour UCB1."""
        self._preference_learner.record(profile, method, rating)
        logger.info(f"Feedback UCB1 : {profile.value}/{method} = {rating}/5")

    def get_statistics(self) -> Dict:
        """Statistiques UCB1."""
        return self._preference_learner.get_statistics()
