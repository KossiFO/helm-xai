"""IG natif XLM-R : probabilité, PAD lexical, contrôles à deux tolérances.

Port du protocole Codex adaptive_verified_code.py du 9 septembre 2026.
Les échecs numériques restent des échecs ; aucun remplacement par LOO.
"""
import re
import time
import numpy as np
from helm.config import Attribution
from helm.evaluation.protocol import probabilities
from .base import BaseExplainer


class VerifiedIGExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, *, max_nodes=32768):
        if getattr(getattr(model, "config", None), "model_type", None) != "xlm-roberta":
            raise ValueError("Ce protocole de positions a été vérifié pour XLM-R uniquement.")
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1152:
            raise ValueError("Budget minimal : 1152 nœuds par tolérance.")
        self.model, self.tokenizer, self.max_nodes = model, tokenizer, max_nodes

    @property
    def method_name(self):
        return "integrated_gradients"

    def estimate_time(self, text_length):
        # Aucune calibration de latence opérationnelle pour ce backend adaptatif.
        return float("inf")

    def explain(self, text, predict_fn, num_features=8):
        import torch
        from captum.attr import IntegratedGradients

        start = time.perf_counter()
        spans = list(re.finditer(r"\S+", text))
        if not spans or isinstance(num_features, bool) or not isinstance(num_features, int) or num_features < 1:
            raise ValueError("Texte non vide et nombre de features positif requis.")
        self.model.eval()
        device = next(self.model.parameters()).device
        enc = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"].to(device)
        if ids.shape[1] > 512:
            raise ValueError("Texte dépassant 512 sous-tokens ; aucune troncature silencieuse.")
        pad = self.tokenizer.pad_token_id
        if pad is None:
            raise ValueError("Identifiant PAD requis.")
        mask = enc["attention_mask"].to(device)
        positions = ids.ne(pad).long().cumsum(1)*ids.ne(pad).long()+pad
        base_ids = ids.clone()
        mapping = np.zeros((len(offsets), len(spans)))
        for j, (a, b) in enumerate(offsets):
            if b > a:
                base_ids[0, j] = pad
            overlap = np.array([max(0, min(b, s.end())-max(a, s.start())) for s in spans])
            if overlap.sum():
                mapping[j] = overlap / overlap.sum()
        target = int(np.argmax(probabilities(predict_fn, [text])[0]))
        with torch.no_grad():
            x = self.model.get_input_embeddings()(ids).detach()
            baseline = self.model.get_input_embeddings()(base_ids).detach()
        direction = x-baseline

        def forward(embeddings, attention, position_ids):
            return self.model(inputs_embeds=embeddings, attention_mask=attention,
                              position_ids=position_ids).logits.softmax(-1)[:, target]

        with torch.no_grad():
            original = float(forward(x, mask, positions).item())
            base_probability = float(forward(baseline, mask, positions).item())
        if abs(original-probabilities(predict_fn, [text])[0, target]) > 1e-5:
            raise ValueError("Le forward IG ne reproduit pas la probabilité du classifieur.")
        gap = original-base_probability
        integrator = IntegratedGradients(forward)

        def integrate(relative_tolerance, nodes_pair):
            used = 0
            def segment(left, right):
                nonlocal used
                vectors, deltas = [], []
                for n in nodes_pair:
                    values, delta = integrator.attribute(
                        baseline+direction*right, baselines=baseline+direction*left,
                        additional_forward_args=(mask, positions), n_steps=n,
                        internal_batch_size=4, return_convergence_delta=True, method="gausslegendre")
                    vectors.append(values.sum(-1)[0].detach().cpu().numpy().astype(float)@mapping)
                    deltas.append(float(delta.item()))
                    used += n
                error = float(np.abs(vectors[1]-vectors[0]).sum())
                return dict(left=left, right=right, scores=vectors[1], delta=deltas[1],
                            error=error, priority=error+abs(deltas[1]))
            leaves = [segment(j/16, (j+1)/16) for j in range(16)]
            while True:
                total = sum((s["scores"] for s in leaves), np.zeros(len(spans)))
                delta = float(total.sum()-gap)
                sum_absolute_deltas = sum(abs(s["delta"]) for s in leaves)
                error = sum(s["error"] for s in leaves)
                gate = .01+.01*abs(gap)
                error_tolerance = .0005+relative_tolerance*float(np.abs(total).sum())
                finite = bool(np.isfinite(total).all() and np.isfinite([delta, sum_absolute_deltas, error]).all())
                passed = finite and abs(delta) <= gate and sum_absolute_deltas <= gate/4 and error <= error_tolerance
                if passed or not finite or used+2*sum(nodes_pair) > self.max_nodes:
                    break
                old = leaves.pop(max(range(len(leaves)), key=lambda i: leaves[i]["priority"]))
                middle = (old["left"]+old["right"])/2
                leaves.extend([segment(old["left"], middle), segment(middle, old["right"])])
            return total, dict(relative_tolerance=relative_tolerance, quadrature_nodes_pair=list(nodes_pair), total_nodes_used=used,
                              passes_internal_checks=bool(passed), delta=delta, gate_tolerance=gate,
                              sum_absolute_segment_deltas=float(sum_absolute_deltas),
                              word_error_estimate=float(error), word_error_tolerance=error_tolerance)

        coarse, first = integrate(.025, (16, 32))
        fine, second = integrate(.0125, (24, 48))
        def top(scores):
            return set(np.argsort(-np.abs(scores), kind="stable")[:min(3, len(spans))].tolist())
        distance = float(np.abs(coarse-fine).sum())
        agreement_tolerance = .001+.05*max(float(np.abs(coarse).sum()), float(np.abs(fine).sum()))
        valid = bool(first["passes_internal_checks"] and second["passes_internal_checks"]
                     and distance <= agreement_tolerance and top(coarse) == top(fine))
        metadata = dict(mode="verified_native", effective_method="integrated_gradients",
                        display_name="IG natif vérifié" if valid else "IG natif — non admissible",
                        protocol="adaptive_ig_xlmr_probability_v1", target_space="probability", target_class=target,
                        baseline="PAD lexical ; tokens spéciaux, attention et positions fixes",
                        input_probability=original, baseline_probability=base_probability,
                        max_nodes_per_tolerance=self.max_nodes, checks=[first, second], valid=valid,
                        agreement_l1=distance, agreement_tolerance=agreement_tolerance,
                        top3_agreement=top(coarse) == top(fine), scores_by_position=fine.tolist())
        # La vue lexicale regroupe les répétitions ; les métriques utilisent les positions complètes.
        token_scores = {}
        if valid:
            for span, score in zip(spans, fine):
                token_scores[span.group()] = token_scores.get(span.group(), 0.)+float(score)
            token_scores = dict(sorted(token_scores.items(), key=lambda item: -abs(item[1]))[:num_features])
        return Attribution(method_name=self.method_name, token_scores=token_scores,
                           computation_time=time.perf_counter()-start, base_value=base_probability,
                           convergence_delta=second["delta"], metadata=metadata)
