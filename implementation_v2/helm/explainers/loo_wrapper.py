"""Remplacement d'un mot à la fois, distincte des gradients intégrés."""
import time
import numpy as np
from helm.config import Attribution
from helm.evaluation.protocol import probabilities
from .base import BaseExplainer


class LOOExplainer(BaseExplainer):
    @property
    def method_name(self):
        return "leave_one_out"

    def estimate_time(self, text_length):
        return (max(1, text_length/5)+1)*.05

    def explain(self, text, predict_fn, num_features=8):
        start = time.perf_counter()
        words = text.split()
        if not words or isinstance(num_features, bool) or not isinstance(num_features, int) or num_features < 1:
            raise ValueError("Texte non vide et nombre de features positif requis.")
        interventions = [" ".join(words[:i]+["le"]+words[i+1:]) for i in range(len(words))]
        p = probabilities(predict_fn, [text, *interventions])
        target = int(np.argmax(p[0]))
        scores = p[0, target]-p[1:, target]
        lexical = {}
        for word, score in zip(words, scores):
            lexical[word] = lexical.get(word, 0.)+float(score)
        lexical = dict(sorted(lexical.items(), key=lambda item: -abs(item[1]))[:num_features])
        return Attribution(method_name=self.method_name, token_scores=lexical,
                           computation_time=time.perf_counter()-start, base_value=float(p[0, target]),
                           metadata={"effective_method": self.method_name, "display_name": "LOO (remplacement par le)",
                                     "mode": "leave_one_out_replacement", "target_class": target,
                                     "scores_by_position": scores.tolist(), "replacement": "le"})
