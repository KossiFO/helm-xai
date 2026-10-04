"""Backends LIME/SHAP positionnels du protocole Codex (septembre 2026)."""
import time
from threading import Lock
import numpy as np
from helm.config import Attribution
from helm.evaluation.protocol import probabilities

_SHAP_RNG_LOCK = Lock()


def _result(name, text, scores, metadata, base_value, start, num_features):
    if len(scores) != len(text.split()) or not np.isfinite(scores).all():
        raise ValueError("Scores non finis ou non alignés sur les positions de mots.")
    lexical = {}
    for word, score in zip(text.split(), scores):
        lexical[word] = lexical.get(word, 0.)+float(score)
    lexical = dict(sorted(lexical.items(), key=lambda item: -abs(item[1]))[:num_features])
    metadata.update(scores_by_position=scores.tolist(), effective_method=name,
                    attribution_protocol="word_positions_v1", display_name=name.upper())
    return Attribution(method_name=name, token_scores=lexical, base_value=base_value,
                       computation_time=time.perf_counter()-start, metadata=metadata)


def lime_positions(text, predict_fn, num_features, *, num_samples, seed):
    from lime.lime_text import LimeTextExplainer
    start = time.perf_counter()
    if not text.split() or num_features < 1:
        raise ValueError("Texte non vide et nombre de features positif requis.")
    target = int(np.argmax(probabilities(predict_fn, [text])[0]))
    explainer = LimeTextExplainer(random_state=seed, split_expression=r"\s+", bow=False)
    result = explainer.explain_instance(text, predict_fn, labels=(target,),
                                      num_features=min(num_features, len(text.split())), num_samples=num_samples)
    scores = np.zeros(len(text.split()))
    for index, value in result.as_map()[target]:
        scores[index] = value
    return _result("lime", text, scores,
                   dict(bow=False, split_expression=r"\s+", num_samples=num_samples,
                        num_features=num_features, seed=seed, target_class=target, local_r2=float(result.score)),
                   float(result.intercept[target]), start, num_features)


def shap_positions(text, predict_fn, num_features, *, max_evals):
    import shap
    start = time.perf_counter()
    if not text.split() or num_features < 1:
        raise ValueError("Texte non vide et nombre de features positif requis.")
    target = int(np.argmax(probabilities(predict_fn, [text])[0]))
    masker = shap.maskers.Text(tokenizer=r"\s+")
    explainer = shap.Explainer(lambda x: predict_fn(list(x)), masker=masker, algorithm="partition")
    with _SHAP_RNG_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(42)
            result = explainer([text], max_evals=max_evals, batch_size=4)
        finally:
            np.random.set_state(state)
    raw = [str(t).strip() for t in result.data[0]]
    keep = np.array([bool(t) for t in raw])
    if [t for t in raw if t] != text.split():
        raise ValueError("SHAP : segmentation non alignée sur les mots du protocole.")
    scores = np.asarray(result.values[0])[:, target]
    return _result("shap", text, scores[keep],
                   dict(algorithm=type(explainer).__name__, max_evals=max_evals, batch_size=4,
                        target_class=target, seed=42, split_expression=r"\s+",
                        nonword_segments_excluded=int((~keep).sum()),
                        nonword_attribution_sum=float(scores[~keep].sum())),
                   float(result.base_values[0][target]), start, num_features)
