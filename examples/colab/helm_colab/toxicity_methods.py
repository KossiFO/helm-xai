"""Five native backends, computed once per comment, reused without refitting."""
from copy import deepcopy
import numpy as np

METHODS = ['shap', 'lime', 'integrated_gradients', 'anchors', 'counterfactual']


def compute_methods(model, comment):
    from helm.explainers import SHAPExplainer, LIMEExplainer, VerifiedIGExplainer, AnchorsExplainer, CounterfactualExplainer
    backends = {
        'shap': lambda: SHAPExplainer(max_evals=100),
        'lime': lambda: LIMEExplainer(num_samples=300, random_state=42),
        'integrated_gradients': lambda: VerifiedIGExplainer(model._model, model._tokenizer, max_nodes=8192),
        'anchors': lambda: AnchorsExplainer(use_native=True, num_samples=100, beam_size=4),
        'counterfactual': lambda: CounterfactualExplainer(max_combination_size=2),
    }
    results, failures = {}, {}
    for name, factory in backends.items():
        np.random.seed(42)
        print('Calcul natif : ' + name, flush=True)
        try:
            attr = factory().explain(comment, model.predict_proba, num_features=8)
            if attr.metadata.get('valid') is False:
                failures[name] = {'error':'Contrôles numériques IG non satisfaits', 'metadata':attr.metadata}
            else:
                results[name] = attr
            print(name + (' : terminé' if name in results else ' : non admissible'), flush=True)
        except Exception as error:
            failures[name] = {'error':str(error), 'type':type(error).__name__}
            print(name + ' : échec — ' + str(error), flush=True)
    return results, failures


class RecordedExplainer:
    def __init__(self, name, comment, result, failure):
        self.method_name, self.comment, self.result, self.failure = name, comment, result, failure

    def explain(self, text, predict_fn, num_features):
        if text != self.comment:
            raise ValueError('Ce résultat enregistré correspond à un autre commentaire.')
        if self.result is None:
            raise RuntimeError(str(self.failure))
        result = deepcopy(self.result)
        result.metadata['reused_across_profiles'] = True
        result.metadata['timing_context'] = 'Original method computation, not replay latency.'
        return result
