"""Separate real-transformer Colab experiment; no heuristic or sentiment proxy."""
import json
import os
from pathlib import Path
from dataclasses import asdict
from importlib.metadata import distributions
import numpy as np

COMMENTS = ["Tu es un idiot, ferme ta gueule !", "Je ne partage pas ton avis, mais merci pour cette explication."]
ROOT = Path('/content/helm_toxicity_five_v1')


def run(comments=None, root=ROOT, prediction_only=False):
    comments = COMMENTS if comments is None else comments
    if not comments or any(not isinstance(c, str) or not c.strip() for c in comments):
        raise ValueError("Saisissez un commentaire non vide.")
    root = Path(root)
    assert Path('/content').is_dir(), 'Exécution requise dans Colab.'
    os.environ['USE_TF'] = '0'
    import torch
    from helm import HELMPipeline
    from helm.config import UserProfile, MODEL_REGISTRY
    from helm.models import ModelManager
    from .toxicity_methods import METHODS, compute_methods, RecordedExplainer
    from helm.visualization.profile_html import context_html
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    model = ModelManager('xlmr').load()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model._model.to(device)
    model._pipeline.device = device
    print('Modèle de toxicité chargé :', MODEL_REGISTRY['xlmr'], 'sur', device, flush=True)
    # Compute method latencies without a cross-method prediction cache.
    results = []
    coverage = []
    for index, comment in enumerate(comments):
        token_count = len(model._tokenizer(comment, add_special_tokens=True)['input_ids'])
        if token_count > 512:
            raise ValueError(f'Commentaire trop long : {token_count} tokens, maximum 512. Raccourcissez-le ; aucun texte tronqué ne sera expliqué.')
        prediction = model.predict_single(comment)
        payload = {'text':comment, 'prediction':asdict(prediction), 'token_count':token_count,
                   'model':MODEL_REGISTRY['xlmr'], 'device':str(device)}
        (root/'prediction.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print('PREDICTION_JSON ' + json.dumps(payload, ensure_ascii=False), flush=True)
        if prediction_only:
            continue
        computed, failures = compute_methods(model, comment)
        coverage.append({"comment_id":index,"successful":list(computed),"failures":failures})
        for profile in UserProfile:
            np.random.seed(42)
            pipeline = HELMPipeline('xlmr', model=model, allowed_methods=METHODS, ig_mode='verified',
                explainers={name:RecordedExplainer(name, comment, computed.get(name), failures.get(name)) for name in METHODS})
            ctx = pipeline.explain(comment, user_profile=profile, evaluate=True, num_features=8, evaluation_k=3)
            selected = [m.method_name for m in ctx.selected_methods]
            ctx.failed_attributions.update({name:failures[name] for name in selected if name in failures})
            record = {'comment_id':index, 'text':comment, 'profile':profile.value, 'prediction':asdict(ctx.prediction),
                      'pool':METHODS, 'computed_attributions':{k:asdict(v) for k,v in computed.items()},
                      'computation_failures':failures, 'selected_methods':selected, 'selection_details':[asdict(m) for m in ctx.selected_methods],
                      'failed_attributions':ctx.failed_attributions, 'attributions':{k:asdict(v) for k,v in ctx.attributions.items()},
                      'evaluation':{k:asdict(v) for k,v in ctx.evaluation.items()},'logs':ctx.logs,
                      'formatted_output':asdict(ctx.formatted_output)}
            results.append(record)
            (root/f'{index}_{profile.value}.html').write_text(context_html(ctx, available=computed, failures=failures))
            print(json.dumps({'comment_id':index,'profile':profile.value,'toxicity_score':ctx.prediction.probabilities[1],
                              'selected':selected,'successful':list(ctx.attributions)}, ensure_ascii=False), flush=True)
            (root/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str))
    if prediction_only:
        return
    assert len({r['profile'] for r in results}) == 4
    for index in range(len(comments)):
        assert len({r['prediction']['probabilities'] for r in results if r['comment_id']==index}) == 1
    audit = {'model':MODEL_REGISTRY['xlmr'], 'environment_versions':{d.metadata['Name']:d.version for d in distributions()},
             'seed':42,'num_features':8,'evaluation_k':3,'results':len(results), 'profiles':4,
             'same_predictions_across_profiles':True,'learned_preferences':False,'pool':METHODS,
             'device':str(device),'method_runs':coverage,'attributions_reused_across_profiles':True}
    (root/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print('AUDIT_FINAL',json.dumps(audit,ensure_ascii=False),flush=True)


def show(root=ROOT, profile=None):
    root = Path(root)
    from helm.visualization.profile_html import toxicity_dashboard
    records = json.loads((root/'results.json').read_text())
    if profile is not None:
        records = [r for r in records if r['profile'] == profile]
    toxicity_dashboard(records).to_html(root/'dashboard.html')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--request')
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--profile')
    parser.add_argument('--prediction-only', action='store_true')
    args = parser.parse_args()
    if args.render:
        show(args.root, args.profile)
    else:
        comments = json.loads(Path(args.request).read_text()) if args.request else None
        run(comments, args.root, args.prediction_only)
