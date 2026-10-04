"""Actual HELM calls, executed by the isolated Python inside Google Colab."""
import argparse
import copy
import json
import time
from importlib.metadata import distributions

import joblib
import numpy as np

from helm import TabularHELM, __version__
from .prepare import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['shap','lime'], required=True)
    args = parser.parse_args()
    model, X_train, X_test, y_test = joblib.load(ROOT/'experiment.joblib')
    helm = TabularHELM(model, X_train, target_class=1, background_size=50, random_state=42)
    native = helm.rank(X_test, k=len(X_test))
    np.testing.assert_allclose(native.target_score.to_numpy(), np.sort(model.predict_proba(X_test)[:,1])[::-1])
    empty = helm.explain_top(X_test, group='predicted_target', k=10, method=args.method)
    assert len(empty.ranking) == int((model.predict(X_test) == 1).sum()) == 0
    audit = {'helm_version': __version__, 'method':args.method, 'target_class':1, 'background_rows':50,
             'background_sha256':helm.background_fingerprint,
             'environment_versions':{d.metadata['Name']:d.version for d in distributions()},
             'explanation_seed':42, 'budget':512 if args.method=='shap' else 2048,
             'input_rows':len(X_test), 'observed_ones':int(y_test.sum()), 'predicted_ones':0,
             'rank_check':True, 'empty_predicted_target_checked':True, 'groups':{}}
    for order in ['highest','lowest']:
        start = time.perf_counter()
        # Native HELM: class of interest stays 1 even when every decision is 0.
        report = helm.explain_top(X_test, k=10, order=order, labels=y_test,
                                 method=args.method, budget=audit['budget'], profile='metier')
        elapsed = time.perf_counter() - start
        joblib.dump(report, ROOT/f'{args.method}_{order}.joblib')
        for profile in ['utilisateur','metier','technique','audit']:
            view = copy.deepcopy(report)
            view.profile = profile
            for local in view.explanations:
                local.profile = profile
            view.to_html(ROOT/f'{args.method}_{order}_{profile}.html')
        scores = report.ranking.target_score
        successful = report.explanations
        stats = {'requested':len(report.ranking), 'successful':len(successful), 'failures':report.errors,
                 'observed_ones':int(report.ranking.observed_label.sum()),
                 'score_min':float(scores.min()), 'score_max':float(scores.max()), 'seconds':elapsed,
                 'summary_top_5':report.summary().head(5).to_dict(orient='records')}
        assert all(e.target_class==1 and e.prediction==0 and len(e.contributions)==17 for e in successful)
        if successful and args.method=='shap':
            stats['max_additivity_residual'] = max(abs(e.quality['additivity_residual']) for e in successful)
            stats['base_value'] = successful[0].quality['base_value']
        if successful and args.method=='lime':
            quality = [e.quality['local_r2'] for e in successful]
            stats['local_r2_min'], stats['local_r2_max'] = min(quality), max(quality)
        audit['groups'][order] = stats
        print(json.dumps({order:stats}, ensure_ascii=False), flush=True)
    (ROOT/f'audit_{args.method}.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    assert all(g['successful']==g['requested'] for g in audit['groups'].values()), 'Explications incomplètes : voir les erreurs conservées.'


if __name__ == '__main__':
    main()
