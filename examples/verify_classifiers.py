"""Run: python examples/verify_classifiers.py --output /tmp/helm-classifiers

Requires helm-xai[boosting,statistics]. Software compatibility test only.
"""
import argparse
import json
import platform
import helm
from importlib.metadata import version
from pathlib import Path

import numpy as np
from helm.tabular import TabularHELM
from classifier_models import fit_classifier, sample


def verify(name, output):
    X_train, X_test, y_train, y_test = sample()
    model, original_probability = fit_classifier(name, X_train, y_train)
    expected = original_probability(X_test)
    before = model.predict_proba(X_test).copy()
    np.testing.assert_allclose(before[:, 1], expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(before.sum(axis=1), 1, atol=1e-7)
    helm = TabularHELM(model, X_train, target_class=1, background_size=20)
    for order in ('highest', 'lowest'):
        ranking = helm.rank(X_test, k=5, order=order)
        idx = np.argsort(-expected if order == 'highest' else expected, kind='stable')[:5]
        assert list(ranking.index) == list(X_test.iloc[idx].index)
    positives = helm.rank(X_test, k=5, group='predicted_target')
    assert len(positives) == 5 and positives.prediction.eq(1).all()
    row = X_test.loc[[positives.index[0]]]
    checks = {}
    for profile in ('metier', 'technique', 'utilisateur', 'audit'):
        report = helm.explain_profile(row, profile=profile, k=1, labels=y_test.loc[row.index],
                                      budgets={'shap': 128, 'lime': 256})
        for method, cohort in report.reports.items():
            assert not cohort.errors and len(cohort.explanations) == 1
            exp = cohort.explanations[0]
            assert exp.prediction == 1 and exp.target_class == 1
            np.testing.assert_allclose(exp.target_score, original_probability(row)[0], atol=1e-12)
            assert set(exp.contributions) == set(X_test.columns)
            assert np.isfinite(list(exp.contributions.values())).all()
            if method == 'shap':
                np.testing.assert_allclose(exp.quality['base_value'] + sum(exp.contributions.values()),
                                           exp.target_score, atol=1e-6)
            else:
                assert np.isfinite(exp.quality['local_r2'])
            checks[method] = exp.quality
        report.to_html(output / f'{name}-{profile}.html')
    np.testing.assert_allclose(before, model.predict_proba(X_test), atol=0, rtol=0)
    return {'status': 'passed', 'train_rows': len(X_train), 'test_rows': len(X_test),
            'features': list(X_test.columns), 'positive_rows_checked': len(positives),
            'profiles': ['metier', 'technique', 'utilisateur', 'audit'], 'explanation_quality': checks,
            'adapter': type(model).__name__}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {'python': platform.python_version(), 'helm_location': str(Path(helm.__file__).resolve()), 'versions': {p: version(p) for p in ('helm-xai', 'xgboost', 'lightgbm', 'statsmodels',
                                                   'pygam', 'scikit-learn', 'numpy', 'pandas', 'scipy')},
               'models': {}}
    for name in ('xgboost', 'lightgbm', 'glm', 'gam'):
        try:
            results['models'][name] = verify(name, args.output)
        except Exception as exc:
            results['models'][name] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        print(name, results['models'][name]['status'], results['models'][name].get('error', ''), flush=True)
    (args.output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    if any(r['status'] != 'passed' for r in results['models'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
