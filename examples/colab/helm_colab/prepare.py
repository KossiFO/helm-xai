"""Execute in Colab only: reproduce the previous MAAF temporal benchmark."""
import hashlib
import json
import platform
import os
from pathlib import Path

import joblib
import pandas as pd

from .data import prepare
from .mixed_preprocessing import normalize, transform

SOURCE_SHA256 = 'babbb9537279d9e930d733c1371ffe8877900bf030059fce1aee3d44935f1289'
ROOT = Path('/content/helm_maaf_v1') / os.environ.get('HELM_RUN_ID', 'manual')


def main():
    assert Path('/content').is_dir(), 'Exécuter cette expérience dans Colab.'
    source = Path('/content/sample_maaf_entrainement_20260825_095534.csv')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    model, schema, _, _, audit = prepare(source, seed=43561735, cutoff='2025-03-01')
    frame = pd.read_csv(source, low_memory=False)
    train = pd.to_datetime(frame.date_survenance) < '2025-03-01'
    X_train = transform(normalize(frame.loc[train]), schema).reset_index(drop=True)
    X_test = transform(normalize(frame.loc[~train]), schema).reset_index(drop=True)
    X_test.index = [f'Test {i:04d}' for i in range(len(X_test))]
    y_test = pd.Series(frame.loc[~train, 'bc_fraude_gmf'].to_numpy(dtype=int), index=X_test.index)
    ROOT.mkdir(exist_ok=True)
    joblib.dump((model, X_train, X_test, y_test), ROOT/'experiment.joblib')
    audit['execution'] = {'platform':platform.platform(), 'python':platform.python_version(), 'environment':'Google Colab', 'target':'bc_fraude_gmf'}
    # No per-record values in this summary; full data remain in the Colab runtime.
    audit.pop('categorical_audit', None)
    audit['unused_preparation_background_rows'] = audit.pop('background_rows')
    (ROOT/'preparation.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({k:audit[k] for k in ['train_rows','test_rows','test_positive_labels','numeric_features','categorical_features','metrics_sample_holdout']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
