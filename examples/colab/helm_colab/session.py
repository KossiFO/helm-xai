"""Short notebook-facing helpers. All model computation stays in Colab."""
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone


class ColabExperiment:
    def __init__(self):
        assert Path('/content').is_dir(), 'Ce test doit être exécuté dans Google Colab.'
        self.source = Path(__file__).resolve().parents[3]
        self.root = Path('/content/helm_maaf_v1') / datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S')
        self.root.mkdir(parents=True, exist_ok=True)
        self.python = Path('/content/helm_tabular_env/bin/python')
        self.env = {**os.environ, 'PYTHONPATH':str(self.source/'examples/colab'), 'USE_TF':'0',
                    'MPLBACKEND':'Agg', 'OMP_NUM_THREADS':'1', 'HELM_RUN_ID':self.root.name}

    def install(self):
        log = self.root/'installation.log'
        with log.open('w') as stream:
            commands = [[sys.executable,'-m','pip','install','-q','uv'],
                        *([[sys.executable,'-m','uv','venv','--python','3.12','/content/helm_tabular_env']] if not self.python.exists() else []),
                        [sys.executable,'-m','uv','pip','install','--python',str(self.python),
                         'numpy==1.26.4','pandas==2.2.3','scikit-learn==1.6.1','shap==0.46.0',
                         'lime==0.2.0.1','matplotlib==3.9.4','joblib==1.4.2'],
                        [sys.executable,'-m','uv','pip','install','--python',str(self.python),'--no-deps','--reinstall',str(self.source)]]
            for command in commands:
                result = subprocess.run(command, stdout=stream, stderr=stream, env=self.env)
                if result.returncode:
                    raise RuntimeError(log.read_text()[-5000:])
        self.call(['-c', 'import helm,sys; print("HELM",helm.__version__,"—",helm.__file__); print("Python",sys.version.split()[0])'])
        print('Installation tabulaire dans Colab terminée. Dépendances NLP non installées ; test limité à TabularHELM.')
        print('Sources GitHub consultables :', self.source/'implementation_v2/helm/tabular')

    def call(self, arguments):
        result = subprocess.run([str(self.python), *arguments], env=self.env, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode:
            print(result.stderr[-6000:])
            raise RuntimeError('Le calcul Colab a échoué ; voir le journal ci-dessus.')

    def prepare(self):
        self.call(['-m','helm_colab.prepare'])

    def explain(self, method='shap'):
        self.call(['-m','helm_colab.worker','--method',method])

    def show(self):
        from IPython.display import HTML, display
        self.call(['-m','helm_colab.profile_view'])
        display(HTML((self.root/'dashboard.html').read_text()))

    def audit(self):
        from IPython.display import Code, display
        for method in ['shap','lime']:
            path = self.root/f'audit_{method}.json'
            if path.exists():
                display(Code(json.dumps(json.loads(path.read_text()),ensure_ascii=False,indent=2), language='json'))
