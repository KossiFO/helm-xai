"""Install and run the separate toxicity experiment inside Colab."""
import os
import subprocess
import sys
from pathlib import Path


class ToxicityExperiment:
    def __init__(self):
        assert Path('/content').is_dir(), 'Exécution requise dans Colab.'
        self.source = Path(__file__).resolve().parents[3]
        self.python = '/content/helm_toxicity_five_env/bin/python'
        self.env = {**os.environ, 'PYTHONPATH':str(self.source/'examples/colab'), 'USE_TF':'0',
                    'MPLBACKEND':'Agg','OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}

    def install(self):
        commands = [[sys.executable,'-m','pip','install','-q','uv','ipywidgets>=8,<9'],
            *([[sys.executable,'-m','uv','venv','--python','3.12','/content/helm_toxicity_five_env']] if not Path(self.python).exists() else []),
            [sys.executable,'-m','uv','pip','install','--python',self.python,'torch==2.6.0'],
            [sys.executable,'-m','uv','pip','install','--python',self.python,'numpy==1.26.4','pandas==2.2.3','scikit-learn==1.6.1','shap==0.46.0','lime==0.2.0.1','matplotlib==3.9.4','transformers==4.48.3','sentencepiece==0.2.0','protobuf<5','captum==0.8.0','anchor-exp==0.0.2','spacy>=3.7,<4'],
            [sys.executable,'-m','uv','pip','install','--python',self.python,'--no-deps','--reinstall',str(self.source)]]
        for command in commands:
            subprocess.run(command,env=self.env,check=True)
        print('Installation terminée dans Colab ; modèle réel XLM-R toxicité ; GPU si disponible.')

    def run(self):
        process = subprocess.Popen([self.python,'-u','-m','helm_colab.toxicity'], env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError('Le calcul toxicité a échoué ; voir le journal ci-dessus.')

    def show(self):
        from IPython.display import HTML, display
        subprocess.run([self.python,'-m','helm_colab.toxicity','--render'],env=self.env,check=True)
        display(HTML(Path('/content/helm_toxicity_five_v1/dashboard.html').read_text()))

    def audit(self):
        from IPython.display import HTML, display
        from html import escape
        source = self.source/'examples/colab/helm_colab'
        content = '<details><summary>Détails techniques et reproductibilité</summary>'
        content += '<p>Pour la relecture scientifique : provenance, versions, paramètres et échecs. Inutile pour explorer les explications.</p>'
        content += '<h3>Audit du calcul</h3><pre>' + escape(Path('/content/helm_toxicity_five_v1/audit.json').read_text()) + '</pre>'
        for filename in ('toxicity.py','toxicity_methods.py'):
            content += '<details><summary>Code exécuté : ' + filename + '</summary><pre>' + escape((source/filename).read_text()) + '</pre></details>'
        display(HTML(content + '</details>'))

    def interactive(self):
        """Free text and live predictions, without altering the saved example recipe."""
        package_path = str(self.source/'implementation_v2')
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
        from helm.visualization.text_input import TextInput
        from .toxicity_live import LiveBackend
        backend = LiveBackend(self)
        self.live_backend = backend
        self.input = TextInput(backend.calculate, backend.render)
        return self.input.show()
