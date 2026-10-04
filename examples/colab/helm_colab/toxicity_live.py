"""Run a freely entered comment in the isolated Colab Python environment."""
import json
import subprocess
import tempfile
from pathlib import Path


class LiveBackend:
    def __init__(self, experiment):
        self.experiment = experiment
        self.last_root = None

    def calculate(self, text, explain, progress):
        # Each run owns its files. Never replace the recorded two-comment recipe.
        root = Path(tempfile.mkdtemp(prefix='helm_comment_', dir='/content'))
        self.last_root = root
        request = root/'request.json'
        request.write_text(json.dumps([text], ensure_ascii=False))
        command = [self.experiment.python, '-u', '-m', 'helm_colab.toxicity',
                   '--request', str(request), '--root', str(root)]
        if not explain:
            command.append('--prediction-only')
        process = subprocess.Popen(command, env=self.experiment.env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        try:
            with (root/'calculation.log').open('w') as log:
                for line in process.stdout:
                    log.write(line)
                    tail = (tail + [line])[-8:]
                    if line.startswith('PREDICTION_JSON '):
                        payload = json.loads(line.removeprefix('PREDICTION_JSON '))
                        progress(f"Score de toxicité : {payload['prediction']['probabilities'][1]:.2%}. " +
                                 ('Calcul des explications en cours…' if explain else 'Prédiction terminée.'))
                    elif line.startswith(('Calcul natif :', 'shap :', 'lime :', 'integrated_gradients :', 'anchors :', 'counterfactual :')):
                        progress(line.strip())
            if process.wait():
                raise RuntimeError(''.join(tail)[-2000:])
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
        return {'root':str(root), 'explained':explain,
                'prediction':json.loads((root/'prediction.json').read_text())}

    def render(self, result, profile):
        if not result['explained']:
            from helm.visualization.text_input import prediction_html
            return prediction_html(result['prediction'])
        subprocess.run([self.experiment.python, '-m', 'helm_colab.toxicity', '--render',
                        '--root', result['root'], '--profile', profile],
                       env=self.experiment.env, check=True, capture_output=True, text=True)
        return (Path(result['root'])/'dashboard.html').read_text()
