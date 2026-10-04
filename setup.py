"""Fail clearly instead of distributing a broken local interface."""
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithInterface(build_py):
    def run(self):
        static = Path(__file__).parent/'implementation_v2/helm/ui/static'
        if not (static/'index.html').is_file() or not list((static/'assets').glob('*.js')):
            raise RuntimeError('Interface absente. Construire les sources avec python tools/build_release.py (Node.js requis) ou installer la wheel publiée.')
        super().run()


setup(cmdclass={'build_py': BuildWithInterface})
