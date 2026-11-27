"""Informations partageables, sans chemins locaux, textes ni identité machine."""
from importlib.metadata import PackageNotFoundError, version
import platform

from helm import __version__


def diagnostics():
    dependencies = {}
    for name in ("numpy", "scipy", "scikit-learn", "shap", "lime", "fastapi"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "absent"
    return {
        "package_version": __version__,
        "python": platform.python_version(),
        "system": platform.system(),
        "dependencies": dependencies,
    }
