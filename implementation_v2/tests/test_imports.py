"""Chaque module du package doit s'importer (précédent : ucb1_auto_reward cassé)."""
import importlib
import pkgutil

import pytest

import helm


def _all_modules():
    for m in pkgutil.walk_packages(helm.__path__, prefix="helm."):
        yield m.name


@pytest.mark.parametrize("name", sorted(_all_modules()))
def test_module_imports(name):
    if name.startswith("helm.visualization"):
        pytest.importorskip("matplotlib")
    if name.startswith("helm.ui"):
        pytest.importorskip("fastapi")
    importlib.import_module(name)


def test_public_api():
    assert helm.__version__
    assert helm.HELMPipeline and helm.UCB1Learner and helm.UCB1AutoReward
    assert helm.UserProfile.MODERATOR.value == "moderateur"
