"""Shared pytest fixtures."""

import shutil
from pathlib import Path

import pytest

from active_inference.llm import models


@pytest.fixture(autouse=True)
def isolated_model_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ModelManager's default registry at a per-test copy.

    ModelManager saves its registry on every change; without this, tests
    rewrite the packaged ``model_registry.json`` and race under parallel runs.
    """
    registry = tmp_path / "model_registry.json"
    if models.DEFAULT_REGISTRY_PATH.exists():
        shutil.copyfile(models.DEFAULT_REGISTRY_PATH, registry)
    monkeypatch.setattr(models, "DEFAULT_REGISTRY_PATH", registry)
    return registry
