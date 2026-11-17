"""Tests for the ``active_inference`` package initializer.

The optional-dependency fallbacks are exercised by executing the package's
``__init__.py`` under a throwaway module name whose subpackages are blocked in
``sys.modules``. The real ``active_inference`` package is never reloaded.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

import active_inference

SUBPACKAGES = (
    "knowledge",
    "research",
    "visualization",
    "applications",
    "tools",
    "platform",
    "llm",
)

FLAG_FOR = {
    "knowledge": "_HAS_KNOWLEDGE",
    "research": "_HAS_RESEARCH",
    "visualization": "_HAS_VISUALIZATION",
    "applications": "_HAS_APPLICATIONS",
    "tools": "_HAS_TOOLS",
    "platform": "_HAS_PLATFORM",
    "llm": "_HAS_LLM",
}

SHADOW = "_ai_init_shadow"


@pytest.fixture
def load_without_subpackages(monkeypatch):
    """Execute ``__init__.py`` with every optional subpackage unavailable."""
    init_path = Path(active_inference.__file__)

    def load():
        for sub in SUBPACKAGES:
            # A ``None`` entry makes ``from .sub import X`` raise ImportError.
            monkeypatch.setitem(sys.modules, f"{SHADOW}.{sub}", None)
        spec = importlib.util.spec_from_file_location(
            SHADOW, init_path, submodule_search_locations=[str(init_path.parent)]
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, SHADOW, module)
        spec.loader.exec_module(module)
        return module

    return load


def test_metadata_and_project_root():
    assert active_inference.__version__ == "0.1.0"
    assert active_inference.__author__ == "Active Inference Community"
    assert active_inference.__email__ == "community@activeinference.org"
    root = active_inference.PROJECT_ROOT
    assert root == Path(active_inference.__file__).parent.parent.parent
    assert (root / "src" / "active_inference" / "__init__.py").is_file()


def test_available_subpackages_are_exported():
    from active_inference.knowledge import KnowledgeRepository
    from active_inference.llm import OllamaClient

    assert active_inference._HAS_KNOWLEDGE is True
    assert active_inference._HAS_LLM is True
    assert active_inference.KnowledgeRepository is KnowledgeRepository
    assert active_inference.OllamaClient is OllamaClient


def test_all_lists_only_defined_unique_names():
    exported = active_inference.__all__
    assert len(exported) == len(set(exported))
    for name in exported:
        assert hasattr(active_inference, name), name
    for flag in FLAG_FOR.values():
        assert flag in exported
        assert isinstance(getattr(active_inference, flag), bool)


def test_missing_subpackages_fall_back_to_none(load_without_subpackages):
    shadow = load_without_subpackages()

    for flag in FLAG_FOR.values():
        assert getattr(shadow, flag) is False, flag

    flags = set(FLAG_FOR.values())
    public = [n for n in shadow.__all__ if n not in flags and n != "PROJECT_ROOT"]
    assert public, "expected exported component names"
    for name in public:
        assert getattr(shadow, name) is None, name

    # Metadata does not depend on optional imports.
    assert shadow.__version__ == active_inference.__version__
    assert shadow.PROJECT_ROOT == active_inference.PROJECT_ROOT
    assert shadow.__all__ == active_inference.__all__


def test_shadow_load_leaves_real_package_untouched(load_without_subpackages):
    from active_inference.knowledge import KnowledgeRepository

    load_without_subpackages()

    assert active_inference.KnowledgeRepository is KnowledgeRepository
    assert active_inference._HAS_KNOWLEDGE is True
