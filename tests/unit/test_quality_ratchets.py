"""Tests for scripts/check_ratchets.py and the coverage thresholds it enforces."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from coverage_config import (  # noqa: E402
    COVERAGE_THRESHOLDS,
    check_coverage_report,
    component_of,
)


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_ratchets", ROOT / "scripts" / "check_ratchets.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_legacy_mypy_lists_agree_and_only_shrink():
    pyproject = checker.load_pyproject()
    listed = set(checker.legacy_override(pyproject)["module"])
    budgets = checker.ratchets(pyproject)["mypy_legacy_errors"]
    assert listed == set(budgets)
    assert listed <= checker.LEGACY_MYPY_MODULES
    assert all(isinstance(n, int) and n > 0 for n in budgets.values())


def test_strict_config_drops_only_the_legacy_override():
    config = checker.strict_mypy_config(checker.load_pyproject())
    assert config.startswith("[mypy]\n")
    assert "disallow_untyped_defs = True" in config
    assert "active_inference.cli" not in config
    assert "[mypy-scipy.*,sympy.*,jinja2.*]" in config
    assert "ignore_missing_imports = True" in config


@pytest.mark.parametrize(
    ("path", "module"),
    [
        ("src/active_inference/__init__.py", "active_inference"),
        ("src/active_inference/cli.py", "active_inference.cli"),
        (
            "src/active_inference/tools/documentation/cli.py",
            "active_inference.tools.documentation.cli",
        ),
    ],
)
def test_module_name(path, module):
    assert checker.module_name(path) == module


@pytest.mark.parametrize(
    ("path", "component"),
    [
        ("src/active_inference/cli.py", "package"),
        ("src/active_inference/__init__.py", "package"),
        ("src/active_inference/llm/client.py", "llm"),
        ("src/active_inference/tools/documentation/cli.py", "tools"),
    ],
)
def test_component_of(path, component):
    assert component_of(path) == component


def _report(tmp_path, files, total):
    data = {
        "files": {
            path: {"summary": {"covered_lines": c, "num_statements": n}}
            for path, (c, n) in files.items()
        },
        "totals": {"percent_covered": total},
    }
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps(data))
    return report


def test_coverage_report_passes_at_thresholds(tmp_path):
    report = _report(
        tmp_path,
        {
            "src/active_inference/llm/a.py": (95, 100),
            "src/active_inference/cli.py": (100, 100),
        },
        99.0,
    )
    assert check_coverage_report(report) == []


def test_coverage_report_flags_component_and_overall(tmp_path):
    report = _report(
        tmp_path,
        {"src/active_inference/llm/a.py": (94, 100)},
        98.9,
    )
    problems = check_coverage_report(report)
    assert problems == ["llm: 94.00% < 95%", "overall: 98.90% < 99%"]


def test_coverage_report_flags_unthresholded_component(tmp_path):
    report = _report(tmp_path, {"src/active_inference/newpkg/a.py": (1, 1)}, 100.0)
    assert check_coverage_report(report) == [
        "component 'newpkg' has no coverage threshold"
    ]


def test_every_source_component_has_a_threshold():
    src = ROOT / "src" / "active_inference"
    components = {
        component_of(str(path.relative_to(ROOT))) for path in src.rglob("*.py")
    }
    assert components <= set(COVERAGE_THRESHOLDS)


def test_cli_rejects_unknown_command(capsys):
    assert checker.main([]) == 2
    assert checker.main(["coverage"]) == 2
    assert "Usage" in capsys.readouterr().out
