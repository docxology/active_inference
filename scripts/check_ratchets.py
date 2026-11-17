"""Enforce the repository's quality ratchets.

Budgets live in ``[tool.active_inference.ratchets]`` in ``pyproject.toml``.
Every budget must equal the measured value: an increase is a regression, and a
decrease must be recorded by lowering the budget in the same change, so the
numbers can only move toward zero.

Usage::

    python scripts/check_ratchets.py mypy      # legacy-module strict-mypy errors
    python scripts/check_ratchets.py sphinx    # documentation build warnings
    python scripts/check_ratchets.py coverage coverage.json
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# The legacy list may only shrink. Adding a module here is an explicit,
# reviewable decision; adding it to pyproject.toml alone fails the check.
LEGACY_MYPY_MODULES = frozenset(
    {
        "active_inference",
        "active_inference.cli",
        "active_inference.knowledge.mathematics",
        "active_inference.knowledge.repository",
        "active_inference.llm.client",
        "active_inference.platform.knowledge_graph",
        "active_inference.platform.search",
        "active_inference.research.analysis",
        "active_inference.tools.documentation.cli",
        "active_inference.tools.documentation.generator",
        "active_inference.tools.documentation.reviewer",
        "active_inference.tools.utilities",
        "active_inference.visualization.animations",
        "active_inference.visualization.comparative",
    }
)


def load_pyproject() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def ratchets(pyproject: dict[str, Any]) -> dict[str, Any]:
    table: dict[str, Any] = pyproject["tool"]["active_inference"]["ratchets"]
    return table


def legacy_override(pyproject: dict[str, Any]) -> dict[str, Any]:
    """Require one exact legacy exemption and only the known test exemptions.

    A second exemption, including a wildcard that reaches project modules,
    must not survive in the configuration used to measure strict diagnostics.
    """
    candidates = []
    for override in pyproject["tool"]["mypy"]["overrides"]:
        if not override.get("ignore_errors"):
            continue
        modules = override["module"]
        modules = [modules] if isinstance(modules, str) else modules
        if modules and set(modules) <= {"tests.*", "test_*"}:
            continue
        candidates.append(override)
    if len(candidates) != 1:
        raise SystemExit("expected exactly one non-test ignore_errors override")
    legacy = dict(candidates[0])
    modules = legacy["module"]
    modules = [modules] if isinstance(modules, str) else modules
    if not modules or not set(modules) <= LEGACY_MYPY_MODULES:
        raise SystemExit("legacy ignore_errors override contains unapproved modules")
    return legacy


def _ini_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def strict_mypy_config(pyproject: dict[str, Any]) -> str:
    """Render the mypy configuration as ini with the legacy override disabled."""
    mypy = dict(pyproject["tool"]["mypy"])
    overrides = mypy.pop("overrides", [])
    legacy = legacy_override(pyproject)
    lines = ["[mypy]"]
    lines += [f"{key} = {_ini_value(value)}" for key, value in mypy.items()]
    for override in overrides:
        if override == legacy:
            continue
        modules = override["module"]
        modules = [modules] if isinstance(modules, str) else modules
        lines.append(f"[mypy-{','.join(modules)}]")
        lines += [
            f"{key} = {_ini_value(value)}"
            for key, value in override.items()
            if key != "module"
        ]
    return "\n".join(lines) + "\n"


def module_name(path: str) -> str:
    parts = Path(path).with_suffix("").parts
    parts = parts[parts.index("active_inference") :]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def check_mypy() -> list[str]:
    pyproject = load_pyproject()
    budgets: dict[str, int] = ratchets(pyproject)["mypy_legacy_errors"]
    listed = set(legacy_override(pyproject)["module"])
    problems = []
    if added := listed - LEGACY_MYPY_MODULES:
        problems.append(f"modules added to the legacy mypy list: {sorted(added)}")
    if listed != set(budgets):
        problems.append(
            "legacy mypy list and mypy_legacy_errors budgets name different "
            f"modules: {sorted(listed ^ set(budgets))}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "mypy.ini"
        config.write_text(strict_mypy_config(pyproject))
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "src/", "--config-file", str(config)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode not in (0, 1):
        return problems + [f"mypy did not run cleanly:\n{result.stdout}{result.stderr}"]
    counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        match = re.match(r"^(src/\S+?\.py):\d+(?::\d+)?: error:", line)
        if match:
            name = module_name(match.group(1))
            counts[name] = counts.get(name, 0) + 1
    for name in sorted(set(counts) - listed):
        problems.append(f"{name}: {counts[name]} strict-mypy errors outside the list")
    for name in sorted(listed):
        found, budget = counts.get(name, 0), budgets.get(name)
        if found == 0:
            problems.append(f"{name}: passes strict mypy; remove it from both lists")
        elif budget is None:
            continue
        elif found > budget:
            problems.append(f"{name}: {found} mypy errors, budget {budget}")
        elif found < budget:
            problems.append(f"{name}: {found} mypy errors; lower its budget to {found}")
    total = sum(counts.get(name, 0) for name in listed)
    print(f"mypy legacy errors: {total} across {len(listed)} modules")
    return problems


def check_sphinx() -> list[str]:
    budget: int = ratchets(load_pyproject())["sphinx_warnings"]
    with tempfile.TemporaryDirectory() as tmp:
        warnings_file = Path(tmp) / "warnings.txt"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-q",
                "-b",
                "html",
                "-w",
                str(warnings_file),
                "docs/",
                str(Path(tmp) / "html"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        text = warnings_file.read_text() if warnings_file.exists() else ""
    if result.returncode != 0:
        return [f"sphinx-build failed:\n{result.stdout}{result.stderr}"]
    found = sum(1 for line in text.splitlines() if "WARNING:" in line)
    print(f"sphinx warnings: {found} (budget {budget})")
    if found > budget:
        return [f"sphinx warnings rose to {found}, budget {budget}:\n{text}"]
    if found < budget:
        return [f"sphinx warnings fell to {found}; lower the budget to {found}"]
    return []


def check_coverage(report: Path) -> list[str]:
    sys.path.insert(0, str(ROOT / "tests"))
    from coverage_config import check_coverage_report

    return check_coverage_report(report)


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in {"mypy", "sphinx", "coverage"}:
        print(__doc__)
        return 2
    if argv[0] == "mypy":
        problems = check_mypy()
    elif argv[0] == "sphinx":
        problems = check_sphinx()
    else:
        if len(argv) != 2:
            print(__doc__)
            return 2
        problems = check_coverage(Path(argv[1]))
    for problem in problems:
        print(f"FAIL {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
