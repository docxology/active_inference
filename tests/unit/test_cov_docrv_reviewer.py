"""Tests for the repository reviewer in tools.documentation.reviewer."""

import ast
import logging
from pathlib import Path

import pytest

from active_inference.tools.documentation.reviewer import (
    CodeMetrics,
    DocumentationMetrics,
    RepositoryReviewer,
)

SAMPLE_MODULE = '''"""Module doc."""
# a comment

import os


class Foo:
    """Foo doc."""

    def bar(self, x: int, y) -> int:
        """Bar doc."""
        if x and y:
            return 1
        for i in range(3):
            pass
        try:
            pass
        except Exception:
            pass
        return 0


def baz():
    with open(os.devnull) as fh:
        while False:
            pass
    return fh
'''

FACTORY_MODULE = '''"""Factory module."""


class WidgetFactory:
    """Creates widgets."""

    def create(self):
        return 1
'''

SINGLETON_MODULE = '''"""Single."""


class S:
    def __new__(cls):
        return cls.instance
'''

TEST_MODULE = '''"""Unit tests."""


def test_a():
    pass


def test_b():
    pass
'''

PYPROJECT = """[project]
name = "x"
dependencies = [
    "numpy>=1.20",
    "requests",
]
"""


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Work from tmp_path so analysed paths are short and relative."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def full_repo(workdir: Path) -> Path:
    """Build a small, varied repository tree and return its relative root."""
    root = Path("repo")
    _write(root / "src" / "pkg" / "__init__.py", '"""Pkg."""\n')
    _write(root / "src" / "pkg" / "core.py", SAMPLE_MODULE)
    _write(root / "src" / "pkg" / "sub" / "factory.py", FACTORY_MODULE)
    _write(root / "src" / "pkg" / "sub" / "single.py", SINGLETON_MODULE)
    _write(root / "src" / "pkg" / "broken.py", "def oops(:\n")
    _write(root / "tests" / "test_core.py", TEST_MODULE)
    _write(root / "docs" / "guide.md", "# Guide\n")
    _write(root / "README.md", "# Readme\n")
    _write(root / "pyproject.toml", PYPROJECT)
    _write(root / "__pycache__" / "junk.py", "x = 1\n")
    _write(root / ".hidden" / "secret.py", "y = 2\n")
    return root


def test_init_sets_paths_and_defaults(workdir: Path) -> None:
    reviewer = RepositoryReviewer("somewhere")
    assert reviewer.repo_path == Path("somewhere")
    assert reviewer.source_dirs == ["src", "tools", "platform"]
    assert "__pycache__" in reviewer.ignore_patterns


def test_should_ignore_patterns_and_hidden_parts(workdir: Path) -> None:
    reviewer = RepositoryReviewer(Path("repo"))
    assert reviewer._should_ignore(Path("repo/__pycache__/a.py")) is True
    assert reviewer._should_ignore(Path("repo/node_modules/x.js")) is True
    assert reviewer._should_ignore(Path("repo/.venv/lib.py")) is True
    assert reviewer._should_ignore(Path("repo/src/mod.py")) is False


def test_analyze_module_quality_counts(workdir: Path) -> None:
    path = _write(Path("mod.py"), SAMPLE_MODULE)
    metrics = RepositoryReviewer(Path("."))._analyze_module_quality(path)

    assert isinstance(metrics, CodeMetrics)
    assert metrics.total_lines == 28
    assert metrics.blank_lines == 7
    assert metrics.comment_lines == 1
    assert metrics.docstring_lines == 3
    assert metrics.code_lines == 17
    assert metrics.function_count == 2
    assert metrics.class_count == 1
    # Foo and bar are documented, baz is not
    assert metrics.documentation_coverage == 2
    assert metrics.type_hints == 1
    assert metrics.complexity == 8
    assert metrics.linting_issues == []


def test_analyze_module_quality_syntax_error(workdir: Path) -> None:
    path = _write(Path("broken.py"), "def oops(:\n")
    metrics = RepositoryReviewer(Path("."))._analyze_module_quality(path)
    assert len(metrics.linting_issues) == 1
    assert metrics.linting_issues[0].startswith("Syntax error:")
    assert metrics.complexity == 0


def test_analyze_module_quality_unreadable_file(workdir: Path) -> None:
    path = Path("bad.py")
    path.write_bytes(b"\xff\xfe\xfa")
    metrics = RepositoryReviewer(Path("."))._analyze_module_quality(path)
    assert len(metrics.linting_issues) == 1
    assert metrics.linting_issues[0].startswith("Analysis error:")
    assert metrics.total_lines == 0


def test_calculate_complexity_counts_branches() -> None:
    reviewer = RepositoryReviewer(Path("."))
    assert reviewer._calculate_complexity(ast.parse("x = 1")) == 1
    tree = ast.parse("if a or b or c:\n    pass\ntry:\n    pass\nexcept E:\n    pass\n")
    # base 1 + if 1 + boolop (3 values -> 2) + try 2
    assert reviewer._calculate_complexity(tree) == 6


def test_analyze_overview(full_repo: Path) -> None:
    overview = RepositoryReviewer(full_repo)._analyze_overview()
    assert overview["total_files"] == 9
    assert overview["total_directories"] == 5
    assert overview["python_files"] == 6
    assert overview["markdown_files"] == 2
    assert overview["json_files"] == 0
    assert overview["rst_files"] == 0
    assert overview["file_types"][".toml"] == 1
    assert overview["source_directories"] == 1
    assert overview["test_directories"] == 1
    assert overview["documentation_directories"] == 1
    assert overview["total_size_mb"] > 0


def test_analyze_code_quality(full_repo: Path) -> None:
    quality = RepositoryReviewer(full_repo)._analyze_code_quality()
    assert quality["total_modules"] == 6
    assert quality["modules_with_issues"] == 1
    assert len(quality["linting_issues"]) == 1
    assert quality["linting_issues"][0].startswith("Syntax error:")
    assert set(quality["module_metrics"]) == {
        "src/pkg/__init__.py",
        "src/pkg/core.py",
        "src/pkg/broken.py",
        "src/pkg/sub/factory.py",
        "src/pkg/sub/single.py",
        "tests/test_core.py",
    }
    core = quality["module_metrics"]["src/pkg/core.py"]
    assert core.complexity == 8
    # one annotated argument over six function definitions
    assert quality["type_hint_coverage"] == pytest.approx(1 / 6)
    assert quality["average_complexity"] > 1


def test_analyze_code_quality_no_modules(workdir: Path) -> None:
    _write(Path("repo") / "README.md", "# Only docs\n")
    quality = RepositoryReviewer(Path("repo"))._analyze_code_quality()
    assert quality["total_modules"] == 0
    assert quality["average_complexity"] == 0.0
    assert quality["type_hint_coverage"] == 0.0
    assert quality["module_metrics"] == {}


def test_analyze_code_quality_zero_functions(workdir: Path) -> None:
    _write(Path("repo") / "consts.py", '"""Constants."""\nX = 1\n')
    quality = RepositoryReviewer(Path("repo"))._analyze_code_quality()
    assert quality["total_modules"] == 1
    assert quality["type_hint_coverage"] == 0.0
    assert quality["average_complexity"] == 1.0


def test_analyze_code_quality_logs_module_failure(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _write(Path("repo") / "a.py", "x = 1\n")
    reviewer = RepositoryReviewer(Path("repo"))

    def boom(path: Path) -> CodeMetrics:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(reviewer, "_analyze_module_quality", boom)
    with caplog.at_level(logging.WARNING):
        quality = reviewer._analyze_code_quality()
    assert quality["total_modules"] == 0
    assert "Could not analyze" in caplog.text
    assert "kaboom" in caplog.text


def test_analyze_documentation(
    full_repo: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        docs = RepositoryReviewer(full_repo)._analyze_documentation()
    assert isinstance(docs, DocumentationMetrics)
    # broken.py cannot be parsed and is skipped with a warning
    assert docs.total_modules == 5
    assert docs.documented_modules == 5
    assert docs.coverage_percentage == 100.0
    assert docs.total_classes == 3
    assert docs.documented_classes == 2
    assert docs.total_functions == 6
    assert docs.documented_functions == 1
    assert "Could not analyze documentation" in caplog.text


def test_analyze_documentation_partial_coverage(workdir: Path) -> None:
    _write(Path("repo") / "a.py", '"""Doc."""\n')
    _write(Path("repo") / "b.py", "x = 1\n")
    docs = RepositoryReviewer(Path("repo"))._analyze_documentation()
    assert docs.total_modules == 2
    assert docs.documented_modules == 1
    assert docs.coverage_percentage == 50.0


def test_analyze_architecture(full_repo: Path) -> None:
    arch = RepositoryReviewer(full_repo)._analyze_architecture()
    pkg = arch["layers"]["pkg"]
    assert sorted(pkg["modules"]) == [
        "__init__.py",
        "broken.py",
        "core.py",
        "factory.py",
        "single.py",
    ]
    assert pkg["subpackages"] == ["sub"]
    assert pkg["size"] > 0
    assert arch["patterns"]["factory"] == ["src/pkg/sub/factory.py"]
    assert arch["patterns"]["singleton"] == ["src/pkg/sub/single.py"]
    assert arch["patterns"]["repository"] == []
    # tests/ holds Python files, docs/ does not
    assert arch["issues"] == ["No documentation directories found"]


def test_analyze_architecture_without_src(workdir: Path) -> None:
    _write(Path("repo") / "repository_impl.py", "class UserRepository:\n    pass\n")
    arch = RepositoryReviewer(Path("repo"))._analyze_architecture()
    assert arch["layers"] == {}
    assert arch["patterns"]["repository"] == ["repository_impl.py"]
    assert "No test directories found" in arch["issues"]
    assert "No documentation directories found" in arch["issues"]


def test_check_architectural_issues_large_file(workdir: Path) -> None:
    big = Path("repo") / "src" / "big.py"
    big.parent.mkdir(parents=True)
    big.write_bytes(b"#" * (1024 * 1024 + 10))
    issues = RepositoryReviewer(Path("repo"))._check_architectural_issues()
    large = [i for i in issues if i.startswith("Large file detected")]
    assert large == [f"Large file detected: {big} (1.00MB)"]


def test_unreadable_files_are_logged_not_fatal(
    workdir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = Path("repo")
    (root / "src").mkdir(parents=True)
    (root / "src" / "bad.py").write_bytes(b"\xff\xfe\xfa")
    (root / "tests").mkdir()
    (root / "tests" / "test_bad.py").write_bytes(b"\xff\xfe\xfa")
    reviewer = RepositoryReviewer(root)
    with caplog.at_level(logging.WARNING):
        patterns = reviewer._identify_patterns()
        issues = reviewer._check_architectural_issues()
        testing = reviewer._analyze_testing()
    assert all(v == [] for v in patterns.values())
    assert "No test directories found" not in issues
    assert testing["test_functions"] == 0
    assert "Could not analyze patterns" in caplog.text
    assert "Could not analyze imports" in caplog.text
    assert "Could not analyze test file" in caplog.text


def test_analyze_testing(full_repo: Path) -> None:
    testing = RepositoryReviewer(full_repo)._analyze_testing()
    # tests/test_core.py matches both the glob and the tests dir: counted once
    assert testing["test_files"] == 1
    assert testing["test_functions"] == 2
    assert dict(testing["test_types"]) == {"unit": 1}
    assert testing["coverage_available"] is True


def test_analyze_testing_categories(workdir: Path) -> None:
    root = Path("repo")
    _write(root / "tests" / "test_i.py", '"""Integration."""\ndef test_x(): pass\n')
    _write(root / "tests" / "test_f.py", '"""Functional."""\ndef test_y(): pass\n')
    _write(root / "tests" / "helper.py", "VALUE = 1\n")
    _write(root / "pkg" / "thing_test.py", "def test_z(): pass\n")
    testing = RepositoryReviewer(root)._analyze_testing()
    assert testing["test_files"] == 4
    assert testing["test_functions"] == 3
    assert dict(testing["test_types"]) == {
        "integration": 1,
        "functional": 1,
        "other": 2,
    }
    assert testing["coverage_available"] is False


def test_analyze_testing_detects_coveragerc(workdir: Path) -> None:
    _write(Path("repo") / ".coveragerc", "[run]\n")
    assert RepositoryReviewer(Path("repo"))._analyze_testing()["coverage_available"]


def test_analyze_dependencies(full_repo: Path) -> None:
    _write(full_repo / "requirements.txt", "numpy\n")
    deps = RepositoryReviewer(full_repo)._analyze_dependencies()
    assert deps["requirements_files"] == ["requirements.txt", "pyproject.toml"]
    assert deps["python_dependencies"] == {
        "numpy": "numpy>=1.20",
        "requests": "requests",
    }
    assert deps["dependency_issues"] == []


def test_analyze_dependencies_unparseable_pyproject(workdir: Path) -> None:
    _write(
        Path("repo") / "pyproject.toml",
        "[project.optional-dependencies]\ndev = []\n",
    )
    deps = RepositoryReviewer(Path("repo"))._analyze_dependencies()
    assert deps["python_dependencies"] == {}
    assert len(deps["dependency_issues"]) == 1
    assert deps["dependency_issues"][0].startswith("Could not parse pyproject.toml")


def test_analyze_dependencies_none(workdir: Path) -> None:
    Path("repo").mkdir()
    deps = RepositoryReviewer(Path("repo"))._analyze_dependencies()
    assert deps["requirements_files"] == []
    assert deps["python_dependencies"] == {}


def test_recommendations_for_bare_repo(workdir: Path) -> None:
    _write(Path("repo") / "a.py", "def f(x):\n    return x\n")
    recs = RepositoryReviewer(Path("repo"))._generate_recommendations()
    assert "Improve documentation coverage: currently 0.0%" in recs
    assert "Add type hints: currently 0.0% coverage" in recs
    assert "Add comprehensive test suite" in recs
    assert "Add documentation structure" in recs
    assert "Add test directory structure" in recs
    assert not any(r.startswith("Improve test coverage") for r in recs)
    assert not any(r.startswith("Fix linting issues") for r in recs)


def test_recommendations_for_full_repo(full_repo: Path) -> None:
    recs = RepositoryReviewer(full_repo)._generate_recommendations()
    assert not any(r.startswith("Improve documentation coverage") for r in recs)
    assert "Improve test coverage: currently 0.0%" in recs
    assert "Fix linting issues in 1 modules" in recs
    assert "Add comprehensive test suite" not in recs
    assert "Add documentation structure" not in recs
    assert "Add test directory structure" not in recs


def test_analyze_repository_sections(full_repo: Path) -> None:
    analysis = RepositoryReviewer(full_repo).analyze_repository()
    assert set(analysis) == {
        "overview",
        "code_quality",
        "documentation",
        "architecture",
        "testing",
        "dependencies",
        "recommendations",
    }
    assert analysis["overview"]["python_files"] == 6
    assert analysis["documentation"].total_modules == 5
    assert "numpy" in analysis["dependencies"]["python_dependencies"]


def test_generate_report_writes_file(full_repo: Path, workdir: Path) -> None:
    out = workdir / "out" / "nested" / "report.md"
    report = RepositoryReviewer(full_repo).generate_report(output_path=out)
    assert out.read_text(encoding="utf-8") == report
    assert report.startswith("# Repository Analysis Report")
    assert "- **Python Files:** 6" in report
    assert "- **Total Modules:** 6" in report
    assert "- **Modules with Issues:** 1" in report
    assert "- **Documented Modules:** 5/5" in report
    assert "- **Test Files:** 1" in report
    assert "- **Test Functions:** 2" in report
    assert "- **Coverage Tool:** Available" in report
    assert "## Recommendations" in report
    assert "Fix linting issues in 1 modules" in report


def test_generate_report_without_output(workdir: Path) -> None:
    Path("repo").mkdir()
    report = RepositoryReviewer(Path("repo")).generate_report()
    assert "- **Total Files:** 0" in report
    assert "- **Coverage Tool:** Not configured" in report
    assert not list(workdir.rglob("*.md"))


def test_format_report_without_recommendations() -> None:
    analysis = {
        "overview": {
            "total_files": 3,
            "total_directories": 1,
            "total_size_mb": 0.5,
            "python_files": 2,
        },
        "code_quality": {
            "total_modules": 2,
            "modules_with_issues": 0,
            "average_complexity": 2.5,
            "average_documentation_coverage": 1.0,
            "type_hint_coverage": 0.25,
        },
        "documentation": DocumentationMetrics(
            total_modules=2,
            documented_modules=1,
            total_functions=4,
            documented_functions=3,
            coverage_percentage=50.0,
        ),
        "testing": {
            "test_files": 0,
            "test_functions": 0,
            "coverage_available": False,
        },
        "recommendations": [],
    }
    report = RepositoryReviewer(Path("."))._format_report(analysis)
    assert "- **Repository Size:** 0.50 MB" in report
    assert "- **Average Complexity:** 2.50" in report
    assert "- **Coverage:** 50.0%" in report
    assert "- **Documented Functions:** 3/4" in report
    assert "## Recommendations" not in report
