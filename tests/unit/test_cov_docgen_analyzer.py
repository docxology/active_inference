"""Coverage tests for the documentation quality analyzer."""

from pathlib import Path

import pytest

from active_inference.tools.documentation.analyzer import (
    DocstringMetrics,
    DocumentationAnalyzer,
    DocumentationQuality,
)

# The analyzer ignores any path containing "test_" or "tests", so fixture trees
# are created with tmp_path_factory under neutral directory names.

SAMPLE_MODULE = '''"""Module providing well documented helpers for checks.

Args:
    none: placeholder entry
"""


def excellent(x, y):
    """Compute the sum of two numbers carefully and return it.

    This function adds two values together and returns the resulting value
    for downstream consumers of the helper.

    Args:
        x: first operand
        y: second operand

    Returns:
        The sum of x and y.

    Raises:
        TypeError: if operands cannot be added.

    Examples:
        >>> excellent(1, 2)
        3
    """
    return x + y


def poor(x):
    """Short."""
    return x


def undocumented(x):
    return x


class Documented:
    """A class with a moderately useful summary line.

    Returns:
        nothing really
    """


class Bare:
    pass
'''


@pytest.fixture
def source_tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("docsrc")
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "sample.py").write_text(SAMPLE_MODULE, encoding="utf-8")
    # Ignored files: test files, caches, hidden directories
    (pkg / "test_ignored.py").write_text("def f():\n    pass\n", encoding="utf-8")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "cached.py").write_text("x = 1\n", encoding="utf-8")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.py").write_text("y = 2\n", encoding="utf-8")
    return root


def _sample_path(root: Path) -> Path:
    return root / "pkg" / "sample.py"


def test_analyze_documentation_counts_and_categorizes(source_tree):
    analyzer = DocumentationAnalyzer([source_tree, source_tree / "missing"])
    quality = analyzer.analyze_documentation()

    sample = _sample_path(source_tree)
    # module + excellent + poor + undocumented + Documented + Bare
    assert quality.total_elements == 6
    assert quality.documented_elements == 4
    assert quality.coverage_percentage == pytest.approx(400 / 6)
    assert quality.excellent_docs == [f"function:{sample}::excellent"]
    assert quality.poor_quality_docs == [f"function:{sample}::poor"]
    assert quality.missing_docs == [f"{sample}::poor"]
    # Placeholder element score averaged over excellent + poor elements
    assert quality.average_quality == pytest.approx(0.7)


def test_analyze_documentation_without_elements(tmp_path_factory):
    empty = tmp_path_factory.mktemp("emptysrc")
    quality = DocumentationAnalyzer([empty]).analyze_documentation()
    assert quality.total_elements == 0
    assert quality.coverage_percentage == 0.0
    assert quality.average_quality == 0.0


def test_average_quality_zero_when_no_categorized_docs(tmp_path_factory):
    root = tmp_path_factory.mktemp("midsrc")
    (root / "mid.py").write_text(
        '"""A module docstring with several words.\n\nReturns:\n    nothing\n"""\n',
        encoding="utf-8",
    )
    quality = DocumentationAnalyzer([root]).analyze_documentation()
    assert quality.total_elements == 1
    assert quality.coverage_percentage == 100.0
    assert quality.average_quality == 0.0
    assert quality.excellent_docs == [] and quality.poor_quality_docs == []


def test_file_with_syntax_error_contributes_nothing(tmp_path_factory, caplog):
    root = tmp_path_factory.mktemp("badsrc")
    broken = root / "broken.py"
    broken.write_text("def broken(:\n", encoding="utf-8")
    analyzer = DocumentationAnalyzer([root])
    with caplog.at_level("WARNING"):
        quality = analyzer._analyze_file_documentation(broken)
    assert quality.total_elements == 0
    assert "Error analyzing" in caplog.text


def test_path_analysis_survives_file_level_exception(source_tree, monkeypatch, caplog):
    root = source_tree
    (root / "pkg" / "other.py").write_text(
        '"""Other module docstring here."""\n', encoding="utf-8"
    )
    analyzer = DocumentationAnalyzer([root])
    original = analyzer._analyze_file_documentation

    def flaky(path):
        if path.name == "sample.py":
            raise RuntimeError("boom")
        return original(path)

    monkeypatch.setattr(analyzer, "_analyze_file_documentation", flaky)
    with caplog.at_level("WARNING"):
        quality = analyzer._analyze_path_documentation(root)
    assert quality.total_elements == 1
    assert quality.documented_elements == 1
    assert "Could not analyze documentation" in caplog.text
    assert "boom" in caplog.text


def test_analyze_docstring_empty_and_blank():
    analyzer = DocumentationAnalyzer([])
    assert analyzer._analyze_docstring("") == DocstringMetrics()
    blank = analyzer._analyze_docstring("   \n   ")
    assert blank.word_count == 0
    assert blank.has_summary is False
    assert blank.quality_score == 0.0


def test_analyze_docstring_sections_and_parameter_count():
    analyzer = DocumentationAnalyzer([])
    docstring = (
        "Summarize the behaviour of this function well.\n"
        "\n"
        "Some extended description line.\n"
        "Note:\n"
        "    a remark\n"
        "Arguments:\n"
        "    alpha: first value\n"
        "        continuation with detail: more text\n"
        "    beta: second value\n"
        "Yields:\n"
        "    int: produced numbers\n"
        "Exceptions:\n"
        "    ValueError: on bad input\n"
        "Example:\n"
        "    >>> f()\n"
    )
    metrics = analyzer._analyze_docstring(docstring)
    assert metrics.has_summary is True
    assert metrics.has_parameters is True
    assert metrics.has_returns is True
    assert metrics.has_raises is True
    assert metrics.has_examples is True
    # Only alpha and beta: continuation and other sections are not parameters
    assert metrics.parameter_count == 2
    assert metrics.quality_score == pytest.approx(1.0)


def test_parameter_count_with_original_indentation():
    analyzer = DocumentationAnalyzer([])
    docstring = """Initialize the component with given values.

        Args:
            source_paths: List of source directories
            strict: Whether to fail fast

        Returns:
            None: nothing
        """
    metrics = analyzer._analyze_docstring(docstring)
    assert metrics.parameter_count == 2


def test_summary_requires_more_than_three_words():
    analyzer = DocumentationAnalyzer([])
    assert analyzer._analyze_docstring("Too few words").has_summary is False
    assert analyzer._analyze_docstring("Now four words here").has_summary is True


@pytest.mark.parametrize(
    "metrics, expected",
    [
        (DocstringMetrics(), 0.0),
        (DocstringMetrics(has_summary=True, word_count=10), 0.3),
        (DocstringMetrics(has_summary=True, word_count=50), 0.4),
        (DocstringMetrics(has_summary=True, word_count=500), 0.35),
        (
            DocstringMetrics(
                has_summary=True,
                has_parameters=True,
                has_returns=True,
                has_examples=True,
                has_raises=True,
                word_count=50,
            ),
            1.0,
        ),
    ],
)
def test_docstring_quality_score(metrics, expected):
    analyzer = DocumentationAnalyzer([])
    assert analyzer._calculate_docstring_quality_score(metrics) == pytest.approx(
        expected
    )


def test_categorize_docstring_quality_bands():
    analyzer = DocumentationAnalyzer([])
    quality = DocumentationQuality()
    analyzer._categorize_docstring_quality(
        "a", "function", DocstringMetrics(quality_score=0.9), quality
    )
    analyzer._categorize_docstring_quality(
        "b", "class", DocstringMetrics(quality_score=0.5), quality
    )
    analyzer._categorize_docstring_quality(
        "c", "module", DocstringMetrics(quality_score=0.3), quality
    )
    assert quality.excellent_docs == ["function:a"]
    assert quality.poor_quality_docs == ["module:c"]
    assert quality.missing_docs == ["c"]


def test_merge_quality_reports_accumulates():
    analyzer = DocumentationAnalyzer([])
    target = DocumentationQuality(
        total_elements=2,
        documented_elements=1,
        excellent_docs=["e1"],
        quality_distribution={"good": 1},
    )
    source = DocumentationQuality(
        total_elements=3,
        documented_elements=2,
        excellent_docs=["e2"],
        poor_quality_docs=["p1"],
        missing_docs=["m1"],
        quality_distribution={"good": 2, "poor": 1},
    )
    analyzer._merge_quality_reports(target, source)
    assert target.total_elements == 5
    assert target.documented_elements == 3
    assert target.excellent_docs == ["e1", "e2"]
    assert target.poor_quality_docs == ["p1"]
    assert target.missing_docs == ["m1"]
    assert target.quality_distribution == {"good": 3, "poor": 1}


@pytest.mark.parametrize(
    "path, ignored",
    [
        ("/proj/pkg/module.py", False),
        ("/proj/pkg/__pycache__/module.py", True),
        ("/proj/build/module.py", True),
        ("/proj/pkg/test_module.py", True),
        ("/proj/tests/module.py", True),
        ("/proj/.venv/module.py", True),
    ],
)
def test_should_ignore(path, ignored):
    assert DocumentationAnalyzer([])._should_ignore(Path(path)) is ignored


def test_generate_quality_report_writes_file(source_tree, tmp_path):
    analyzer = DocumentationAnalyzer([source_tree])
    output = tmp_path / "reports" / "quality.md"
    report = analyzer.generate_quality_report(output)

    assert output.read_text(encoding="utf-8") == report
    assert report.startswith("# Documentation Quality Report")
    assert "- **Coverage:** 66.7%" in report
    assert "- **Total Elements:** 6" in report
    assert "- **Excellent Documentation:** 1 elements" in report
    assert "- **Good Documentation:** 4 elements" in report
    assert "- **Poor Documentation:** 1 elements" in report
    assert "## Missing Documentation" in report
    assert "1. Improve documentation coverage from 66.7% to 80%+" in report
    assert "2. Review and improve 1 poorly documented elements" in report


def test_generate_quality_report_without_output(source_tree):
    report = DocumentationAnalyzer([source_tree]).generate_quality_report()
    assert "## Poor Quality Documentation" in report


def test_format_quality_report_truncates_long_lists():
    analyzer = DocumentationAnalyzer([])
    quality = DocumentationQuality(
        total_elements=60,
        documented_elements=30,
        coverage_percentage=50.0,
        average_quality=0.5,
        missing_docs=[f"m{i}" for i in range(25)],
        poor_quality_docs=[f"p{i}" for i in range(23)],
    )
    report = analyzer._format_quality_report(quality)
    assert "- m19" in report and "- m20" not in report
    assert "- ... and 5 more" in report
    assert "- p19" in report and "- p20" not in report
    assert "- ... and 3 more" in report
    assert "3. Add comprehensive docstrings" in report
    assert "Excellent! Documentation quality" not in report


def test_format_quality_report_excellent():
    analyzer = DocumentationAnalyzer([])
    quality = DocumentationQuality(
        total_elements=4,
        documented_elements=4,
        coverage_percentage=100.0,
        average_quality=0.9,
        excellent_docs=["a", "b"],
    )
    report = analyzer._format_quality_report(quality)
    assert "- **Good Documentation:** 2 elements" in report
    assert "1. Excellent! Documentation quality is very good." in report
    assert "Improve documentation coverage" not in report
    assert "## Missing Documentation" not in report


def test_suggest_improvements_from_real_analysis(source_tree):
    suggestions = DocumentationAnalyzer([source_tree]).suggest_improvements(90.0)
    assert suggestions["current_coverage"] == pytest.approx(400 / 6)
    assert suggestions["target_coverage"] == 90.0
    assert suggestions["gap"] == pytest.approx(90.0 - 400 / 6)
    assert suggestions["missing_elements"] == 1
    assert suggestions["poor_quality_elements"] == 1
    assert suggestions["priority_actions"] == [
        "Add documentation for 1 more elements",
        "Improve 1 poor quality docstrings",
    ]
    assert suggestions["estimated_effort"] == "medium"


def test_suggest_improvements_when_quality_is_clean(monkeypatch):
    analyzer = DocumentationAnalyzer([])
    monkeypatch.setattr(
        analyzer,
        "analyze_documentation",
        lambda: DocumentationQuality(total_elements=5, coverage_percentage=100.0),
    )
    suggestions = analyzer.suggest_improvements(90.0)
    assert suggestions["priority_actions"] == [
        "Focus on adding examples and comprehensive parameter descriptions"
    ]
    assert suggestions["estimated_effort"] == "low"


def test_suggest_improvements_with_no_actions(monkeypatch):
    analyzer = DocumentationAnalyzer([])
    monkeypatch.setattr(
        analyzer,
        "analyze_documentation",
        lambda: DocumentationQuality(
            total_elements=5, coverage_percentage=95.0, missing_docs=["x"]
        ),
    )
    suggestions = analyzer.suggest_improvements(90.0)
    assert suggestions["priority_actions"] == []
    assert suggestions["estimated_effort"] == "none"
