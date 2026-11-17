"""Tests for the documentation validator in tools.documentation.validator."""

import ast
import logging
from pathlib import Path
from typing import List

import pytest

from active_inference.tools.documentation.validator import (
    DocumentationValidator,
    ValidationResult,
    _node_kind,
)

CHECK_METHODS = [
    "_check_class_docstring",
    "_check_function_docstring",
    "_check_docstring_summary",
    "_check_docstring_description",
    "_check_parameter_documentation",
    "_check_return_documentation",
    "_check_docstring_format",
    "_check_broken_links",
    "_check_examples",
    "_check_exception_documentation",
]

CLEAN_MODULE = '''"""Utility helpers for numeric work

Provides a small set of helpers.
Used by the package internals.
"""


def scale_values(alpha):
    """Scale numeric values by a factor

    Multiplies the given alpha by two.
    Intended for internal use.
    """
    return alpha * 2
'''


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use relative paths so tmp dir names (e.g. ``test_*``) are not ignored."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def validator() -> DocumentationValidator:
    return DocumentationValidator([Path("pkgsrc")])


def _names(results: List[ValidationResult]) -> List[str]:
    return [r.element_name for r in results]


def test_node_kind_labels() -> None:
    tree = ast.parse("def f():\n    pass\nclass C:\n    pass\n")
    func, cls = tree.body
    assert _node_kind(func) == "Function"
    assert _node_kind(cls) == "Class"
    assert _node_kind(tree) == "Class"
    assert _node_kind(tree, default="Module") == "Module"


def test_init_registers_rules() -> None:
    v = DocumentationValidator(["a", Path("b")])
    assert v.source_paths == [Path("a"), Path("b")]
    names = [r.name for r in v.rules]
    assert len(names) == 11
    assert names[0] == "module_docstring_required"
    assert "raises_documented" in names
    severities = {r.name: r.severity for r in v.rules}
    assert severities["class_docstring_required"] == "error"
    assert severities["function_docstring_required"] == "warning"
    assert severities["examples_provided"] == "info"


def test_check_module_docstring(workdir: Path, validator: DocumentationValidator):
    missing = _write(Path("missing.py"), "x = 1\n")
    present = _write(Path("present.py"), '"""Has a docstring."""\n')
    broken = _write(Path("broken.py"), "def oops(:\n")

    results = validator._check_module_docstring(missing)
    assert len(results) == 1
    assert results[0].severity == "error"
    assert results[0].message == "Module is missing docstring"
    assert results[0].file_path == missing

    assert validator._check_module_docstring(present) == []

    parse_failure = validator._check_module_docstring(broken)
    assert len(parse_failure) == 1
    assert parse_failure[0].message.startswith("Could not parse module:")


def test_check_class_docstring(workdir: Path, validator: DocumentationValidator):
    path = _write(
        Path("classes.py"),
        'class A:\n    pass\n\n\nclass B:\n    """B doc."""\n',
    )
    results = validator._check_class_docstring(path)
    assert _names(results) == ["A"]
    assert results[0].line_number == 1
    assert results[0].message == "Class 'A' is missing docstring"


def test_check_function_docstring(workdir: Path, validator: DocumentationValidator):
    path = _write(
        Path("funcs.py"),
        "def public():\n    pass\n\n\n"
        "def _private():\n    pass\n\n\n"
        'def documented():\n    """Doc."""\n',
    )
    results = validator._check_function_docstring(path)
    assert _names(results) == ["public"]
    assert results[0].severity == "warning"
    assert results[0].line_number == 1


def test_check_docstring_summary(workdir: Path, validator: DocumentationValidator):
    path = _write(
        Path("summary.py"),
        '"""Tiny"""\n\n\n'
        'class Short:\n    """Too short"""\n\n\n'
        'def brief():\n    """Hi"""\n\n\n'
        'def good():\n    """Computes the requested sensible value"""\n',
    )
    results = validator._check_docstring_summary(path)
    messages = [r.message for r in results]
    assert messages == [
        "Module docstring missing proper summary",
        "Class 'Short' docstring missing proper summary",
        "Function 'brief' docstring missing proper summary",
    ]
    assert results[0].line_number is None
    assert results[1].element_name == "Short"


def test_check_docstring_summary_module_ok(
    workdir: Path, validator: DocumentationValidator
):
    path = _write(Path("ok.py"), '"""Computes the requested sensible value"""\n')
    assert validator._check_docstring_summary(path) == []


def test_check_docstring_description(workdir: Path, validator: DocumentationValidator):
    path = _write(
        Path("desc.py"),
        'class Thin:\n    """Only a summary."""\n\n\n'
        'def thin():\n    """Only a summary."""\n\n\n'
        'def rich():\n    """Summary line here.\n\n    First detail.\n'
        '    Second detail.\n    """\n',
    )
    results = validator._check_docstring_description(path)
    assert [r.message for r in results] == [
        "Class 'Thin' docstring could use more detailed description",
        "Function 'thin' docstring could use more detailed description",
    ]
    assert all(r.severity == "info" for r in results)


def test_check_parameter_documentation(
    workdir: Path, validator: DocumentationValidator
):
    path = _write(
        Path("params.py"),
        'def undocumented(alpha, beta):\n    """Does something."""\n\n\n'
        'class K:\n    def method(self, gamma):\n        """Uses gamma."""\n\n'
        '    def noargs(self):\n        """No parameters."""\n\n\n'
        "def nodoc(delta):\n    pass\n",
    )
    results = validator._check_parameter_documentation(path)
    assert _names(results) == ["undocumented"]
    assert results[0].suggestion == (
        "Add parameter documentation in Args section: alpha, beta"
    )


def test_check_return_documentation(workdir: Path, validator: DocumentationValidator):
    path = _write(
        Path("returns.py"),
        'def silent() -> int:\n    """Compute value."""\n    return 1\n\n\n'
        'def told() -> int:\n    """Compute value.\n\n    Returns:\n'
        '        The value.\n    """\n    return 1\n\n\n'
        'def unannotated():\n    """Compute value."""\n    return 1\n',
    )
    results = validator._check_return_documentation(path)
    assert _names(results) == ["silent"]
    assert results[0].message == "Function 'silent' return value not documented"


def test_check_docstring_format(workdir: Path, validator: DocumentationValidator):
    long_module_doc = "\n".join(f"line {i}" for i in range(12))
    path = _write(
        Path("fmt.py"),
        f'"""{long_module_doc}"""\n\n\n'
        'def ragged():\n    """Summary.\n\n    level0\n        level1\n'
        '            level2\n    """\n\n\n'
        'def tidy():\n    """Summary.\n\n    Args:\n        x: value\n    """\n',
    )
    results = validator._check_docstring_format(path)
    assert [(r.message, r.element_name, r.line_number) for r in results] == [
        ("Module docstring format could be improved", "module", None),
        ("Function docstring format could be improved", "ragged", 15),
    ]


def test_check_broken_links(workdir: Path, validator: DocumentationValidator):
    path = _write(
        Path("links.py"),
        '"""See https://example.org/docs for details."""\n'
        "# local: http://localhost:8000/x\n"
        "# api: https://127.0.0.1/api\n",
    )
    results = validator._check_broken_links(path)
    assert [(r.line_number, r.message) for r in results] == [
        (2, "Localhost URL found in documentation: http://localhost:8000/x"),
        (3, "Localhost URL found in documentation: https://127.0.0.1/api"),
    ]


def test_check_examples(workdir: Path, validator: DocumentationValidator):
    path = _write(
        Path("examples.py"),
        'def run_job():\n    """Run the job."""\n\n\n'
        'def generate():\n    """Generate things.\n\n    Example:\n'
        '        generate()\n    """\n\n\n'
        "def build_nodoc():\n    pass\n\n\n"
        'def helper():\n    """Helps."""\n',
    )
    results = validator._check_examples(path)
    assert _names(results) == ["run_job"]
    assert results[0].message == (
        "Important function 'run_job' should have usage examples"
    )


def test_check_exception_documentation(
    workdir: Path, validator: DocumentationValidator
):
    path = _write(
        Path("raises.py"),
        'def raiser():\n    """Doc."""\n    raise ValueError("x")\n\n\n'
        'def handler():\n    """Doc."""\n    try:\n        pass\n'
        "    except OSError:\n        pass\n\n\n"
        'def declared():\n    """Doc.\n\n    Raises:\n        ValueError: bad\n'
        '    """\n    raise ValueError("x")\n\n\n'
        'def calm():\n    """Doc."""\n    return 1\n\n\n'
        "def nodoc():\n    raise ValueError\n",
    )
    results = validator._check_exception_documentation(path)
    assert _names(results) == ["raiser", "handler"]
    assert all(r.rule_name == "raises_documented" for r in results)


@pytest.mark.parametrize("method", CHECK_METHODS)
def test_checks_log_and_skip_unreadable_files(
    workdir: Path,
    validator: DocumentationValidator,
    caplog: pytest.LogCaptureFixture,
    method: str,
) -> None:
    path = Path("bad.py")
    path.write_bytes(b"\xff\xfe\xfa")
    with caplog.at_level(logging.WARNING):
        results = getattr(validator, method)(path)
    assert results == []
    assert "bad.py" in caplog.text


def test_has_proper_summary(validator: DocumentationValidator) -> None:
    assert validator._has_proper_summary("Computes the requested value") is True
    assert validator._has_proper_summary("Too short") is False
    assert validator._has_proper_summary("") is False


def test_has_description(validator: DocumentationValidator) -> None:
    assert validator._has_description("Summary only") is False
    assert validator._has_description("Summary\n\nshort detail") is False
    assert validator._has_description("Summary\n\nfirst\nsecond") is True
    long_line = " ".join(["word"] * 11)
    assert validator._has_description(f"Summary\n{long_line}") is True


def test_text_helpers(validator: DocumentationValidator) -> None:
    assert validator._parameters_documented("Uses ALPHA here", ["alpha"]) is True
    assert validator._parameters_documented("Nothing here", ["zeta"]) is False
    assert validator._return_documented("Yields items") is True
    assert validator._return_documented("Plain text") is False
    assert validator._exceptions_documented("Raises: KeyError") is True
    assert validator._exceptions_documented("Plain text") is False


def test_is_valid_docstring_format(validator: DocumentationValidator) -> None:
    long_without_sections = "\n".join(f"line {i}" for i in range(11))
    long_with_sections = long_without_sections + "\nArgs: none"
    assert validator._is_valid_docstring_format(long_without_sections) is False
    assert validator._is_valid_docstring_format(long_with_sections) is True
    assert validator._is_valid_docstring_format("One line") is True
    ragged = "Summary\n  a\n    b\n      c"
    assert validator._is_valid_docstring_format(ragged) is False
    two_levels = "Summary\na\n    b\nc"
    assert validator._is_valid_docstring_format(two_levels) is True


def test_should_ignore(validator: DocumentationValidator) -> None:
    assert validator._should_ignore(Path("pkg/__pycache__/m.py")) is True
    assert validator._should_ignore(Path("pkg/tests/m.py")) is True
    assert validator._should_ignore(Path("pkg/test_m.py")) is True
    assert validator._should_ignore(Path("pkg/m_test.py")) is True
    assert validator._should_ignore(Path("pkg/m.py")) is False


def test_group_results(validator: DocumentationValidator) -> None:
    a = ValidationResult("class_docstring_required", "error", "a", Path("x.py"))
    b = ValidationResult("examples_provided", "info", "b", Path("y.py"))
    c = ValidationResult("unknown_rule", "info", "c")
    by_category = validator._group_results_by_category([a, b, c])
    assert by_category == {"structure": [a], "completeness": [b]}
    by_file = validator._group_results_by_file([a, b, c])
    assert by_file == {"x.py": [a], "y.py": [b]}


def test_validate_documentation_counts(workdir: Path) -> None:
    src = Path("pkgsrc")
    _write(src / "clean.py", CLEAN_MODULE)
    _write(src / "messy.py", "class Bare:\n    pass\n")
    _write(src / "tests" / "ignored.py", "class Ignored:\n    pass\n")
    v = DocumentationValidator([src, Path("does-not-exist")])
    summary = v.validate_documentation()

    # messy.py: missing module docstring + missing class docstring
    assert summary["errors"] == 2
    assert summary["warnings"] == 0
    assert summary["info"] == 0
    assert summary["total_issues"] == 2
    assert summary["validation_passed"] is False
    assert set(summary["results_by_file"]) == {str(src / "messy.py")}
    assert list(summary["results_by_category"]) == ["structure"]
    assert {r.rule_name for r in summary["results"]} == {
        "module_docstring_required",
        "class_docstring_required",
    }


def test_validate_documentation_counts_warnings_and_info(workdir: Path) -> None:
    src = Path("pkgsrc")
    _write(
        src / "mod.py",
        '"""Utility helpers for numeric work"""\n\n\n'
        "def nodoc(x):\n    pass\n\n\n"
        'def run_it():\n    """Runs the requested job now"""\n',
    )
    summary = DocumentationValidator([src]).validate_documentation()
    assert summary["errors"] == 0
    # nodoc missing docstring
    assert summary["warnings"] == 1
    # run_it: no description, no examples
    assert summary["info"] == 2
    assert summary["validation_passed"] is True


def test_run_validation_rule_logs_check_failure(
    workdir: Path, validator: DocumentationValidator, caplog: pytest.LogCaptureFixture
) -> None:
    _write(Path("pkgsrc") / "m.py", "x = 1\n")
    rule = validator.rules[0]

    def boom(path: Path) -> List[ValidationResult]:
        raise RuntimeError("check exploded")

    rule.check_function = boom
    with caplog.at_level(logging.WARNING):
        assert validator._run_validation_rule(rule) == []
    assert "Error checking module_docstring_required" in caplog.text
    assert "check exploded" in caplog.text


def test_validate_documentation_survives_rule_failure(
    workdir: Path,
    validator: DocumentationValidator,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write(Path("pkgsrc") / "m.py", "class Bare:\n    pass\n")
    original = validator._run_validation_rule

    def flaky(rule):
        if rule.name == "module_docstring_required":
            raise RuntimeError("rule exploded")
        return original(rule)

    monkeypatch.setattr(validator, "_run_validation_rule", flaky)
    with caplog.at_level(logging.ERROR):
        summary = validator.validate_documentation()
    assert summary["errors"] == 1
    assert summary["results"][0].rule_name == "class_docstring_required"
    assert "Error running validation rule module_docstring_required" in caplog.text


def test_generate_validation_report_writes_file(workdir: Path) -> None:
    src = Path("pkgsrc")
    classes = "".join(f"class C{i}:\n    pass\n\n\n" for i in range(12))
    _write(src / "many.py", classes)
    _write(
        src / "warn.py",
        '"""Utility helpers for numeric work"""\n\n\n'
        "def nodoc(x):\n    pass\n\n\n"
        'def run_it():\n    """Runs the requested job now"""\n',
    )
    out = Path("reports") / "deep" / "validation.md"
    report = DocumentationValidator([src]).generate_validation_report(out)

    assert out.read_text(encoding="utf-8") == report
    assert report.startswith("# Documentation Validation Report")
    assert "- **Errors:** 13" in report
    assert "- **Warnings:** 1" in report
    assert "- **Info:** 2" in report
    assert "- **Validation Passed:** ❌ No" in report
    assert "### Structure (14 issues)" in report
    assert "- ... and 4 more" in report
    assert f"  - File: {src / 'many.py'}\n" in report
    assert f"  - File: {src / 'many.py'}, Element: C0" in report
    assert f"### {src / 'many.py'} (13 issues)" in report
    assert "1. **Fix all errors** before proceeding" in report
    assert "2. **Address 1 warnings**" in report
    assert "3. **Review 2 suggestions**" in report
    assert "All validation checks passed" not in report


def test_generate_validation_report_clean(workdir: Path) -> None:
    Path("pkgsrc").mkdir()
    report = DocumentationValidator([Path("pkgsrc")]).generate_validation_report()
    assert "- **Total Issues:** 0" in report
    assert "- **Validation Passed:** ✅ Yes" in report
    assert "## Issues by File" not in report
    assert "🎉 **All validation checks passed!**" in report
    assert "Fix all errors" not in report
    assert not Path("reports").exists()


def test_format_validation_report_optional_fields(
    validator: DocumentationValidator,
) -> None:
    bare = ValidationResult("examples_provided", "info", "bare message")
    located = ValidationResult(
        "no_broken_links", "warning", "link message", file_path=Path("f.py")
    )
    validation = {
        "total_issues": 2,
        "errors": 0,
        "warnings": 1,
        "info": 1,
        "validation_passed": True,
        "results_by_category": {
            "completeness": [bare],
            "content": [located],
            "style": [],
        },
        "results_by_file": {"f.py": [located], "empty.py": []},
    }
    report = validator._format_validation_report(validation)
    lines = report.split("\n")
    bare_idx = lines.index("- **INFO:** bare message")
    assert lines[bare_idx + 1] == ""
    assert "  - File: f.py" in lines
    assert "### Style" not in report
    assert "### empty.py" not in report
    file_section = lines.index("### f.py (1 issues)")
    assert lines[file_section + 1] == "- **WARNING:** link message"
    assert lines[file_section + 2] == ""
    assert "🎉 **All validation checks passed!**" in report
