"""Coverage tests for the documentation command-line interface."""

import argparse
import json
import subprocess
import sys

import pytest

from active_inference.tools.documentation import cli as cli_module
from active_inference.tools.documentation import generator as gen_module
from active_inference.tools.documentation.analyzer import DocumentationQuality
from active_inference.tools.documentation.cli import DocumentationCLI

MODULE_SOURCE = '''"""Package module with a reasonably descriptive summary."""


def documented(value):
    """Return the value unchanged for callers.

    Args:
        value: anything

    Returns:
        The same value.
    """
    return value


def undocumented(value):
    return value
'''


def _write_knowledge(root):
    (root / "foundations").mkdir(parents=True)
    (root / "metadata").mkdir()
    node = {
        "id": "entropy",
        "title": "Entropy",
        "content_type": "foundation",
        "difficulty": "beginner",
        "description": "Uncertainty measure.",
        "prerequisites": [],
        "tags": ["information"],
        "learning_objectives": ["understand"],
        "content": {"overview": "text"},
        "metadata": {},
    }
    (root / "foundations" / "entropy.json").write_text(json.dumps(node))
    (root / "metadata" / "learning_paths.json").write_text(
        json.dumps(
            [
                {
                    "id": "intro",
                    "name": "Intro",
                    "description": "Start.",
                    "nodes": ["entropy"],
                    "difficulty": "beginner",
                }
            ]
        )
    )


@pytest.fixture
def repo_root(tmp_path_factory):
    # Analyzer/validator ignore any path containing "test_", so avoid tmp_path.
    root = tmp_path_factory.mktemp("clirepo")
    pkg = root / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "module.py").write_text(MODULE_SOURCE, encoding="utf-8")
    _write_knowledge(root / "knowledge")
    return root


@pytest.fixture
def cli(repo_root):
    return DocumentationCLI(repo_root)


@pytest.fixture
def fake_apidoc(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(gen_module.subprocess, "run", fake_run)
    return calls


def _raise(*args, **kwargs):
    raise RuntimeError("tool exploded")


def test_init_wires_tools_to_repo(cli, repo_root):
    assert cli.repo_path == repo_root
    assert cli.generator.output_dir == repo_root / "docs"
    assert (repo_root / "docs").is_dir()
    assert cli.analyzer.source_paths == [repo_root / "src"]
    assert cli.reviewer.repo_path == repo_root
    assert cli.validator.source_paths == [repo_root / "src"]


def test_init_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert DocumentationCLI().repo_path == tmp_path


def test_parser_parses_subcommands(cli):
    parser = cli.create_parser()
    args = parser.parse_args(["analyze", "coverage", "-f", "json", "-t", "70"])
    assert args.command == "analyze"
    assert args.analysis_type == "coverage"
    assert args.format == "json"
    assert args.threshold == 70.0
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "bogus"])


def test_run_without_command_prints_help(cli, capsys):
    assert cli.run([]) == 0
    assert "Active Inference Documentation Tools" in capsys.readouterr().out


def test_run_reads_sys_argv(cli, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ai-docs", "generate", "research"])
    assert cli.run() == 0
    assert "Documentation type 'research' not yet implemented" in (
        capsys.readouterr().out
    )


def test_run_returns_one_when_dispatch_raises(cli, monkeypatch):
    monkeypatch.setattr(cli, "_dispatch_command", _raise)
    assert cli.run(["analyze", "quality"]) == 1


def test_dispatch_unknown_command(cli, capsys):
    assert cli._dispatch_command(argparse.Namespace(command="bogus")) == 1
    assert "Unknown command: bogus" in capsys.readouterr().out


def test_generate_api(cli, repo_root, fake_apidoc, capsys):
    code = cli.run(["generate", "api", "--source", str(repo_root / "src")])
    assert code == 0
    assert "API documentation generated successfully" in capsys.readouterr().out
    assert len(fake_apidoc) == 1
    assert "pkg/index" in (repo_root / "docs" / "api" / "index.rst").read_text()


def test_generate_all(cli, repo_root, fake_apidoc, capsys):
    code = cli.run(["generate", "all", "--source", str(repo_root / "src")])
    assert code == 0
    assert "All documentation generated successfully" in capsys.readouterr().out
    docs = repo_root / "docs"
    assert (docs / "knowledge" / "learning_paths" / "intro.rst").exists()
    assert (docs / "knowledge" / "search" / "index.rst").exists()


def test_generate_knowledge(cli, repo_root, capsys):
    assert cli.run(["generate", "knowledge"]) == 0
    assert "Knowledge documentation generated successfully" in (capsys.readouterr().out)
    docs = repo_root / "docs" / "knowledge"
    assert "Intro" in (docs / "learning_paths" / "intro.rst").read_text()
    assert (docs / "concepts" / "knowledge_graph.json").exists()
    assert (docs / "statistics" / "index.rst").exists()
    assert not (docs / "search").exists()


def test_generate_failure_reports_error(cli, repo_root, fake_apidoc, capsys):
    code = cli.run(["generate", "api", "--source", str(repo_root / "nope")])
    assert code == 1
    assert "Documentation generation failed" in capsys.readouterr().out


def test_analyze_quality_text(cli, capsys):
    assert cli.run(["analyze", "quality"]) == 0
    out = capsys.readouterr().out
    # module + documented + undocumented: two of three documented
    assert "Coverage: 66.7%" in out
    assert "Total Elements: 3" in out
    assert "Documented Elements: 2" in out


def test_analyze_quality_json_to_file(cli, tmp_path, capsys):
    output = tmp_path / "reports" / "quality.json"
    assert cli.run(["analyze", "quality", "-f", "json", "-o", str(output)]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["total_elements"] == 3
    assert data["documented_elements"] == 2
    assert data["coverage_percentage"] == pytest.approx(200 / 3)
    assert f"Report saved to: {output}" in capsys.readouterr().out


def test_analyze_coverage_text_and_json(cli, capsys):
    assert cli.run(["analyze", "coverage"]) == 0
    text = capsys.readouterr().out
    assert "Overall Coverage: 66.7%" in text
    assert "- Total: 3" in text

    assert cli.run(["analyze", "coverage", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["documented_elements"] == 2
    # The terse module docstring scores as poor quality
    assert [p.rsplit("/", 2)[-2:] for p in data["missing_docs"]] == [
        ["pkg", "module.py"]
    ]
    assert len(data["poor_quality_docs"]) == 1
    assert data["poor_quality_docs"][0].startswith("module:")


def test_analyze_unimplemented_and_failure(cli, monkeypatch, capsys):
    assert cli.run(["analyze", "complexity"]) == 0
    assert "Analysis type 'complexity' not yet implemented" in (capsys.readouterr().out)
    monkeypatch.setattr(cli.analyzer, "analyze_documentation", _raise)
    assert cli.run(["analyze", "quality"]) == 1
    assert "Analysis failed: tool exploded" in capsys.readouterr().out


def test_review_repository_to_file(cli, tmp_path, capsys):
    output = tmp_path / "review.md"
    assert cli.run(["review", "repository", "-o", str(output)]) == 0
    assert output.read_text(encoding="utf-8").startswith("# Repository Analysis Report")
    assert "Repository review completed" in capsys.readouterr().out


def test_review_json_format_handles_json_and_text(cli, monkeypatch, capsys):
    monkeypatch.setattr(cli.reviewer, "generate_report", lambda: '{"score": 1}')
    assert cli.run(["review", "repository", "-f", "json"]) == 0
    assert '{\n  "score": 1\n}' in capsys.readouterr().out

    monkeypatch.setattr(cli.reviewer, "generate_report", lambda: "# plain text")
    assert cli.run(["review", "repository", "-f", "json"]) == 0
    assert "# plain text" in capsys.readouterr().out


def test_review_unimplemented_and_failure(cli, monkeypatch, capsys):
    assert cli.run(["review", "code"]) == 0
    assert "Review type 'code' not yet implemented" in capsys.readouterr().out
    monkeypatch.setattr(cli.reviewer, "generate_report", _raise)
    assert cli.run(["review", "repository"]) == 1
    assert "Review failed: tool exploded" in capsys.readouterr().out


def test_validate_structure_text(cli, capsys):
    assert cli.run(["validate", "structure"]) == 0
    out = capsys.readouterr().out
    assert "Documentation Validation Results" in out
    assert "Validation completed" in out
    validation = cli.validator.validate_documentation()
    assert f"- Total Issues: {validation['total_issues']}" in out
    for category, results in validation["results_by_category"].items():
        assert f"- {category}: {len(results)} issues" in out


def test_validate_structure_json_serializes_results(cli, tmp_path):
    output = tmp_path / "validation.json"
    assert cli.run(["validate", "structure", "-f", "json", "-o", str(output)]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["total_issues"] == len(data["results"]) > 0
    first = data["results"][0]
    assert set(first) >= {"rule_name", "severity", "message", "file_path"}
    assert first["file_path"].endswith("module.py")


def test_validate_unimplemented_and_failure(cli, monkeypatch, capsys):
    assert cli.run(["validate", "style"]) == 0
    assert "Validation type 'style' not yet implemented" in (capsys.readouterr().out)
    monkeypatch.setattr(cli.validator, "validate_documentation", _raise)
    assert cli.run(["validate", "structure"]) == 1
    assert "Validation failed: tool exploded" in capsys.readouterr().out


def test_report_summary(cli, capsys):
    assert cli.run(["report", "summary"]) == 0
    out = capsys.readouterr().out
    assert "Active Inference Documentation Summary Report" in out
    assert "Coverage: 66.7%" in out
    assert "- Improve documentation coverage from 66.7% to 80%+" in out
    assert "Summary report generated" in out


def test_report_unimplemented_and_failure(cli, monkeypatch, capsys):
    assert cli.run(["report", "health"]) == 0
    assert "Report type 'health' not yet implemented" in capsys.readouterr().out
    monkeypatch.setattr(cli, "_generate_summary_report", _raise)
    assert cli.run(["report", "summary"]) == 1
    assert "Report generation failed: tool exploded" in capsys.readouterr().out


def test_get_recommendations(cli):
    poor = DocumentationQuality(coverage_percentage=40.0)
    recs = cli._get_recommendations(poor, {"errors": 2, "warnings": 3})
    assert recs == [
        "Improve documentation coverage from 40.0% to 80%+",
        "Fix 2 documentation errors",
        "Address 3 documentation warnings",
    ]
    good = DocumentationQuality(coverage_percentage=95.0)
    assert cli._get_recommendations(good, {"errors": 0, "warnings": 0}) == [
        "Documentation quality is excellent!"
    ]


def test_display_validation_results_skips_empty_categories(cli, capsys):
    validation = {
        "total_issues": 1,
        "errors": 0,
        "warnings": 1,
        "info": 0,
        "validation_passed": True,
        "results_by_category": {"style": ["x"], "content": []},
    }
    cli._display_validation_results(validation, "text", None)
    out = capsys.readouterr().out
    assert "- style: 1 issues" in out
    assert "content" not in out
    assert "Validation Passed: ✅ Yes" in out


def test_main_exits_with_run_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ai-docs"])
    with pytest.raises(SystemExit) as exc:
        cli_module.main()
    assert exc.value.code == 0
