"""
Coverage Configuration

Comprehensive test coverage configuration for the Active Inference Knowledge
Environment.
Defines coverage requirements, exclusions, and reporting configuration.
"""

from pathlib import Path

# Coverage configuration
COVERAGE_CONFIG = {
    "source": ["src/active_inference"],
    "omit": [
        "*/tests/*",
        "*/test_*",
        "setup.py",
        "*/__pycache__/*",
        "*/.venv/*",
        "docs/*",
        "*/migrations/*",
    ],
    "include": [
        "src/active_inference/**/*.py",
    ],
    "exclude_lines": [
        "pragma: no cover",
        "def __repr__",
        "if self.debug:",
        "if settings.DEBUG",
        "raise AssertionError",
        "raise NotImplementedError",
        "if 0:",
        "if __name__ == .__main__.:",
        "class .*\bProtocol\\):",
        "@(abc\\.)?abstractmethod",
    ],
}

# Coverage thresholds, enforced in CI by `scripts/check_ratchets.py coverage`.
# AGENTS.md requires >80% overall and >95% for core components; every
# component below is held to the core bar. "overall" matches the
# `--cov-fail-under` floor in .github/workflows/ci.yml and sits just under the
# measured line coverage (99.9% when it was set), so it works as a ratchet:
# raise it as coverage rises and never lower it to admit a regression.
COVERAGE_THRESHOLDS = {
    "overall": 99,
    "package": 95,  # top-level modules: __init__.py and cli.py
    "applications": 95,
    "knowledge": 95,
    "llm": 95,
    "platform": 95,
    "research": 95,
    "tools": 95,
    "visualization": 95,
}

# Coverage reporting configuration
REPORT_CONFIG = {
    "html": {
        "directory": "htmlcov",
        "title": "Active Inference Knowledge Environment - Test Coverage",
    },
    "xml": {"output": "coverage.xml"},
    "json": {"output": "coverage.json", "pretty_print": True},
    "term": {"show_missing": True, "skip_covered": False, "skip_empty": True},
}


def get_coverage_threshold(component: str = "overall") -> int:
    """Get coverage threshold for a component"""
    return COVERAGE_THRESHOLDS.get(component, COVERAGE_THRESHOLDS["overall"])


def is_coverage_acceptable(coverage_percent: float, component: str = "overall") -> bool:
    """Check if coverage meets the required threshold"""
    threshold = get_coverage_threshold(component)
    return coverage_percent >= threshold


def component_of(path: str) -> str:
    """Map a source path from a coverage report to its component name."""
    parts = Path(path).parts
    index = parts.index("active_inference")
    return parts[index + 1] if len(parts) > index + 2 else "package"


def check_coverage_report(report_path: Path) -> list[str]:
    """Check a coverage.py JSON report against COVERAGE_THRESHOLDS.

    Returns one message per threshold that is not met, and one for any
    component without a threshold, so a new package cannot go unchecked.
    """
    import json

    report = json.loads(Path(report_path).read_text())
    covered: dict[str, int] = {}
    statements: dict[str, int] = {}
    for path, data in report["files"].items():
        component = component_of(path)
        summary = data["summary"]
        covered[component] = covered.get(component, 0) + summary["covered_lines"]
        statements[component] = statements.get(component, 0) + summary["num_statements"]
    problems = []
    for component in sorted(statements):
        if component not in COVERAGE_THRESHOLDS:
            problems.append(f"component {component!r} has no coverage threshold")
            continue
        percent = 100.0 * covered[component] / max(statements[component], 1)
        threshold = COVERAGE_THRESHOLDS[component]
        print(f"{component}: {percent:.2f}% (threshold {threshold}%)")
        if percent < threshold:
            problems.append(f"{component}: {percent:.2f}% < {threshold}%")
    total = report["totals"]["percent_covered"]
    print(f"overall: {total:.2f}% (threshold {COVERAGE_THRESHOLDS['overall']}%)")
    if total < COVERAGE_THRESHOLDS["overall"]:
        problems.append(f"overall: {total:.2f}% < {COVERAGE_THRESHOLDS['overall']}%")
    return problems


def generate_coverage_report(output_formats: list = None) -> dict:
    """Generate comprehensive coverage report"""
    if output_formats is None:
        output_formats = ["html", "term", "xml", "json"]

    import subprocess
    import sys

    # Build pytest command with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        f"--cov={'src/active_inference'}",
        "--cov-report=" + ",".join(output_formats),
        "--cov-fail-under=0",  # Don't fail, just report
        "-v",
    ]

    # Run tests with coverage
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Coverage report generated successfully")
            return {"success": True, "output": result.stdout}
        else:
            print("❌ Coverage report generation failed")
            print(f"Error: {result.stderr}")
            return {"success": False, "error": result.stderr}

    except Exception as e:
        print(f"❌ Failed to generate coverage report: {e}")
        return {"success": False, "error": str(e)}


def analyze_coverage_gaps() -> dict:
    """Analyze coverage gaps and provide recommendations"""
    try:
        import json
        import subprocess

        # Generate JSON coverage report
        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "tests/",
                "--cov=src/active_inference",
                "--cov-report=json:coverage_temp.json",
                "--cov-fail-under=0",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {"success": False, "error": "Failed to generate coverage data"}

        # Read coverage data
        coverage_file = Path("coverage_temp.json")
        if coverage_file.exists():
            with open(coverage_file, "r") as f:
                coverage_data = json.load(f)

            # Clean up temp file
            coverage_file.unlink()

            # Analyze gaps
            gaps = []
            files_data = coverage_data.get("files", {})

            for file_path, file_data in files_data.items():
                if file_path.startswith("src/active_inference/"):
                    missing_lines = file_data.get("missing_lines", [])
                    if missing_lines:
                        gaps.append(
                            {
                                "file": file_path,
                                "missing_lines": len(missing_lines),
                                "coverage_percent": (
                                    file_data.get("summary", {}).get(
                                        "percent_covered", 0
                                    )
                                ),
                            }
                        )

            # Sort by coverage (lowest first)
            gaps.sort(key=lambda x: x["coverage_percent"])

            return {
                "success": True,
                "total_files": len(files_data),
                "files_with_gaps": len(gaps),
                "gaps": gaps[:20],  # Top 20 files with worst coverage
                "overall_coverage": (
                    coverage_data.get("totals", {}).get("percent_covered", 0)
                ),
            }

        return {"success": False, "error": "Coverage file not found"}

    except Exception as e:
        return {"success": False, "error": str(e)}
