"""
Coverage Configuration

Comprehensive test coverage configuration for the Active Inference Knowledge Environment.
Defines coverage requirements, exclusions, and reporting configuration.
"""

import os
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

# Coverage thresholds by component
COVERAGE_THRESHOLDS = {
    "overall": 85,
    "knowledge": 90,
    "llm": 88,
    "research": 80,
    "platform": 75,
    "visualization": 70,
    "applications": 75,
}

# Coverage reporting configuration
REPORT_CONFIG = {
    "html": {
        "directory": "htmlcov",
        "title": "Active Inference Knowledge Environment - Test Coverage"
    },
    "xml": {
        "output": "coverage.xml"
    },
    "json": {
        "output": "coverage.json",
        "pretty_print": True
    },
    "term": {
        "show_missing": True,
        "skip_covered": False,
        "skip_empty": True
    }
}

def get_coverage_threshold(component: str = "overall") -> int:
    """Get coverage threshold for a component"""
    return COVERAGE_THRESHOLDS.get(component, COVERAGE_THRESHOLDS["overall"])

def is_coverage_acceptable(coverage_percent: float, component: str = "overall") -> bool:
    """Check if coverage meets the required threshold"""
    threshold = get_coverage_threshold(component)
    return coverage_percent >= threshold

def generate_coverage_report(output_formats: list = None) -> dict:
    """Generate comprehensive coverage report"""
    if output_formats is None:
        output_formats = ["html", "term", "xml", "json"]

    import subprocess
    import sys

    # Build pytest command with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        f"--cov={'src/active_inference'}",
        "--cov-report=" + ",".join(output_formats),
        "--cov-fail-under=0",  # Don't fail, just report
        "-v"
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
        import subprocess
        import json

        # Generate JSON coverage report
        result = subprocess.run([
            "python", "-m", "pytest",
            "tests/",
            "--cov=src/active_inference",
            "--cov-report=json:coverage_temp.json",
            "--cov-fail-under=0"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            return {"success": False, "error": "Failed to generate coverage data"}

        # Read coverage data
        coverage_file = Path("coverage_temp.json")
        if coverage_file.exists():
            with open(coverage_file, 'r') as f:
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
                        gaps.append({
                            "file": file_path,
                            "missing_lines": len(missing_lines),
                            "coverage_percent": file_data.get("summary", {}).get("percent_covered", 0)
                        })

            # Sort by coverage (lowest first)
            gaps.sort(key=lambda x: x["coverage_percent"])

            return {
                "success": True,
                "total_files": len(files_data),
                "files_with_gaps": len(gaps),
                "gaps": gaps[:20],  # Top 20 files with worst coverage
                "overall_coverage": coverage_data.get("totals", {}).get("percent_covered", 0)
            }

        return {"success": False, "error": "Coverage file not found"}

    except Exception as e:
        return {"success": False, "error": str(e)}
