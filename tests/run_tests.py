#!/usr/bin/env python3
"""
Test Runner

Comprehensive test runner for the Active Inference Knowledge Environment.
Provides multiple ways to run tests with different configurations and reporting.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional


def run_command(command: List[str], description: str = "") -> bool:
    """Run a shell command and return success status"""
    try:
        print(f"🔧 {description}")
        print(f"   Command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False


def run_all_tests() -> bool:
    """Run all tests"""
    return run_command([
        "python", "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--durations=10"
    ], "Running all tests")


def run_unit_tests() -> bool:
    """Run only unit tests"""
    return run_command([
        "python", "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short"
    ], "Running unit tests")


def run_integration_tests() -> bool:
    """Run only integration tests"""
    return run_command([
        "python", "-m", "pytest",
        "tests/integration/",
        "-v",
        "--tb=short",
        "-m", "integration"
    ], "Running integration tests")


def run_knowledge_tests() -> bool:
    """Run knowledge validation tests"""
    return run_command([
        "python", "-m", "pytest",
        "tests/knowledge/",
        "-v",
        "--tb=short",
        "-m", "knowledge"
    ], "Running knowledge tests")


def run_tests_with_coverage() -> bool:
    """Run tests with coverage reporting"""
    return run_command([
        "python", "-m", "pytest",
        "tests/",
        "--cov=src/active_inference",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        "-v"
    ], "Running tests with coverage")


def run_tests_parallel() -> bool:
    """Run tests in parallel"""
    return run_command([
        "python", "-m", "pytest",
        "tests/",
        "-n", "auto",
        "--tb=short",
        "-q"
    ], "Running tests in parallel")


def run_specific_test(test_path: str) -> bool:
    """Run a specific test file"""
    return run_command([
        "python", "-m", "pytest",
        test_path,
        "-v",
        "--tb=short"
    ], f"Running {test_path}")


def run_tests_by_component(component: str) -> bool:
    """Run tests for a specific component"""
    test_patterns = {
        "knowledge": "tests/unit/test_knowledge_*",
        "llm": "tests/unit/test_llm_*",
        "research": "tests/unit/test_research_*",
        "platform": "tests/unit/test_platform_*",
        "visualization": "tests/unit/test_visualization_*"
    }

    if component not in test_patterns:
        print(f"❌ Unknown component: {component}")
        print(f"   Available: {list(test_patterns.keys())}")
        return False

    pattern = test_patterns[component]
    return run_command([
        "python", "-m", "pytest",
        pattern,
        "-v",
        "--tb=short"
    ], f"Running {component} tests")


def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description="Active Inference Test Runner")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=[
            "all", "unit", "integration", "knowledge", "coverage",
            "parallel", "component"
        ],
        help="Test type to run"
    )
    parser.add_argument(
        "--component",
        help="Component to test (for component command)"
    )
    parser.add_argument(
        "--file",
        help="Specific test file to run"
    )

    args = parser.parse_args()

    print("🧪 Active Inference Knowledge Environment - Test Runner")
    print("=" * 60)

    success = True

    if args.command == "all":
        success = run_all_tests()
    elif args.command == "unit":
        success = run_unit_tests()
    elif args.command == "integration":
        success = run_integration_tests()
    elif args.command == "knowledge":
        success = run_knowledge_tests()
    elif args.command == "coverage":
        success = run_tests_with_coverage()
    elif args.command == "parallel":
        success = run_tests_parallel()
    elif args.command == "component":
        if not args.component:
            print("❌ Component name required for component command")
            return 1
        success = run_tests_by_component(args.component)
    elif args.file:
        success = run_specific_test(args.file)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1

    print()
    if success:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
