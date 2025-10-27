"""
Repository Reviewer

Comprehensive code analysis and repository review tools for the Active Inference
Knowledge Environment. Provides automated code quality analysis, documentation
coverage, and architectural insights.
"""

import ast
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re

import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeMetrics:
    """Code quality metrics for a module or repository"""
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    function_count: int = 0
    class_count: int = 0
    docstring_lines: int = 0
    complexity: int = 0
    test_coverage: float = 0.0
    documentation_coverage: float = 0.0
    linting_issues: List[str] = field(default_factory=list)
    type_hints: int = 0
    type_hint_coverage: float = 0.0


@dataclass
class DocumentationMetrics:
    """Documentation quality metrics"""
    total_modules: int = 0
    documented_modules: int = 0
    total_functions: int = 0
    documented_functions: int = 0
    total_classes: int = 0
    documented_classes: int = 0
    missing_docstrings: List[str] = field(default_factory=list)
    incomplete_docstrings: List[str] = field(default_factory=list)
    coverage_percentage: float = 0.0


class RepositoryReviewer:
    """
    Comprehensive repository analysis and review system.

    Analyzes code quality, documentation coverage, architectural patterns,
    and provides actionable insights for improvement.
    """

    def __init__(self, repo_path: Path):
        """
        Initialize repository reviewer

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)
        self.source_dirs = ['src', 'tools', 'platform']
        self.test_dirs = ['tests', 'test']
        self.doc_dirs = ['docs', 'documentation']
        self.ignore_patterns = [
            '__pycache__',
            '.git',
            'node_modules',
            '.pytest_cache',
            'build',
            'dist',
            '.coverage',
            '*.egg-info'
        ]

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Repository reviewer initialized for {self.repo_path}")

    def analyze_repository(self) -> Dict[str, Any]:
        """
        Perform comprehensive repository analysis

        Returns:
            Complete analysis report
        """
        self.logger.info("Starting comprehensive repository analysis")

        analysis = {
            'overview': self._analyze_overview(),
            'code_quality': self._analyze_code_quality(),
            'documentation': self._analyze_documentation(),
            'architecture': self._analyze_architecture(),
            'testing': self._analyze_testing(),
            'dependencies': self._analyze_dependencies(),
            'recommendations': self._generate_recommendations()
        }

        self.logger.info("Repository analysis completed")
        return analysis

    def _analyze_overview(self) -> Dict[str, Any]:
        """Analyze repository overview metrics"""
        self.logger.info("Analyzing repository overview")

        # Count files by type
        file_counts = defaultdict(int)
        total_size = 0
        total_files = 0

        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file() and not self._should_ignore(file_path):
                total_files += 1
                total_size += file_path.stat().st_size

                # Categorize by extension
                suffix = file_path.suffix.lower()
                file_counts[suffix] += 1

        # Count directories
        total_dirs = 0
        source_dirs = 0
        test_dirs = 0
        doc_dirs = 0

        for dir_path in self.repo_path.rglob('*'):
            if dir_path.is_dir() and not self._should_ignore(dir_path):
                total_dirs += 1
                dir_name = dir_path.name

                if any(dir_name == s for s in self.source_dirs):
                    source_dirs += 1
                elif any(dir_name.startswith(t) for t in self.test_dirs):
                    test_dirs += 1
                elif any(dir_name.startswith(d) for d in self.doc_dirs):
                    doc_dirs += 1

        return {
            'total_files': total_files,
            'total_directories': total_dirs,
            'total_size_mb': total_size / (1024 * 1024),
            'file_types': dict(file_counts),
            'source_directories': source_dirs,
            'test_directories': test_dirs,
            'documentation_directories': doc_dirs,
            'python_files': file_counts.get('.py', 0),
            'json_files': file_counts.get('.json', 0),
            'markdown_files': file_counts.get('.md', 0),
            'rst_files': file_counts.get('.rst', 0)
        }

    def _analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        self.logger.info("Analyzing code quality")

        quality_metrics = {
            'total_modules': 0,
            'modules_with_issues': 0,
            'average_complexity': 0.0,
            'average_test_coverage': 0.0,
            'average_documentation_coverage': 0.0,
            'linting_issues': [],
            'type_hint_coverage': 0.0,
            'module_metrics': {}
        }

        total_complexity = 0
        total_test_coverage = 0.0
        total_doc_coverage = 0.0
        total_type_hints = 0
        total_functions = 0

        # Analyze Python modules
        python_files = list(self.repo_path.rglob('*.py'))
        python_files = [f for f in python_files if not self._should_ignore(f)]

        for file_path in python_files:
            try:
                metrics = self._analyze_module_quality(file_path)
                quality_metrics['module_metrics'][str(file_path.relative_to(self.repo_path))] = metrics

                quality_metrics['total_modules'] += 1
                total_complexity += metrics.complexity
                total_test_coverage += metrics.test_coverage
                total_doc_coverage += metrics.documentation_coverage
                total_type_hints += metrics.type_hints
                total_functions += metrics.function_count

                if metrics.linting_issues:
                    quality_metrics['modules_with_issues'] += 1
                    quality_metrics['linting_issues'].extend(metrics.linting_issues)

            except Exception as e:
                self.logger.warning(f"Could not analyze {file_path}: {e}")

        # Calculate averages
        if quality_metrics['total_modules'] > 0:
            quality_metrics['average_complexity'] = total_complexity / quality_metrics['total_modules']
            quality_metrics['average_test_coverage'] = total_test_coverage / quality_metrics['total_modules']
            quality_metrics['average_documentation_coverage'] = total_doc_coverage / quality_metrics['total_modules']
            quality_metrics['type_hint_coverage'] = total_type_hints / total_functions if total_functions > 0 else 0.0

        return quality_metrics

    def _analyze_module_quality(self, file_path: Path) -> CodeMetrics:
        """Analyze quality metrics for a single module"""
        metrics = CodeMetrics()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Basic metrics
            lines = content.split('\n')
            metrics.total_lines = len(lines)

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    metrics.blank_lines += 1
                elif stripped.startswith('#'):
                    metrics.comment_lines += 1
                elif '"""' in line or "'''" in line:
                    metrics.docstring_lines += 1
                else:
                    metrics.code_lines += 1

            # Parse AST for structural analysis
            try:
                tree = ast.parse(content, filename=str(file_path))

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        metrics.function_count += 1
                        if ast.get_docstring(node):
                            metrics.documentation_coverage += 1

                    elif isinstance(node, ast.ClassDef):
                        metrics.class_count += 1
                        if ast.get_docstring(node):
                            metrics.documentation_coverage += 1

                    elif isinstance(node, ast.arg) and node.annotation:
                        metrics.type_hints += 1

                # Calculate complexity (simplified)
                metrics.complexity = self._calculate_complexity(tree)

            except SyntaxError as e:
                metrics.linting_issues.append(f"Syntax error: {e}")

        except Exception as e:
            metrics.linting_issues.append(f"Analysis error: {e}")

        return metrics

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate code complexity (simplified)"""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += 2  # Try blocks are more complex
            elif isinstance(node, ast.BoolOp):  # and/or operations
                complexity += len(node.values) - 1

        return complexity

    def _analyze_documentation(self) -> DocumentationMetrics:
        """Analyze documentation coverage and quality"""
        self.logger.info("Analyzing documentation")

        metrics = DocumentationMetrics()

        python_files = list(self.repo_path.rglob('*.py'))
        python_files = [f for f in python_files if not self._should_ignore(f)]

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content, filename=str(file_path))

                # Check module docstring
                if ast.get_docstring(tree):
                    metrics.documented_modules += 1
                metrics.total_modules += 1

                # Check functions and classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        metrics.total_functions += 1
                        if ast.get_docstring(node):
                            metrics.documented_functions += 1

                    elif isinstance(node, ast.ClassDef):
                        metrics.total_classes += 1
                        if ast.get_docstring(node):
                            metrics.documented_classes += 1

            except Exception as e:
                self.logger.warning(f"Could not analyze documentation for {file_path}: {e}")

        # Calculate coverage
        if metrics.total_modules > 0:
            metrics.coverage_percentage = (metrics.documented_modules / metrics.total_modules) * 100

        return metrics

    def _analyze_architecture(self) -> Dict[str, Any]:
        """Analyze architectural patterns and structure"""
        self.logger.info("Analyzing architecture")

        architecture = {
            'layers': {},
            'dependencies': {},
            'patterns': {},
            'issues': []
        }

        # Analyze package structure
        src_path = self.repo_path / 'src'
        if src_path.exists():
            architecture['layers'] = self._analyze_package_layers(src_path)

        # Check for common patterns
        architecture['patterns'] = self._identify_patterns()

        # Identify potential issues
        architecture['issues'] = self._check_architectural_issues()

        return architecture

    def _analyze_package_layers(self, src_path: Path) -> Dict[str, Any]:
        """Analyze package layering and organization"""
        layers = {}

        for package_path in src_path.iterdir():
            if package_path.is_dir() and not self._should_ignore(package_path):
                package_name = package_path.name
                layers[package_name] = {
                    'modules': [],
                    'subpackages': [],
                    'dependencies': set(),
                    'size': 0
                }

                # Count modules and size
                for file_path in package_path.rglob('*.py'):
                    if not self._should_ignore(file_path):
                        layers[package_name]['modules'].append(file_path.name)
                        layers[package_name]['size'] += file_path.stat().st_size

                # Count subpackages
                for subpackage in package_path.iterdir():
                    if subpackage.is_dir() and not self._should_ignore(subpackage):
                        layers[package_name]['subpackages'].append(subpackage.name)

        return layers

    def _identify_patterns(self) -> Dict[str, List[str]]:
        """Identify common design patterns in the codebase"""
        patterns = {
            'factory': [],
            'singleton': [],
            'observer': [],
            'decorator': [],
            'strategy': [],
            'repository': []
        }

        # Simple pattern detection (could be enhanced)
        python_files = list(self.repo_path.rglob('*.py'))
        python_files = [f for f in python_files if not self._should_ignore(f)]

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Look for pattern indicators
                if 'factory' in content.lower() and ('create' in content.lower() or 'build' in content.lower()):
                    patterns['factory'].append(str(file_path.relative_to(self.repo_path)))

                if '__new__' in content and 'cls.' in content:
                    patterns['singleton'].append(str(file_path.relative_to(self.repo_path)))

                if 'repository' in content.lower():
                    patterns['repository'].append(str(file_path.relative_to(self.repo_path)))

            except Exception as e:
                self.logger.warning(f"Could not analyze patterns for {file_path}: {e}")

        return patterns

    def _check_architectural_issues(self) -> List[str]:
        """Check for common architectural issues"""
        issues = []

        # Check for circular imports (simplified)
        python_files = list(self.repo_path.rglob('*.py'))
        python_files = [f for f in python_files if not self._should_ignore(f)]

        imports = defaultdict(set)

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract imports
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('from ') or line.startswith('import '):
                        module = line.split()[1] if 'import' in line else line.split()[1]
                        imports[str(file_path.relative_to(self.repo_path))].add(module)

            except Exception as e:
                self.logger.warning(f"Could not analyze imports for {file_path}: {e}")

        # Check for potential issues
        if not any('tests' in str(f) for f in python_files):
            issues.append("No test directories found")

        if not any('docs' in str(f) for f in python_files):
            issues.append("No documentation directories found")

        # Check for very large files
        for file_path in python_files:
            if file_path.stat().st_size > 1024 * 1024:  # 1MB
                issues.append(f"Large file detected: {file_path} ({file_path.stat().st_size / 1024 / 1024:.2f}MB)")

        return issues

    def _analyze_testing(self) -> Dict[str, Any]:
        """Analyze testing structure and coverage"""
        self.logger.info("Analyzing testing")

        testing = {
            'test_files': 0,
            'test_functions': 0,
            'coverage_available': False,
            'test_types': defaultdict(int),
            'untested_modules': []
        }

        # Find test files
        test_files = []
        for pattern in ['test_*.py', '*_test.py']:
            test_files.extend(list(self.repo_path.glob(f'**/{pattern}')))

        # Also check in test directories
        for test_dir in self.test_dirs:
            test_path = self.repo_path / test_dir
            if test_path.exists():
                test_files.extend(list(test_path.rglob('*.py')))

        testing['test_files'] = len(test_files)

        # Analyze test structure
        for test_file in test_files:
            if not self._should_ignore(test_file):
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Count test functions
                    test_count = len(re.findall(r'def test_', content))
                    testing['test_functions'] += test_count

                    # Categorize test types
                    if 'unit' in content.lower():
                        testing['test_types']['unit'] += 1
                    elif 'integration' in content.lower():
                        testing['test_types']['integration'] += 1
                    elif 'functional' in content.lower():
                        testing['test_types']['functional'] += 1
                    else:
                        testing['test_types']['other'] += 1

                except Exception as e:
                    self.logger.warning(f"Could not analyze test file {test_file}: {e}")

        # Check for coverage configuration
        coverage_config = self.repo_path / '.coveragerc' or self.repo_path / 'pyproject.toml'
        testing['coverage_available'] = coverage_config.exists()

        return testing

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies"""
        self.logger.info("Analyzing dependencies")

        dependencies = {
            'requirements_files': [],
            'python_dependencies': {},
            'dev_dependencies': {},
            'optional_dependencies': {},
            'dependency_issues': []
        }

        # Find requirements files
        for req_file in ['requirements.txt', 'requirements-dev.txt', 'requirements-test.txt', 'pyproject.toml']:
            req_path = self.repo_path / req_file
            if req_path.exists():
                dependencies['requirements_files'].append(req_file)

        # Analyze pyproject.toml if it exists
        pyproject_path = self.repo_path / 'pyproject.toml'
        if pyproject_path.exists():
            try:
                with open(pyproject_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse dependencies (simplified)
                if 'dependencies' in content:
                    deps_section = content.split('dependencies = [')[1].split(']')[0]
                    deps = [line.strip().strip('",\'') for line in deps_section.split('\n') if line.strip()]
                    dependencies['python_dependencies'] = {dep.split('>=')[0]: dep for dep in deps if dep}

            except Exception as e:
                dependencies['dependency_issues'].append(f"Could not parse pyproject.toml: {e}")

        return dependencies

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        self.logger.info("Generating recommendations")

        recommendations = []

        # Get analysis data
        overview = self._analyze_overview()
        quality = self._analyze_code_quality()
        docs = self._analyze_documentation()
        testing = self._analyze_testing()

        # Generate recommendations based on analysis

        # Documentation recommendations
        if docs.coverage_percentage < 80:
            recommendations.append(f"Improve documentation coverage: currently {docs.coverage_percentage:.1f}%")

        if quality['type_hint_coverage'] < 70:
            recommendations.append(f"Add type hints: currently {quality['type_hint_coverage']:.1f}% coverage")

        # Testing recommendations
        if testing['test_files'] == 0:
            recommendations.append("Add comprehensive test suite")

        if testing['coverage_available'] and quality['average_test_coverage'] < 80:
            recommendations.append(f"Improve test coverage: currently {quality['average_test_coverage']:.1f}%")

        # Architecture recommendations
        if quality['modules_with_issues'] > 0:
            recommendations.append(f"Fix linting issues in {quality['modules_with_issues']} modules")

        # General recommendations
        if overview['documentation_directories'] == 0:
            recommendations.append("Add documentation structure")

        if overview['test_directories'] == 0:
            recommendations.append("Add test directory structure")

        return recommendations

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored in analysis"""
        path_str = str(path)

        for pattern in self.ignore_patterns:
            if pattern in path_str:
                return True

        # Check if it's a hidden file/directory
        parts = path.parts
        if any(part.startswith('.') for part in parts):
            return True

        return False

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive analysis report

        Args:
            output_path: Optional path to save report

        Returns:
            Report content as string
        """
        self.logger.info("Generating comprehensive report")

        analysis = self.analyze_repository()

        # Generate report
        report = self._format_report(analysis)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Report saved to {output_path}")

        return report

    def _format_report(self, analysis: Dict[str, Any]) -> str:
        """Format analysis results into readable report"""
        report = []
        report.append("# Repository Analysis Report")
        report.append("=" * 50)
        report.append("")

        # Overview
        overview = analysis['overview']
        report.append("## Overview")
        report.append(f"- **Total Files:** {overview['total_files']}")
        report.append(f"- **Total Directories:** {overview['total_directories']}")
        report.append(f"- **Repository Size:** {overview['total_size_mb']:.2f} MB")
        report.append(f"- **Python Files:** {overview['python_files']}")
        report.append("")

        # Code Quality
        quality = analysis['code_quality']
        report.append("## Code Quality")
        report.append(f"- **Total Modules:** {quality['total_modules']}")
        report.append(f"- **Modules with Issues:** {quality['modules_with_issues']}")
        report.append(f"- **Average Complexity:** {quality['average_complexity']:.2f}")
        report.append(f"- **Documentation Coverage:** {quality['average_documentation_coverage']:.1f}%")
        report.append(f"- **Type Hint Coverage:** {quality['type_hint_coverage']:.1f}%")
        report.append("")

        # Documentation
        docs = analysis['documentation']
        report.append("## Documentation")
        report.append(f"- **Coverage:** {docs.coverage_percentage:.1f}%")
        report.append(f"- **Documented Modules:** {docs.documented_modules}/{docs.total_modules}")
        report.append(f"- **Documented Functions:** {docs.documented_functions}/{docs.total_functions}")
        report.append("")

        # Testing
        testing = analysis['testing']
        report.append("## Testing")
        report.append(f"- **Test Files:** {testing['test_files']}")
        report.append(f"- **Test Functions:** {testing['test_functions']}")
        report.append(f"- **Coverage Tool:** {'Available' if testing['coverage_available'] else 'Not configured'}")
        report.append("")

        # Recommendations
        recommendations = analysis['recommendations']
        if recommendations:
            report.append("## Recommendations")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")

        return "\n".join(report)
