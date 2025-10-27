"""
Documentation Analyzer

Analyzes documentation quality, coverage, and provides insights for improvement.
Provides comprehensive analysis of docstrings, documentation structure, and
content quality metrics.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

import logging

logger = logging.getLogger(__name__)


@dataclass
class DocstringMetrics:
    """Metrics for a single docstring"""
    has_summary: bool = False
    has_description: bool = False
    has_parameters: bool = False
    has_returns: bool = False
    has_examples: bool = False
    has_raises: bool = False
    parameter_count: int = 0
    word_count: int = 0
    quality_score: float = 0.0


@dataclass
class DocumentationQuality:
    """Overall documentation quality metrics"""
    total_elements: int = 0
    documented_elements: int = 0
    coverage_percentage: float = 0.0
    average_quality: float = 0.0
    missing_docs: List[str] = field(default_factory=list)
    poor_quality_docs: List[str] = field(default_factory=list)
    excellent_docs: List[str] = field(default_factory=list)
    quality_distribution: Dict[str, int] = field(default_factory=dict)


class DocumentationAnalyzer:
    """
    Comprehensive documentation quality analyzer.

    Analyzes docstring quality, coverage, and provides actionable insights
    for improving documentation standards.
    """

    def __init__(self, source_paths: List[Path]):
        """
        Initialize documentation analyzer

        Args:
            source_paths: List of source directories to analyze
        """
        self.source_paths = [Path(p) for p in source_paths]
        self.logger = setup_logger(__name__)

        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'poor': 0.3
        }

        self.logger.info(f"Documentation analyzer initialized for {len(self.source_paths)} source paths")

    def analyze_documentation(self) -> DocumentationQuality:
        """
        Perform comprehensive documentation analysis

        Returns:
            Complete documentation quality report
        """
        self.logger.info("Starting comprehensive documentation analysis")

        quality = DocumentationQuality()

        # Analyze each source path
        for source_path in self.source_paths:
            if source_path.exists():
                path_quality = self._analyze_path_documentation(source_path)
                self._merge_quality_reports(quality, path_quality)

        # Calculate overall metrics
        if quality.total_elements > 0:
            quality.coverage_percentage = (quality.documented_elements / quality.total_elements) * 100
            quality.average_quality = sum(
                score for score in [self._calculate_element_quality_score(elem) for elem in quality.excellent_docs + quality.poor_quality_docs]
            ) / len(quality.excellent_docs + quality.poor_quality_docs) if (quality.excellent_docs + quality.poor_quality_docs) else 0.0

        self.logger.info(f"Documentation analysis completed: {quality.coverage_percentage:.1f}% coverage")
        return quality

    def _analyze_path_documentation(self, source_path: Path) -> DocumentationQuality:
        """Analyze documentation for a specific source path"""
        quality = DocumentationQuality()

        # Find all Python files
        python_files = list(source_path.rglob('*.py'))
        python_files = [f for f in python_files if not self._should_ignore(f)]

        for file_path in python_files:
            try:
                file_quality = self._analyze_file_documentation(file_path)
                self._merge_quality_reports(quality, file_quality)
            except Exception as e:
                self.logger.warning(f"Could not analyze documentation for {file_path}: {e}")

        return quality

    def _analyze_file_documentation(self, file_path: Path) -> DocumentationQuality:
        """Analyze documentation for a single file"""
        quality = DocumentationQuality()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            # Analyze module docstring
            module_docstring = ast.get_docstring(tree)
            quality.total_elements += 1
            if module_docstring:
                quality.documented_elements += 1
                docstring_metrics = self._analyze_docstring(module_docstring)
                self._categorize_docstring_quality(str(file_path), "module", docstring_metrics, quality)

            # Analyze functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    quality.total_elements += 1
                    if ast.get_docstring(node):
                        quality.documented_elements += 1
                        func_docstring = ast.get_docstring(node)
                        docstring_metrics = self._analyze_docstring(func_docstring)
                        self._categorize_docstring_quality(f"{file_path}::{node.name}", "function", docstring_metrics, quality)

                elif isinstance(node, ast.ClassDef):
                    quality.total_elements += 1
                    if ast.get_docstring(node):
                        quality.documented_elements += 1
                        class_docstring = ast.get_docstring(node)
                        docstring_metrics = self._analyze_docstring(class_docstring)
                        self._categorize_docstring_quality(f"{file_path}::{node.name}", "class", docstring_metrics, quality)

        except Exception as e:
            self.logger.warning(f"Error analyzing {file_path}: {e}")

        return quality

    def _analyze_docstring(self, docstring: str) -> DocstringMetrics:
        """Analyze quality metrics for a single docstring"""
        metrics = DocstringMetrics()

        if not docstring:
            return metrics

        lines = docstring.strip().split('\n')
        metrics.word_count = len(docstring.split())

        # Remove leading/trailing whitespace and empty lines
        content_lines = [line.strip() for line in lines if line.strip()]

        if len(content_lines) == 0:
            return metrics

        # Check for summary (first non-empty line)
        first_line = content_lines[0]
        metrics.has_summary = len(first_line.split()) > 3

        # Check for sections
        current_section = None
        section_content = []

        for line in content_lines[1:]:  # Skip first line (summary)
            if line.startswith(('Args:', 'Arguments:', 'Parameters:')):
                metrics.has_parameters = True
                current_section = 'parameters'
                section_content = []
            elif line.startswith(('Returns:', 'Return:', 'Yields:')):
                metrics.has_returns = True
                current_section = 'returns'
                section_content = []
            elif line.startswith(('Examples:', 'Example:')):
                metrics.has_examples = True
                current_section = 'examples'
                section_content = []
            elif line.startswith(('Raises:', 'Exceptions:')):
                metrics.has_raises = True
                current_section = 'raises'
                section_content = []
            elif line.startswith(('Note:', 'Notes:', 'Warning:', 'See also:')):
                current_section = 'other'
                section_content = []
            elif current_section and line and not line.startswith(' '):
                # New section or end of current section
                current_section = None
                section_content = []
            elif current_section:
                section_content.append(line)

        # Count parameters in Args section
        if metrics.has_parameters:
            param_lines = [line for line in content_lines if '    ' in line and ':' in line]
            metrics.parameter_count = len(param_lines)

        # Calculate quality score
        metrics.quality_score = self._calculate_docstring_quality_score(metrics)

        return metrics

    def _calculate_docstring_quality_score(self, metrics: DocstringMetrics) -> float:
        """Calculate quality score for a docstring (0.0 to 1.0)"""
        score = 0.0

        # Basic sections (weighted)
        if metrics.has_summary:
            score += 0.3
        if metrics.has_parameters:
            score += 0.2
        if metrics.has_returns:
            score += 0.2
        if metrics.has_examples:
            score += 0.2
        if metrics.has_raises:
            score += 0.1

        # Length bonus (prefer informative but not verbose)
        if 20 <= metrics.word_count <= 200:
            score += 0.1
        elif metrics.word_count > 200:
            score += 0.05  # Still good but very long

        return min(score, 1.0)

    def _categorize_docstring_quality(self, element_name: str, element_type: str,
                                    metrics: DocstringMetrics, quality: DocumentationQuality) -> None:
        """Categorize docstring quality and update quality report"""
        if metrics.quality_score >= self.quality_thresholds['excellent']:
            quality.excellent_docs.append(f"{element_type}:{element_name}")
        elif metrics.quality_score <= self.quality_thresholds['poor']:
            quality.poor_quality_docs.append(f"{element_type}:{element_name}")
            quality.missing_docs.append(element_name)

    def _merge_quality_reports(self, target: DocumentationQuality, source: DocumentationQuality) -> None:
        """Merge two quality reports"""
        target.total_elements += source.total_elements
        target.documented_elements += source.documented_elements
        target.excellent_docs.extend(source.excellent_docs)
        target.poor_quality_docs.extend(source.poor_quality_docs)
        target.missing_docs.extend(source.missing_docs)

        # Merge quality distribution
        for key, value in source.quality_distribution.items():
            target.quality_distribution[key] = target.quality_distribution.get(key, 0) + value

    def _calculate_element_quality_score(self, element_name: str) -> float:
        """Calculate quality score for a documented element"""
        # This would be implemented based on specific quality criteria
        # For now, return a placeholder score
        return 0.7

    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored in analysis"""
        path_str = str(file_path)

        # Ignore common patterns
        ignore_patterns = [
            '__pycache__',
            '.git',
            'node_modules',
            '.pytest_cache',
            'build',
            'dist',
            '.coverage',
            '*.egg-info',
            'test_',
            '_test',
            'tests'
        ]

        for pattern in ignore_patterns:
            if pattern in path_str:
                return True

        # Check if it's a hidden file/directory
        parts = file_path.parts
        if any(part.startswith('.') for part in parts):
            return True

        return False

    def generate_quality_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate detailed quality report

        Args:
            output_path: Optional path to save report

        Returns:
            Quality report as string
        """
        self.logger.info("Generating documentation quality report")

        quality = self.analyze_documentation()
        report = self._format_quality_report(quality)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Quality report saved to {output_path}")

        return report

    def _format_quality_report(self, quality: DocumentationQuality) -> str:
        """Format quality analysis into readable report"""
        report = []
        report.append("# Documentation Quality Report")
        report.append("=" * 40)
        report.append("")

        # Overall metrics
        report.append("## Overview")
        report.append(f"- **Coverage:** {quality.coverage_percentage:.1f}%")
        report.append(f"- **Total Elements:** {quality.total_elements}")
        report.append(f"- **Documented Elements:** {quality.documented_elements}")
        report.append(f"- **Average Quality Score:** {quality.average_quality:.2f}")
        report.append("")

        # Quality distribution
        report.append("## Quality Distribution")
        excellent_count = len(quality.excellent_docs)
        good_count = quality.total_elements - excellent_count - len(quality.poor_quality_docs)
        poor_count = len(quality.poor_quality_docs)

        report.append(f"- **Excellent Documentation:** {excellent_count} elements")
        report.append(f"- **Good Documentation:** {good_count} elements")
        report.append(f"- **Poor Documentation:** {poor_count} elements")
        report.append("")

        # Missing documentation
        if quality.missing_docs:
            report.append("## Missing Documentation")
            for element in quality.missing_docs[:20]:  # Show first 20
                report.append(f"- {element}")
            if len(quality.missing_docs) > 20:
                report.append(f"- ... and {len(quality.missing_docs) - 20} more")
            report.append("")

        # Poor quality documentation
        if quality.poor_quality_docs:
            report.append("## Poor Quality Documentation")
            for element in quality.poor_quality_docs[:20]:  # Show first 20
                report.append(f"- {element}")
            if len(quality.poor_quality_docs) > 20:
                report.append(f"- ... and {len(quality.poor_quality_docs) - 20} more")
            report.append("")

        # Recommendations
        report.append("## Recommendations")
        if quality.coverage_percentage < 80:
            report.append(f"1. Improve documentation coverage from {quality.coverage_percentage:.1f}% to 80%+")
        if poor_count > 0:
            report.append(f"2. Review and improve {poor_count} poorly documented elements")
        if quality.average_quality < 0.7:
            report.append("3. Add comprehensive docstrings with examples and parameter descriptions")
        if not quality.missing_docs and not quality.poor_quality_docs:
            report.append("1. Excellent! Documentation quality is very good.")
        report.append("")

        return "\n".join(report)

    def suggest_improvements(self, target_coverage: float = 90.0) -> Dict[str, Any]:
        """
        Suggest specific improvements to reach target coverage

        Args:
            target_coverage: Target documentation coverage percentage

        Returns:
            Improvement suggestions
        """
        self.logger.info(f"Generating improvement suggestions for {target_coverage}% coverage")

        current_quality = self.analyze_documentation()

        suggestions = {
            'current_coverage': current_quality.coverage_percentage,
            'target_coverage': target_coverage,
            'gap': target_coverage - current_quality.coverage_percentage,
            'missing_elements': len(current_quality.missing_docs),
            'poor_quality_elements': len(current_quality.poor_quality_docs),
            'priority_actions': [],
            'estimated_effort': 'medium'
        }

        # Generate priority actions
        if current_quality.coverage_percentage < target_coverage:
            elements_needed = int((target_coverage - current_quality.coverage_percentage) / 100 * current_quality.total_elements)
            suggestions['priority_actions'].append(f"Add documentation for {elements_needed} more elements")

        if current_quality.poor_quality_docs:
            suggestions['priority_actions'].append(f"Improve {len(current_quality.poor_quality_docs)} poor quality docstrings")

        if not current_quality.missing_docs and not current_quality.poor_quality_docs:
            suggestions['priority_actions'].append("Focus on adding examples and comprehensive parameter descriptions")
            suggestions['estimated_effort'] = 'low'

        # Estimate effort
        total_actions = len(suggestions['priority_actions'])
        if total_actions > 5:
            suggestions['estimated_effort'] = 'high'
        elif total_actions == 0:
            suggestions['estimated_effort'] = 'none'

        return suggestions
