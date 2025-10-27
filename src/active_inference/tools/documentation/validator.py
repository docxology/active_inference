"""
Documentation Validator

Validates documentation structure, content, and adherence to standards.
Provides comprehensive validation of documentation completeness and quality.
"""

import ast
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """A documentation validation rule"""
    name: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'structure', 'content', 'style', 'completeness'
    check_function: callable


@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule_name: str
    severity: str
    message: str
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    element_name: Optional[str] = None
    suggestion: Optional[str] = None


class DocumentationValidator:
    """
    Comprehensive documentation validator.

    Validates documentation against established standards, checks for
    completeness, consistency, and quality issues.
    """

    def __init__(self, source_paths: List[Path]):
        """
        Initialize documentation validator

        Args:
            source_paths: List of source directories to validate
        """
        self.source_paths = [Path(p) for p in source_paths]
        self.logger = logging.getLogger(__name__)

        # Initialize validation rules
        self.rules = self._initialize_validation_rules()

        self.logger.info(f"Documentation validator initialized with {len(self.rules)} rules")

    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize validation rules"""
        rules = []

        # Structure rules
        rules.append(ValidationRule(
            name="module_docstring_required",
            description="Module must have a docstring",
            severity="error",
            category="structure",
            check_function=self._check_module_docstring
        ))

        rules.append(ValidationRule(
            name="class_docstring_required",
            description="Class must have a docstring",
            severity="error",
            category="structure",
            check_function=self._check_class_docstring
        ))

        rules.append(ValidationRule(
            name="function_docstring_required",
            description="Public function must have a docstring",
            severity="warning",
            category="structure",
            check_function=self._check_function_docstring
        ))

        # Content rules
        rules.append(ValidationRule(
            name="docstring_has_summary",
            description="Docstring must have a summary line",
            severity="warning",
            category="content",
            check_function=self._check_docstring_summary
        ))

        rules.append(ValidationRule(
            name="docstring_has_description",
            description="Docstring should have a description beyond just the summary",
            severity="info",
            category="content",
            check_function=self._check_docstring_description
        ))

        rules.append(ValidationRule(
            name="parameters_documented",
            description="Function parameters should be documented",
            severity="warning",
            category="content",
            check_function=self._check_parameter_documentation
        ))

        rules.append(ValidationRule(
            name="returns_documented",
            description="Function return values should be documented",
            severity="warning",
            category="content",
            check_function=self._check_return_documentation
        ))

        # Style rules
        rules.append(ValidationRule(
            name="docstring_format_consistent",
            description="Docstring format should be consistent",
            severity="info",
            category="style",
            check_function=self._check_docstring_format
        ))

        rules.append(ValidationRule(
            name="no_broken_links",
            description="Documentation should not contain broken links",
            severity="warning",
            category="content",
            check_function=self._check_broken_links
        ))

        # Completeness rules
        rules.append(ValidationRule(
            name="examples_provided",
            description="Important functions should have usage examples",
            severity="info",
            category="completeness",
            check_function=self._check_examples
        ))

        rules.append(ValidationRule(
            name="raises_documented",
            description="Functions that can raise exceptions should document them",
            severity="info",
            category="completeness",
            check_function=self._check_exception_documentation
        ))

        return rules

    def validate_documentation(self) -> Dict[str, Any]:
        """
        Perform comprehensive documentation validation

        Returns:
            Complete validation report
        """
        self.logger.info("Starting comprehensive documentation validation")

        results = []
        errors = 0
        warnings = 0
        info = 0

        # Run all validation rules
        for rule in self.rules:
            try:
                rule_results = self._run_validation_rule(rule)
                results.extend(rule_results)

                # Count by severity
                for result in rule_results:
                    if result.severity == 'error':
                        errors += 1
                    elif result.severity == 'warning':
                        warnings += 1
                    elif result.severity == 'info':
                        info += 1

            except Exception as e:
                self.logger.error(f"Error running validation rule {rule.name}: {e}")

        # Generate summary
        summary = {
            'total_issues': len(results),
            'errors': errors,
            'warnings': warnings,
            'info': info,
            'validation_passed': errors == 0,
            'results_by_category': self._group_results_by_category(results),
            'results_by_file': self._group_results_by_file(results),
            'results': results
        }

        self.logger.info(f"Documentation validation completed: {errors} errors, {warnings} warnings, {info} info")
        return summary

    def _run_validation_rule(self, rule: ValidationRule) -> List[ValidationResult]:
        """Run a single validation rule"""
        results = []

        for source_path in self.source_paths:
            if not source_path.exists():
                continue

            # Find Python files
            python_files = list(source_path.rglob('*.py'))
            python_files = [f for f in python_files if not self._should_ignore(f)]

            for file_path in python_files:
                try:
                    file_results = rule.check_function(file_path)
                    results.extend(file_results)
                except Exception as e:
                    self.logger.warning(f"Error checking {rule.name} for {file_path}: {e}")

        return results

    def _check_module_docstring(self, file_path: Path) -> List[ValidationResult]:
        """Check if module has docstring"""
        results = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            docstring = ast.get_docstring(tree)

            if not docstring:
                results.append(ValidationResult(
                    rule_name="module_docstring_required",
                    severity="error",
                    message="Module is missing docstring",
                    file_path=file_path,
                    suggestion="Add a module docstring describing the module's purpose and functionality"
                ))

        except Exception as e:
            results.append(ValidationResult(
                rule_name="module_docstring_required",
                severity="error",
                message=f"Could not parse module: {e}",
                file_path=file_path
            ))

        return results

    def _check_class_docstring(self, file_path: Path) -> List[ValidationResult]:
        """Check if classes have docstrings"""
        results = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    if not docstring:
                        results.append(ValidationResult(
                            rule_name="class_docstring_required",
                            severity="error",
                            message=f"Class '{node.name}' is missing docstring",
                            file_path=file_path,
                            line_number=node.lineno,
                            element_name=node.name,
                            suggestion=f"Add a class docstring describing the class purpose, parameters, and methods"
                        ))

        except Exception as e:
            self.logger.warning(f"Error checking classes in {file_path}: {e}")

        return results

    def _check_function_docstring(self, file_path: Path) -> List[ValidationResult]:
        """Check if public functions have docstrings"""
        results = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip private functions (starting with _)
                    if node.name.startswith('_'):
                        continue

                    docstring = ast.get_docstring(node)
                    if not docstring:
                        results.append(ValidationResult(
                            rule_name="function_docstring_required",
                            severity="warning",
                            message=f"Function '{node.name}' is missing docstring",
                            file_path=file_path,
                            line_number=node.lineno,
                            element_name=node.name,
                            suggestion=f"Add a function docstring describing parameters, return value, and behavior"
                        ))

        except Exception as e:
            self.logger.warning(f"Error checking functions in {file_path}: {e}")

        return results

    def _check_docstring_summary(self, file_path: Path) -> List[ValidationResult]:
        """Check if docstrings have proper summaries"""
        results = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            # Check module docstring
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                if not self._has_proper_summary(module_docstring):
                    results.append(ValidationResult(
                        rule_name="docstring_has_summary",
                        severity="warning",
                        message="Module docstring missing proper summary",
                        file_path=file_path,
                        suggestion="Add a concise summary line at the beginning of the module docstring"
                    ))

            # Check function and class docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)
                    if docstring and not self._has_proper_summary(docstring):
                        results.append(ValidationResult(
                            rule_name="docstring_has_summary",
                            severity="warning",
                            message=f"{'Function' if isinstance(node, ast.FunctionDef) else 'Class'} '{node.name}' docstring missing proper summary",
                            file_path=file_path,
                            line_number=node.lineno,
                            element_name=node.name,
                            suggestion="Add a concise summary line at the beginning of the docstring"
                        ))

        except Exception as e:
            self.logger.warning(f"Error checking docstring summaries in {file_path}: {e}")

        return results

    def _check_docstring_description(self, file_path: Path) -> List[ValidationResult]:
        """Check if docstrings have descriptions beyond summary"""
        results = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)
                    if docstring and not self._has_description(docstring):
                        results.append(ValidationResult(
                            rule_name="docstring_has_description",
                            severity="info",
                            message=f"{'Function' if isinstance(node, ast.FunctionDef) else 'Class'} '{node.name}' docstring could use more detailed description",
                            file_path=file_path,
                            line_number=node.lineno,
                            element_name=node.name,
                            suggestion="Add a more detailed description of the functionality and behavior"
                        ))

        except Exception as e:
            self.logger.warning(f"Error checking docstring descriptions in {file_path}: {e}")

        return results

    def _check_parameter_documentation(self, file_path: Path) -> List[ValidationResult]:
        """Check if function parameters are documented"""
        results = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Get function parameters
                        args = [arg.arg for arg in node.args.args if arg.arg != 'self']

                        # Check if parameters are documented in docstring
                        if args and not self._parameters_documented(docstring, args):
                            results.append(ValidationResult(
                                rule_name="parameters_documented",
                                severity="warning",
                                message=f"Function '{node.name}' parameters not documented",
                                file_path=file_path,
                                line_number=node.lineno,
                                element_name=node.name,
                                suggestion=f"Add parameter documentation in Args section: {', '.join(args)}"
                            ))

        except Exception as e:
            self.logger.warning(f"Error checking parameter documentation in {file_path}: {e}")

        return results

    def _check_return_documentation(self, file_path: Path) -> List[ValidationResult]:
        """Check if function return values are documented"""
        results = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if docstring and node.returns:
                        # Check if return value is documented
                        if not self._return_documented(docstring):
                            results.append(ValidationResult(
                                rule_name="returns_documented",
                                severity="warning",
                                message=f"Function '{node.name}' return value not documented",
                                file_path=file_path,
                                line_number=node.lineno,
                                element_name=node.name,
                                suggestion="Add return value documentation in Returns section"
                            ))

        except Exception as e:
            self.logger.warning(f"Error checking return documentation in {file_path}: {e}")

        return results

    def _check_docstring_format(self, file_path: Path) -> List[ValidationResult]:
        """Check docstring format consistency"""
        results = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring and not self._is_valid_docstring_format(docstring):
                        results.append(ValidationResult(
                            rule_name="docstring_format_consistent",
                            severity="info",
                            message=f"{'Function' if isinstance(node, ast.FunctionDef) else 'Class' if isinstance(node, ast.ClassDef) else 'Module'} docstring format could be improved",
                            file_path=file_path,
                            line_number=getattr(node, 'lineno', None),
                            element_name=getattr(node, 'name', 'module'),
                            suggestion="Use consistent docstring format with proper sections and formatting"
                        ))

        except Exception as e:
            self.logger.warning(f"Error checking docstring format in {file_path}: {e}")

        return results

    def _check_broken_links(self, file_path: Path) -> List[ValidationResult]:
        """Check for broken links in documentation"""
        results = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for links in docstrings and comments
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                # Check for URLs
                urls = re.findall(r'https?://[^\s)]+', line)
                for url in urls:
                    # Simple validation - could be enhanced with actual HTTP checks
                    if 'localhost' in url or '127.0.0.1' in url:
                        results.append(ValidationResult(
                            rule_name="no_broken_links",
                            severity="warning",
                            message=f"Localhost URL found in documentation: {url}",
                            file_path=file_path,
                            line_number=i,
                            suggestion="Replace localhost URLs with relative paths or proper documentation links"
                        ))

        except Exception as e:
            self.logger.warning(f"Error checking links in {file_path}: {e}")

        return results

    def _check_examples(self, file_path: Path) -> List[ValidationResult]:
        """Check if important functions have examples"""
        results = []

        # Define which functions should have examples (could be configurable)
        important_functions = ['main', 'run', 'execute', 'process', 'generate', 'create', 'build']

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if any(important in node.name.lower() for important in important_functions):
                        docstring = ast.get_docstring(node)
                        if docstring and 'example' not in docstring.lower():
                            results.append(ValidationResult(
                                rule_name="examples_provided",
                                severity="info",
                                message=f"Important function '{node.name}' should have usage examples",
                                file_path=file_path,
                                line_number=node.lineno,
                                element_name=node.name,
                                suggestion="Add usage examples in Examples section of docstring"
                            ))

        except Exception as e:
            self.logger.warning(f"Error checking examples in {file_path}: {e}")

        return results

    def _check_exception_documentation(self, file_path: Path) -> List[ValidationResult]:
        """Check if functions document exceptions they can raise"""
        results = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Check if function has exception handling
                        has_exceptions = any(
                            isinstance(n, (ast.Try, ast.Raise))
                            for n in ast.walk(node)
                        )

                        if has_exceptions and not self._exceptions_documented(docstring):
                            results.append(ValidationResult(
                                rule_name="raises_documented",
                                severity="info",
                                message=f"Function '{node.name}' should document exceptions it can raise",
                                file_path=file_path,
                                line_number=node.lineno,
                                element_name=node.name,
                                suggestion="Add exception documentation in Raises section"
                            ))

        except Exception as e:
            self.logger.warning(f"Error checking exception documentation in {file_path}: {e}")

        return results

    # Helper methods for validation checks

    def _has_proper_summary(self, docstring: str) -> bool:
        """Check if docstring has a proper summary line"""
        lines = docstring.strip().split('\n')
        if not lines:
            return False

        summary = lines[0].strip()
        return len(summary.split()) > 3 and not summary.endswith('.')

    def _has_description(self, docstring: str) -> bool:
        """Check if docstring has description beyond summary"""
        lines = docstring.strip().split('\n')
        if len(lines) < 2:
            return False

        # Check if there are lines after the summary that contain substantial content
        content_lines = [line.strip() for line in lines[1:] if line.strip()]
        return len(content_lines) > 1 or any(len(line.split()) > 10 for line in content_lines)

    def _parameters_documented(self, docstring: str, parameters: List[str]) -> bool:
        """Check if function parameters are documented"""
        docstring_lower = docstring.lower()
        return any(param.lower() in docstring_lower for param in parameters)

    def _return_documented(self, docstring: str) -> bool:
        """Check if return value is documented"""
        return any(keyword in docstring.lower() for keyword in ['returns', 'return', 'yields', 'yield'])

    def _is_valid_docstring_format(self, docstring: str) -> bool:
        """Check if docstring follows good formatting practices"""
        # Check for common formatting issues
        lines = docstring.split('\n')

        # Should not be too long without sections
        if len(lines) > 10 and not any(':' in line for line in lines):
            return False

        # Should not have inconsistent indentation
        non_empty_lines = [line for line in lines if line.strip()]
        if len(non_empty_lines) > 1:
            indentations = [len(line) - len(line.lstrip()) for line in non_empty_lines[1:]]
            if len(set(indentations)) > 2:  # More than 2 different indentations
                return False

        return True

    def _exceptions_documented(self, docstring: str) -> bool:
        """Check if exceptions are documented"""
        return any(keyword in docstring.lower() for keyword in ['raises', 'exceptions', 'exception', 'error'])

    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored in validation"""
        path_str = str(file_path)

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

        return False

    def _group_results_by_category(self, results: List[ValidationResult]) -> Dict[str, List[ValidationResult]]:
        """Group validation results by category"""
        categories = defaultdict(list)
        for result in results:
            # Find the rule that generated this result
            for rule in self.rules:
                if rule.name == result.rule_name:
                    categories[rule.category].append(result)
                    break

        return dict(categories)

    def _group_results_by_file(self, results: List[ValidationResult]) -> Dict[str, List[ValidationResult]]:
        """Group validation results by file"""
        files = defaultdict(list)
        for result in results:
            if result.file_path:
                files[str(result.file_path)].append(result)

        return dict(files)

    def generate_validation_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate detailed validation report

        Args:
            output_path: Optional path to save report

        Returns:
            Validation report as string
        """
        self.logger.info("Generating validation report")

        validation = self.validate_documentation()
        report = self._format_validation_report(validation)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Validation report saved to {output_path}")

        return report

    def _format_validation_report(self, validation: Dict[str, Any]) -> str:
        """Format validation results into readable report"""
        report = []
        report.append("# Documentation Validation Report")
        report.append("=" * 45)
        report.append("")

        # Summary
        report.append("## Summary")
        report.append(f"- **Total Issues:** {validation['total_issues']}")
        report.append(f"- **Errors:** {validation['errors']}")
        report.append(f"- **Warnings:** {validation['warnings']}")
        report.append(f"- **Info:** {validation['info']}")
        report.append(f"- **Validation Passed:** {'✅ Yes' if validation['validation_passed'] else '❌ No'}")
        report.append("")

        # Results by category
        report.append("## Issues by Category")
        for category, results in validation['results_by_category'].items():
            if results:
                report.append(f"### {category.title()} ({len(results)} issues)")
                for result in results[:10]:  # Show first 10
                    report.append(f"- **{result.severity.upper()}:** {result.message}")
                    if result.file_path and result.element_name:
                        report.append(f"  - File: {result.file_path}, Element: {result.element_name}")
                    elif result.file_path:
                        report.append(f"  - File: {result.file_path}")
                    if result.suggestion:
                        report.append(f"  - Suggestion: {result.suggestion}")
                if len(results) > 10:
                    report.append(f"- ... and {len(results) - 10} more")
                report.append("")

        # Results by file
        if validation['results_by_file']:
            report.append("## Issues by File")
            for file_path, results in validation['results_by_file'].items():
                if results:
                    report.append(f"### {file_path} ({len(results)} issues)")
                    for result in results:
                        report.append(f"- **{result.severity.upper()}:** {result.message}")
                        if result.element_name:
                            report.append(f"  - Element: {result.element_name}")
                        if result.suggestion:
                            report.append(f"  - Suggestion: {result.suggestion}")
                    report.append("")

        # Recommendations
        report.append("## Recommendations")
        if validation['errors'] > 0:
            report.append("1. **Fix all errors** before proceeding")
        if validation['warnings'] > 0:
            report.append(f"2. **Address {validation['warnings']} warnings** to improve documentation quality")
        if validation['info'] > 0:
            report.append(f"3. **Review {validation['info']} suggestions** for best practices")

        if validation['validation_passed']:
            report.append("🎉 **All validation checks passed!**")

        report.append("")

        return "\n".join(report)
