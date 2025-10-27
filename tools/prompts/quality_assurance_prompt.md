# Comprehensive Quality Assurance and Validation Prompt

**"Active Inference for, with, by Generative AI"**

## 🎯 Mission: Establish Comprehensive Quality Assurance Framework

You are tasked with developing comprehensive quality assurance and validation systems for the Active Inference Knowledge Environment. This involves creating automated quality checking, code review processes, performance monitoring, security validation, and compliance verification that ensures the platform meets the highest standards of quality and reliability.

## 📋 Quality Assurance Requirements

### Core Quality Standards (MANDATORY)
1. **Automated Quality Gates**: All code and content must pass automated quality checks
2. **Continuous Validation**: Ongoing monitoring and validation of system components
3. **Security First**: Comprehensive security assessment and vulnerability prevention
4. **Performance Standards**: Guaranteed performance levels and scalability validation
5. **Compliance Verification**: Regulatory and standards compliance validation
6. **User Experience Validation**: Comprehensive UX testing and accessibility verification

### Quality Assurance Architecture
```
qa/
├── automated_testing/        # Automated quality gate testing
│   ├── code_quality.py       # Code quality and style validation
│   ├── security_scanning.py  # Security vulnerability scanning
│   ├── performance_testing.py # Performance benchmark validation
│   ├── integration_validation.py # System integration verification
│   └── compliance_checking.py # Regulatory compliance validation
├── code_review/              # Automated code review systems
│   ├── static_analysis.py    # Static code analysis
│   ├── complexity_analysis.py # Code complexity measurement
│   ├── dependency_checking.py # Dependency vulnerability assessment
│   ├── documentation_validation.py # Documentation completeness checking
│   └── standards_compliance.py # Coding standards verification
├── monitoring/               # Continuous quality monitoring
│   ├── health_monitoring.py  # System health and availability monitoring
│   ├── performance_monitoring.py # Performance metrics tracking
│   ├── error_tracking.py     # Error detection and alerting
│   ├── user_experience_monitoring.py # UX quality assessment
│   └── quality_metrics_dashboard.py # Quality metrics visualization
└── validation/               # Content and system validation
    ├── knowledge_validation.py # Knowledge base accuracy validation
    ├── api_validation.py      # API contract validation
    ├── data_validation.py     # Data integrity and quality validation
    ├── accessibility_testing.py # Accessibility compliance testing
    └── user_acceptance_testing.py # End-to-end user validation
```

## 🏗️ Automated Quality Gate System

### Phase 1: Code Quality Validation Framework

#### 1.1 Comprehensive Code Quality Checker
```python
import ast
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import subprocess
import json

@dataclass
class QualityIssue:
    """Represents a quality issue found during validation"""
    issue_type: str
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    file_path: str
    line_number: Optional[int]
    column_number: Optional[int]
    message: str
    rule_id: str
    suggestion: Optional[str] = None
    context: Optional[str] = None

@dataclass
class QualityReport:
    """Comprehensive quality report"""
    report_id: str
    timestamp: datetime
    target_path: str
    total_files: int
    issues_found: List[QualityIssue]
    quality_score: float
    passed_gates: List[str]
    failed_gates: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]

class CodeQualityValidator:
    """Comprehensive code quality validation system"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize code quality validator"""
        self.config = config
        self.logger = logging.getLogger('CodeQualityValidator')

        # Quality thresholds
        self.thresholds = {
            'max_complexity': config.get('max_complexity', 10),
            'max_line_length': config.get('max_line_length', 88),
            'max_function_length': config.get('max_function_length', 50),
            'min_test_coverage': config.get('min_test_coverage', 80),
            'max_dependencies': config.get('max_dependencies', 20)
        }

        # Validation rules
        self.validation_rules = self.initialize_validation_rules()

    def initialize_validation_rules(self) -> Dict[str, Callable]:
        """Initialize validation rules"""
        return {
            'syntax_check': self.check_syntax,
            'import_organization': self.check_import_organization,
            'naming_conventions': self.check_naming_conventions,
            'docstring_completeness': self.check_docstring_completeness,
            'type_annotations': self.check_type_annotations,
            'error_handling': self.check_error_handling,
            'security_vulnerabilities': self.check_security_vulnerabilities,
            'performance_issues': self.check_performance_issues,
            'code_complexity': self.check_code_complexity,
            'test_coverage': self.check_test_coverage
        }

    def validate_codebase(self, target_path: str, rules_to_run: Optional[List[str]] = None) -> QualityReport:
        """Run comprehensive code quality validation"""
        start_time = datetime.now()

        if rules_to_run is None:
            rules_to_run = list(self.validation_rules.keys())

        all_issues = []
        total_files = 0

        # Find all Python files
        python_files = self.find_python_files(target_path)

        for file_path in python_files:
            total_files += 1
            file_issues = []

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Run validation rules
                for rule_name in rules_to_run:
                    if rule_name in self.validation_rules:
                        rule_issues = self.validation_rules[rule_name](file_path, content)
                        file_issues.extend(rule_issues)

            except Exception as e:
                # File reading error
                error_issue = QualityIssue(
                    issue_type='file_error',
                    severity='high',
                    file_path=file_path,
                    line_number=None,
                    column_number=None,
                    message=f"Failed to read file: {e}",
                    rule_id='file_access'
                )
                file_issues.append(error_issue)

            all_issues.extend(file_issues)

        # Calculate quality score
        quality_score = self.calculate_quality_score(all_issues, total_files)

        # Determine gate status
        passed_gates, failed_gates = self.evaluate_quality_gates(all_issues, quality_score)

        # Generate recommendations
        recommendations = self.generate_recommendations(all_issues, quality_score)

        report = QualityReport(
            report_id=f"quality_report_{int(start_time.timestamp())}",
            timestamp=start_time,
            target_path=target_path,
            total_files=total_files,
            issues_found=all_issues,
            quality_score=quality_score,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            recommendations=recommendations,
            metadata={
                'validation_rules_run': rules_to_run,
                'thresholds_used': self.thresholds,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        )

        return report

    def find_python_files(self, target_path: str) -> List[str]:
        """Find all Python files in target path"""
        import os

        python_files = []

        for root, dirs, files in os.walk(target_path):
            # Skip common exclude directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', 'node_modules']]

            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        return python_files

    def check_syntax(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check Python syntax"""
        issues = []

        try:
            ast.parse(content)
        except SyntaxError as e:
            issues.append(QualityIssue(
                issue_type='syntax_error',
                severity='critical',
                file_path=file_path,
                line_number=e.lineno,
                column_number=e.offset,
                message=f"Syntax error: {e.msg}",
                rule_id='syntax_check',
                suggestion="Fix the syntax error in the code"
            ))

        return issues

    def check_import_organization(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check import organization and style"""
        issues = []
        lines = content.split('\n')

        # Check for imports after code
        code_started = False
        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped and not stripped.startswith('#') and not stripped.startswith('import') and not stripped.startswith('from'):
                code_started = True
            elif code_started and (stripped.startswith('import') or stripped.startswith('from')):
                issues.append(QualityIssue(
                    issue_type='import_organization',
                    severity='medium',
                    file_path=file_path,
                    line_number=i + 1,
                    column_number=None,
                    message="Import statement found after code has started",
                    rule_id='import_organization',
                    suggestion="Move all imports to the top of the file"
                ))

        return issues

    def check_naming_conventions(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check naming conventions"""
        issues = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                        issues.append(QualityIssue(
                            issue_type='naming_convention',
                            severity='low',
                            file_path=file_path,
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            message=f"Function name '{node.name}' does not follow snake_case convention",
                            rule_id='naming_conventions',
                            suggestion="Use snake_case for function names"
                        ))

                elif isinstance(node, ast.ClassDef):
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                        issues.append(QualityIssue(
                            issue_type='naming_convention',
                            severity='low',
                            file_path=file_path,
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            message=f"Class name '{node.name}' does not follow PascalCase convention",
                            rule_id='naming_conventions',
                            suggestion="Use PascalCase for class names"
                        ))

        except SyntaxError:
            # Skip naming checks if syntax is invalid
            pass

        return issues

    def check_docstring_completeness(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check docstring completeness"""
        issues = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        node_type = "function" if isinstance(node, ast.FunctionDef) else "class"
                        issues.append(QualityIssue(
                            issue_type='missing_docstring',
                            severity='medium',
                            file_path=file_path,
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            message=f"{node_type.capitalize()} '{node.name}' is missing a docstring",
                            rule_id='docstring_completeness',
                            suggestion="Add a docstring describing the function/class purpose and parameters"
                        ))

        except SyntaxError:
            pass

        return issues

    def check_type_annotations(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check type annotation completeness"""
        issues = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check function parameters
                    for arg in node.args.args:
                        if not arg.annotation and arg.arg != 'self':
                            issues.append(QualityIssue(
                                issue_type='missing_type_annotation',
                                severity='low',
                                file_path=file_path,
                                line_number=node.lineno,
                                column_number=node.col_offset,
                                message=f"Parameter '{arg.arg}' in function '{node.name}' is missing type annotation",
                                rule_id='type_annotations',
                                suggestion="Add type annotation for parameter"
                            ))

                    # Check return annotation
                    if not node.returns:
                        issues.append(QualityIssue(
                            issue_type='missing_return_annotation',
                            severity='low',
                            file_path=file_path,
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            message=f"Function '{node.name}' is missing return type annotation",
                            rule_id='type_annotations',
                            suggestion="Add return type annotation"
                        ))

        except SyntaxError:
            pass

        return issues

    def check_error_handling(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check error handling completeness"""
        issues = []

        try:
            tree = ast.parse(content)

            # Find all try-except blocks
            try_blocks = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    try_blocks.append(node)

            # Check for bare except clauses
            for try_block in try_blocks:
                for handler in try_block.handlers:
                    if handler.type is None:  # Bare except
                        issues.append(QualityIssue(
                            issue_type='bare_except',
                            severity='medium',
                            file_path=file_path,
                            line_number=handler.lineno,
                            column_number=handler.col_offset,
                            message="Bare 'except:' clause found",
                            rule_id='error_handling',
                            suggestion="Specify exception types to catch"
                        ))

        except SyntaxError:
            pass

        return issues

    def check_security_vulnerabilities(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check for common security vulnerabilities"""
        issues = []

        # Check for eval usage
        if 'eval(' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'eval(' in line:
                    issues.append(QualityIssue(
                        issue_type='security_vulnerability',
                        severity='high',
                        file_path=file_path,
                        line_number=i + 1,
                        column_number=None,
                        message="Use of 'eval()' detected - potential security risk",
                        rule_id='security_vulnerabilities',
                        suggestion="Avoid using eval(); use safer alternatives like ast.literal_eval()"
                    ))

        # Check for SQL injection patterns
        sql_patterns = [r'execute\s*\(.+\s*%\s*.*\)', r'execute\s*\(.+\s*\+\s*.*\)']
        for pattern in sql_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    issue_type='security_vulnerability',
                    severity='high',
                    file_path=file_path,
                    line_number=line_num,
                    column_number=None,
                    message="Potential SQL injection vulnerability detected",
                    rule_id='security_vulnerabilities',
                    suggestion="Use parameterized queries or prepared statements"
                ))

        return issues

    def check_performance_issues(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check for performance issues"""
        issues = []

        # Check for inefficient list operations
        if '.append(' in content and 'for ' in content:
            # Simple heuristic - could be improved
            issues.append(QualityIssue(
                issue_type='performance_issue',
                severity='low',
                file_path=file_path,
                line_number=None,
                column_number=None,
                message="Potential inefficient list operations detected",
                rule_id='performance_issues',
                suggestion="Consider using list comprehensions or generator expressions"
            ))

        return issues

    def check_code_complexity(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check code complexity metrics"""
        issues = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Simple complexity metric: number of statements
                    statements = len([n for n in ast.walk(node) if isinstance(n, (ast.Expr, ast.Assign, ast.Return, ast.If, ast.For, ast.While))])

                    if statements > self.thresholds['max_function_length']:
                        issues.append(QualityIssue(
                            issue_type='high_complexity',
                            severity='medium',
                            file_path=file_path,
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            message=f"Function '{node.name}' has high complexity ({statements} statements)",
                            rule_id='code_complexity',
                            suggestion="Consider breaking down the function into smaller, more focused functions"
                        ))

        except SyntaxError:
            pass

        return issues

    def check_test_coverage(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check if corresponding tests exist"""
        issues = []

        # This is a simplified check - in practice, would integrate with coverage tools
        if not file_path.startswith('test_') and not file_path.endswith('_test.py'):
            # Look for corresponding test file
            test_file_candidates = [
                file_path.replace('.py', '_test.py'),
                f"test_{file_path}",
                f"tests/test_{os.path.basename(file_path)}"
            ]

            test_exists = any(os.path.exists(candidate) for candidate in test_file_candidates)

            if not test_exists:
                issues.append(QualityIssue(
                    issue_type='missing_tests',
                    severity='medium',
                    file_path=file_path,
                    line_number=None,
                    column_number=None,
                    message="No corresponding test file found",
                    rule_id='test_coverage',
                    suggestion="Create unit tests for this module"
                ))

        return issues

    def calculate_quality_score(self, issues: List[QualityIssue], total_files: int) -> float:
        """Calculate overall quality score"""
        if not issues:
            return 100.0

        # Weight issues by severity
        severity_weights = {
            'critical': 10,
            'high': 5,
            'medium': 3,
            'low': 1,
            'info': 0.5
        }

        total_weighted_score = sum(severity_weights.get(issue.severity, 1) for issue in issues)

        # Normalize by files and ideal score
        max_expected_issues = total_files * 5  # Expect some issues per file
        normalized_score = max(0, 100 - (total_weighted_score / max(max_expected_issues, 1)) * 100)

        return round(normalized_score, 2)

    def evaluate_quality_gates(self, issues: List[QualityIssue], quality_score: float) -> Tuple[List[str], List[str]]:
        """Evaluate quality gates"""
        passed_gates = []
        failed_gates = []

        # Critical issues gate
        critical_issues = [i for i in issues if i.severity == 'critical']
        if len(critical_issues) == 0:
            passed_gates.append('no_critical_issues')
        else:
            failed_gates.append('no_critical_issues')

        # Quality score gate
        if quality_score >= 70:
            passed_gates.append('quality_score_threshold')
        else:
            failed_gates.append('quality_score_threshold')

        # Security gate
        security_issues = [i for i in issues if i.rule_id == 'security_vulnerabilities']
        if len(security_issues) == 0:
            passed_gates.append('no_security_issues')
        else:
            failed_gates.append('no_security_issues')

        return passed_gates, failed_gates

    def generate_recommendations(self, issues: List[QualityIssue], quality_score: float) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Group issues by type
        issue_types = {}
        for issue in issues:
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1

        # Generate recommendations based on common issues
        if 'missing_docstring' in issue_types:
            recommendations.append("Add comprehensive docstrings to all public functions and classes")

        if 'missing_type_annotation' in issue_types:
            recommendations.append("Add type annotations to function parameters and return values")

        if 'high_complexity' in issue_types:
            recommendations.append("Refactor complex functions into smaller, more focused functions")

        if 'security_vulnerability' in issue_types:
            recommendations.append("Address security vulnerabilities - avoid eval() and prevent SQL injection")

        if 'missing_tests' in issue_types:
            recommendations.append("Increase test coverage by adding unit tests for uncovered code")

        if quality_score < 70:
            recommendations.append("Focus on fixing critical and high-severity issues to improve overall quality score")

        if not recommendations:
            recommendations.append("Code quality is good - consider adding more advanced checks like performance profiling")

        return recommendations

    def export_report(self, report: QualityReport, format: str = 'json') -> str:
        """Export quality report"""
        if format == 'json':
            return json.dumps({
                'report_id': report.report_id,
                'timestamp': report.timestamp.isoformat(),
                'target_path': report.target_path,
                'total_files': report.total_files,
                'quality_score': report.quality_score,
                'issues_count': len(report.issues_found),
                'passed_gates': report.passed_gates,
                'failed_gates': report.failed_gates,
                'recommendations': report.recommendations,
                'issues': [
                    {
                        'type': issue.issue_type,
                        'severity': issue.severity,
                        'file': issue.file_path,
                        'line': issue.line_number,
                        'message': issue.message,
                        'suggestion': issue.suggestion
                    }
                    for issue in report.issues_found
                ]
            }, indent=2)

        elif format == 'html':
            # Generate HTML report
            html = f"""
            <html>
            <head><title>Quality Report {report.report_id}</title></head>
            <body>
                <h1>Code Quality Report</h1>
                <p><strong>Quality Score:</strong> {report.quality_score}/100</p>
                <p><strong>Files Analyzed:</strong> {report.total_files}</p>
                <p><strong>Issues Found:</strong> {len(report.issues_found)}</p>

                <h2>Issues by Severity</h2>
                <ul>
            """

            severity_counts = {}
            for issue in report.issues_found:
                severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

            for severity, count in severity_counts.items():
                html += f"<li>{severity.title()}: {count}</li>"

            html += """
                </ul>

                <h2>Recommendations</h2>
                <ul>
            """

            for rec in report.recommendations:
                html += f"<li>{rec}</li>"

            html += """
                </ul>
            </body>
            </html>
            """

            return html

        return ""
```

#### 1.2 Security Vulnerability Scanner
```python
import re
from typing import Dict, List, Any, Optional
import logging

class SecurityScanner:
    """Comprehensive security vulnerability scanner"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize security scanner"""
        self.config = config
        self.logger = logging.getLogger('SecurityScanner')

        # Security patterns to check
        self.security_patterns = self.initialize_security_patterns()

    def initialize_security_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize security vulnerability patterns"""
        return {
            'hardcoded_secrets': {
                'pattern': r'(?i)(password|secret|key|token)\s*[=:]\s*["\'][^"\']+["\']',
                'severity': 'high',
                'description': 'Hardcoded secrets or credentials detected'
            },
            'sql_injection': {
                'pattern': r'(?i)(execute|query)\s*\(\s*.*\+.*\s*\)',
                'severity': 'high',
                'description': 'Potential SQL injection vulnerability'
            },
            'command_injection': {
                'pattern': r'(?i)(subprocess\.|os\.system|os\.popen)\s*\(\s*.*\+.*\s*\)',
                'severity': 'high',
                'description': 'Potential command injection vulnerability'
            },
            'pickle_usage': {
                'pattern': r'\b(pickle\.|cPickle\.)',
                'severity': 'medium',
                'description': 'Use of pickle module - potential security risk'
            },
            'eval_usage': {
                'pattern': r'\beval\s*\(',
                'severity': 'high',
                'description': 'Use of eval() function - arbitrary code execution risk'
            },
            'weak_crypto': {
                'pattern': r'(?i)(md5|sha1)\s*\(',
                'severity': 'medium',
                'description': 'Weak cryptographic hash function usage'
            },
            'debug_enabled': {
                'pattern': r'debug\s*=\s*True',
                'severity': 'low',
                'description': 'Debug mode enabled in production code'
            },
            'insecure_random': {
                'pattern': r'(?i)random\.(randint|choice|sample)',
                'severity': 'low',
                'description': 'Use of cryptographically insecure random functions'
            }
        }

    def scan_file(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Scan a file for security vulnerabilities"""
        vulnerabilities = []

        for vuln_name, vuln_config in self.security_patterns.items():
            pattern = vuln_config['pattern']
            matches = re.finditer(pattern, content)

            for match in matches:
                line_num = content[:match.start()].count('\n') + 1

                vulnerability = {
                    'vulnerability_type': vuln_name,
                    'severity': vuln_config['severity'],
                    'file_path': file_path,
                    'line_number': line_num,
                    'description': vuln_config['description'],
                    'matched_text': match.group(),
                    'recommendation': self.get_recommendation(vuln_name)
                }

                vulnerabilities.append(vulnerability)

        return vulnerabilities

    def get_recommendation(self, vulnerability_type: str) -> str:
        """Get remediation recommendation for vulnerability type"""
        recommendations = {
            'hardcoded_secrets': 'Use environment variables or secure credential management systems',
            'sql_injection': 'Use parameterized queries or prepared statements',
            'command_injection': 'Validate and sanitize input; avoid shell command construction',
            'pickle_usage': 'Use safer serialization formats like JSON, or implement custom unpickling restrictions',
            'eval_usage': 'Use ast.literal_eval() for safe evaluation, or avoid eval entirely',
            'weak_crypto': 'Use SHA-256 or stronger hash functions for security-critical applications',
            'debug_enabled': 'Disable debug mode in production environments',
            'insecure_random': 'Use secrets module or cryptography library for secure random generation'
        }

        return recommendations.get(vulnerability_type, 'Review and fix the security issue')

    def scan_codebase(self, target_path: str) -> Dict[str, Any]:
        """Scan entire codebase for security vulnerabilities"""
        import os

        all_vulnerabilities = []
        total_files = 0

        for root, dirs, files in os.walk(target_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', 'node_modules']]

            for file in files:
                if file.endswith('.py'):
                    total_files += 1
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        file_vulnerabilities = self.scan_file(file_path, content)
                        all_vulnerabilities.extend(file_vulnerabilities)

                    except Exception as e:
                        self.logger.error(f"Failed to scan file {file_path}: {e}")

        # Analyze results
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

        for vuln in all_vulnerabilities:
            severity = vuln['severity']
            if severity in severity_counts:
                severity_counts[severity] += 1

        # Determine overall security status
        if severity_counts['critical'] > 0 or severity_counts['high'] > 5:
            security_status = 'critical'
        elif severity_counts['high'] > 0 or severity_counts['medium'] > 10:
            security_status = 'high_risk'
        elif severity_counts['medium'] > 0 or severity_counts['low'] > 20:
            security_status = 'medium_risk'
        else:
            security_status = 'low_risk'

        return {
            'scan_summary': {
                'total_files_scanned': total_files,
                'total_vulnerabilities': len(all_vulnerabilities),
                'severity_breakdown': severity_counts,
                'security_status': security_status
            },
            'vulnerabilities': all_vulnerabilities,
            'recommendations': self.generate_security_recommendations(all_vulnerabilities)
        }

    def generate_security_recommendations(self, vulnerabilities: List[Dict[str, Any]]) -> List[str]:
        """Generate security improvement recommendations"""
        recommendations = []

        # Group vulnerabilities by type
        vuln_types = {}
        for vuln in vulnerabilities:
            vuln_type = vuln['vulnerability_type']
            vuln_types[vuln_type] = vuln_types.get(vuln_type, 0) + 1

        # Generate recommendations based on common issues
        if 'hardcoded_secrets' in vuln_types:
            recommendations.append("Implement secure credential management - use environment variables or secret management systems")

        if 'sql_injection' in vuln_types:
            recommendations.append("Implement parameterized queries for all database operations")

        if 'eval_usage' in vuln_types:
            recommendations.append("Remove all uses of eval() and replace with safe alternatives")

        if vuln_types.get('high', 0) > 0:
            recommendations.append("Address all high-severity security issues before deployment")

        if not recommendations:
            recommendations.append("Security scanning completed - no major issues found")

        return recommendations

class PerformanceValidator:
    """Performance validation and benchmarking system"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize performance validator"""
        self.config = config
        self.logger = logging.getLogger('PerformanceValidator')
        self.baselines = config.get('performance_baselines', {})

    def validate_performance(self, component_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component performance against baselines"""
        validation_result = {
            'component': component_name,
            'passed': True,
            'failed_checks': [],
            'warnings': [],
            'performance_score': 0.0
        }

        if component_name not in self.baselines:
            validation_result['warnings'].append(f"No baseline defined for {component_name}")
            return validation_result

        baseline = self.baselines[component_name]

        # Check response time
        if 'response_time' in metrics and 'max_response_time' in baseline:
            actual_time = metrics['response_time']
            max_time = baseline['max_response_time']

            if actual_time > max_time:
                validation_result['passed'] = False
                validation_result['failed_checks'].append({
                    'check': 'response_time',
                    'actual': actual_time,
                    'threshold': max_time,
                    'message': f"Response time {actual_time:.3f}s exceeds threshold {max_time:.3f}s"
                })

        # Check throughput
        if 'throughput' in metrics and 'min_throughput' in baseline:
            actual_throughput = metrics['throughput']
            min_throughput = baseline['min_throughput']

            if actual_throughput < min_throughput:
                validation_result['passed'] = False
                validation_result['failed_checks'].append({
                    'check': 'throughput',
                    'actual': actual_throughput,
                    'threshold': min_throughput,
                    'message': f"Throughput {actual_throughput:.1f} req/s below minimum {min_throughput:.1f} req/s"
                })

        # Check memory usage
        if 'memory_usage' in metrics and 'max_memory_usage' in baseline:
            actual_memory = metrics['memory_usage']
            max_memory = baseline['max_memory_usage']

            if actual_memory > max_memory:
                validation_result['passed'] = False
                validation_result['failed_checks'].append({
                    'check': 'memory_usage',
                    'actual': actual_memory,
                    'threshold': max_memory,
                    'message': f"Memory usage {actual_memory:.1f}MB exceeds limit {max_memory:.1f}MB"
                })

        # Calculate performance score
        validation_result['performance_score'] = self.calculate_performance_score(
            metrics, baseline, validation_result['failed_checks']
        )

        return validation_result

    def calculate_performance_score(self, metrics: Dict[str, Any],
                                  baseline: Dict[str, Any],
                                  failed_checks: List[Dict[str, Any]]) -> float:
        """Calculate overall performance score"""
        score = 100.0

        # Deduct points for failed checks
        score -= len(failed_checks) * 20

        # Adjust based on performance ratios
        if 'response_time' in metrics and 'max_response_time' in baseline:
            ratio = metrics['response_time'] / baseline['max_response_time']
            if ratio > 1:
                score -= min(30, (ratio - 1) * 50)

        if 'throughput' in metrics and 'min_throughput' in baseline:
            ratio = metrics['throughput'] / baseline['min_throughput']
            if ratio < 1:
                score -= min(30, (1 - ratio) * 50)

        return max(0.0, score)

    def run_performance_benchmark(self, component_name: str, test_function: callable,
                                 iterations: int = 100) -> Dict[str, Any]:
        """Run performance benchmark for component"""
        import time
        import psutil
        import os

        self.logger.info(f"Running performance benchmark for {component_name}")

        process = psutil.Process(os.getpid())
        results = {
            'component': component_name,
            'iterations': iterations,
            'response_times': [],
            'memory_usage': [],
            'cpu_usage': []
        }

        for i in range(iterations):
            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Execute function
            start_time = time.perf_counter()
            result = test_function()
            end_time = time.perf_counter()

            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB

            # Measure CPU
            cpu_percent = process.cpu_percent()

            response_time = end_time - start_time

            results['response_times'].append(response_time)
            results['memory_usage'].append(memory_after - memory_before)
            results['cpu_usage'].append(cpu_percent)

            # Small delay between iterations
            time.sleep(0.01)

        # Calculate statistics
        import statistics
        results['statistics'] = {
            'avg_response_time': statistics.mean(results['response_times']),
            'min_response_time': min(results['response_times']),
            'max_response_time': max(results['response_times']),
            'p95_response_time': statistics.quantiles(results['response_times'], n=20)[18],
            'avg_memory_delta': statistics.mean(results['memory_usage']),
            'avg_cpu_percent': statistics.mean(results['cpu_usage'])
        }

        return results

class ComplianceValidator:
    """Regulatory and standards compliance validator"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize compliance validator"""
        self.config = config
        self.logger = logging.getLogger('ComplianceValidator')

        # Compliance frameworks to check
        self.compliance_frameworks = {
            'gdpr': self.check_gdpr_compliance,
            'hipaa': self.check_hipaa_compliance,
            'soc2': self.check_soc2_compliance,
            'iso27001': self.check_iso27001_compliance
        }

    def validate_compliance(self, target_path: str, frameworks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate compliance with specified frameworks"""
        if frameworks is None:
            frameworks = list(self.compliance_frameworks.keys())

        compliance_results = {
            'target_path': target_path,
            'frameworks_checked': frameworks,
            'overall_compliant': True,
            'framework_results': {}
        }

        for framework in frameworks:
            if framework in self.compliance_frameworks:
                result = self.compliance_frameworks[framework](target_path)
                compliance_results['framework_results'][framework] = result

                if not result.get('compliant', True):
                    compliance_results['overall_compliant'] = False

        return compliance_results

    def check_gdpr_compliance(self, target_path: str) -> Dict[str, Any]:
        """Check GDPR compliance"""
        # Simplified GDPR compliance check
        issues = []

        # This would check for:
        # - Data collection consent
        # - Right to erasure implementation
        # - Data minimization
        # - Privacy by design

        return {
            'framework': 'GDPR',
            'compliant': len(issues) == 0,
            'issues': issues,
            'recommendations': ['Implement data subject access controls', 'Add data retention policies']
        }

    def check_hipaa_compliance(self, target_path: str) -> Dict[str, Any]:
        """Check HIPAA compliance for health data"""
        issues = []

        return {
            'framework': 'HIPAA',
            'compliant': len(issues) == 0,
            'issues': issues,
            'recommendations': ['Implement PHI encryption', 'Add audit logging for health data access']
        }

    def check_soc2_compliance(self, target_path: str) -> Dict[str, Any]:
        """Check SOC 2 compliance"""
        issues = []

        return {
            'framework': 'SOC 2',
            'compliant': len(issues) == 0,
            'issues': issues,
            'recommendations': ['Implement access controls', 'Add security monitoring']
        }

    def check_iso27001_compliance(self, target_path: str) -> Dict[str, Any]:
        """Check ISO 27001 compliance"""
        issues = []

        return {
            'framework': 'ISO 27001',
            'compliant': len(issues) == 0,
            'issues': issues,
            'recommendations': ['Implement information security management system', 'Conduct risk assessments']
        }

class TestQualityAssuranceFramework:
    """Tests for quality assurance framework"""

    @pytest.fixture
    def quality_validator(self):
        """Create quality validator for testing"""
        config = {
            'max_complexity': 10,
            'max_line_length': 88,
            'min_test_coverage': 80
        }
        return CodeQualityValidator(config)

    def test_code_quality_validation(self, quality_validator):
        """Test code quality validation"""
        # Create a test file
        test_code = '''
def bad_function(x,y,z,a,b,c,d,e,f,g):  # Too many parameters
    result = x+y+z+a+b+c+d+e+f+g  # Complex expression
    return result

class badclass:  # Wrong naming
    pass

def another_func():  # Missing docstring
    pass
'''

        with open('/tmp/test_quality.py', 'w') as f:
            f.write(test_code)

        report = quality_validator.validate_codebase('/tmp')

        # Should find several issues
        assert len(report.issues_found) > 0
        assert report.quality_score < 100

        # Clean up
        import os
        os.remove('/tmp/test_quality.py')

    def test_security_scanning(self):
        """Test security vulnerability scanning"""
        scanner = SecurityScanner({})

        # Test code with vulnerabilities
        test_code = '''
password = "secret123"  # Hardcoded password
query = "SELECT * FROM users WHERE id = " + user_id  # SQL injection
result = eval(user_input)  # Eval usage
'''

        vulnerabilities = scanner.scan_file('test.py', test_code)

        # Should detect multiple vulnerabilities
        assert len(vulnerabilities) >= 3

        vuln_types = [v['vulnerability_type'] for v in vulnerabilities]
        assert 'hardcoded_secrets' in vuln_types
        assert 'sql_injection' in vuln_types
        assert 'eval_usage' in vuln_types

    def test_performance_validation(self):
        """Test performance validation"""
        validator = PerformanceValidator({
            'performance_baselines': {
                'test_component': {
                    'max_response_time': 1.0,
                    'min_throughput': 10.0
                }
            }
        })

        # Test passing metrics
        good_metrics = {
            'response_time': 0.5,
            'throughput': 15.0
        }

        result = validator.validate_performance('test_component', good_metrics)
        assert result['passed'] == True
        assert result['performance_score'] > 80

        # Test failing metrics
        bad_metrics = {
            'response_time': 2.0,  # Too slow
            'throughput': 5.0     # Too low
        }

        result = validator.validate_performance('test_component', bad_metrics)
        assert result['passed'] == False
        assert len(result['failed_checks']) > 0
```

---

**"Active Inference for, with, by Generative AI"** - Establishing comprehensive quality assurance frameworks that ensure platform reliability, security, performance, and compliance through automated validation, continuous monitoring, and rigorous testing standards.
