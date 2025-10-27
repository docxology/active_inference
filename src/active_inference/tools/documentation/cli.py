"""
Documentation CLI Tool

Command-line interface for documentation generation, analysis, and repository review.
Provides comprehensive tools for maintaining high-quality documentation and codebase.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from .generator import DocumentationGenerator
from .analyzer import DocumentationAnalyzer
from .reviewer import RepositoryReviewer
from .validator import DocumentationValidator

logger = logging.getLogger(__name__)


class DocumentationCLI:
    """
    Command-line interface for documentation tools.

    Provides commands for generating documentation, analyzing quality,
    validating standards, and reviewing repository health.
    """

    def __init__(self, repo_path: Optional[Path] = None):
        """
        Initialize documentation CLI

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = repo_path or Path.cwd()
        self.logger = logging.getLogger(__name__)

        # Initialize tools
        self.generator = DocumentationGenerator(self.repo_path / "docs")
        self.analyzer = DocumentationAnalyzer([self.repo_path / "src"])
        self.reviewer = RepositoryReviewer(self.repo_path)
        self.validator = DocumentationValidator([self.repo_path / "src"])

        self.logger.info(f"Documentation CLI initialized for {self.repo_path}")

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser"""
        parser = argparse.ArgumentParser(
            description="Active Inference Documentation Tools",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Generate all documentation
  ai-docs generate all

  # Analyze documentation quality
  ai-docs analyze quality

  # Review repository health
  ai-docs review repository

  # Validate documentation standards
  ai-docs validate structure

  # Generate API documentation only
  ai-docs generate api --output docs/api

  # Check documentation coverage
  ai-docs analyze coverage --format json
            """
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Generate command
        generate_parser = subparsers.add_parser('generate', help='Generate documentation')
        generate_parser.add_argument('type', choices=['all', 'api', 'knowledge', 'research', 'visualization', 'applications', 'platform'],
                                   help='Type of documentation to generate')
        generate_parser.add_argument('--output', '-o', type=Path, default=Path('docs'),
                                   help='Output directory')
        generate_parser.add_argument('--source', '-s', type=Path, default=Path('src'),
                                   help='Source directory')
        generate_parser.add_argument('--force', '-f', action='store_true',
                                   help='Force regeneration even if files exist')

        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze documentation and code')
        analyze_parser.add_argument('analysis_type', choices=['quality', 'coverage', 'structure', 'complexity', 'dependencies'],
                                  help='Type of analysis to perform')
        analyze_parser.add_argument('--format', '-f', choices=['text', 'json', 'html'], default='text',
                                  help='Output format')
        analyze_parser.add_argument('--output', '-o', type=Path,
                                  help='Output file path')
        analyze_parser.add_argument('--threshold', '-t', type=float, default=80.0,
                                  help='Quality threshold percentage')

        # Review command
        review_parser = subparsers.add_parser('review', help='Review repository health')
        review_parser.add_argument('review_type', choices=['repository', 'code', 'documentation', 'architecture', 'testing'],
                                 help='Type of review to perform')
        review_parser.add_argument('--format', '-f', choices=['text', 'json', 'html'], default='text',
                                 help='Output format')
        review_parser.add_argument('--output', '-o', type=Path,
                                 help='Output file path')
        review_parser.add_argument('--include-tests', action='store_true',
                                 help='Include test analysis')

        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate documentation standards')
        validate_parser.add_argument('validation_type', choices=['structure', 'content', 'style', 'completeness', 'all'],
                                   help='Type of validation to perform')
        validate_parser.add_argument('--format', '-f', choices=['text', 'json', 'html'], default='text',
                                   help='Output format')
        validate_parser.add_argument('--output', '-o', type=Path,
                                   help='Output file path')
        validate_parser.add_argument('--strict', action='store_true',
                                   help='Treat warnings as errors')

        # Report command
        report_parser = subparsers.add_parser('report', help='Generate comprehensive reports')
        report_parser.add_argument('report_type', choices=['summary', 'detailed', 'health', 'quality', 'all'],
                                 help='Type of report to generate')
        report_parser.add_argument('--format', '-f', choices=['text', 'json', 'html', 'pdf'], default='text',
                                 help='Output format')
        report_parser.add_argument('--output', '-o', type=Path,
                                 help='Output file path')
        report_parser.add_argument('--include-charts', action='store_true',
                                 help='Include charts and visualizations')

        return parser

    def run(self, args: Optional[list] = None) -> int:
        """
        Run the CLI with given arguments

        Args:
            args: Command line arguments

        Returns:
            Exit code
        """
        parser = self.create_parser()

        if args is None:
            args = sys.argv[1:]

        parsed_args = parser.parse_args(args)

        if not parsed_args.command:
            parser.print_help()
            return 0

        try:
            return self._dispatch_command(parsed_args)
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return 1

    def _dispatch_command(self, args) -> int:
        """Dispatch command to appropriate handler"""
        if args.command == 'generate':
            return self._handle_generate_command(args)
        elif args.command == 'analyze':
            return self._handle_analyze_command(args)
        elif args.command == 'review':
            return self._handle_review_command(args)
        elif args.command == 'validate':
            return self._handle_validate_command(args)
        elif args.command == 'report':
            return self._handle_report_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1

    def _handle_generate_command(self, args) -> int:
        """Handle documentation generation commands"""
        self.logger.info(f"Generating {args.type} documentation")

        try:
            if args.type == 'all':
                self.generator.generate_all_docs(args.source, self.repo_path / "knowledge")
                print("✅ All documentation generated successfully")
            elif args.type == 'api':
                self.generator.generate_api_docs(args.source)
                print("✅ API documentation generated successfully")
            elif args.type == 'knowledge':
                # Initialize knowledge repository
                from ...knowledge.repository import KnowledgeRepository, KnowledgeRepositoryConfig
                repo_config = KnowledgeRepositoryConfig(
                    root_path=self.repo_path / "knowledge",
                    auto_index=True
                )
                repo = KnowledgeRepository(repo_config)
                self.generator.generate_learning_paths(repo)
                self.generator.generate_concept_maps(repo)
                self.generator.generate_statistics(repo)
                print("✅ Knowledge documentation generated successfully")
            else:
                print(f"⚠️  Documentation type '{args.type}' not yet implemented")
                return 0

            return 0

        except Exception as e:
            print(f"❌ Documentation generation failed: {e}")
            return 1

    def _handle_analyze_command(self, args) -> int:
        """Handle documentation analysis commands"""
        self.logger.info(f"Analyzing {args.analysis_type}")

        try:
            if args.analysis_type == 'quality':
                quality = self.analyzer.analyze_documentation()
                self._display_quality_analysis(quality, args.format, args.output)
            elif args.analysis_type == 'coverage':
                quality = self.analyzer.analyze_documentation()
                self._display_coverage_analysis(quality, args.format, args.output)
            else:
                print(f"⚠️  Analysis type '{args.analysis_type}' not yet implemented")
                return 0

            return 0

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return 1

    def _handle_review_command(self, args) -> int:
        """Handle repository review commands"""
        self.logger.info(f"Reviewing {args.review_type}")

        try:
            if args.review_type == 'repository':
                report = self.reviewer.generate_report()
                self._display_report(report, args.format, args.output)
                print("✅ Repository review completed")
            else:
                print(f"⚠️  Review type '{args.review_type}' not yet implemented")
                return 0

            return 0

        except Exception as e:
            print(f"❌ Review failed: {e}")
            return 1

    def _handle_validate_command(self, args) -> int:
        """Handle validation commands"""
        self.logger.info(f"Validating {args.validation_type}")

        try:
            if args.validation_type == 'structure':
                validation = self.validator.validate_documentation()
                self._display_validation_results(validation, args.format, args.output)
                print("✅ Validation completed")
            else:
                print(f"⚠️  Validation type '{args.validation_type}' not yet implemented")
                return 0

            return 0

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return 1

    def _handle_report_command(self, args) -> int:
        """Handle report generation commands"""
        self.logger.info(f"Generating {args.report_type} report")

        try:
            if args.report_type == 'summary':
                # Generate summary report combining all analyses
                summary = self._generate_summary_report()
                self._display_report(summary, args.format, args.output)
                print("✅ Summary report generated")
            else:
                print(f"⚠️  Report type '{args.report_type}' not yet implemented")
                return 0

            return 0

        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            return 1

    def _display_quality_analysis(self, quality, format_type: str, output_path: Optional[Path]) -> None:
        """Display documentation quality analysis"""
        if format_type == 'json':
            import json
            content = json.dumps({
                'coverage_percentage': quality.coverage_percentage,
                'total_elements': quality.total_elements,
                'documented_elements': quality.documented_elements,
                'average_quality': quality.average_quality
            }, indent=2)
        else:
            content = f"""
📊 Documentation Quality Analysis
{'=' * 40}

Coverage: {quality.coverage_percentage:.1f}%
Total Elements: {quality.total_elements}
Documented Elements: {quality.documented_elements}
Average Quality Score: {quality.average_quality:.2f}

Quality Distribution:
- Excellent: {len(quality.excellent_docs)} elements
- Poor: {len(quality.poor_quality_docs)} elements

Missing Documentation: {len(quality.missing_docs)} elements
"""

        self._output_content(content, output_path)

    def _display_coverage_analysis(self, quality, format_type: str, output_path: Optional[Path]) -> None:
        """Display coverage analysis"""
        if format_type == 'json':
            import json
            content = json.dumps({
                'coverage_percentage': quality.coverage_percentage,
                'total_elements': quality.total_elements,
                'documented_elements': quality.documented_elements,
                'missing_docs': quality.missing_docs,
                'poor_quality_docs': quality.poor_quality_docs
            }, indent=2)
        else:
            content = f"""
📈 Documentation Coverage Analysis
{'=' * 40}

Overall Coverage: {quality.coverage_percentage:.1f}%

Elements:
- Total: {quality.total_elements}
- Documented: {quality.documented_elements}
- Missing: {len(quality.missing_docs)}

Missing Documentation:
{chr(10).join(f'- {doc}' for doc in quality.missing_docs[:10])}

Poor Quality Documentation:
{chr(10).join(f'- {doc}' for doc in quality.poor_quality_docs[:10])}
"""

        self._output_content(content, output_path)

    def _display_validation_results(self, validation, format_type: str, output_path: Optional[Path]) -> None:
        """Display validation results"""
        if format_type == 'json':
            import json
            content = json.dumps(validation, indent=2)
        else:
            content = f"""
🔍 Documentation Validation Results
{'=' * 40}

Summary:
- Total Issues: {validation['total_issues']}
- Errors: {validation['errors']}
- Warnings: {validation['warnings']}
- Info: {validation['info']}
- Validation Passed: {'✅ Yes' if validation['validation_passed'] else '❌ No'}

Issues by Category:
{chr(10).join(f'- {category}: {len(results)} issues' for category, results in validation['results_by_category'].items() if results)}
"""

        self._output_content(content, output_path)

    def _display_report(self, report: str, format_type: str, output_path: Optional[Path]) -> None:
        """Display report content"""
        if format_type == 'json':
            import json
            # Try to parse as JSON for structured output
            try:
                report_data = json.loads(report)
                content = json.dumps(report_data, indent=2)
            except:
                content = report
        else:
            content = report

        self._output_content(content, output_path)

    def _generate_summary_report(self) -> str:
        """Generate comprehensive summary report"""
        # Combine multiple analyses
        quality = self.analyzer.analyze_documentation()
        validation = self.validator.validate_documentation()

        return f"""
📋 Active Inference Documentation Summary Report
{'=' * 50}

📊 Documentation Quality
Coverage: {quality.coverage_percentage:.1f}%
Quality Score: {quality.average_quality:.2f}

🔍 Validation Status
Errors: {validation['errors']}
Warnings: {validation['warnings']}
Validation Passed: {'✅ Yes' if validation['validation_passed'] else '❌ No'}

📈 Repository Health
Total Issues: {validation['total_issues']}
Documentation Score: {quality.coverage_percentage:.1f}%

🔧 Recommendations
{chr(10).join(f'- {rec}' for rec in self._get_recommendations(quality, validation))}
"""

    def _get_recommendations(self, quality, validation) -> list:
        """Generate recommendations based on analysis"""
        recommendations = []

        if quality.coverage_percentage < 80:
            recommendations.append(f"Improve documentation coverage from {quality.coverage_percentage:.1f}% to 80%+")

        if validation['errors'] > 0:
            recommendations.append(f"Fix {validation['errors']} documentation errors")

        if validation['warnings'] > 0:
            recommendations.append(f"Address {validation['warnings']} documentation warnings")

        if not recommendations:
            recommendations.append("Documentation quality is excellent!")

        return recommendations

    def _output_content(self, content: str, output_path: Optional[Path]) -> None:
        """Output content to console or file"""
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📄 Report saved to: {output_path}")
        else:
            print(content)


def main():
    """Main entry point for documentation CLI"""
    cli = DocumentationCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
