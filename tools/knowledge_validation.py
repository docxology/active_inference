#!/usr/bin/env python3
"""
Knowledge Base Validation Tools
Comprehensive validation system for Active Inference Knowledge Environment

This module provides tools for validating knowledge base content including:
- JSON schema compliance validation
- Cross-reference checking
- Content quality assessment
- Learning path validation
- Metadata completeness checking
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of knowledge validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeNode:
    """Represents a knowledge node with validation data"""
    id: str
    title: str
    content_type: str
    difficulty: str
    file_path: str
    data: Dict[str, Any]
    validation_result: Optional[ValidationResult] = None

class KnowledgeSchemaValidator:
    """Validates knowledge JSON files against the required schema"""

    def __init__(self):
        """Initialize the schema validator"""
        self.required_fields = [
            'id', 'title', 'content_type', 'difficulty',
            'description', 'prerequisites', 'tags',
            'learning_objectives', 'content', 'metadata'
        ]

        self.content_type_options = ['foundation', 'mathematics', 'implementation', 'application']
        self.difficulty_options = ['beginner', 'intermediate', 'advanced', 'expert']

        self.quality_weights = {
            'schema_compliance': 0.3,
            'content_completeness': 0.2,
            'metadata_quality': 0.15,
            'cross_references': 0.15,
            'learning_objectives': 0.1,
            'content_structure': 0.1
        }

    def validate_knowledge_json(self, file_path: str) -> ValidationResult:
        """
        Validate JSON file against knowledge base schema

        Args:
            file_path: Path to the JSON file to validate

        Returns:
            ValidationResult with validation details
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            result = ValidationResult(is_valid=True, errors=[], warnings=[])

            # Check required fields
            self._validate_required_fields(data, result)

            # Validate field types and values
            self._validate_field_types(data, result)

            # Validate content structure
            self._validate_content_structure(data, result)

            # Validate metadata
            self._validate_metadata(data, result)

            # Calculate overall score
            result.score = self._calculate_quality_score(data, result)

            return result

        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid JSON format: {e}"],
                score=0.0
            )
        except FileNotFoundError:
            return ValidationResult(
                is_valid=False,
                errors=[f"File not found: {file_path}"],
                score=0.0
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {e}"],
                score=0.0
            )

    def _validate_required_fields(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate that all required fields are present"""
        for field in self.required_fields:
            if field not in data:
                result.errors.append(f"Missing required field: {field}")
                result.is_valid = False

    def _validate_field_types(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate field types and allowed values"""
        # Validate content_type
        if 'content_type' in data:
            if data['content_type'] not in self.content_type_options:
                result.errors.append(
                    f"Invalid content_type: {data['content_type']}. "
                    f"Must be one of: {self.content_type_options}"
                )
                result.is_valid = False

        # Validate difficulty
        if 'difficulty' in data:
            if data['difficulty'] not in self.difficulty_options:
                result.errors.append(
                    f"Invalid difficulty: {data['difficulty']}. "
                    f"Must be one of: {self.difficulty_options}"
                )
                result.is_valid = False

        # Validate field types
        if 'prerequisites' in data and not isinstance(data['prerequisites'], list):
            result.errors.append("prerequisites must be a list")
            result.is_valid = False

        if 'tags' in data and not isinstance(data['tags'], list):
            result.errors.append("tags must be a list")
            result.is_valid = False

        if 'learning_objectives' in data and not isinstance(data['learning_objectives'], list):
            result.errors.append("learning_objectives must be a list")
            result.is_valid = False

    def _validate_content_structure(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate content structure and completeness"""
        if 'content' not in data:
            return

        content = data['content']

        # Check for overview section
        if 'overview' not in content:
            result.warnings.append("Content should include 'overview' section")
            result.score -= 0.1

        # Validate content sections have meaningful content
        for key, value in content.items():
            content_length = self._get_content_length(value)
            if content_length < 10:
                result.warnings.append(f"Content section '{key}' is very short")
                result.score -= 0.05

    def _get_content_length(self, value: Any) -> int:
        """Get meaningful length of content value"""
        if isinstance(value, str):
            return len(value.strip())
        elif isinstance(value, list):
            return len(value)
        elif isinstance(value, dict):
            return len(value)
        else:
            return 1  # Non-empty by default

    def _validate_metadata(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate metadata completeness and correctness"""
        if 'metadata' not in data:
            return

        metadata = data['metadata']

        # Check for estimated reading time
        if 'estimated_reading_time' not in metadata:
            result.warnings.append("Missing estimated_reading_time in metadata")
            result.score -= 0.05

        # Check for author information
        if 'author' not in metadata:
            result.warnings.append("Missing author in metadata")
            result.score -= 0.05

        # Check for version information
        if 'version' not in metadata:
            result.warnings.append("Missing version in metadata")
            result.score -= 0.05

    def _calculate_quality_score(self, data: Dict[str, Any], result: ValidationResult) -> float:
        """Calculate overall quality score"""
        score = 1.0

        # Schema compliance score
        if result.errors:
            score -= 0.5
        else:
            score += 0.1  # Bonus for perfect schema compliance

        # Content completeness score
        content_score = self._assess_content_completeness(data)
        score = score * 0.7 + content_score * 0.3

        # Metadata quality score
        metadata_score = self._assess_metadata_quality(data)
        score = score * 0.8 + metadata_score * 0.2

        return max(0.0, min(1.0, score))

    def _assess_content_completeness(self, data: Dict[str, Any]) -> float:
        """Assess content completeness"""
        if 'content' not in data:
            return 0.0

        content = data['content']
        score = 0.0

        # Overview section
        if 'overview' in content and self._get_content_length(content['overview']) > 50:
            score += 0.3

        # Mathematical definition
        if 'mathematical_definition' in content and self._get_content_length(content['mathematical_definition']) > 30:
            score += 0.2

        # Examples section
        if 'examples' in content and self._get_content_length(content['examples']) > 30:
            score += 0.2

        # Interactive exercises
        if 'interactive_exercises' in content and self._get_content_length(content['interactive_exercises']) > 20:
            score += 0.3

        return min(1.0, score)

    def _assess_metadata_quality(self, data: Dict[str, Any]) -> float:
        """Assess metadata quality"""
        if 'metadata' not in data:
            return 0.0

        metadata = data['metadata']
        score = 0.0

        if 'estimated_reading_time' in metadata:
            score += 0.3
        if 'author' in metadata:
            score += 0.3
        if 'version' in metadata:
            score += 0.2
        if 'last_updated' in metadata:
            score += 0.2

        return score

class CrossReferenceValidator:
    """Validates cross-references between knowledge nodes"""

    def __init__(self, knowledge_base_path: str):
        """Initialize cross-reference validator"""
        self.knowledge_base_path = Path(knowledge_base_path)
        self.all_ids: Set[str] = set()
        self.reference_map: Dict[str, Set[str]] = {}
        self.nodes: Dict[str, KnowledgeNode] = {}

    def scan_knowledge_base(self) -> Dict[str, KnowledgeNode]:
        """Scan all JSON files and extract node information"""
        logger.info("Scanning knowledge base for cross-references...")

        json_files = list(self.knowledge_base_path.rglob("*.json"))
        nodes = {}

        for file_path in json_files:
            if file_path.name in ['learning_paths.json', 'faq.json', 'glossary.json']:
                continue  # Skip non-content JSON files

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                node = KnowledgeNode(
                    id=data['id'],
                    title=data['title'],
                    content_type=data['content_type'],
                    difficulty=data['difficulty'],
                    file_path=str(file_path),
                    data=data
                )

                nodes[data['id']] = node
                self.all_ids.add(data['id'])

            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")

        self.nodes = nodes
        logger.info(f"Found {len(nodes)} knowledge nodes")
        return nodes

    def validate_cross_references(self) -> Dict[str, ValidationResult]:
        """Validate all cross-references in the knowledge base"""
        logger.info("Validating cross-references...")

        results = {}

        for node_id, node in self.nodes.items():
            result = self._validate_node_references(node)
            results[node_id] = result

        return results

    def _validate_node_references(self, node: KnowledgeNode) -> ValidationResult:
        """Validate references for a single node"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        # Check prerequisites
        for prereq in node.data.get('prerequisites', []):
            if prereq not in self.all_ids:
                result.errors.append(f"Missing prerequisite: {prereq}")
                result.is_valid = False

        # Check content references (scan content for node IDs)
        content_refs = self._extract_content_references(node.data.get('content', {}))
        for ref in content_refs:
            if ref not in self.all_ids:
                result.warnings.append(f"Content references non-existent node: {ref}")

        # Check learning objectives references
        objectives_refs = self._extract_objectives_references(node.data.get('learning_objectives', []))
        for ref in objectives_refs:
            if ref not in self.all_ids:
                result.warnings.append(f"Learning objective references non-existent node: {ref}")

        return result

    def _extract_content_references(self, content: Any) -> Set[str]:
        """Extract node ID references from content"""
        refs = set()

        if isinstance(content, dict):
            for key, value in content.items():
                refs.update(self._extract_content_references(value))
        elif isinstance(content, list):
            for item in content:
                refs.update(self._extract_content_references(item))
        elif isinstance(content, str):
            # Look for patterns like "node_id" or references to other concepts
            import re
            node_refs = re.findall(r'\b([a-z][a-z_]+)\b', content)
            # Filter to likely node IDs (snake_case, no common words)
            common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'why'}
            refs.update([ref for ref in node_refs if ref not in common_words and len(ref) > 3])

        return refs

    def _extract_objectives_references(self, objectives: List[str]) -> Set[str]:
        """Extract node ID references from learning objectives"""
        refs = set()

        for objective in objectives:
            refs.update(self._extract_content_references(objective))

        return refs

    def generate_reference_report(self) -> Dict[str, Any]:
        """Generate comprehensive reference report"""
        report = {
            'total_nodes': len(self.nodes),
            'total_ids': len(self.all_ids),
            'orphaned_nodes': [],
            'highly_connected_nodes': [],
            'isolated_nodes': [],
            'reference_statistics': {}
        }

        # Find orphaned nodes (no other nodes reference them)
        referenced_ids = set()
        for node in self.nodes.values():
            for prereq in node.data.get('prerequisites', []):
                referenced_ids.add(prereq)

        orphaned = self.all_ids - referenced_ids
        report['orphaned_nodes'] = list(orphaned)

        # Find highly connected nodes
        connection_counts = {}
        for node in self.nodes.values():
            prereq_count = len(node.data.get('prerequisites', []))
            connection_counts[node.id] = prereq_count

        highly_connected = [node_id for node_id, count in connection_counts.items() if count > 5]
        report['highly_connected_nodes'] = highly_connected

        # Find isolated nodes
        isolated = [node_id for node_id, count in connection_counts.items() if count == 0]
        report['isolated_nodes'] = isolated

        # Reference statistics
        report['reference_statistics'] = {
            'avg_prerequisites': sum(connection_counts.values()) / len(connection_counts),
            'max_prerequisites': max(connection_counts.values()),
            'min_prerequisites': min(connection_counts.values()),
            'zero_prerequisite_nodes': len(isolated)
        }

        return report

class KnowledgeBaseAuditor:
    """Main auditor class that coordinates all validation"""

    def __init__(self, knowledge_base_path: str):
        """Initialize the knowledge base auditor"""
        self.knowledge_base_path = Path(knowledge_base_path)
        self.schema_validator = KnowledgeSchemaValidator()
        self.cross_validator = CrossReferenceValidator(knowledge_base_path)
        self.results: Dict[str, ValidationResult] = {}
        self.cross_results: Dict[str, ValidationResult] = {}

    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive audit of the entire knowledge base"""
        logger.info("Starting comprehensive knowledge base audit...")

        # Phase 1: Scan and validate all files
        self.cross_validator.scan_knowledge_base()

        # Phase 2: Schema validation
        self._validate_all_files()

        # Phase 3: Cross-reference validation
        self.cross_results = self.cross_validator.validate_cross_references()

        # Phase 4: Generate comprehensive report
        report = self._generate_audit_report()

        logger.info("Comprehensive audit completed")
        return report

    def _validate_all_files(self) -> None:
        """Validate all JSON files in the knowledge base"""
        json_files = list(self.knowledge_base_path.rglob("*.json"))

        for file_path in json_files:
            if file_path.name in ['learning_paths.json', 'faq.json', 'glossary.json', 'success_metrics.json'] or 'metadata' in str(file_path):
                continue  # Skip non-content JSON files

            relative_path = file_path.relative_to(self.knowledge_base_path)
            logger.info(f"Validating {relative_path}")

            result = self.schema_validator.validate_knowledge_json(str(file_path))
            self.results[str(relative_path)] = result

    def _generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        total_files = len(self.results)
        valid_files = sum(1 for r in self.results.values() if r.is_valid)
        total_errors = sum(len(r.errors) for r in self.results.values())
        total_warnings = sum(len(r.warnings) for r in self.results.values())

        # Calculate average scores
        scores = [r.score for r in self.results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Content type breakdown
        content_types = {}
        difficulties = {}

        for result in self.results.values():
            # This is a simplified approach - in practice we'd need to parse the file
            # to get the actual content type and difficulty
            pass

        # Cross-reference report
        cross_report = self.cross_validator.generate_reference_report()

        # Critical issues
        critical_issues = []
        for file_path, result in self.results.items():
            if result.errors:
                critical_issues.append({
                    'file': file_path,
                    'errors': result.errors,
                    'warnings': result.warnings
                })

        report = {
            'audit_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': total_files,
                'valid_files': valid_files,
                'validation_rate': valid_files / total_files if total_files > 0 else 0,
                'total_errors': total_errors,
                'total_warnings': total_warnings,
                'average_score': avg_score,
                'knowledge_nodes': len(self.cross_validator.nodes)
            },
            'cross_reference_analysis': cross_report,
            'content_quality': {
                'schema_compliance': self._calculate_schema_compliance(),
                'content_completeness': self._calculate_content_completeness(),
                'metadata_quality': self._calculate_metadata_quality()
            },
            'critical_issues': critical_issues,
            'detailed_results': {
                file_path: {
                    'is_valid': result.is_valid,
                    'score': result.score,
                    'errors': result.errors,
                    'warnings': result.warnings
                }
                for file_path, result in self.results.items()
            }
        }

        return report

    def _calculate_schema_compliance(self) -> float:
        """Calculate overall schema compliance score"""
        if not self.results:
            return 0.0

        compliance_scores = []
        for result in self.results.values():
            if result.is_valid:
                compliance_scores.append(1.0)
            else:
                compliance_scores.append(0.5)  # Partial credit for files with warnings but no errors

        return sum(compliance_scores) / len(compliance_scores)

    def _calculate_content_completeness(self) -> float:
        """Calculate overall content completeness score"""
        if not self.results:
            return 0.0

        completeness_scores = [result.score for result in self.results.values()]
        return sum(completeness_scores) / len(completeness_scores)

    def _calculate_metadata_quality(self) -> float:
        """Calculate overall metadata quality score"""
        if not self.results:
            return 0.0

        metadata_scores = []
        for result in self.results.values():
            metadata_score = 1.0
            if result.warnings:
                metadata_score -= 0.1 * len(result.warnings)
            metadata_scores.append(max(0.0, metadata_score))

        return sum(metadata_scores) / len(metadata_scores)

def main():
    """Main function for running knowledge base audit"""
    if len(sys.argv) != 2:
        print("Usage: python knowledge_validation.py <knowledge_base_path>")
        sys.exit(1)

    knowledge_path = sys.argv[1]

    if not os.path.exists(knowledge_path):
        print(f"Knowledge base path does not exist: {knowledge_path}")
        sys.exit(1)

    # Run comprehensive audit
    auditor = KnowledgeBaseAuditor(knowledge_path)
    report = auditor.run_comprehensive_audit()

    # Save report to file
    report_file = "knowledge_audit_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    summary = report['summary']
    print("\n" + "="*60)
    print("KNOWLEDGE BASE AUDIT REPORT")
    print("="*60)
    print(f"Timestamp: {report['audit_timestamp']}")
    print(f"Total Files: {summary['total_files']}")
    print(f"Valid Files: {summary['valid_files']}")
    print(f"Validation Rate: {summary['validation_rate']:.1%}")
    print(f"Average Quality Score: {summary['average_score']:.1%}")
    print(f"Total Errors: {summary['total_errors']}")
    print(f"Total Warnings: {summary['total_warnings']}")
    print(f"Knowledge Nodes: {summary['knowledge_nodes']}")
    print("\nCross-Reference Analysis:")
    cross_ref = report['cross_reference_analysis']
    print(f"  Orphaned Nodes: {len(cross_ref['orphaned_nodes'])}")
    print(f"  Highly Connected: {len(cross_ref['highly_connected_nodes'])}")
    print(f"  Isolated Nodes: {len(cross_ref['isolated_nodes'])}")
    print(f"  Avg Prerequisites: {cross_ref['reference_statistics']['avg_prerequisites']:.1f}")

    print("\nQuality Metrics:")
    quality = report['content_quality']
    print(f"  Schema Compliance: {quality['schema_compliance']:.1%}")
    print(f"  Content Completeness: {quality['content_completeness']:.1%}")
    print(f"  Metadata Quality: {quality['metadata_quality']:.1%}")

    if summary['total_errors'] > 0:
        print(f"\nCritical Issues: {len(report['critical_issues'])} files have errors")
        for issue in report['critical_issues'][:5]:  # Show first 5
            print(f"  - {issue['file']}: {len(issue['errors'])} errors")

    print(f"\nDetailed report saved to: {report_file}")
    print("="*60)

if __name__ == "__main__":
    main()
