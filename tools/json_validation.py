#!/usr/bin/env python3
"""
JSON Schema Validation Tool for Active Inference Knowledge Base

This tool validates all JSON files in the knowledge base against the established schema
and identifies missing content, broken references, and quality issues.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeBaseValidator:
    """Validator for Active Inference knowledge base JSON files"""

    def __init__(self, knowledge_base_path: str = "knowledge"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.all_ids: Set[str] = set()
        self.validation_errors: List[Dict[str, Any]] = []
        self.missing_references: List[Dict[str, Any]] = []
        self.schema_issues: List[Dict[str, Any]] = []

    def get_all_json_files(self) -> List[Path]:
        """Get all JSON files in the knowledge base"""
        json_files = []
        skip_files = ['learning_paths.json', 'faq.json', 'glossary.json', 'success_metrics.json']

        for json_file in self.knowledge_base_path.rglob("*.json"):
            if json_file.name in skip_files or 'metadata' in str(json_file):
                continue  # Skip non-knowledge-node JSON files
            json_files.append(json_file)
        return json_files

    def load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file safely"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {}

    def validate_schema(self, data: Dict[str, Any], file_path: Path) -> List[str]:
        """Validate JSON file against knowledge base schema"""
        errors = []

        # Debug logging for specific files
        # if 'differential_geometry' in str(file_path):
        #     logger.info(f"Debug: differential_geometry data keys: {list(data.keys())}")
        #     logger.info(f"Debug: learning_objectives in data: {'learning_objectives' in data}")
        #     if 'learning_objectives' in data:
        #         logger.info(f"Debug: learning_objectives value: {data['learning_objectives']}")

        # Required fields
        required_fields = [
            'id', 'title', 'content_type', 'difficulty',
            'description', 'prerequisites', 'tags',
            'learning_objectives', 'content', 'metadata'
        ]

        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Validate content_type
        if 'content_type' in data:
            valid_types = ['foundation', 'mathematics', 'implementation', 'application']
            if data['content_type'] not in valid_types:
                errors.append(f"Invalid content_type: {data['content_type']}")

        # Validate difficulty
        if 'difficulty' in data:
            valid_difficulties = ['beginner', 'intermediate', 'advanced', 'expert']
            if data['difficulty'] not in valid_difficulties:
                errors.append(f"Invalid difficulty: {data['difficulty']}")

        # Validate content structure
        if 'content' in data:
            content = data['content']
            if 'overview' not in content:
                errors.append("Content must include 'overview' section")

        # Validate metadata
        if 'metadata' in data:
            metadata = data['metadata']
            required_metadata = ['estimated_reading_time', 'difficulty_level', 'last_updated', 'version', 'author']
            for field in required_metadata:
                if field not in metadata:
                    errors.append(f"Missing metadata field: {field}")

        return errors

    def collect_all_ids(self, json_files: List[Path]) -> None:
        """Collect all IDs from JSON files"""
        for file_path in json_files:
            data = self.load_json_file(file_path)
            if data and 'id' in data:
                self.all_ids.add(data['id'])
                logger.info(f"Found ID: {data['id']} in {file_path}")

    def check_cross_references(self, json_files: List[Path]) -> None:
        """Check that all referenced concepts exist"""
        for file_path in json_files:
            data = self.load_json_file(file_path)
            if not data:
                continue

            file_id = data.get('id', 'unknown')

            # Check prerequisites
            for prereq in data.get('prerequisites', []):
                if prereq not in self.all_ids:
                    self.missing_references.append({
                        'file': str(file_path),
                        'id': file_id,
                        'missing_reference': prereq,
                        'type': 'prerequisite'
                    })
                    logger.warning(f"Missing prerequisite: {prereq} referenced by {file_id}")

            # Check related_concepts (if exists)
            for concept in data.get('related_concepts', []):
                if concept not in self.all_ids:
                    self.missing_references.append({
                        'file': str(file_path),
                        'id': file_id,
                        'missing_reference': concept,
                        'type': 'related_concept'
                    })
                    logger.warning(f"Missing related concept: {concept} referenced by {file_id}")

    def validate_all_files(self) -> Dict[str, Any]:
        """Run complete validation of all JSON files"""
        json_files = self.get_all_json_files()
        logger.info(f"Found {len(json_files)} JSON files to validate")

        # First pass: collect all IDs
        self.collect_all_ids(json_files)

        # Second pass: validate each file and check references
        for file_path in json_files:
            data = self.load_json_file(file_path)
            if not data:
                continue

            # Schema validation
            schema_errors = self.validate_schema(data, file_path)
            if schema_errors:
                self.schema_issues.append({
                    'file': str(file_path),
                    'errors': schema_errors
                })
                for error in schema_errors:
                    logger.error(f"Schema error in {file_path}: {error}")

        # Check cross-references
        self.check_cross_references(json_files)

        # Generate report
        report = {
            'total_files': len(json_files),
            'total_ids': len(self.all_ids),
            'schema_issues': self.schema_issues,
            'missing_references': self.missing_references,
            'validation_summary': {
                'files_with_schema_errors': len(self.schema_issues),
                'missing_prerequisites': len([r for r in self.missing_references if r['type'] == 'prerequisite']),
                'missing_related_concepts': len([r for r in self.missing_references if r['type'] == 'related_concept'])
            }
        }

        return report

    def print_report(self, report: Dict[str, Any]) -> None:
        """Print validation report"""
        print("\n" + "="*60)
        print("KNOWLEDGE BASE VALIDATION REPORT")
        print("="*60)

        print("\n📊 SUMMARY:")
        print(f"   Total JSON files: {report['total_files']}")
        print(f"   Total concept IDs: {report['total_ids']}")
        print(f"   Files with schema errors: {report['validation_summary']['files_with_schema_errors']}")
        print(f"   Missing prerequisites: {report['validation_summary']['missing_prerequisites']}")
        print(f"   Missing related concepts: {report['validation_summary']['missing_related_concepts']}")

        if report['schema_issues']:
            print("\n❌ SCHEMA ISSUES:")
            for issue in report['schema_issues'][:10]:  # Show first 10
                print(f"   {issue['file']}:")
                for error in issue['errors']:
                    print(f"     - {error}")
            if len(report['schema_issues']) > 10:
                print(f"   ... and {len(report['schema_issues']) - 10} more files")

        if report['missing_references']:
            print("\n🔗 MISSING REFERENCES:")
            for ref in report['missing_references'][:10]:  # Show first 10
                print(f"   {ref['id']} -> {ref['missing_reference']} ({ref['type']})")
            if len(report['missing_references']) > 10:
                print(f"   ... and {len(report['missing_references']) - 10} more references")

        print("\n✅ VALIDATION COMPLETE")
        print("="*60)

def main():
    """Main validation function"""
    validator = KnowledgeBaseValidator()
    report = validator.validate_all_files()
    validator.print_report(report)

    # Save detailed report
    with open('output/json_validation/validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    logger.info("Detailed report saved to output/json_validation/validation_report.json")

if __name__ == "__main__":
    main()
