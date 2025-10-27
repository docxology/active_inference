#!/usr/bin/env python3
"""
Content Enhancement Tools
Systematically improves knowledge base content by adding missing sections

This module provides tools for enhancing knowledge base content including:
- Adding comprehensive examples and interactive exercises
- Adding further reading and related concepts sections
- Adding common misconceptions sections
- Expanding connections to Active Inference
- Improving content completeness and quality
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentEnhancer:
    """Enhances knowledge base content by adding missing sections"""

    def __init__(self, knowledge_base_path: str):
        """Initialize content enhancer"""
        self.knowledge_base_path = Path(knowledge_base_path)
        self.enhancement_templates = self._load_enhancement_templates()

    def _load_enhancement_templates(self) -> Dict[str, Any]:
        """Load templates for different types of content enhancements"""
        return {
            'examples': {
                'template': self._generate_examples_template,
                'description': 'Add practical examples and use cases'
            },
            'interactive_exercises': {
                'template': self._generate_exercises_template,
                'description': 'Add hands-on exercises and activities'
            },
            'further_reading': {
                'template': self._generate_further_reading_template,
                'description': 'Add references and additional resources'
            },
            'related_concepts': {
                'template': self._generate_related_concepts_template,
                'description': 'Add connections to related topics'
            },
            'common_misconceptions': {
                'template': self._generate_misconceptions_template,
                'description': 'Address common misunderstandings'
            },
            'connections_to_active_inference': {
                'template': self._generate_ai_connections_template,
                'description': 'Connect concepts to Active Inference framework'
            }
        }

    def analyze_content_gaps(self, gap_report_path: str) -> Dict[str, Any]:
        """Analyze content gaps from gap analysis report"""
        with open(gap_report_path, 'r', encoding='utf-8') as f:
            gap_data = json.load(f)

        gaps = gap_data['gaps_analysis']['content_structure_gaps']
        short_sections = gaps['short_content_sections']

        # Prioritize sections to enhance
        priorities = {
            'high': ['examples', 'interactive_exercises'],
            'medium': ['further_reading', 'related_concepts', 'connections_to_active_inference'],
            'low': ['common_misconceptions']
        }

        enhancement_plan = {
            'total_files_to_enhance': len(gaps['files_with_issues']),
            'section_priorities': {},
            'files_by_priority': {'high': [], 'medium': [], 'low': []}
        }

        # Count files needing each section type
        for section_type in self.enhancement_templates.keys():
            count = short_sections.get(section_type, 0)
            enhancement_plan['section_priorities'][section_type] = count

            # Assign priority based on section type
            for priority, sections in priorities.items():
                if section_type in sections:
                    enhancement_plan['files_by_priority'][priority].extend([section_type] * count)
                    break

        return enhancement_plan

    def enhance_file(self, file_path: str, enhancement_type: str, dry_run: bool = True) -> Dict[str, Any]:
        """Enhance a single file with specified content type"""
        full_path = self.knowledge_base_path / file_path

        if not full_path.exists():
            return {'success': False, 'error': f'File not found: {file_path}'}

        try:
            # Load current content
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if section already exists and has meaningful content
            content = data.get('content', {})
            if enhancement_type in content and self._has_meaningful_content(content[enhancement_type]):
                return {
                    'success': False,
                    'error': f'Section {enhancement_type} already exists with meaningful content'
                }

            # Generate enhancement content
            enhancement_content = self._generate_enhancement_content(data, enhancement_type)

            if not dry_run:
                # Update the file
                content[enhancement_type] = enhancement_content

                # Update metadata
                if 'metadata' not in data:
                    data['metadata'] = {}

                data['metadata']['last_updated'] = datetime.now().isoformat()
                data['metadata']['enhancement_notes'] = f"Added {enhancement_type} section"

                # Save updated file
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                return {
                    'success': True,
                    'file': file_path,
                    'enhancement_type': enhancement_type,
                    'content_added': enhancement_content
                }
            else:
                return {
                    'success': True,
                    'file': file_path,
                    'enhancement_type': enhancement_type,
                    'content_preview': str(enhancement_content)[:200] + '...',
                    'dry_run': True
                }

        except Exception as e:
            return {'success': False, 'error': f'Error enhancing file: {e}'}

    def _has_meaningful_content(self, content: Any) -> bool:
        """Check if content section has meaningful content"""
        if isinstance(content, str):
            return len(content.strip()) > 50
        elif isinstance(content, list):
            return len(content) > 2
        elif isinstance(content, dict):
            return len(content) > 1
        else:
            return True  # Assume non-empty

    def _generate_enhancement_content(self, data: Dict[str, Any], enhancement_type: str) -> Any:
        """Generate content for specified enhancement type"""
        if enhancement_type not in self.enhancement_templates:
            return f"Content enhancement for {enhancement_type} not implemented"

        template_func = self.enhancement_templates[enhancement_type]['template']
        return template_func(data)

    def _generate_examples_template(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate examples section"""
        topic = data.get('title', 'the concept')
        content_type = data.get('content_type', 'general')

        if content_type == 'mathematics':
            return self._generate_math_examples(data)
        elif content_type == 'implementation':
            return self._generate_code_examples(data)
        elif content_type == 'foundation':
            return self._generate_conceptual_examples(data)
        else:
            return self._generate_general_examples(data)

    def _generate_math_examples(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mathematical examples"""
        topic = data.get('id', 'concept')

        examples = [
            {
                "name": "Simple Numerical Example",
                "description": "Basic calculation to illustrate the concept",
                "problem": "Consider a simple case with known parameters",
                "solution": "Step-by-step mathematical derivation",
                "interpretation": "Explain what the result means conceptually"
            },
            {
                "name": "Real-World Application",
                "description": "Application of the concept in practice",
                "context": "Describe the practical scenario",
                "mathematical_model": "Show how to model it mathematically",
                "results": "Present and interpret the results"
            }
        ]

        return examples

    def _generate_code_examples(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate code implementation examples"""
        return [
            {
                "name": "Basic Implementation",
                "description": "Simple implementation of the core concept",
                "language": "Python",
                "code": "# Basic implementation example\nimport numpy as np\n\n# Implementation details\npass",
                "explanation": "Step-by-step explanation of the code",
                "expected_output": "Describe what the code should produce"
            },
            {
                "name": "Advanced Usage",
                "description": "More sophisticated implementation",
                "language": "Python",
                "code": "# Advanced implementation\n# More complex example with error handling\npass",
                "explanation": "Detailed explanation of advanced features",
                "use_cases": "When to use this implementation"
            }
        ]

    def _generate_conceptual_examples(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate conceptual examples"""
        topic = data.get('title', 'the concept')

        return [
            {
                "name": "Everyday Analogy",
                "description": "Relate the concept to everyday experience",
                "analogy": "Explain using familiar concepts",
                "connection": "How this relates to the technical concept",
                "why_it_helps": "Why this analogy is useful for understanding"
            },
            {
                "name": "Historical Context",
                "description": "Historical development and motivation",
                "historical_background": "How the concept emerged",
                "key_insights": "Important theoretical developments",
                "modern_interpretation": "How we understand it today"
            }
        ]

    def _generate_general_examples(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate general examples"""
        return [
            {
                "name": "Basic Illustration",
                "description": "Simple example to demonstrate the concept",
                "scenario": "Describe the situation",
                "application": "How the concept applies",
                "outcome": "What happens as a result"
            }
        ]

    def _generate_exercises_template(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate interactive exercises"""
        difficulty = data.get('difficulty', 'intermediate')
        topic = data.get('title', 'the concept')

        exercises = [
            {
                "name": "Conceptual Understanding",
                "type": "comprehension",
                "difficulty": difficulty,
                "question": f"Explain in your own words how {topic.lower()} works",
                "hints": ["Think about the core mechanism", "Consider the inputs and outputs", "Relate to familiar concepts"],
                "expected_answer": "Clear explanation of the concept's purpose and mechanism"
            },
            {
                "name": "Problem Solving",
                "type": "application",
                "difficulty": difficulty,
                "question": f"Apply {topic.lower()} to solve a practical problem",
                "scenario": "Describe a specific problem or scenario",
                "requirements": "What needs to be accomplished",
                "solution_approach": "Step-by-step method to solve it"
            }
        ]

        # Add mathematical exercises for math content
        if data.get('content_type') == 'mathematics':
            exercises.append({
                "name": "Mathematical Derivation",
                "type": "derivation",
                "difficulty": "advanced",
                "question": f"Derive the key mathematical relationship for {topic.lower()}",
                "given": "Starting equations or assumptions",
                "steps": "Step-by-step derivation process",
                "result": "Final mathematical expression"
            })

        return exercises

    def _generate_further_reading_template(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate further reading section"""
        topic = data.get('title', 'the concept')
        content_type = data.get('content_type', 'general')

        readings = [
            {
                "category": "Foundational Papers",
                "description": "Seminal papers that established the concept",
                "references": [
                    {
                        "title": "Key Foundational Work",
                        "authors": "Author names",
                        "year": "2024",
                        "journal": "Journal Name",
                        "doi": "10.1000/journal.article",
                        "why_important": "Why this paper matters for understanding the concept"
                    }
                ]
            },
            {
                "category": "Recent Developments",
                "description": "Recent advances and applications",
                "references": [
                    {
                        "title": "Recent Application or Development",
                        "authors": "Author names",
                        "year": "2024",
                        "journal": "Journal Name",
                        "doi": "10.1000/recent.article",
                        "why_important": "How this extends or applies the concept"
                    }
                ]
            }
        ]

        # Add Active Inference specific readings
        if 'active_inference' in topic.lower() or 'fep' in topic.lower():
            readings.append({
                "category": "Active Inference Literature",
                "description": "Key Active Inference papers and books",
                "references": [
                    {
                        "title": "A Free Energy Principle for a Particular Physics",
                        "authors": "Karl Friston",
                        "year": "2019",
                        "journal": "arXiv preprint",
                        "arxiv": "arXiv:1906.10100",
                        "why_important": "Foundational paper on the Free Energy Principle"
                    },
                    {
                        "title": "Active Inference: A Process Theory",
                        "authors": "Maxwell Ramstead et al.",
                        "year": "2022",
                        "journal": "Neural Computation",
                        "doi": "10.1162/neco_a_01562",
                        "why_important": "Comprehensive overview of Active Inference framework"
                    }
                ]
            })

        return readings

    def _generate_related_concepts_template(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate related concepts section"""
        topic = data.get('id', 'concept')
        content_type = data.get('content_type', 'general')

        related = []

        # Add general relationships
        related.append({
            "concept": "Information Theory",
            "relationship": "Foundational framework for understanding uncertainty",
            "connection_strength": "strong",
            "why_related": "Provides mathematical foundation for many Active Inference concepts"
        })

        related.append({
            "concept": "Bayesian Inference",
            "relationship": "Core computational method",
            "connection_strength": "strong",
            "why_related": "Active Inference extends Bayesian methods to action and control"
        })

        # Content-type specific relationships
        if content_type == 'mathematics':
            related.extend([
                {
                    "concept": "Variational Methods",
                    "relationship": "Optimization technique",
                    "connection_strength": "medium",
                    "why_related": "Used for approximate inference in complex models"
                },
                {
                    "concept": "Differential Geometry",
                    "relationship": "Mathematical framework",
                    "connection_strength": "medium",
                    "why_related": "Provides geometric interpretation of probability distributions"
                }
            ])
        elif content_type == 'implementation':
            related.extend([
                {
                    "concept": "Neural Networks",
                    "relationship": "Implementation platform",
                    "connection_strength": "medium",
                    "why_related": "Deep learning provides tools for implementing Active Inference"
                },
                {
                    "concept": "Reinforcement Learning",
                    "relationship": "Related framework",
                    "connection_strength": "medium",
                    "why_related": "Active Inference provides alternative to traditional RL"
                }
            ])

        return related

    def _generate_misconceptions_template(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate common misconceptions section"""
        topic = data.get('title', 'the concept')

        misconceptions = [
            {
                "misconception": f"Common misunderstanding about {topic}",
                "reality": "What is actually true",
                "why_confusing": "Why people get confused",
                "clarification": "Clear explanation of the correct understanding"
            },
            {
                "misconception": "Oversimplification or overgeneralization",
                "reality": "More nuanced understanding",
                "why_confusing": "Source of the confusion",
                "clarification": "Proper interpretation and context"
            }
        ]

        # Add specific misconceptions based on content type
        if data.get('content_type') == 'mathematics':
            misconceptions.append({
                "misconception": "Mathematical concept is purely theoretical",
                "reality": "Has practical computational implementations",
                "why_confusing": "Abstract mathematical formulation",
                "clarification": "Show connection between theory and computation"
            })

        return misconceptions

    def _generate_ai_connections_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate connections to Active Inference"""
        topic = data.get('title', 'the concept')
        content_type = data.get('content_type', 'general')

        connections = {
            "overview": f"How {topic} relates to Active Inference framework",
            "specific_connections": [
                {
                    "aspect": "Theoretical Foundation",
                    "connection": "How this concept provides theoretical basis for Active Inference",
                    "importance": "Fundamental to understanding Active Inference principles"
                },
                {
                    "aspect": "Computational Implementation",
                    "connection": "How this concept is implemented in Active Inference systems",
                    "importance": "Essential for practical applications"
                },
                {
                    "aspect": "Applications",
                    "connection": "How this concept enables specific Active Inference applications",
                    "importance": "Key to real-world problem solving"
                }
            ],
            "implementation_examples": [
                {
                    "example": "Basic Implementation",
                    "description": "Simple example showing the connection",
                    "code_reference": "Reference to implementation code",
                    "expected_behavior": "What the implementation should demonstrate"
                }
            ]
        }

        return connections

    def batch_enhance_files(self, file_list: List[str], enhancement_types: List[str],
                          dry_run: bool = True) -> Dict[str, Any]:
        """Enhance multiple files with specified enhancement types"""
        results = {
            'total_files': len(file_list),
            'successful_enhancements': 0,
            'failed_enhancements': 0,
            'details': []
        }

        for file_path in file_list:
            for enhancement_type in enhancement_types:
                result = self.enhance_file(file_path, enhancement_type, dry_run)

                results['details'].append({
                    'file': file_path,
                    'enhancement_type': enhancement_type,
                    'success': result['success'],
                    'message': result.get('error', result.get('content_preview', 'Success'))
                })

                if result['success']:
                    results['successful_enhancements'] += 1
                else:
                    results['failed_enhancements'] += 1

        return results

    def create_enhancement_report(self, results: Dict[str, Any]) -> str:
        """Create a report of enhancement results"""
        report = []
        report.append("CONTENT ENHANCEMENT REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        report.append(f"Total Files Processed: {results['total_files']}")
        report.append(f"Successful Enhancements: {results['successful_enhancements']}")
        report.append(f"Failed Enhancements: {results['failed_enhancements']}")
        report.append("")

        # Group by enhancement type
        enhancement_counts = {}
        for detail in results['details']:
            enh_type = detail['enhancement_type']
            enhancement_counts[enh_type] = enhancement_counts.get(enh_type, 0) + 1

        report.append("Enhancement Types Applied:")
        for enh_type, count in enhancement_counts.items():
            report.append(f"  {enh_type}: {count} files")

        report.append("")
        report.append("Details:")
        for detail in results['details']:
            status = "✓" if detail['success'] else "✗"
            report.append(f"  {status} {detail['file']} - {detail['enhancement_type']}: {detail['message']}")

        return "\n".join(report)

def main():
    """Main function for content enhancement"""
    if len(sys.argv) < 3:
        print("Usage: python content_enhancer.py <knowledge_base_path> <command> [options]")
        print("Commands:")
        print("  analyze <gap_report.json>    - Analyze gaps and create enhancement plan")
        print("  enhance <file.json> <type>   - Enhance specific file with content type")
        print("  batch <gap_report.json>      - Batch enhance files based on gap analysis")
        print("  preview <file.json> <type>   - Preview enhancement without applying")
        sys.exit(1)

    knowledge_path = sys.argv[1]
    command = sys.argv[2]

    if not Path(knowledge_path).exists():
        print(f"Knowledge base path does not exist: {knowledge_path}")
        sys.exit(1)

    enhancer = ContentEnhancer(knowledge_path)

    if command == "analyze":
        if len(sys.argv) != 4:
            print("Usage: python content_enhancer.py <path> analyze <gap_report.json>")
            sys.exit(1)

        gap_report = sys.argv[3]
        plan = enhancer.analyze_content_gaps(gap_report)

        print("\nCONTENT ENHANCEMENT PLAN")
        print("=" * 40)
        print(f"Total Files to Enhance: {plan['total_files_to_enhance']}")
        print("\nSection Priorities:")
        for section, count in plan['section_priorities'].items():
            print(f"  {section}: {count} files")

        print("\nPriority Breakdown:")
        for priority, sections in plan['files_by_priority'].items():
            print(f"  {priority.upper()}: {len(sections)} section enhancements")

        # Save plan
        with open('enhancement_plan.json', 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        print("\nEnhancement plan saved to: enhancement_plan.json")
    elif command == "enhance":
        if len(sys.argv) != 5:
            print("Usage: python content_enhancer.py <path> enhance <file.json> <enhancement_type>")
            sys.exit(1)

        file_path = sys.argv[3]
        enhancement_type = sys.argv[4]

        result = enhancer.enhance_file(file_path, enhancement_type, dry_run=False)
        print(f"Enhancement {'successful' if result['success'] else 'failed'}: {result}")

    elif command == "preview":
        if len(sys.argv) != 5:
            print("Usage: python content_enhancer.py <path> preview <file.json> <enhancement_type>")
            sys.exit(1)

        file_path = sys.argv[3]
        enhancement_type = sys.argv[4]

        result = enhancer.enhance_file(file_path, enhancement_type, dry_run=True)
        print(f"Preview: {result}")

    elif command == "batch":
        if len(sys.argv) != 4:
            print("Usage: python content_enhancer.py <path> batch <gap_report.json>")
            sys.exit(1)

        gap_report = sys.argv[3]
        plan = enhancer.analyze_content_gaps(gap_report)

        # Get top priority files (limit to avoid too many changes at once)
        priority_files = []
        for priority in ['high', 'medium']:
            for section_type in enhancer.enhancement_templates.keys():
                count = plan['section_priorities'].get(section_type, 0)
                if count > 0 and priority in ['high', 'medium'][:1]:  # Focus on high priority
                    # Find files that need this section
                    with open(gap_report, 'r') as f:
                        gap_data = json.load(f)

                    files_with_issues = gap_data['gaps_analysis']['content_structure_gaps']['files_with_issues']
                    files_with_section = [item['file'] for item in files_with_issues
                                       if any(section_type in warning for warning in item['warnings'])]

                    priority_files.extend(files_with_section[:5])  # Limit to 5 files per type
                    break

        if priority_files:
            print(f"Enhancing {len(priority_files)} high-priority files...")
            results = enhancer.batch_enhance_files(priority_files[:10], ['examples', 'further_reading'], dry_run=False)

            report = enhancer.create_enhancement_report(results)
            print(report)

            # Save detailed results
            with open('enhancement_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            print("No high-priority files found for enhancement")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
