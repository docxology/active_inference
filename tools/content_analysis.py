#!/usr/bin/env python3
"""
Comprehensive Content Quality Analysis Tool for Active Inference Knowledge Base

This tool analyzes the depth, completeness, and quality of existing content
to identify areas for enhancement beyond basic concept coverage.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """Analyzer for content quality and completeness"""

    def __init__(self, knowledge_base_path: str = "knowledge"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.content_metrics: Dict[str, Any] = {}

        # Quality indicators to check
        self.quality_indicators = {
            'mathematical_depth': [
                'equations', 'derivations', 'proofs', 'mathematical_formulation'
            ],
            'practical_examples': [
                'examples', 'code_examples', 'implementations', 'applications'
            ],
            'interactive_elements': [
                'exercises', 'interactive_exercises', 'simulations', 'tutorials'
            ],
            'cross_references': [
                'related_concepts', 'further_reading', 'references'
            ],
            'clarity_indicators': [
                'overview', 'common_misconceptions', 'intuitive_explanation'
            ]
        }

    def analyze_content_structure(self) -> Dict[str, Any]:
        """Analyze the structure and completeness of content"""
        content_analysis = {
            'total_files': 0,
            'files_by_category': {},
            'quality_scores': {},
            'missing_elements': {},
            'content_patterns': {}
        }

        for file_path in self.knowledge_base_path.rglob("*.json"):
            if file_path.name in ['faq.json', 'glossary.json', 'learning_paths.json']:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not self._is_valid_content_node(data):
                    continue

                content_analysis['total_files'] += 1
                category = data.get('content_type', 'unknown')

                if category not in content_analysis['files_by_category']:
                    content_analysis['files_by_category'][category] = 0
                content_analysis['files_by_category'][category] += 1

                # Analyze content quality for this file
                quality_score = self._analyze_file_quality(data)
                file_id = data.get('id', 'unknown')

                if category not in content_analysis['quality_scores']:
                    content_analysis['quality_scores'][category] = []
                content_analysis['quality_scores'][category].append({
                    'file': file_id,
                    'score': quality_score,
                    'difficulty': data.get('difficulty', 'unknown')
                })

            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")

        return content_analysis

    def _is_valid_content_node(self, data: Dict[str, Any]) -> bool:
        """Check if data represents a valid content node"""
        required_fields = ['id', 'title', 'content_type', 'difficulty', 'content']
        return all(field in data for field in required_fields)

    def _analyze_file_quality(self, data: Dict[str, Any]) -> float:
        """Analyze quality score for a single file (0-1 scale)"""
        score = 0.0
        max_score = 5.0

        # Check for overview (essential)
        if 'overview' in data.get('content', {}):
            score += 1.0

        # Check for mathematical content (for math files)
        content = data.get('content', {})
        content_type = data.get('content_type', '')

        if content_type == 'mathematics':
            has_math = any(indicator in str(content).lower() for indicator in
                          ['equation', 'formula', 'derivation', 'proof', '\\', '$'])
            if has_math:
                score += 1.0
        else:
            score += 1.0  # Non-math content gets full score for this

        # Check for examples
        has_examples = any(indicator in str(content).lower() for indicator in
                          ['example', 'illustration', 'case study'])
        if has_examples:
            score += 1.0

        # Check for interactive elements
        has_interactive = any(indicator in str(content).lower() for indicator in
                            ['exercise', 'tutorial', 'simulation', 'practice'])
        if has_interactive:
            score += 1.0

        # Check for cross-references
        has_references = ('related_concepts' in data and data['related_concepts']) or \
                        ('further_reading' in str(content))
        if has_references:
            score += 1.0

        return score / max_score

    def analyze_content_depth(self) -> Dict[str, Any]:
        """Analyze depth and completeness of content by category"""
        depth_analysis = {
            'foundation_concepts': {},
            'mathematics_content': {},
            'implementation_examples': {},
            'application_domains': {}
        }

        # Analyze foundation concepts
        foundation_files = [
            'causal_inference', 'optimization_methods', 'neural_dynamics',
            'information_bottleneck', 'decision_theory', 'fep_introduction',
            'active_inference_introduction', 'bayesian_basics'
        ]

        for concept in foundation_files:
            file_path = self.knowledge_base_path / 'foundations' / f'{concept}.json'
            if file_path.exists():
                depth_analysis['foundation_concepts'][concept] = self._analyze_concept_depth(file_path)

        # Analyze mathematics content
        math_files = [
            'variational_free_energy', 'expected_free_energy', 'information_geometry',
            'predictive_coding', 'stochastic_processes', 'dynamical_systems'
        ]

        for concept in math_files:
            file_path = self.knowledge_base_path / 'mathematics' / f'{concept}.json'
            if file_path.exists():
                depth_analysis['mathematics_content'][concept] = self._analyze_concept_depth(file_path)

        # Analyze implementation examples
        impl_files = [
            'active_inference_basic', 'variational_inference', 'planning_algorithms',
            'reinforcement_learning', 'control_systems', 'neural_network_implementation'
        ]

        for concept in impl_files:
            file_path = self.knowledge_base_path / 'implementations' / f'{concept}.json'
            if file_path.exists():
                depth_analysis['implementation_examples'][concept] = self._analyze_concept_depth(file_path)

        # Analyze application domains
        app_files = [
            'robotics_control', 'neuroscience_perception', 'ai_safety',
            'clinical_applications', 'adaptive_learning', 'climate_decision_making'
        ]

        for concept in app_files:
            # Check in applications directory and subdirectories
            file_path = self.knowledge_base_path / 'applications' / f'{concept}.json'
            if not file_path.exists():
                file_path = self.knowledge_base_path / 'applications' / 'domains' / '**' / f'{concept}.json'

            if file_path.exists():
                depth_analysis['application_domains'][concept] = self._analyze_concept_depth(file_path)

        return depth_analysis

    def _analyze_concept_depth(self, file_path: Union[Path, str]) -> Dict[str, Any]:
        """Analyze depth of a specific concept"""
        if isinstance(file_path, str):
            file_path = Path(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            content = data.get('content', {})
            content_str = json.dumps(content).lower()

            depth_metrics = {
                'word_count': len(content_str.split()),
                'has_mathematical_content': any(indicator in content_str for indicator in
                                               ['equation', 'formula', '\\', '$', 'theorem']),
                'has_examples': any(indicator in content_str for indicator in
                                  ['example', 'illustration', 'case study']),
                'has_exercises': any(indicator in content_str for indicator in
                                   ['exercise', 'practice', 'tutorial']),
                'has_references': 'further_reading' in str(content) or 'references' in content_str,
                'section_count': len([k for k in content.keys() if not k.startswith('_')]),
                'quality_score': self._analyze_file_quality(data)
            }

            return depth_metrics

        except Exception as e:
            logger.error(f"Error analyzing depth for {file_path}: {e}")
            return {}

    def identify_enhancement_opportunities(self) -> Dict[str, Any]:
        """Identify specific opportunities for content enhancement"""
        opportunities = {
            'shallow_content': [],
            'missing_examples': [],
            'missing_exercises': [],
            'missing_references': [],
            'incomplete_concepts': []
        }

        # Check content depth analysis
        depth_analysis = self.analyze_content_depth()

        for category, concepts in depth_analysis.items():
            for concept, metrics in concepts.items():
                if metrics.get('quality_score', 0) < 0.6:
                    opportunities['shallow_content'].append({
                        'concept': concept,
                        'category': category,
                        'score': metrics.get('quality_score', 0)
                    })

                if not metrics.get('has_examples', False):
                    opportunities['missing_examples'].append(concept)

                if not metrics.get('has_exercises', False):
                    opportunities['missing_exercises'].append(concept)

                if not metrics.get('has_references', False):
                    opportunities['missing_references'].append(concept)

                if metrics.get('word_count', 0) < 500:
                    opportunities['incomplete_concepts'].append({
                        'concept': concept,
                        'word_count': metrics.get('word_count', 0)
                    })

        return opportunities

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive content analysis report"""
        structure_analysis = self.analyze_content_structure()
        depth_analysis = self.analyze_content_depth()
        opportunities = self.identify_enhancement_opportunities()

        # Calculate overall quality metrics
        all_quality_scores = []
        for category_scores in structure_analysis['quality_scores'].values():
            all_quality_scores.extend([item['score'] for item in category_scores])

        overall_metrics = {
            'average_quality_score': sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0,
            'quality_distribution': {
                'excellent': len([s for s in all_quality_scores if s >= 0.8]),
                'good': len([s for s in all_quality_scores if 0.6 <= s < 0.8]),
                'needs_improvement': len([s for s in all_quality_scores if s < 0.6])
            }
        }

        report = {
            'summary': {
                'total_content_nodes': structure_analysis['total_files'],
                'category_distribution': structure_analysis['files_by_category'],
                'overall_quality_metrics': overall_metrics
            },
            'structure_analysis': structure_analysis,
            'depth_analysis': depth_analysis,
            'enhancement_opportunities': opportunities,
            'recommendations': self._generate_specific_recommendations(opportunities, depth_analysis)
        }

        return report

    def _generate_specific_recommendations(self, opportunities: Dict[str, Any],
                                        depth_analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for content enhancement"""
        recommendations = []

        if opportunities['shallow_content']:
            recommendations.append(f"Enhance {len(opportunities['shallow_content'])} shallow content areas: "
                                f"{[item['concept'] for item in opportunities['shallow_content'][:5]]}")

        if opportunities['missing_examples']:
            recommendations.append(f"Add practical examples to {len(opportunities['missing_examples'])} concepts: "
                                f"{opportunities['missing_examples'][:5]}")

        if opportunities['missing_exercises']:
            recommendations.append(f"Add interactive exercises to {len(opportunities['missing_exercises'])} concepts: "
                                f"{opportunities['missing_exercises'][:5]}")

        if opportunities['missing_references']:
            recommendations.append(f"Add references and further reading to {len(opportunities['missing_references'])} concepts: "
                                f"{opportunities['missing_references'][:5]}")

        if opportunities['incomplete_concepts']:
            recommendations.append(f"Expand {len(opportunities['incomplete_concepts'])} incomplete concepts "
                                f"(under 500 words): {[item['concept'] for item in opportunities['incomplete_concepts'][:5]]}")

        # Check for advanced topics that need more depth
        recommendations.append("Consider adding advanced topics: measure theory, optimal transport applications, "
                            "multi-scale modeling, quantum information theory")

        # Check for emerging applications
        recommendations.append("Expand emerging application areas: AI governance, quantum computing, "
                            "collective intelligence, synthetic biology")

        return recommendations

    def print_report(self, report: Dict[str, Any]) -> None:
        """Print formatted comprehensive content analysis report"""
        print("\n" + "="*90)
        print("COMPREHENSIVE CONTENT QUALITY ANALYSIS REPORT")
        print("="*90)

        summary = report['summary']
        print("
📊 SUMMARY:"        print(f"   Total content nodes: {summary['total_content_nodes']}")
        print(f"   Category distribution: {summary['category_distribution']}")
        print(f"   Average quality score: {summary['overall_quality_metrics']['average_quality_score']:.2f}")
        print(f"   Quality distribution: {summary['overall_quality_metrics']['quality_distribution']}")

        opportunities = report['enhancement_opportunities']
        print("
🔍 ENHANCEMENT OPPORTUNITIES:"        print(f"   Shallow content areas: {len(opportunities['shallow_content'])}")
        print(f"   Missing examples: {len(opportunities['missing_examples'])}")
        print(f"   Missing exercises: {len(opportunities['missing_exercises'])}")
        print(f"   Missing references: {len(opportunities['missing_references'])}")
        print(f"   Incomplete concepts: {len(opportunities['incomplete_concepts'])}")

        print("
💡 SPECIFIC RECOMMENDATIONS:"        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")

        print("
🎯 PRIORITY AREAS:"        print("   1. Enhance shallow content with deeper explanations and examples"        print("   2. Add practical examples and case studies"        print("   3. Develop interactive exercises and tutorials"        print("   4. Expand reference sections with current research"        print("   5. Complete underdeveloped concepts with comprehensive coverage"        print("   6. Add emerging topics and advanced applications"

        print("\n" + "="*90)

def main():
    """Main content analysis function"""
    analyzer = ContentAnalyzer()
    report = analyzer.generate_comprehensive_report()
    analyzer.print_report(report)

    # Save detailed report
    with open('output/json_validation/content_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Detailed content analysis report saved to output/json_validation/content_analysis_report.json")

if __name__ == "__main__":
    main()
