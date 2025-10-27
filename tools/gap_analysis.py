#!/usr/bin/env python3
"""
Comprehensive Gap Analysis Tool for Active Inference Knowledge Base

This tool systematically identifies missing content, incomplete coverage,
and priority areas for enhancement based on the comprehensive audit requirements.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class GapAnalyzer:
    """Analyzer for identifying content gaps in the knowledge base"""

    def __init__(self, knowledge_base_path: str = "knowledge"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.existing_concepts: Set[str] = set()
        self.content_by_category: Dict[str, List[str]] = {
            'foundation': [], 'mathematics': [], 'implementation': [], 'application': []
        }

        # Define comprehensive requirements from audit prompt
        self.required_foundation_topics = {
            'causal_inference', 'optimization_methods', 'neural_dynamics',
            'information_bottleneck', 'decision_theory', 'info_theory_entropy',
            'info_theory_kl_divergence', 'info_theory_mutual_information',
            'bayesian_basics', 'bayesian_models', 'belief_updating',
            'fep_introduction', 'fep_mathematical_formulation', 'fep_biological_systems',
            'active_inference_introduction', 'ai_generative_models', 'ai_policy_selection',
            'hierarchical_models', 'conjugate_priors', 'empirical_bayes', 'cross_entropy',
            'multi_agent_systems', 'continuous_control'
        }

        self.required_mathematics_topics = {
            'variational_free_energy', 'expected_free_energy', 'information_geometry',
            'predictive_coding', 'stochastic_processes', 'optimal_transport',
            'markov_chain_monte_carlo', 'advanced_variational_methods',
            'dynamical_systems', 'differential_geometry', 'advanced_probability'
        }

        self.required_implementation_topics = {
            'active_inference_basic', 'variational_inference', 'expected_free_energy_calculation',
            'mcmc_sampling', 'neural_network_implementation', 'planning_algorithms',
            'reinforcement_learning', 'control_systems', 'deep_generative_models',
            'simulation_methods', 'benchmarking', 'uncertainty_quantification'
        }

        self.required_application_domains = {
            'ai_safety', 'autonomous_systems', 'brain_imaging', 'clinical_applications',
            'intelligent_tutoring', 'market_behavior', 'game_theory', 'climate_decision_making',
            'ai_alignment', 'adaptive_learning', 'control_systems', 'cognitive_science',
            'robotics_control', 'neuroscience_perception', 'decision_making'
        }

    def collect_existing_concepts(self) -> None:
        """Collect all existing concept IDs and categorize them"""
        json_files = list(self.knowledge_base_path.rglob("*.json"))

        for file_path in json_files:
            # Skip special files that don't follow the standard schema
            if file_path.name in ['faq.json', 'glossary.json', 'learning_paths.json']:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'id' in data and 'content_type' in data:
                    concept_id = data['id']
                    content_type = data['content_type']

                    self.existing_concepts.add(concept_id)

                    if content_type in self.content_by_category:
                        self.content_by_category[content_type].append(concept_id)
                    else:
                        logger.warning(f"Unknown content_type: {content_type} for {concept_id}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Collected {len(self.existing_concepts)} concepts")
        for category, concepts in self.content_by_category.items():
            logger.info(f"  {category}: {len(concepts)} concepts")

    def identify_missing_concepts(self) -> Dict[str, Set[str]]:
        """Identify missing concepts by category"""
        missing = {}

        # Foundation gaps
        foundation_missing = self.required_foundation_topics - self.existing_concepts
        missing['foundation'] = foundation_missing

        # Mathematics gaps
        math_missing = self.required_mathematics_topics - self.existing_concepts
        missing['mathematics'] = math_missing

        # Implementation gaps
        implementation_missing = self.required_implementation_topics - self.existing_concepts
        missing['implementation'] = implementation_missing

        # Application gaps
        application_missing = self.required_application_domains - self.existing_concepts
        missing['application'] = application_missing

        return missing

    def analyze_learning_path_completeness(self) -> Dict[str, Any]:
        """Analyze completeness of learning paths"""
        try:
            with open(self.knowledge_base_path / 'learning_paths.json', 'r') as f:
                learning_paths = json.load(f)

            analysis = {
                'total_paths': len(learning_paths.get('learning_paths', [])),
                'missing_concepts': [],
                'path_completeness': {}
            }

            for path in learning_paths.get('learning_paths', []):
                path_id = path.get('id', 'unknown')
                required_nodes = set(path.get('nodes', []))

                # Check which nodes are missing
                missing_nodes = required_nodes - self.existing_concepts
                existing_nodes = required_nodes & self.existing_concepts

                completeness = len(existing_nodes) / len(required_nodes) if required_nodes else 0

                analysis['path_completeness'][path_id] = {
                    'completeness': completeness,
                    'total_nodes': len(required_nodes),
                    'existing_nodes': len(existing_nodes),
                    'missing_nodes': len(missing_nodes),
                    'missing_node_list': list(missing_nodes)
                }

                if missing_nodes:
                    analysis['missing_concepts'].extend(list(missing_nodes))

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing learning paths: {e}")
            return {}

    def assess_content_quality_indicators(self) -> Dict[str, Any]:
        """Assess various quality indicators of the content"""
        quality_metrics = {
            'total_concepts': len(self.existing_concepts),
            'concepts_by_difficulty': {},
            'schema_compliance': {},
            'cross_reference_density': 0,
            'content_completeness': {}
        }

        difficulty_counts = {'beginner': 0, 'intermediate': 0, 'advanced': 0, 'expert': 0}
        schema_compliance = {'complete': 0, 'incomplete': 0}

        total_prerequisites = 0
        total_related_concepts = 0
        valid_prerequisites = 0
        valid_related_concepts = 0

        for file_path in self.knowledge_base_path.rglob("*.json"):
            if file_path.name in ['faq.json', 'glossary.json', 'learning_paths.json']:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'id' not in data:
                    continue

                # Count by difficulty
                if 'difficulty' in data:
                    difficulty = data['difficulty']
                    if difficulty in difficulty_counts:
                        difficulty_counts[difficulty] += 1

                # Check schema compliance
                required_fields = [
                    'id', 'title', 'content_type', 'difficulty',
                    'description', 'prerequisites', 'tags',
                    'learning_objectives', 'content', 'metadata'
                ]

                is_complete = all(field in data for field in required_fields)
                if is_complete:
                    schema_compliance['complete'] += 1
                else:
                    schema_compliance['incomplete'] += 1

                # Count cross-references
                total_prerequisites += len(data.get('prerequisites', []))
                total_related_concepts += len(data.get('related_concepts', []))

                for prereq in data.get('prerequisites', []):
                    if prereq in self.existing_concepts:
                        valid_prerequisites += 1

                for concept in data.get('related_concepts', []):
                    if concept in self.existing_concepts:
                        valid_related_concepts += 1

            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")

        quality_metrics['concepts_by_difficulty'] = difficulty_counts
        quality_metrics['schema_compliance'] = schema_compliance

        # Calculate cross-reference density
        total_possible_refs = len(self.existing_concepts) * (len(self.existing_concepts) - 1)
        if total_possible_refs > 0:
            quality_metrics['cross_reference_density'] = (valid_prerequisites + valid_related_concepts) / total_possible_refs

        return quality_metrics

    def prioritize_missing_content(self, missing_concepts: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        """Prioritize missing content based on impact and dependencies"""
        priorities = {'high': [], 'medium': [], 'low': []}

        # High priority - core foundational concepts
        high_priority_foundation = {
            'causal_inference', 'optimization_methods', 'neural_dynamics',
            'information_bottleneck', 'decision_theory'
        }

        # High priority - essential mathematics
        high_priority_math = {
            'advanced_variational_methods', 'dynamical_systems', 'differential_geometry'
        }

        # High priority - core implementations
        high_priority_impl = {
            'reinforcement_learning', 'control_systems', 'planning_algorithms',
            'deep_generative_models', 'simulation_methods'
        }

        # High priority - critical applications
        high_priority_apps = {
            'ai_safety', 'autonomous_systems', 'brain_imaging', 'clinical_applications'
        }

        # Categorize missing content
        for category, concepts in missing_concepts.items():
            for concept in concepts:
                if (category == 'foundation' and concept in high_priority_foundation) or \
                   (category == 'mathematics' and concept in high_priority_math) or \
                   (category == 'implementation' and concept in high_priority_impl) or \
                   (category == 'application' and concept in high_priority_apps):
                    priorities['high'].append(concept)
                elif category in ['foundation', 'mathematics']:
                    priorities['medium'].append(concept)
                else:
                    priorities['low'].append(concept)

        return priorities

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive gap analysis report"""
        # Collect data
        self.collect_existing_concepts()
        missing_concepts = self.identify_missing_concepts()
        learning_path_analysis = self.analyze_learning_path_completeness()
        quality_metrics = self.assess_content_quality_indicators()
        priorities = self.prioritize_missing_content(missing_concepts)

        # Generate report
        report = {
            'summary': {
                'total_concepts': len(self.existing_concepts),
                'concepts_by_category': {k: len(v) for k, v in self.content_by_category.items()},
                'missing_concepts_total': sum(len(v) for v in missing_concepts.values()),
                'learning_paths_analyzed': len(learning_path_analysis.get('path_completeness', {}))
            },
            'missing_concepts': missing_concepts,
            'quality_metrics': quality_metrics,
            'learning_path_analysis': learning_path_analysis,
            'priorities': priorities,
            'recommendations': self.generate_recommendations(missing_concepts, priorities, quality_metrics)
        }

        return report

    def generate_recommendations(self, missing_concepts: Dict[str, Set[str]],
                               priorities: Dict[str, List[str]],
                               quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for improvement"""
        recommendations = []

        # High priority recommendations
        if priorities['high']:
            recommendations.append(f"CRITICAL: Create {len(priorities['high'])} high-priority concepts: {', '.join(priorities['high'])}")

        # Category-specific recommendations
        for category, missing in missing_concepts.items():
            if missing:
                completion_rate = len(self.content_by_category[category]) / (len(self.content_by_category[category]) + len(missing))
                recommendations.append(f"Enhance {category} coverage: {len(missing)} missing concepts, {completion_rate:.1%} current completion")

        # Quality recommendations
        incomplete_files = quality_metrics['schema_compliance'].get('incomplete', 0)
        if incomplete_files > 0:
            recommendations.append(f"Fix schema compliance: {incomplete_files} files missing required fields")

        # Learning path recommendations
        incomplete_paths = [k for k, v in quality_metrics.get('learning_path_analysis', {}).get('path_completeness', {}).items()
                          if v.get('completeness', 0) < 0.8]
        if incomplete_paths:
            recommendations.append(f"Complete learning paths: {len(incomplete_paths)} paths are less than 80% complete")

        return recommendations

    def print_report(self, report: Dict[str, Any]) -> None:
        """Print formatted gap analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE KNOWLEDGE BASE GAP ANALYSIS REPORT")
        print("="*80)

        summary = report['summary']
        print("\n📊 SUMMARY:")
        print(f"   Total concepts: {summary['total_concepts']}")
        print(f"   Concepts by category: {summary['concepts_by_category']}")
        print(f"   Missing concepts: {summary['missing_concepts_total']}")
        print(f"   Learning paths analyzed: {summary['learning_paths_analyzed']}")

        print("\n📚 MISSING CONCEPTS BY CATEGORY:")
        missing = report['missing_concepts']
        for category, concepts in missing.items():
            if concepts:
                print(f"   {category.title()}: {len(concepts)} missing")
                for concept in sorted(concepts):
                    print(f"     - {concept}")

        priorities = report['priorities']
        print("\n🎯 PRIORITIZATION:")
        print(f"   High priority ({len(priorities['high'])}): {', '.join(priorities['high'])}")
        print(f"   Medium priority ({len(priorities['medium'])}): {', '.join(priorities['medium'])}")
        print(f"   Low priority ({len(priorities['low'])}): {', '.join(priorities['low'])}")

        print("\n✅ QUALITY METRICS:")
        quality = report['quality_metrics']
        difficulty = quality['concepts_by_difficulty']
        print(f"   Difficulty distribution: {difficulty}")
        schema = quality['schema_compliance']
        print(f"   Schema compliance: {schema['complete']} complete, {schema['incomplete']} incomplete")
        print(f"   Cross-reference density: {quality['cross_reference_density']:.3f}")

        print("\n🔗 LEARNING PATH ANALYSIS:")
        paths = report['learning_path_analysis']
        if paths.get('path_completeness'):
            incomplete_paths = [(k, v['completeness']) for k, v in paths['path_completeness'].items() if v['completeness'] < 1.0]
            if incomplete_paths:
                print(f"   Incomplete paths: {len(incomplete_paths)}")
                for path_id, completeness in incomplete_paths[:5]:  # Show top 5
                    print(f"     - {path_id}: {completeness:.1%} complete")

        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")

        print("\n📋 NEXT STEPS:")
        print("   1. Create high-priority missing concepts")
        print("   2. Fix schema compliance issues")
        print("   3. Complete learning paths")
        print("   4. Enhance cross-referencing")
        print("   5. Validate content quality")

        print("\n" + "="*80)

def main():
    """Main gap analysis function"""
    analyzer = GapAnalyzer()
    report = analyzer.generate_comprehensive_report()
    analyzer.print_report(report)

    # Save detailed report
    with open('output/json_validation/gap_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Detailed gap analysis report saved to output/json_validation/gap_analysis_report.json")

if __name__ == "__main__":
    main()
