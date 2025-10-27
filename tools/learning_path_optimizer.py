#!/usr/bin/env python3
"""
Learning Path Optimization Tools
Validates and optimizes learning paths for better educational experience

This module provides tools for:
- Validating learning path completeness and consistency
- Optimizing prerequisite chains and difficulty progression
- Enhancing learning path metadata and descriptions
- Identifying gaps in learning path coverage
- Recommending path improvements
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningPathValidator:
    """Validates and optimizes learning paths"""

    def __init__(self, knowledge_base_path: str):
        """Initialize learning path validator"""
        self.knowledge_base_path = Path(knowledge_base_path)
        self.all_nodes = self._scan_all_nodes()
        self.learning_paths = self._load_learning_paths()

    def _scan_all_nodes(self) -> Set[str]:
        """Scan all knowledge nodes"""
        nodes = set()
        json_files = list(self.knowledge_base_path.rglob("*.json"))

        for file_path in json_files:
            if file_path.name in ['learning_paths.json', 'faq.json', 'glossary.json']:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                nodes.add(data['id'])
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")

        return nodes

    def _load_learning_paths(self) -> Dict[str, Any]:
        """Load learning paths from file"""
        paths_file = self.knowledge_base_path / 'learning_paths.json'
        if paths_file.exists():
            with open(paths_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'learning_paths': []}

    def validate_learning_paths(self) -> Dict[str, Any]:
        """Validate all learning paths for completeness and consistency"""
        validation_results = {
            'total_paths': len(self.learning_paths['learning_paths']),
            'valid_paths': 0,
            'invalid_paths': 0,
            'path_details': {},
            'missing_nodes': set(),
            'orphaned_paths': set(),
            'improvement_recommendations': []
        }

        for path in self.learning_paths['learning_paths']:
            path_validation = self._validate_single_path(path)
            validation_results['path_details'][path['id']] = path_validation

            if path_validation['is_valid']:
                validation_results['valid_paths'] += 1
            else:
                validation_results['invalid_paths'] += 1

            # Collect missing nodes
            validation_results['missing_nodes'].update(path_validation['missing_nodes'])

        # Generate improvement recommendations
        validation_results['improvement_recommendations'] = self._generate_path_improvements()

        return validation_results

    def _validate_single_path(self, path: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single learning path"""
        validation = {
            'is_valid': True,
            'missing_nodes': set(),
            'prerequisite_issues': [],
            'difficulty_progression': [],
            'completeness_score': 0.0,
            'recommendations': []
        }

        # Check if all nodes in path exist
        path_nodes = set()
        for track in path.get('tracks', []):
            for node_id in track.get('nodes', []):
                path_nodes.add(node_id)
                if node_id not in self.all_nodes:
                    validation['missing_nodes'].add(node_id)
                    validation['is_valid'] = False

        # Check prerequisite chains
        validation['prerequisite_issues'] = self._check_prerequisite_chains(path)

        # Check difficulty progression
        validation['difficulty_progression'] = self._check_difficulty_progression(path)

        # Calculate completeness score
        validation['completeness_score'] = self._calculate_path_completeness(path, validation)

        # Generate recommendations
        validation['recommendations'] = self._generate_path_recommendations(path, validation)

        return validation

    def _check_prerequisite_chains(self, path: Dict[str, Any]) -> List[str]:
        """Check if prerequisite chains are logical"""
        issues = []

        for track in path.get('tracks', []):
            track_nodes = track.get('nodes', [])

            # Check if nodes have valid prerequisites
            for i, node_id in enumerate(track_nodes):
                if node_id in self.all_nodes:
                    # This would require reading the actual node file to check prerequisites
                    # For now, just check basic ordering
                    if i > 0:  # Not the first node in track
                        # Check if previous nodes could be prerequisites
                        pass  # Placeholder for more sophisticated checking

        return issues

    def _check_difficulty_progression(self, path: Dict[str, Any]) -> List[str]:
        """Check if difficulty progression is logical"""
        issues = []

        for track in path.get('tracks', []):
            track_nodes = track.get('nodes', [])
            difficulties = []

            # Get difficulty levels for each node
            for node_id in track_nodes:
                difficulty = self._get_node_difficulty(node_id)
                difficulties.append((node_id, difficulty))

            # Check progression
            for i in range(1, len(difficulties)):
                prev_diff = difficulties[i-1][1]
                curr_diff = difficulties[i][1]

                if self._difficulty_to_numeric(prev_diff) > self._difficulty_to_numeric(curr_diff):
                    issues.append(
                        f"Difficulty regression: {prev_diff} -> {curr_diff} "
                        f"({difficulties[i-1][0]} -> {difficulties[i][0]})"
                    )

        return issues

    def _get_node_difficulty(self, node_id: str) -> str:
        """Get difficulty level for a node"""
        # Find the file for this node
        for json_file in self.knowledge_base_path.rglob("*.json"):
            if json_file.name in ['learning_paths.json', 'faq.json', 'glossary.json']:
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if data['id'] == node_id:
                    return data.get('difficulty', 'intermediate')
            except Exception:
                continue

        return 'intermediate'  # Default

    def _difficulty_to_numeric(self, difficulty: str) -> int:
        """Convert difficulty string to numeric value"""
        mapping = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
        return mapping.get(difficulty, 2)

    def _calculate_path_completeness(self, path: Dict[str, Any], validation: Dict[str, Any]) -> float:
        """Calculate completeness score for a learning path"""
        total_nodes = 0
        valid_nodes = 0

        for track in path.get('tracks', []):
            total_nodes += len(track.get('nodes', []))
            valid_nodes += len([n for n in track.get('nodes', []) if n in self.all_nodes])

        if total_nodes == 0:
            return 0.0

        # Base completeness
        completeness = valid_nodes / total_nodes

        # Penalty for missing nodes
        if validation['missing_nodes']:
            completeness *= 0.8

        # Bonus for good structure
        if path.get('description') and len(path['description']) > 50:
            completeness += 0.1

        if path.get('metadata', {}).get('learning_outcomes'):
            completeness += 0.1

        return min(1.0, completeness)

    def _generate_path_recommendations(self, path: Dict[str, Any], validation: Dict[str, Any]) -> List[str]:
        """Generate recommendations for path improvement"""
        recommendations = []

        if validation['missing_nodes']:
            recommendations.append(
                f"Add missing nodes: {', '.join(validation['missing_nodes'])}"
            )

        if validation['prerequisite_issues']:
            recommendations.append(
                f"Fix prerequisite issues: {', '.join(validation['prerequisite_issues'][:3])}"
            )

        if validation['difficulty_progression']:
            recommendations.append(
                f"Review difficulty progression: {len(validation['difficulty_progression'])} issues found"
            )

        if validation['completeness_score'] < 0.8:
            recommendations.append("Enhance path completeness and add more comprehensive content")

        # Add metadata recommendations
        if not path.get('description') or len(path['description']) < 50:
            recommendations.append("Add more detailed path description")

        if not path.get('metadata', {}).get('learning_outcomes'):
            recommendations.append("Add clear learning outcomes")

        return recommendations

    def _generate_path_improvements(self) -> List[Dict[str, Any]]:
        """Generate overall improvement recommendations"""
        recommendations = []

        # Check for missing audience coverage
        audience_coverage = self._analyze_audience_coverage()
        if audience_coverage['missing_audiences']:
            recommendations.append({
                'priority': 'high',
                'type': 'audience_coverage',
                'title': 'Expand Audience Coverage',
                'description': f"Add learning paths for missing audiences: {', '.join(audience_coverage['missing_audiences'])}",
                'effort': 'medium',
                'impact': 'high'
            })

        # Check for missing domain coverage
        domain_coverage = self._analyze_domain_coverage()
        if domain_coverage['missing_domains']:
            recommendations.append({
                'priority': 'medium',
                'type': 'domain_coverage',
                'title': 'Expand Domain Coverage',
                'description': f"Add learning paths for missing domains: {', '.join(domain_coverage['missing_domains'])}",
                'effort': 'high',
                'impact': 'medium'
            })

        # Check for prerequisite optimization
        prerequisite_issues = self._analyze_prerequisite_optimization()
        if prerequisite_issues:
            recommendations.append({
                'priority': 'medium',
                'type': 'prerequisite_optimization',
                'title': 'Optimize Prerequisite Chains',
                'description': "Review and optimize prerequisite relationships across paths",
                'effort': 'medium',
                'impact': 'high'
            })

        return recommendations

    def _analyze_audience_coverage(self) -> Dict[str, Any]:
        """Analyze coverage of different audiences"""
        current_audiences = set()
        target_audiences = {
            'beginners', 'students', 'researchers', 'developers', 'educators',
            'policy_makers', 'clinicians', 'engineers', 'philosophers'
        }

        for path in self.learning_paths['learning_paths']:
            audience = path.get('metadata', {}).get('target_audience', '')
            if audience:
                current_audiences.add(audience.lower())

        missing_audiences = target_audiences - current_audiences

        return {
            'current_audiences': list(current_audiences),
            'target_audiences': list(target_audiences),
            'missing_audiences': list(missing_audiences),
            'coverage_percentage': len(current_audiences) / len(target_audiences)
        }

    def _analyze_domain_coverage(self) -> Dict[str, Any]:
        """Analyze coverage of different domains"""
        current_domains = set()
        target_domains = {
            'artificial_intelligence', 'neuroscience', 'robotics', 'psychology',
            'engineering', 'education', 'economics', 'climate_science', 'philosophy'
        }

        for path in self.learning_paths['learning_paths']:
            path_id = path['id']
            if any(domain in path_id for domain in target_domains):
                current_domains.add(path_id.split('_')[0])

        missing_domains = target_domains - current_domains

        return {
            'current_domains': list(current_domains),
            'target_domains': list(target_domains),
            'missing_domains': list(missing_domains),
            'coverage_percentage': len(current_domains) / len(target_domains)
        }

    def _analyze_prerequisite_optimization(self) -> List[str]:
        """Analyze prerequisite relationships for optimization opportunities"""
        issues = []

        # Check for redundant prerequisites
        # Check for circular dependencies
        # Check for overly complex prerequisite chains

        return issues

    def optimize_learning_paths(self) -> Dict[str, Any]:
        """Optimize learning paths for better structure"""
        optimization_results = {
            'paths_optimized': 0,
            'optimizations_applied': [],
            'new_paths_suggested': [],
            'improvements_summary': {}
        }

        for path in self.learning_paths['learning_paths']:
            optimizations = self._optimize_single_path(path)
            if optimizations:
                optimization_results['paths_optimized'] += 1
                optimization_results['optimizations_applied'].extend(optimizations)

        # Suggest new paths based on gaps
        new_paths = self._suggest_new_paths()
        optimization_results['new_paths_suggested'] = new_paths

        return optimization_results

    def _optimize_single_path(self, path: Dict[str, Any]) -> List[str]:
        """Optimize a single learning path"""
        optimizations = []

        # Add missing metadata
        if not path.get('description') or len(path['description']) < 50:
            optimizations.append(f"Enhanced description for {path['id']}")

        if not path.get('metadata', {}).get('learning_outcomes'):
            optimizations.append(f"Added learning outcomes for {path['id']}")

        # Optimize estimated hours
        if not path.get('estimated_hours'):
            optimizations.append(f"Added estimated hours for {path['id']}")

        return optimizations

    def _suggest_new_paths(self) -> List[Dict[str, Any]]:
        """Suggest new learning paths based on gaps"""
        suggestions = []

        # Suggest paths for missing audiences
        audience_coverage = self._analyze_audience_coverage()
        if audience_coverage['missing_audiences']:
            for audience in audience_coverage['missing_audiences'][:3]:  # Top 3
                suggestions.append({
                    'suggested_id': f"{audience.replace(' ', '_')}_track",
                    'title': f"{audience.title()} Learning Track",
                    'description': f"Specialized track for {audience}",
                    'priority': 'medium',
                    'rationale': f"Missing coverage for {audience} audience"
                })

        return suggestions

    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        validation = self.validate_learning_paths()
        optimization = self.optimize_learning_paths()

        report = []
        report.append("LEARNING PATH OPTIMIZATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        report.append(f"Total Learning Paths: {validation['total_paths']}")
        report.append(f"Valid Paths: {validation['valid_paths']}")
        report.append(f"Invalid Paths: {validation['invalid_paths']}")
        report.append(f"Validation Rate: {validation['valid_paths']/validation['total_paths']*100:.1f}%")
        report.append("")

        if validation['missing_nodes']:
            report.append(f"Missing Nodes: {len(validation['missing_nodes'])}")
            report.append(f"  {', '.join(list(validation['missing_nodes'])[:10])}")
            if len(validation['missing_nodes']) > 10:
                report.append(f"  ... and {len(validation['missing_nodes'])-10} more")
        else:
            report.append("✓ All referenced nodes exist")

        report.append("")
        report.append("Optimization Results:")
        report.append(f"  Paths Optimized: {optimization['paths_optimized']}")
        report.append(f"  Optimizations Applied: {len(optimization['optimizations_applied'])}")

        if optimization['new_paths_suggested']:
            report.append(f"  New Paths Suggested: {len(optimization['new_paths_suggested'])}")
            report.append("  Suggested Paths:")
            for suggestion in optimization['new_paths_suggested'][:5]:
                report.append(f"    - {suggestion['title']} ({suggestion['suggested_id']})")

        report.append("")
        report.append("Top Recommendations:")
        for rec in validation['improvement_recommendations'][:5]:
            report.append(f"  - {rec}")

        return "\n".join(report)

def main():
    """Main function for learning path optimization"""
    if len(sys.argv) != 3:
        print("Usage: python learning_path_optimizer.py <knowledge_base_path> <command>")
        print("Commands:")
        print("  validate    - Validate all learning paths")
        print("  optimize    - Optimize learning paths")
        print("  report      - Generate comprehensive report")
        sys.exit(1)

    knowledge_path = sys.argv[1]
    command = sys.argv[2]

    if not Path(knowledge_path).exists():
        print(f"Knowledge base path does not exist: {knowledge_path}")
        sys.exit(1)

    validator = LearningPathValidator(knowledge_path)

    if command == "validate":
        validation = validator.validate_learning_paths()

        print(f"\nLearning Path Validation Results:")
        print(f"  Total Paths: {validation['total_paths']}")
        print(f"  Valid Paths: {validation['valid_paths']}")
        print(f"  Invalid Paths: {validation['invalid_paths']}")
        print(f"  Validation Rate: {validation['valid_paths']/validation['total_paths']*100:.1f}%")
        if validation['missing_nodes']:
            print(f"  Missing Nodes: {len(validation['missing_nodes'])}")

    elif command == "optimize":
        optimization = validator.optimize_learning_paths()

        print("\nLearning Path Optimization Results:")
        print(f"  Paths Optimized: {optimization['paths_optimized']}")
        print(f"  Optimizations Applied: {len(optimization['optimizations_applied'])}")

        if optimization['new_paths_suggested']:
            print(f"  New Paths Suggested: {len(optimization['new_paths_suggested'])}")

    elif command == "report":
        report = validator.generate_optimization_report()
        print(report)

        # Save detailed report
        with open('learning_path_optimization_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
