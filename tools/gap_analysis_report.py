#!/usr/bin/env python3
"""
Comprehensive Gap Analysis Report
Analyzes knowledge base audit results to identify gaps and prioritize improvements

This script processes the validation results and creates a detailed gap analysis
including missing content, quality issues, and enhancement priorities.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GapAnalyzer:
    """Analyzes gaps in the knowledge base based on audit results"""

    def __init__(self, audit_report_path: str):
        """Initialize gap analyzer with audit report"""
        with open(audit_report_path, 'r', encoding='utf-8') as f:
            self.audit_data = json.load(f)

        self.summary = self.audit_data.get('summary', {})
        self.cross_ref = self.audit_data.get('cross_reference_analysis', {})
        self.detailed_results = self.audit_data.get('detailed_results', {})

    def analyze_content_gaps(self) -> Dict[str, Any]:
        """Analyze content gaps by category and type"""
        gaps = {
            'content_structure_gaps': {},
            'content_completeness_gaps': {},
            'metadata_gaps': {},
            'cross_reference_gaps': {},
            'learning_path_gaps': {}
        }

        # Analyze content structure gaps
        gaps['content_structure_gaps'] = self._analyze_content_structure_gaps()

        # Analyze content completeness gaps
        gaps['content_completeness_gaps'] = self._analyze_content_completeness_gaps()

        # Analyze metadata gaps
        gaps['metadata_gaps'] = self._analyze_metadata_gaps()

        # Analyze cross-reference gaps
        gaps['cross_reference_gaps'] = self._analyze_cross_reference_gaps()

        # Analyze learning path gaps
        gaps['learning_path_gaps'] = self._analyze_learning_path_gaps()

        return gaps

    def _analyze_content_structure_gaps(self) -> Dict[str, Any]:
        """Analyze gaps in content structure"""
        structure_gaps = {
            'missing_overview_sections': 0,
            'short_content_sections': defaultdict(int),
            'missing_examples': 0,
            'missing_exercises': 0,
            'files_with_issues': []
        }

        for file_path, result in self.detailed_results.items():
            if result['warnings']:
                structure_gaps['files_with_issues'].append({
                    'file': file_path,
                    'warnings': result['warnings'],
                    'score': result['score']
                })

                # Count specific warning types
                for warning in result['warnings']:
                    if 'overview' in warning.lower():
                        structure_gaps['missing_overview_sections'] += 1
                    elif 'examples' in warning.lower():
                        structure_gaps['missing_examples'] += 1
                    elif 'exercises' in warning.lower():
                        structure_gaps['missing_exercises'] += 1
                    elif 'very short' in warning.lower():
                        section_name = warning.split("'")[1]
                        structure_gaps['short_content_sections'][section_name] += 1

        return structure_gaps

    def _analyze_content_completeness_gaps(self) -> Dict[str, Any]:
        """Analyze gaps in content completeness"""
        completeness_gaps = {
            'low_score_files': [],
            'score_distribution': defaultdict(int),
            'content_type_scores': defaultdict(list),
            'improvement_priorities': []
        }

        for file_path, result in self.detailed_results.items():
            score = result['score']
            completeness_gaps['score_distribution'][self._bucket_score(score)] += 1

            # Extract content type from file path
            content_type = self._extract_content_type(file_path)
            completeness_gaps['content_type_scores'][content_type].append(score)

            if score < 0.8:
                completeness_gaps['low_score_files'].append({
                    'file': file_path,
                    'score': score,
                    'content_type': content_type
                })

        # Sort low score files by priority
        completeness_gaps['improvement_priorities'] = sorted(
            completeness_gaps['low_score_files'],
            key=lambda x: (x['score'], x['content_type'])
        )

        return completeness_gaps

    def _analyze_metadata_gaps(self) -> Dict[str, Any]:
        """Analyze gaps in metadata completeness"""
        metadata_gaps = {
            'missing_metadata_fields': defaultdict(int),
            'incomplete_metadata_files': [],
            'metadata_completeness_scores': []
        }

        # Sample a few files to check metadata structure
        sample_files = list(self.detailed_results.keys())[:10]

        for file_path in sample_files:
            # Read actual file to check metadata
            full_path = Path('knowledge') / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if 'metadata' in data:
                        metadata = data['metadata']
                        expected_fields = ['estimated_reading_time', 'author', 'version', 'last_updated']

                        for field in expected_fields:
                            if field not in metadata:
                                metadata_gaps['missing_metadata_fields'][field] += 1

                        # Calculate completeness score for this file
                        completeness = sum(1 for field in expected_fields if field in metadata) / len(expected_fields)
                        metadata_gaps['metadata_completeness_scores'].append(completeness)

                        if completeness < 1.0:
                            metadata_gaps['incomplete_metadata_files'].append({
                                'file': file_path,
                                'missing_fields': [f for f in expected_fields if f not in metadata],
                                'completeness': completeness
                            })

                except Exception as e:
                    logger.warning(f"Error reading {full_path}: {e}")

        return metadata_gaps

    def _analyze_cross_reference_gaps(self) -> Dict[str, Any]:
        """Analyze gaps in cross-references"""
        cross_gaps = {
            'orphaned_nodes_analysis': {},
            'isolated_nodes_analysis': {},
            'reference_quality': {}
        }

        orphaned = self.cross_ref['orphaned_nodes']
        isolated = self.cross_ref['isolated_nodes']

        # Analyze orphaned nodes by type
        cross_gaps['orphaned_nodes_analysis'] = self._categorize_nodes(orphaned)

        # Analyze isolated nodes
        cross_gaps['isolated_nodes_analysis'] = self._categorize_nodes(isolated)

        # Reference quality metrics
        cross_gaps['reference_quality'] = {
            'connectivity_ratio': 1 - (len(isolated) / self.cross_ref['total_nodes']),
            'orphaned_ratio': len(orphaned) / self.cross_ref['total_nodes'],
            'avg_prerequisites': self.cross_ref['reference_statistics']['avg_prerequisites'],
            'max_prerequisites': self.cross_ref['reference_statistics']['max_prerequisites']
        }

        return cross_gaps

    def _analyze_learning_path_gaps(self) -> Dict[str, Any]:
        """Analyze gaps in learning paths"""
        learning_gaps = {
            'missing_prerequisites': [],
            'broken_paths': [],
            'coverage_gaps': {},
            'path_completeness': {}
        }

        # Load learning paths
        try:
            with open('knowledge/learning_paths.json', 'r', encoding='utf-8') as f:
                learning_paths = json.load(f)

            # Check each learning path
            for path in learning_paths['learning_paths']:
                path_issues = self._validate_learning_path(path)
                if path_issues:
                    learning_gaps['broken_paths'].append({
                        'path_id': path['id'],
                        'path_name': path['name'],
                        'issues': path_issues
                    })

        except Exception as e:
            logger.warning(f"Error analyzing learning paths: {e}")

        return learning_gaps

    def _categorize_nodes(self, node_ids: List[str]) -> Dict[str, int]:
        """Categorize nodes by content type"""
        categories = defaultdict(int)

        for node_id in node_ids:
            # Find the file for this node
            for file_path, result in self.detailed_results.items():
                if file_path.endswith(f'{node_id}.json'):
                    content_type = self._extract_content_type(file_path)
                    categories[content_type] += 1
                    break

        return dict(categories)

    def _extract_content_type(self, file_path: str) -> str:
        """Extract content type from file path"""
        parts = file_path.split('/')
        if 'foundations' in parts:
            return 'foundation'
        elif 'mathematics' in parts:
            return 'mathematics'
        elif 'implementations' in parts:
            return 'implementation'
        elif 'applications' in parts:
            return 'application'
        else:
            return 'other'

    def _bucket_score(self, score: float) -> str:
        """Bucket scores into categories"""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'good'
        elif score >= 0.7:
            return 'fair'
        elif score >= 0.6:
            return 'poor'
        else:
            return 'critical'

    def _validate_learning_path(self, path: Dict[str, Any]) -> List[str]:
        """Validate a single learning path"""
        issues = []

        # Check if all nodes in path exist
        tracks = path.get('tracks', [])
        if isinstance(tracks, list):
            for track in tracks:
                if isinstance(track, dict):
                    for node_id in track.get('nodes', []):
                        if hasattr(self.cross_ref, 'get') and self.cross_ref.get('total_ids'):
                            if node_id not in self.cross_ref['total_ids']:
                                issues.append(f"Missing node: {node_id}")

        # Check prerequisites
        path_nodes = set()
        for track in path.get('tracks', []):
            path_nodes.update(track.get('nodes', []))

        # This would need more sophisticated validation
        # For now, just check basic structure

        return issues

    def generate_priority_recommendations(self, gaps: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations for improvement"""
        recommendations = []

        # High priority: Fix broken cross-references
        if gaps['cross_reference_gaps']['reference_quality']['orphaned_ratio'] > 0.1:
            recommendations.append({
                'priority': 'high',
                'category': 'cross_references',
                'title': 'Fix Orphaned Knowledge Nodes',
                'description': f"Connect {len(self.cross_ref['orphaned_nodes'])} orphaned nodes to improve knowledge connectivity",
                'effort': 'medium',
                'impact': 'high',
                'affected_nodes': self.cross_ref['orphaned_nodes']
            })

        # High priority: Improve low-scoring content
        low_score_count = len(gaps['content_completeness_gaps']['improvement_priorities'])
        if low_score_count > 0:
            recommendations.append({
                'priority': 'high',
                'category': 'content_quality',
                'title': 'Enhance Low-Quality Content',
                'description': f"Improve {low_score_count} files with quality scores below 80%",
                'effort': 'high',
                'impact': 'high',
                'affected_files': [item['file'] for item in gaps['content_completeness_gaps']['improvement_priorities'][:10]]
            })

        # Medium priority: Add missing metadata
        missing_metadata = gaps['metadata_gaps']['missing_metadata_fields']
        if any(count > 0 for count in missing_metadata.values()):
            recommendations.append({
                'priority': 'medium',
                'category': 'metadata',
                'title': 'Complete Missing Metadata',
                'description': f"Add missing metadata fields to improve discoverability and quality tracking",
                'effort': 'low',
                'impact': 'medium',
                'missing_fields': missing_metadata
            })

        # Medium priority: Fix short content sections
        short_sections = gaps['content_structure_gaps']['short_content_sections']
        if len(short_sections) > 20:
            recommendations.append({
                'priority': 'medium',
                'category': 'content_structure',
                'title': 'Expand Short Content Sections',
                'description': f"Expand {len(short_sections)} content sections that are too brief",
                'effort': 'medium',
                'impact': 'medium',
                'short_sections': short_sections
            })

        # Low priority: Add missing examples and exercises
        if gaps['content_structure_gaps']['missing_examples'] > 10:
            recommendations.append({
                'priority': 'low',
                'category': 'content_enhancement',
                'title': 'Add Missing Examples',
                'description': f"Add practical examples to {gaps['content_structure_gaps']['missing_examples']} files",
                'effort': 'medium',
                'impact': 'medium'
            })

        return recommendations

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive gap analysis report"""
        gaps = self.analyze_content_gaps()
        recommendations = self.generate_priority_recommendations(gaps)

        report = {
            'analysis_timestamp': self.audit_data['audit_timestamp'],
            'summary': self.summary,
            'gaps_analysis': gaps,
            'recommendations': recommendations,
            'priority_matrix': self._create_priority_matrix(recommendations),
            'implementation_plan': self._create_implementation_plan(recommendations)
        }

        return report

    def _create_priority_matrix(self, recommendations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Create priority matrix for implementation"""
        matrix = {'high': [], 'medium': [], 'low': []}

        for rec in recommendations:
            priority = rec['priority']
            matrix[priority].append(rec)

        return matrix

    def _create_implementation_plan(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create phased implementation plan"""
        plan = []

        # Phase 1: High priority items (Week 1-2)
        high_priority = [r for r in recommendations if r['priority'] == 'high']
        plan.append({
            'phase': 1,
            'title': 'Critical Fixes and High-Impact Improvements',
            'duration': '1-2 weeks',
            'items': high_priority,
            'success_criteria': 'All critical issues resolved, quality scores improved by 10%'
        })

        # Phase 2: Medium priority items (Week 3-4)
        medium_priority = [r for r in recommendations if r['priority'] == 'medium']
        plan.append({
            'phase': 2,
            'title': 'Content Enhancement and Metadata Completion',
            'duration': '1-2 weeks',
            'items': medium_priority,
            'success_criteria': 'Metadata completeness >95%, all short sections expanded'
        })

        # Phase 3: Low priority items (Week 5-6)
        low_priority = [r for r in recommendations if r['priority'] == 'low']
        plan.append({
            'phase': 3,
            'title': 'Quality Polish and Enhancement',
            'duration': '1-2 weeks',
            'items': low_priority,
            'success_criteria': 'All examples and exercises added, final quality audit >95%'
        })

        return plan

def main():
    """Main function for gap analysis"""
    if len(sys.argv) != 2:
        print("Usage: python gap_analysis_report.py <audit_report.json>")
        sys.exit(1)

    audit_report_path = sys.argv[1]

    if not Path(audit_report_path).exists():
        print(f"Audit report not found: {audit_report_path}")
        sys.exit(1)

    # Analyze gaps
    analyzer = GapAnalyzer(audit_report_path)
    report = analyzer.generate_comprehensive_report()

    # Save detailed report
    output_file = "knowledge_gap_analysis_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*70)
    print("COMPREHENSIVE KNOWLEDGE BASE GAP ANALYSIS REPORT")
    print("="*70)
    print(f"Analysis Date: {report['analysis_timestamp']}")

    summary = report['summary']
    print(f"\nOverall Status:")
    print(f"  Total Files: {summary['total_files']}")
    print(f"  Validation Rate: {summary['validation_rate']:.1%}")
    print(f"  Average Quality Score: {summary['average_score']:.1%}")
    print(f"  Knowledge Nodes: {summary['knowledge_nodes']}")

    cross_ref = report.get('cross_reference_analysis', {})
    print(f"\nCross-Reference Analysis:")
    print(f"  Orphaned Nodes: {len(cross_ref.get('orphaned_nodes', []))}")
    print(f"  Isolated Nodes: {len(cross_ref.get('isolated_nodes', []))}")
    print(f"  Avg Prerequisites: {cross_ref.get('reference_statistics', {}).get('avg_prerequisites', 0):.1f}")

    # Priority recommendations
    print(f"\nTop Priority Recommendations:")
    priorities = report.get('priority_matrix', {})

    for priority, items in priorities.items():
        if items:
            print(f"\n  {priority.upper()} PRIORITY ({len(items)} items):")
            for i, item in enumerate(items[:3], 1):  # Show top 3 per priority
                print(f"    {i}. {item['title']}")
                print(f"       Impact: {item['impact']}, Effort: {item['effort']}")

    # Implementation phases
    print("\nImplementation Plan:")
    implementation_plan = report.get('implementation_plan', [])
    for phase in implementation_plan:
        print(f"\n  Phase {phase['phase']}: {phase['title']}")
        print(f"    Duration: {phase['duration']}")
        print(f"    Items: {len(phase['items'])}")
        print(f"    Success: {phase['success_criteria']}")

    print(f"\nDetailed gap analysis saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
