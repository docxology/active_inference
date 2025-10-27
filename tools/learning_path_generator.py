#!/usr/bin/env python3
"""
Learning Path Generator
Creates specialized learning paths for different audiences based on audit findings
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class LearningPathGenerator:
    """Generates specialized learning paths for different audiences"""

    def __init__(self, knowledge_base_path: str):
        """Initialize learning path generator"""
        self.knowledge_base_path = Path(knowledge_base_path)
        self.all_nodes = self._load_all_nodes()
        self.existing_paths = self._load_existing_paths()

    def _load_all_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Load all knowledge nodes"""
        nodes = {}

        # Find all JSON files
        json_files = list(self.knowledge_base_path.rglob("*.json"))

        for file_path in json_files:
            if file_path.name in ['learning_paths.json', 'faq.json', 'glossary.json']:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                nodes[data['id']] = data
            except Exception:
                continue

        return nodes

    def _load_existing_paths(self) -> Dict[str, Any]:
        """Load existing learning paths"""
        paths_file = self.knowledge_base_path / 'learning_paths.json'
        if paths_file.exists():
            with open(paths_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'learning_paths': []}

    def generate_specialized_paths(self) -> List[Dict[str, Any]]:
        """Generate specialized learning paths for missing audiences"""

        new_paths = []

        # Generate Beginners Path
        beginners_path = self._create_beginners_path()
        if beginners_path:
            new_paths.append(beginners_path)

        # Generate Policy Makers Path
        policy_path = self._create_policy_makers_path()
        if policy_path:
            new_paths.append(policy_path)

        # Generate Philosophers Path
        philosophy_path = self._create_philosophers_path()
        if philosophy_path:
            new_paths.append(philosophy_path)

        return new_paths

    def _create_beginners_path(self) -> Dict[str, Any]:
        """Create a beginner-friendly learning path"""

        # Find beginner-level foundation nodes
        beginner_nodes = []
        for node_id, node_data in self.all_nodes.items():
            if node_data.get('difficulty') == 'beginner':
                beginner_nodes.append(node_id)

        # Create a gentle introduction path
        path = {
            'id': 'beginners_active_inference_journey',
            'title': 'Beginner\'s Journey into Active Inference',
            'description': 'A gentle introduction to Active Inference concepts for newcomers to the field. Start with foundational ideas and build understanding progressively.',
            'difficulty': 'beginner',
            'estimated_hours': 20,
            'metadata': {
                'target_audience': 'beginners',
                'learning_outcomes': [
                    'Understand basic Active Inference concepts',
                    'Recognize the relationship between perception and action',
                    'Appreciate the mathematical foundations',
                    'See real-world applications'
                ],
                'prerequisites': 'None - designed for complete beginners',
                'pace': 'gentle',
                'focus': 'conceptual understanding'
            },
            'tracks': [
                {
                    'id': 'foundational_concepts',
                    'title': 'Building Foundations',
                    'description': 'Start with the basic ideas that underpin Active Inference',
                    'estimated_hours': 8,
                    'nodes': [
                        'active_inference_introduction',
                        'bayesian_basics',
                        'info_theory_entropy',
                        'fep_introduction'
                    ]
                },
                {
                    'id': 'core_mechanisms',
                    'title': 'Core Mechanisms',
                    'description': 'Explore the key mechanisms of Active Inference',
                    'estimated_hours': 7,
                    'nodes': [
                        'belief_updating',
                        'ai_policy_selection',
                        'perception'
                    ]
                },
                {
                    'id': 'applications',
                    'title': 'Real-World Applications',
                    'description': 'See how Active Inference applies to real problems',
                    'estimated_hours': 5,
                    'nodes': [
                        'decision_making',
                        'neuroscience_perception'
                    ]
                }
            ]
        }

        return path

    def _create_policy_makers_path(self) -> Dict[str, Any]:
        """Create a policy-focused learning path"""

        path = {
            'id': 'active_inference_for_policy_makers',
            'title': 'Active Inference for Policy Makers',
            'description': 'Understanding Active Inference for informed policy decisions in science, technology, and governance.',
            'difficulty': 'intermediate',
            'estimated_hours': 15,
            'metadata': {
                'target_audience': 'policy_makers',
                'learning_outcomes': [
                    'Understand Active Inference implications for policy',
                    'Evaluate AI safety and governance frameworks',
                    'Assess neuroscience and psychology applications',
                    'Make informed decisions about research funding'
                ],
                'prerequisites': 'Basic understanding of science policy',
                'focus': 'policy implications',
                'emphasis': 'decision-making frameworks'
            },
            'tracks': [
                {
                    'id': 'foundations',
                    'title': 'Theoretical Foundations',
                    'description': 'Understand the core theory of Active Inference',
                    'estimated_hours': 5,
                    'nodes': [
                        'active_inference_introduction',
                        'fep_introduction',
                        'fep_mathematical_formulation'
                    ]
                },
                {
                    'id': 'policy_applications',
                    'title': 'Policy Applications',
                    'description': 'Explore applications relevant to policy makers',
                    'estimated_hours': 7,
                    'nodes': [
                        'ai_alignment',
                        'ai_safety',
                        'climate_decision_making',
                        'behavioral_economics'
                    ]
                },
                {
                    'id': 'future_implications',
                    'title': 'Future Implications',
                    'description': 'Consider future policy challenges and opportunities',
                    'estimated_hours': 3,
                    'nodes': [
                        'multi_agent_systems',
                        'decision_theory'
                    ]
                }
            ]
        }

        return path

    def _create_philosophers_path(self) -> Dict[str, Any]:
        """Create a philosophy-focused learning path"""

        path = {
            'id': 'philosophical_foundations_active_inference',
            'title': 'Philosophical Foundations of Active Inference',
            'description': 'Explore the deep philosophical implications of Active Inference for understanding mind, consciousness, and intelligence.',
            'difficulty': 'advanced',
            'estimated_hours': 25,
            'metadata': {
                'target_audience': 'philosophers',
                'learning_outcomes': [
                    'Understand philosophical implications of Active Inference',
                    'Connect to theories of mind and consciousness',
                    'Explore epistemology and ontology of intelligence',
                    'Critique traditional philosophical frameworks'
                ],
                'prerequisites': 'Philosophy background recommended',
                'focus': 'philosophical analysis',
                'emphasis': 'fundamental questions'
            },
            'tracks': [
                {
                    'id': 'epistemology',
                    'title': 'Epistemological Foundations',
                    'description': 'How Active Inference changes our understanding of knowledge and belief',
                    'estimated_hours': 8,
                    'nodes': [
                        'bayesian_inference',
                        'belief_updating',
                        'information_bottleneck',
                        'empirical_bayes'
                    ]
                },
                {
                    'id': 'philosophy_of_mind',
                    'title': 'Philosophy of Mind',
                    'description': 'Active Inference perspectives on consciousness and cognition',
                    'estimated_hours': 10,
                    'nodes': [
                        'fep_biological_systems',
                        'neural_dynamics',
                        'hierarchical_models',
                        'perception'
                    ]
                },
                {
                    'id': 'metaphysical_questions',
                    'title': 'Metaphysical Questions',
                    'description': 'Fundamental questions about reality and intelligence',
                    'estimated_hours': 7,
                    'nodes': [
                        'information_geometry',
                        'causal_inference',
                        'optimal_control'
                    ]
                }
            ]
        }

        return path

    def validate_and_save_paths(self, new_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate new paths and save to learning paths file"""

        results = {
            'paths_validated': 0,
            'paths_added': 0,
            'validation_errors': [],
            'saved_successfully': False
        }

        # Validate each path
        validated_paths = []
        for path in new_paths:
            validation = self._validate_path(path)
            if validation['valid']:
                validated_paths.append(path)
                results['paths_validated'] += 1
            else:
                results['validation_errors'].extend(validation['errors'])

        if validated_paths:
            # Add to existing paths
            self.existing_paths['learning_paths'].extend(validated_paths)
            results['paths_added'] = len(validated_paths)

            # Save updated learning paths
            try:
                paths_file = self.knowledge_base_path / 'learning_paths.json'
                with open(paths_file, 'w', encoding='utf-8') as f:
                    json.dump(self.existing_paths, f, indent=2, ensure_ascii=False)
                results['saved_successfully'] = True
            except Exception as e:
                results['validation_errors'].append(f"Save failed: {str(e)}")

        return results

    def _validate_path(self, path: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a learning path"""

        validation = {
            'valid': True,
            'errors': []
        }

        # Check required fields
        required_fields = ['id', 'title', 'description', 'tracks']
        for field in required_fields:
            if field not in path:
                validation['errors'].append(f"Missing required field: {field}")
                validation['valid'] = False

        # Check tracks
        if 'tracks' in path:
            for track in path['tracks']:
                if 'nodes' in track:
                    for node_id in track['nodes']:
                        if node_id not in self.all_nodes:
                            validation['errors'].append(f"Referenced node does not exist: {node_id}")
                            validation['valid'] = False

        return validation

def main():
    """Main function for learning path generation"""

    knowledge_path = "knowledge"

    if not Path(knowledge_path).exists():
        print(f"Knowledge base path does not exist: {knowledge_path}")
        return

    generator = LearningPathGenerator(knowledge_path)

    print("🎓 LEARNING PATH GENERATOR")
    print("=" * 35)

    # Generate specialized paths
    new_paths = generator.generate_specialized_paths()

    print(f"📋 Generated {len(new_paths)} specialized learning paths:")
    for path in new_paths:
        print(f"  • {path['title']} (for {path['metadata']['target_audience']})")

    # Validate and save
    results = generator.validate_and_save_paths(new_paths)

    print("\n📊 VALIDATION RESULTS")
    print("-" * 25)
    print(f"Paths Validated: {results['paths_validated']}")
    print(f"Paths Added: {results['paths_added']}")

    if results['validation_errors']:
        print(f"Validation Errors: {len(results['validation_errors'])}")
        for error in results['validation_errors'][:5]:  # Show first 5
            print(f"  ✗ {error}")

    if results['saved_successfully']:
        print("✅ LEARNING PATHS SUCCESSFULLY ADDED")
        print(f"Total learning paths now: {len(generator.existing_paths['learning_paths'])}")
    else:
        print("❌ FAILED TO SAVE LEARNING PATHS")

    # Summary
    if results['paths_added'] > 0:
        print("\n🎯 SPECIALIZED LEARNING PATHS CREATED")
        print("The knowledge base now supports:")
        for path in new_paths:
            audience = path['metadata']['target_audience']
            print(f"  • {audience.title()} with {len(path['tracks'])} learning tracks")
    else:
        print("\n⚠️  NO NEW PATHS WERE ADDED")
        print("Check validation errors above")

if __name__ == "__main__":
    main()
