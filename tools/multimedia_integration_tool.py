#!/usr/bin/env python3
"""
Multimedia Integration Tool for Active Inference Knowledge Base

This tool integrates diagrams, visualizations, and multimedia content
into the educational materials to enhance learning and understanding.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MultimediaIntegrationTool:
    """Integrates multimedia content into knowledge nodes"""

    def __init__(self, knowledge_base_path: str = "knowledge", media_path: str = "media"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.media_path = Path(media_path)
        self.media_path.mkdir(exist_ok=True)

    def integrate_multimedia_for_node(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate multimedia content for a knowledge node

        Args:
            node_data: Knowledge node data

        Returns:
            Enhanced node data with multimedia content
        """
        node_id = node_data.get('id', '')
        content_type = node_data.get('content_type', 'foundation')
        title = node_data.get('title', '')

        # Initialize multimedia sections if they don't exist
        if 'multimedia' not in node_data:
            node_data['multimedia'] = {}

        multimedia = node_data['multimedia']

        # Add diagrams based on content type
        if content_type == 'foundation':
            multimedia['diagrams'] = self._generate_concept_diagrams(node_data)
        elif content_type == 'mathematics':
            multimedia['diagrams'] = self._generate_mathematical_diagrams(node_data)
        elif content_type == 'implementation':
            multimedia['diagrams'] = self._generate_implementation_diagrams(node_data)
        elif content_type == 'application':
            multimedia['diagrams'] = self._generate_application_diagrams(node_data)

        # Add animations for dynamic concepts
        if self._needs_animation(node_data):
            multimedia['animations'] = self._generate_animations(node_data)

        # Add interactive visualizations
        if self._needs_interactive_viz(node_data):
            multimedia['interactive_visualizations'] = self._generate_interactive_visualizations(node_data)

        # Add video content references
        multimedia['videos'] = self._generate_video_references(node_data)

        # Add image galleries
        multimedia['images'] = self._generate_image_gallery(node_data)

        # Add audio explanations for complex topics
        if self._needs_audio_explanation(node_data):
            multimedia['audio'] = self._generate_audio_content(node_data)

        # Update metadata to reflect multimedia integration
        if 'metadata' not in node_data:
            node_data['metadata'] = {}

        node_data['metadata']['multimedia_integrated'] = True
        node_data['metadata']['multimedia_count'] = sum(len(v) if isinstance(v, list) else 1 for v in multimedia.values() if v)

        return node_data

    def _generate_concept_diagrams(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate concept diagrams for foundation content"""
        diagrams = []
        title = node_data.get('title', '')
        node_id = node_data.get('id', '')

        # Main concept diagram
        diagrams.append({
            'type': 'concept_map',
            'title': f'{title} Concept Map',
            'description': f'Visual representation of {title.lower()} and its relationships',
            'file_path': f'diagrams/{node_id}_concept_map.svg',
            'format': 'svg',
            'interactive': True,
            'elements': self._extract_concept_elements(node_data)
        })

        # Relationship diagram
        if len(node_data.get('prerequisites', [])) > 0 or len(node_data.get('tags', [])) > 3:
            diagrams.append({
                'type': 'relationship_graph',
                'title': f'{title} Relationships',
                'description': f'Connections between {title.lower()} and related concepts',
                'file_path': f'diagrams/{node_id}_relationships.svg',
                'format': 'svg',
                'interactive': True,
                'connections': self._extract_relationships(node_data)
            })

        # Process flow diagram for dynamic concepts
        if any(keyword in title.lower() for keyword in ['process', 'cycle', 'flow', 'inference']):
            diagrams.append({
                'type': 'flow_diagram',
                'title': f'{title} Process Flow',
                'description': f'Step-by-step process of {title.lower()}',
                'file_path': f'diagrams/{node_id}_flow.svg',
                'format': 'svg',
                'interactive': True,
                'steps': self._extract_process_steps(node_data)
            })

        return diagrams

    def _generate_mathematical_diagrams(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mathematical diagrams and visualizations"""
        diagrams = []
        title = node_data.get('title', '')
        node_id = node_data.get('id', '')

        # Formula visualization
        diagrams.append({
            'type': 'formula_diagram',
            'title': f'{title} Mathematical Formulation',
            'description': f'Visual representation of the mathematical concepts in {title.lower()}',
            'file_path': f'math/{node_id}_formula.svg',
            'format': 'svg',
            'interactive': False,
            'equations': self._extract_mathematical_equations(node_data)
        })

        # Geometric interpretation
        if any(keyword in title.lower() for keyword in ['geometry', 'space', 'manifold', 'metric']):
            diagrams.append({
                'type': 'geometric_visualization',
                'title': f'{title} Geometric Interpretation',
                'description': f'Geometric visualization of {title.lower()} concepts',
                'file_path': f'math/{node_id}_geometry.svg',
                'format': 'svg',
                'interactive': True,
                'dimensions': self._determine_geometric_dimensions(node_data)
            })

        # Probability distribution plots
        if any(keyword in title.lower() for keyword in ['probability', 'distribution', 'entropy', 'kl']):
            diagrams.append({
                'type': 'probability_plot',
                'title': f'{title} Probability Distributions',
                'description': f'Visual plots of probability distributions related to {title.lower()}',
                'file_path': f'math/{node_id}_distributions.svg',
                'format': 'svg',
                'interactive': True,
                'distributions': self._generate_distribution_examples(node_data)
            })

        # Convergence plots for iterative methods
        if any(keyword in title.lower() for keyword in ['variational', 'optimization', 'inference']):
            diagrams.append({
                'type': 'convergence_plot',
                'title': f'{title} Convergence Behavior',
                'description': f'Visualization of convergence properties in {title.lower()}',
                'file_path': f'math/{node_id}_convergence.svg',
                'format': 'svg',
                'interactive': True,
                'metrics': ['free_energy', 'kl_divergence', 'log_likelihood']
            })

        return diagrams

    def _generate_implementation_diagrams(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation diagrams and code visualizations"""
        diagrams = []
        title = node_data.get('title', '')
        node_id = node_data.get('id', '')

        # Algorithm flowchart
        diagrams.append({
            'type': 'algorithm_flowchart',
            'title': f'{title} Algorithm Flow',
            'description': f'Visual flowchart of the {title.lower()} algorithm',
            'file_path': f'code/{node_id}_flowchart.svg',
            'format': 'svg',
            'interactive': True,
            'steps': self._extract_algorithm_steps(node_data)
        })

        # Data flow diagram
        diagrams.append({
            'type': 'data_flow',
            'title': f'{title} Data Flow',
            'description': f'Data flow and transformation in {title.lower()} implementation',
            'file_path': f'code/{node_id}_dataflow.svg',
            'format': 'svg',
            'interactive': True,
            'data_elements': self._extract_data_elements(node_data)
        })

        # Performance comparison charts
        diagrams.append({
            'type': 'performance_chart',
            'title': f'{title} Performance Comparison',
            'description': f'Performance comparison of different {title.lower()} implementations',
            'file_path': f'code/{node_id}_performance.svg',
            'format': 'svg',
            'interactive': True,
            'metrics': ['time_complexity', 'space_complexity', 'accuracy', 'convergence_speed']
        })

        return diagrams

    def _generate_application_diagrams(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate application diagrams and case study visualizations"""
        diagrams = []
        title = node_data.get('title', '')
        node_id = node_data.get('id', '')

        # Application architecture diagram
        diagrams.append({
            'type': 'architecture_diagram',
            'title': f'{title} System Architecture',
            'description': f'High-level architecture of {title.lower()} application',
            'file_path': f'apps/{node_id}_architecture.svg',
            'format': 'svg',
            'interactive': True,
            'components': self._extract_architecture_components(node_data)
        })

        # Use case diagram
        diagrams.append({
            'type': 'use_case_diagram',
            'title': f'{title} Use Cases',
            'description': f'Use cases and scenarios for {title.lower()} application',
            'file_path': f'apps/{node_id}_usecases.svg',
            'format': 'svg',
            'interactive': False,
            'actors': self._extract_use_case_actors(node_data),
            'scenarios': self._extract_use_case_scenarios(node_data)
        })

        # Results visualization
        diagrams.append({
            'type': 'results_visualization',
            'title': f'{title} Results and Outcomes',
            'description': f'Visualization of results and outcomes from {title.lower()} application',
            'file_path': f'apps/{node_id}_results.svg',
            'format': 'svg',
            'interactive': True,
            'metrics': self._extract_result_metrics(node_data)
        })

        return diagrams

    def _generate_animations(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate animations for dynamic concepts"""
        animations = []
        title = node_data.get('title', '')
        node_id = node_data.get('id', '')

        # Process animations
        if any(keyword in title.lower() for keyword in ['inference', 'learning', 'optimization', 'control']):
            animations.append({
                'type': 'process_animation',
                'title': f'{title} Process Animation',
                'description': f'Animated visualization of {title.lower()} process over time',
                'file_path': f'animations/{node_id}_process.mp4',
                'format': 'mp4',
                'duration': 30,
                'frames': 900,  # 30 seconds at 30fps
                'keyframes': self._generate_animation_keyframes(node_data)
            })

        # Concept evolution animations
        if any(keyword in title.lower() for keyword in ['belief', 'probability', 'distribution']):
            animations.append({
                'type': 'evolution_animation',
                'title': f'{title} Evolution Over Time',
                'description': f'Animation showing how {title.lower()} evolves during inference',
                'file_path': f'animations/{node_id}_evolution.mp4',
                'format': 'mp4',
                'duration': 20,
                'frames': 600,
                'parameters': ['time', 'evidence', 'uncertainty']
            })

        return animations

    def _generate_interactive_visualizations(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate interactive visualizations"""
        visualizations = []
        title = node_data.get('title', '')
        node_id = node_data.get('id', '')

        # Parameter exploration
        if any(keyword in title.lower() for keyword in ['model', 'system', 'process']):
            visualizations.append({
                'type': 'parameter_explorer',
                'title': f'{title} Parameter Explorer',
                'description': f'Interactive exploration of {title.lower()} parameters and their effects',
                'file_path': f'interactive/{node_id}_explorer.html',
                'format': 'html',
                'interactive_elements': ['sliders', 'dropdowns', 'plots'],
                'parameters': self._extract_explorable_parameters(node_data)
            })

        # Simulation interface
        if any(keyword in title.lower() for keyword in ['simulation', 'model', 'inference']):
            visualizations.append({
                'type': 'simulation_interface',
                'title': f'{title} Simulation Interface',
                'description': f'Interactive simulation of {title.lower()} dynamics',
                'file_path': f'interactive/{node_id}_simulation.html',
                'format': 'html',
                'interactive_elements': ['play_pause', 'reset', 'parameter_controls'],
                'simulation_parameters': self._extract_simulation_parameters(node_data)
            })

        return visualizations

    def _generate_video_references(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate video content references"""
        videos = []
        title = node_data.get('title', '')
        content_type = node_data.get('content_type', '')

        # Educational videos
        videos.append({
            'type': 'educational_video',
            'title': f'Introduction to {title}',
            'description': f'Comprehensive introduction to {title.lower()} concepts',
            'platform': 'youtube',
            'url': f'https://youtube.com/watch?v={self._generate_video_id(node_data, "intro")}',
            'duration': 600,  # 10 minutes
            'level': 'beginner'
        })

        # Advanced tutorials
        if content_type in ['mathematics', 'implementation']:
            videos.append({
                'type': 'tutorial_video',
                'title': f'{title} Deep Dive',
                'description': f'Detailed technical tutorial on {title.lower()}',
                'platform': 'youtube',
                'url': f'https://youtube.com/watch?v={self._generate_video_id(node_data, "deepdive")}',
                'duration': 1800,  # 30 minutes
                'level': 'advanced'
            })

        # Case studies
        if content_type == 'application':
            videos.append({
                'type': 'case_study_video',
                'title': f'{title} Case Study',
                'description': f'Real-world application of {title.lower()}',
                'platform': 'youtube',
                'url': f'https://youtube.com/watch?v={self._generate_video_id(node_data, "casestudy")}',
                'duration': 900,  # 15 minutes
                'level': 'intermediate'
            })

        return videos

    def _generate_image_gallery(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate image gallery for visual content"""
        images = []
        node_id = node_data.get('id', '')

        # Key concept illustrations
        images.append({
            'type': 'concept_illustration',
            'title': 'Key Concept Visualization',
            'description': 'Visual representation of the main concept',
            'file_path': f'images/{node_id}_concept.svg',
            'format': 'svg',
            'alt_text': f'Visual illustration of {node_data.get("title", "concept")}'
        })

        # Formula visualizations
        if node_data.get('content_type') == 'mathematics':
            images.append({
                'type': 'formula_visualization',
                'title': 'Mathematical Formula',
                'description': 'Visual representation of key mathematical formulas',
                'file_path': f'images/{node_id}_formula.svg',
                'format': 'svg',
                'alt_text': f'Mathematical notation for {node_data.get("title", "concept")}'
            })

        # Example visualizations
        images.append({
            'type': 'example_visualization',
            'title': 'Practical Example',
            'description': 'Visual representation of a practical example',
            'file_path': f'images/{node_id}_example.svg',
            'format': 'svg',
            'alt_text': f'Example illustration for {node_data.get("title", "concept")}'
        })

        return images

    def _generate_audio_content(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate audio content for complex topics"""
        audio = []
        title = node_data.get('title', '')
        node_id = node_data.get('id', '')

        # For complex mathematical or theoretical content
        if node_data.get('difficulty') in ['advanced', 'expert'] or node_data.get('content_type') == 'mathematics':
            audio.append({
                'type': 'explanation_audio',
                'title': f'{title} Audio Explanation',
                'description': f'Audio explanation of {title.lower()} concepts',
                'file_path': f'audio/{node_id}_explanation.mp3',
                'format': 'mp3',
                'duration': 300,  # 5 minutes
                'language': 'en',
                'transcript_available': True
            })

        return audio

    def _needs_animation(self, node_data: Dict[str, Any]) -> bool:
        """Determine if content needs animation"""
        title = node_data.get('title', '').lower()
        content_type = node_data.get('content_type', '')

        # Dynamic processes, learning algorithms, inference processes
        animation_keywords = [
            'inference', 'learning', 'optimization', 'evolution', 'dynamics',
            'process', 'algorithm', 'control', 'adaptation', 'convergence'
        ]

        return (content_type in ['implementation', 'mathematics'] or
                any(keyword in title for keyword in animation_keywords))

    def _needs_interactive_viz(self, node_data: Dict[str, Any]) -> bool:
        """Determine if content needs interactive visualization"""
        content_type = node_data.get('content_type', '')
        difficulty = node_data.get('difficulty', '')

        # Interactive visualizations for implementations and applications
        return content_type in ['implementation', 'application'] or difficulty in ['intermediate', 'advanced']

    def _needs_audio_explanation(self, node_data: Dict[str, Any]) -> bool:
        """Determine if content needs audio explanation"""
        difficulty = node_data.get('difficulty', '')
        content_type = node_data.get('content_type', '')

        # Audio for complex theoretical content
        return (difficulty in ['advanced', 'expert'] or
                content_type == 'mathematics')

    def _extract_concept_elements(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract concept elements for diagrams"""
        elements = []
        tags = node_data.get('tags', [])
        prerequisites = node_data.get('prerequisites', [])

        # Main concept
        elements.append({
            'id': 'main_concept',
            'label': node_data.get('title', ''),
            'type': 'central',
            'description': node_data.get('description', '')
        })

        # Prerequisites
        for i, prereq in enumerate(prerequisites[:5]):  # Limit to 5
            elements.append({
                'id': f'prereq_{i}',
                'label': prereq,
                'type': 'prerequisite',
                'description': f'Required knowledge: {prereq}'
            })

        # Related concepts from tags
        for i, tag in enumerate(tags[:3]):  # Limit to 3
            elements.append({
                'id': f'tag_{i}',
                'label': tag,
                'type': 'related',
                'description': f'Related concept: {tag}'
            })

        return elements

    def _extract_relationships(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relationships for diagrams"""
        relationships = []
        prerequisites = node_data.get('prerequisites', [])

        for prereq in prerequisites:
            relationships.append({
                'source': f'prereq_{prerequisites.index(prereq)}',
                'target': 'main_concept',
                'type': 'prerequisite',
                'label': 'requires'
            })

        return relationships

    def _extract_process_steps(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract process steps for flow diagrams"""
        # Generic process steps - could be made more specific based on content
        return [
            {'id': 'input', 'label': 'Input', 'description': 'Process inputs'},
            {'id': 'processing', 'label': 'Processing', 'description': 'Core processing'},
            {'id': 'output', 'label': 'Output', 'description': 'Process outputs'}
        ]

    def _extract_mathematical_equations(self, node_data: Dict[str, Any]) -> List[str]:
        """Extract mathematical equations from content"""
        content = node_data.get('content', {})

        equations = []
        # Look for mathematical content in various sections
        for section_name, section_content in content.items():
            if isinstance(section_content, str):
                # Simple regex to find mathematical expressions
                math_patterns = [
                    r'\$[^$]+\$',  # LaTeX inline math
                    r'\\\[.*?\\\]',  # LaTeX display math
                    r'[A-Z]\([^)]+\)',  # Function notation
                    r'\\frac\{[^}]+\}\{[^}]+\}',  # Fractions
                    r'\\sum[^}]*',  # Sums
                    r'\\int[^}]*'  # Integrals
                ]

                for pattern in math_patterns:
                    matches = re.findall(pattern, section_content)
                    equations.extend(matches[:3])  # Limit matches per section

        return list(set(equations))[:5]  # Unique equations, max 5

    def _determine_geometric_dimensions(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine geometric dimensions for visualization"""
        title = node_data.get('title', '').lower()

        if 'geometry' in title:
            return {'dimensions': 2, 'type': 'manifold', 'coordinates': ['x', 'y']}
        elif 'information' in title:
            return {'dimensions': 'n', 'type': 'parameter_space', 'coordinates': ['θ₁', 'θ₂', 'θ₃']}
        else:
            return {'dimensions': 2, 'type': 'concept_space', 'coordinates': ['complexity', 'applicability']}

    def _generate_distribution_examples(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate probability distribution examples"""
        return [
            {'name': 'Prior Distribution', 'type': 'beta', 'parameters': [2, 2]},
            {'name': 'Likelihood', 'type': 'normal', 'parameters': [0, 1]},
            {'name': 'Posterior Distribution', 'type': 'beta', 'parameters': [3, 3]}
        ]

    def _extract_algorithm_steps(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract algorithm steps for flowcharts"""
        # Generic algorithm steps
        return [
            {'id': 'initialize', 'label': 'Initialize', 'description': 'Set up initial conditions'},
            {'id': 'process', 'label': 'Process', 'description': 'Execute main algorithm'},
            {'id': 'converge', 'label': 'Check Convergence', 'description': 'Verify termination conditions'},
            {'id': 'output', 'label': 'Output Results', 'description': 'Return final results'}
        ]

    def _extract_data_elements(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data elements for data flow diagrams"""
        return [
            {'id': 'input_data', 'label': 'Input Data', 'type': 'input'},
            {'id': 'processed_data', 'label': 'Processed Data', 'type': 'intermediate'},
            {'id': 'output_data', 'label': 'Output Data', 'type': 'output'}
        ]

    def _extract_architecture_components(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract architecture components"""
        return [
            {'id': 'input_layer', 'label': 'Input Processing', 'type': 'input'},
            {'id': 'core_engine', 'label': 'Core Engine', 'type': 'processing'},
            {'id': 'output_layer', 'label': 'Output Generation', 'type': 'output'}
        ]

    def _extract_use_case_actors(self, node_data: Dict[str, Any]) -> List[str]:
        """Extract use case actors"""
        return ['User', 'System', 'Administrator', 'External Service']

    def _extract_use_case_scenarios(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract use case scenarios"""
        return [
            {'actor': 'User', 'action': 'Interact with system', 'outcome': 'Expected results'},
            {'actor': 'System', 'action': 'Process request', 'outcome': 'Response generated'}
        ]

    def _extract_result_metrics(self, node_data: Dict[str, Any]) -> List[str]:
        """Extract result metrics"""
        return ['accuracy', 'performance', 'efficiency', 'reliability']

    def _generate_animation_keyframes(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate animation keyframes"""
        return [
            {'time': 0, 'description': 'Initial state', 'visual_elements': []},
            {'time': 15, 'description': 'Processing state', 'visual_elements': []},
            {'time': 30, 'description': 'Final state', 'visual_elements': []}
        ]

    def _extract_explorable_parameters(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract parameters that can be explored interactively"""
        return [
            {'name': 'learning_rate', 'type': 'slider', 'range': [0.001, 1.0], 'default': 0.01},
            {'name': 'iterations', 'type': 'slider', 'range': [10, 1000], 'default': 100},
            {'name': 'noise_level', 'type': 'slider', 'range': [0.0, 1.0], 'default': 0.1}
        ]

    def _extract_simulation_parameters(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract simulation parameters"""
        return [
            {'name': 'time_steps', 'value': 1000, 'description': 'Number of simulation steps'},
            {'name': 'dt', 'value': 0.01, 'description': 'Time step size'},
            {'name': 'initial_conditions', 'value': [0.0, 0.0], 'description': 'Starting conditions'}
        ]

    def _generate_video_id(self, node_data: Dict[str, Any], video_type: str) -> str:
        """Generate placeholder video ID"""
        node_id = node_data.get('id', 'unknown')
        # Generate a pseudo-random but deterministic video ID based on content
        import hashlib
        content_hash = hashlib.md5(f"{node_id}_{video_type}".encode()).hexdigest()[:11]
        return content_hash

    def integrate_multimedia_into_knowledge_base(self) -> Dict[str, Any]:
        """
        Integrate multimedia content into the entire knowledge base

        Returns:
            Report on multimedia integration
        """
        logger.info("Starting multimedia integration...")

        files = self._get_all_knowledge_files()
        total_multimedia_items = 0

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    node_data = json.load(f)

                # Integrate multimedia
                enhanced_data = self.integrate_multimedia_for_node(node_data)

                # Save enhanced node
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

                # Count multimedia items
                multimedia = enhanced_data.get('multimedia', {})
                file_multimedia_count = sum(len(v) if isinstance(v, list) else 1 for v in multimedia.values() if v)
                total_multimedia_items += file_multimedia_count

                logger.info(f"Enhanced {file_path.name} with {file_multimedia_count} multimedia items")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        report = {
            "timestamp": datetime.now().isoformat(),
            "files_processed": len(files),
            "total_multimedia_items": total_multimedia_items,
            "average_multimedia_per_file": total_multimedia_items / len(files) if files else 0
        }

        logger.info(f"Multimedia integration completed: {total_multimedia_items} items added")
        return report

    def _get_all_knowledge_files(self) -> List[Path]:
        """Get all knowledge files"""
        json_files = []
        skip_files = ['learning_paths.json', 'faq.json', 'glossary.json', 'success_metrics.json']

        for json_file in self.knowledge_base_path.rglob("*.json"):
            if json_file.name in skip_files or 'metadata' in str(json_file):
                continue
            json_files.append(json_file)

        return sorted(json_files)

def main():
    """Main function"""
    tool = MultimediaIntegrationTool()
    report = tool.integrate_multimedia_into_knowledge_base()

    # Save report
    report_file = "multimedia_integration_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Multimedia integration completed. Report saved to {report_file}")

if __name__ == "__main__":
    main()

