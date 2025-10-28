#!/usr/bin/env python3
"""
Interactive Exercise Generator for Active Inference Knowledge Base

This tool generates interactive exercises, quizzes, and hands-on learning components
to enhance the educational experience and provide practical application opportunities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class InteractiveExerciseGenerator:
    """Generates interactive exercises for Active Inference content"""

    def __init__(self, knowledge_base_path: str = "knowledge"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.exercises_generated = 0
        self.exercises = []

    def generate_exercises_for_node(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate interactive exercises for a knowledge node

        Args:
            node_data: Knowledge node data

        Returns:
            List of generated exercises
        """
        exercises = []
        node_id = node_data.get('id', '')
        content_type = node_data.get('content_type', 'foundation')
        difficulty = node_data.get('difficulty', 'intermediate')

        # Generate different types of exercises based on content
        if content_type == 'foundation':
            exercises.extend(self._generate_concept_exercises(node_data))
        elif content_type == 'mathematics':
            exercises.extend(self._generate_mathematical_exercises(node_data))
        elif content_type == 'implementation':
            exercises.extend(self._generate_coding_exercises(node_data))
        elif content_type == 'application':
            exercises.extend(self._generate_application_exercises(node_data))

        # Add general exercises applicable to all content types
        exercises.extend(self._generate_general_exercises(node_data))

        # Add difficulty-appropriate exercises
        if difficulty in ['beginner', 'intermediate']:
            exercises.extend(self._generate_guided_exercises(node_data))
        elif difficulty in ['advanced', 'expert']:
            exercises.extend(self._generate_challenge_exercises(node_data))

        return exercises

    def _generate_concept_exercises(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate concept understanding exercises"""
        exercises = []
        title = node_data.get('title', '')
        tags = node_data.get('tags', [])

        # Multiple choice questions
        if any(tag in ['entropy', 'information', 'probability'] for tag in tags):
            exercises.append({
                'type': 'multiple_choice',
                'title': f'Understanding {title}',
                'question': f'Which of the following best describes the concept of {title.lower()}?',
                'options': self._generate_concept_options(node_data),
                'correct_answer': 0,
                'explanation': f'This exercise tests understanding of {title.lower()} and its key characteristics.',
                'difficulty': node_data.get('difficulty', 'intermediate'),
                'estimated_time': 3
            })

        # True/False questions
        exercises.append({
            'type': 'true_false',
            'title': f'Concept Verification: {title}',
            'question': self._generate_true_false_question(node_data),
            'correct_answer': True,
            'explanation': 'This helps verify understanding of key conceptual relationships.',
            'difficulty': node_data.get('difficulty', 'intermediate'),
            'estimated_time': 2
        })

        # Concept mapping
        exercises.append({
            'type': 'concept_mapping',
            'title': f'Connect Concepts: {title}',
            'instructions': f'Drag and drop to connect related concepts from {title.lower()}:',
            'concepts': self._extract_key_concepts(node_data),
            'difficulty': node_data.get('difficulty', 'intermediate'),
            'estimated_time': 5
        })

        return exercises

    def _generate_mathematical_exercises(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mathematical exercises"""
        exercises = []
        title = node_data.get('title', '')

        # Calculation exercises
        if 'entropy' in title.lower() or 'kl' in title.lower():
            exercises.append({
                'type': 'calculation',
                'title': f'Calculate {title}',
                'problem': self._generate_calculation_problem(node_data),
                'solution_steps': self._generate_solution_steps(node_data),
                'difficulty': node_data.get('difficulty', 'intermediate'),
                'estimated_time': 10,
                'hints': [
                    'Remember the formula for entropy: H(X) = -∑p(x)log₂p(x)',
                    'KL divergence measures the difference between two distributions'
                ]
            })

        # Proof exercises
        exercises.append({
            'type': 'proof',
            'title': f'Prove: {title} Properties',
            'problem': f'Prove that {title.lower()} satisfies the following properties:',
            'properties': self._generate_proof_properties(node_data),
            'difficulty': 'advanced',
            'estimated_time': 15
        })

        # Mathematical visualization
        exercises.append({
            'type': 'visualization',
            'title': f'Visualize {title}',
            'instructions': 'Create a visualization of the mathematical concept:',
            'requirements': self._generate_visualization_requirements(node_data),
            'difficulty': node_data.get('difficulty', 'intermediate'),
            'estimated_time': 8
        })

        return exercises

    def _generate_coding_exercises(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate coding exercises"""
        exercises = []
        title = node_data.get('title', '')

        # Implementation exercises
        exercises.append({
            'type': 'coding',
            'title': f'Implement {title}',
            'instructions': f'Write code to implement {title.lower()}:',
            'starter_code': self._generate_starter_code(node_data),
            'test_cases': self._generate_test_cases(node_data),
            'solution': self._generate_solution_code(node_data),
            'difficulty': node_data.get('difficulty', 'intermediate'),
            'estimated_time': 20,
            'programming_language': 'python'
        })

        # Debugging exercises
        exercises.append({
            'type': 'debugging',
            'title': f'Debug {title} Implementation',
            'instructions': 'Find and fix the bugs in this code:',
            'buggy_code': self._generate_buggy_code(node_data),
            'hints': self._generate_debugging_hints(node_data),
            'solution': self._generate_corrected_code(node_data),
            'difficulty': node_data.get('difficulty', 'intermediate'),
            'estimated_time': 15
        })

        # Optimization exercises
        exercises.append({
            'type': 'optimization',
            'title': f'Optimize {title} Algorithm',
            'instructions': 'Improve the performance of this algorithm:',
            'code': self._generate_optimization_target(node_data),
            'performance_metrics': ['time_complexity', 'space_complexity', 'numerical_stability'],
            'difficulty': 'advanced',
            'estimated_time': 25
        })

        return exercises

    def _generate_application_exercises(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate application exercises"""
        exercises = []
        title = node_data.get('title', '')

        # Case study analysis
        exercises.append({
            'type': 'case_analysis',
            'title': f'Analyze {title} Application',
            'scenario': self._generate_application_scenario(node_data),
            'questions': self._generate_analysis_questions(node_data),
            'difficulty': node_data.get('difficulty', 'intermediate'),
            'estimated_time': 15
        })

        # Design exercises
        exercises.append({
            'type': 'design',
            'title': f'Design {title} System',
            'requirements': self._generate_design_requirements(node_data),
            'constraints': self._generate_design_constraints(node_data),
            'evaluation_criteria': self._generate_evaluation_criteria(node_data),
            'difficulty': 'advanced',
            'estimated_time': 30
        })

        # Implementation planning
        exercises.append({
            'type': 'planning',
            'title': f'Plan {title} Implementation',
            'objectives': self._generate_implementation_objectives(node_data),
            'steps': self._generate_implementation_steps(node_data),
            'resources': self._generate_required_resources(node_data),
            'difficulty': node_data.get('difficulty', 'intermediate'),
            'estimated_time': 20
        })

        return exercises

    def _generate_general_exercises(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate general exercises applicable to all content types"""
        exercises = []
        title = node_data.get('title', '')

        # Discussion questions
        exercises.append({
            'type': 'discussion',
            'title': f'Discuss: {title}',
            'questions': [
                f'How does {title.lower()} relate to other Active Inference concepts?',
                f'What are the practical implications of {title.lower()}?',
                f'What challenges arise when applying {title.lower()}?',
                f'How might {title.lower()} evolve in the future?'
            ],
            'difficulty': node_data.get('difficulty', 'intermediate'),
            'estimated_time': 10
        })

        # Research questions
        exercises.append({
            'type': 'research',
            'title': f'Research: {title} Applications',
            'questions': [
                f'Find a real-world application of {title.lower()}',
                f'What research papers discuss {title.lower()}?',
                f'How is {title.lower()} used in industry?',
                f'What are current research challenges related to {title.lower()}?'
            ],
            'difficulty': node_data.get('difficulty', 'intermediate'),
            'estimated_time': 20
        })

        # Peer review exercises
        exercises.append({
            'type': 'peer_review',
            'title': f'Peer Review: {title} Explanation',
            'instructions': 'Review and provide feedback on this explanation of the concept:',
            'rubric': self._generate_peer_review_rubric(),
            'difficulty': node_data.get('difficulty', 'intermediate'),
            'estimated_time': 12
        })

        return exercises

    def _generate_guided_exercises(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate guided exercises for beginners and intermediates"""
        exercises = []

        # Step-by-step tutorials
        exercises.append({
            'type': 'tutorial',
            'title': f'Step-by-Step: Understanding {node_data.get("title", "")}',
            'steps': self._generate_tutorial_steps(node_data),
            'checkpoints': self._generate_checkpoints(node_data),
            'difficulty': 'beginner',
            'estimated_time': 15
        })

        # Scaffolded problem solving
        exercises.append({
            'type': 'scaffolded',
            'title': f'Guided Problem: {node_data.get("title", "")}',
            'hints': self._generate_progressive_hints(node_data),
            'partial_solutions': self._generate_partial_solutions(node_data),
            'difficulty': 'intermediate',
            'estimated_time': 12
        })

        return exercises

    def _generate_challenge_exercises(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate challenge exercises for advanced and expert learners"""
        exercises = []

        # Research challenges
        exercises.append({
            'type': 'research_challenge',
            'title': f'Research Challenge: Extend {node_data.get("title", "")}',
            'problem': f'Develop a novel extension or application of {node_data.get("title", "").lower()}',
            'requirements': self._generate_research_requirements(node_data),
            'difficulty': 'expert',
            'estimated_time': 40
        })

        # Critical analysis
        exercises.append({
            'type': 'critical_analysis',
            'title': f'Critique: {node_data.get("title", "")} Assumptions',
            'questions': self._generate_critical_questions(node_data),
            'alternative_views': self._generate_alternative_views(node_data),
            'difficulty': 'advanced',
            'estimated_time': 25
        })

        return exercises

    def _generate_concept_options(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate multiple choice options for concept questions"""
        title = node_data.get('title', '')
        content_type = node_data.get('content_type', '')

        # Generic options based on content type
        if content_type == 'foundation':
            return [
                f'{title} is a fundamental concept in Active Inference theory',
                f'{title} is primarily used in machine learning applications',
                f'{title} deals with quantum mechanics principles',
                f'{title} is related to classical physics only'
            ]
        elif content_type == 'mathematics':
            return [
                f'{title} provides mathematical tools for Active Inference',
                f'{title} is used for web development',
                f'{title} deals with database management',
                f'{title} is for image processing only'
            ]
        else:
            return [
                f'{title} is an important Active Inference concept',
                f'{title} is unrelated to cognitive science',
                f'{title} is only used in theoretical physics',
                f'{title} has no practical applications'
            ]

    def _generate_true_false_question(self, node_data: Dict[str, Any]) -> str:
        """Generate true/false questions"""
        title = node_data.get('title', '')
        return f'{title} is a core concept in Active Inference theory.'

    def _extract_key_concepts(self, node_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract key concepts for mapping exercises"""
        tags = node_data.get('tags', [])
        return {
            'concepts': tags[:5],  # Limit to 5 concepts
            'relationships': [
                {'from': tags[0] if tags else 'concept', 'to': tags[1] if len(tags) > 1 else 'related'}
            ]
        }

    def _generate_calculation_problem(self, node_data: Dict[str, Any]) -> str:
        """Generate calculation problems"""
        if 'entropy' in node_data.get('title', '').lower():
            return "Calculate the entropy of a fair six-sided die. Show your work."
        elif 'kl' in node_data.get('title', '').lower():
            return "Compute the KL divergence between two Bernoulli distributions with p=0.3 and q=0.7."
        else:
            return f"Solve a representative problem involving {node_data.get('title', 'the concept')}."

    def _generate_solution_steps(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate solution steps"""
        return [
            "Identify the relevant formula or principle",
            "Substitute the given values",
            "Perform the necessary calculations",
            "Interpret the result in context"
        ]

    def _generate_proof_properties(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate properties to prove"""
        return [
            "The concept satisfies mathematical consistency",
            "It maintains the required theoretical properties",
            "It connects properly to related concepts"
        ]

    def _generate_visualization_requirements(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate visualization requirements"""
        return [
            "Create a clear diagram showing the key components",
            "Include labels and explanations",
            "Show relationships between elements",
            "Use appropriate visual metaphors"
        ]

    def _generate_starter_code(self, node_data: Dict[str, Any]) -> str:
        """Generate starter code for coding exercises"""
        return f'''# Implement {node_data.get('title', 'the concept')}
import numpy as np

def implement_concept():
    """
    Implement the {node_data.get('title', 'concept')} functionality
    """
    # Your implementation here
    pass

if __name__ == "__main__":
    result = implement_concept()
    print(f"Result: {result}")'''

    def _generate_test_cases(self, node_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases"""
        return [
            {
                'input': 'sample_input',
                'expected_output': 'expected_result',
                'description': 'Basic functionality test'
            }
        ]

    def _generate_solution_code(self, node_data: Dict[str, Any]) -> str:
        """Generate solution code"""
        return f'''# Complete implementation of {node_data.get('title', 'the concept')}
import numpy as np

def implement_concept():
    """
    Complete implementation
    """
    return "implementation_complete"

if __name__ == "__main__":
    result = implement_concept()
    print(f"Result: {result}")'''

    def _generate_buggy_code(self, node_data: Dict[str, Any]) -> str:
        """Generate buggy code for debugging exercises"""
        return f'''# Buggy implementation of {node_data.get('title', 'the concept')}
import numpy as np

def buggy_function():
    # This code has bugs - find and fix them!
    x = [1, 2, 3, 4, 5]
    result = 0
    for i in x:
        result += i  # Bug: should be multiplication
    return result

print(buggy_function())'''

    def _generate_debugging_hints(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate debugging hints"""
        return [
            "Check the mathematical operation in the loop",
            "Verify the expected vs actual output",
            "Look for off-by-one errors"
        ]

    def _generate_corrected_code(self, node_data: Dict[str, Any]) -> str:
        """Generate corrected code"""
        return f'''# Corrected implementation
import numpy as np

def corrected_function():
    x = [1, 2, 3, 4, 5]
    result = 1  # Start with 1 for multiplication
    for i in x:
        result *= i  # Corrected: multiplication instead of addition
    return result

print(corrected_function())  # Should print 120'''

    def _generate_optimization_target(self, node_data: Dict[str, Any]) -> str:
        """Generate code for optimization exercises"""
        return '''# Inefficient implementation - optimize this!
def slow_function(n):
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append(i ** 2)
    return result'''

    def _generate_application_scenario(self, node_data: Dict[str, Any]) -> str:
        """Generate application scenarios"""
        return f"A company is trying to apply {node_data.get('title', 'the concept')} to solve a complex problem in their domain."

    def _generate_analysis_questions(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate analysis questions"""
        return [
            "What are the key challenges in this application?",
            "How does the Active Inference approach help?",
            "What are the potential benefits and drawbacks?",
            "What metrics would you use to evaluate success?"
        ]

    def _generate_design_requirements(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate design requirements"""
        return [
            "Must incorporate Active Inference principles",
            "Should be scalable and maintainable",
            "Needs to handle uncertainty appropriately",
            "Should provide clear interfaces and documentation"
        ]

    def _generate_design_constraints(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate design constraints"""
        return [
            "Limited computational resources",
            "Real-time performance requirements",
            "Data privacy and security concerns",
            "Integration with existing systems"
        ]

    def _generate_evaluation_criteria(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate evaluation criteria"""
        return [
            "Technical correctness",
            "Performance efficiency",
            "Practical applicability",
            "Code quality and maintainability"
        ]

    def _generate_implementation_objectives(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate implementation objectives"""
        return [
            "Create a working prototype",
            "Validate against theoretical predictions",
            "Ensure computational efficiency",
            "Document the implementation thoroughly"
        ]

    def _generate_implementation_steps(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate implementation steps"""
        return [
            "Review theoretical foundations",
            "Design the system architecture",
            "Implement core algorithms",
            "Test and validate the implementation",
            "Document and share results"
        ]

    def _generate_required_resources(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate required resources"""
        return [
            "Python programming environment",
            "Scientific computing libraries (NumPy, SciPy)",
            "Data for testing and validation",
            "Computational resources for experiments"
        ]

    def _generate_tutorial_steps(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate tutorial steps"""
        return [
            "Read the concept introduction",
            "Review the key definitions",
            "Work through the examples",
            "Complete the practice exercises",
            "Apply the concept to a new problem"
        ]

    def _generate_checkpoints(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate checkpoints for guided exercises"""
        return [
            "Can you explain the concept in your own words?",
            "Can you identify examples in real life?",
            "Can you solve basic problems using the concept?",
            "Can you explain how it relates to other concepts?"
        ]

    def _generate_progressive_hints(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate progressive hints"""
        return [
            "Start by understanding the basic definition",
            "Look for similar examples in the content",
            "Try breaking the problem into smaller parts",
            "Check if you're using the right approach"
        ]

    def _generate_partial_solutions(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate partial solutions"""
        return [
            "The first step involves...",
            "You need to consider...",
            "The key insight is...",
            "Finally, you should..."
        ]

    def _generate_research_requirements(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate research requirements"""
        return [
            "Identify a novel application or extension",
            "Develop a theoretical foundation",
            "Create computational implementation",
            "Validate against existing results",
            "Document findings and implications"
        ]

    def _generate_critical_questions(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate critical questions"""
        return [
            "What are the underlying assumptions?",
            "How robust is the approach to violations of assumptions?",
            "What are the limitations and edge cases?",
            "How does this compare to alternative approaches?"
        ]

    def _generate_alternative_views(self, node_data: Dict[str, Any]) -> List[str]:
        """Generate alternative views"""
        return [
            "Connectionist perspective",
            "Symbolic AI approach",
            "Bayesian statistical viewpoint",
            "Neuroscience-based interpretation"
        ]

    def _generate_peer_review_rubric(self) -> Dict[str, List[str]]:
        """Generate peer review rubric"""
        return {
            'clarity': ['Unclear', 'Somewhat clear', 'Clear', 'Very clear'],
            'accuracy': ['Inaccurate', 'Mostly accurate', 'Accurate', 'Highly accurate'],
            'completeness': ['Incomplete', 'Partially complete', 'Mostly complete', 'Complete'],
            'usefulness': ['Not useful', 'Somewhat useful', 'Useful', 'Very useful']
        }

    def generate_exercises_for_knowledge_base(self) -> Dict[str, Any]:
        """
        Generate interactive exercises for the entire knowledge base

        Returns:
            Report on exercise generation
        """
        logger.info("Starting interactive exercise generation...")

        files = self._get_all_knowledge_files()
        total_exercises = 0

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    node_data = json.load(f)

                exercises = self.generate_exercises_for_node(node_data)

                if exercises:
                    # Add exercises to node data
                    if 'interactive_exercises' not in node_data:
                        node_data['interactive_exercises'] = []

                    node_data['interactive_exercises'].extend(exercises)

                    # Save updated node
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(node_data, f, indent=2, ensure_ascii=False)

                    total_exercises += len(exercises)
                    logger.info(f"Added {len(exercises)} exercises to {file_path.name}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        report = {
            "timestamp": datetime.now().isoformat(),
            "files_processed": len(files),
            "total_exercises_generated": total_exercises,
            "average_exercises_per_file": total_exercises / len(files) if files else 0
        }

        logger.info(f"Exercise generation completed: {total_exercises} exercises generated")
        return report

    def _get_all_knowledge_files(self) -> List[Path]:
        """Get all knowledge files (excluding special files)"""
        json_files = []
        skip_files = ['learning_paths.json', 'faq.json', 'glossary.json', 'success_metrics.json']

        for json_file in self.knowledge_base_path.rglob("*.json"):
            if json_file.name in skip_files or 'metadata' in str(json_file):
                continue
            json_files.append(json_file)

        return sorted(json_files)

def main():
    """Main function"""
    generator = InteractiveExerciseGenerator()
    report = generator.generate_exercises_for_knowledge_base()

    # Save report
    report_file = "interactive_exercises_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Interactive exercises generated. Report saved to {report_file}")

if __name__ == "__main__":
    main()

