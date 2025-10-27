# Knowledge Repository - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Knowledge module of the Active Inference Knowledge Environment source code. It outlines implementation patterns, development workflows, and best practices for creating and managing educational content systems.

## Knowledge Module Overview

The Knowledge module provides the source code implementation for the Active Inference educational content management system, including knowledge repository, learning path management, content validation, and semantic organization of educational materials.

## Source Code Architecture

### Module Responsibilities
- **Knowledge Repository**: Core content management and organization system
- **Foundations System**: Theoretical foundations management and organization
- **Mathematics System**: Mathematical formulations and computational implementations
- **Content Validation**: Quality assurance and content integrity validation
- **Learning Path Management**: Educational workflow and progression systems

### Integration Points
- **Research Tools**: Integration with experiment management and analysis
- **Visualization Engine**: Content visualization and interactive exploration
- **Applications Framework**: Connection to practical implementation templates
- **Platform Services**: Deployment and collaboration features

## Core Implementation Responsibilities

### Knowledge Repository Implementation
**Central content management and organization system**
- Implement comprehensive knowledge node management with validation
- Create efficient search and indexing systems
- Develop learning path generation and prerequisite validation
- Implement content organization and metadata management

**Key Methods to Implement:**
```python
def initialize_knowledge_base(self) -> None:
    """Initialize knowledge base with content loading and validation"""

def add_knowledge_node(self, node_data: Dict[str, Any]) -> bool:
    """Add knowledge node with comprehensive validation and indexing"""

def update_knowledge_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
    """Update knowledge node with dependency checking and re-indexing"""

def delete_knowledge_node(self, node_id: str) -> bool:
    """Delete knowledge node with cascade deletion and reference updates"""

def search_with_semantic_ranking(self, query: str, filters: Dict[str, Any]) -> List[KnowledgeNode]:
    """Implement semantic search with intelligent ranking and filtering"""

def build_prerequisite_graph(self, node_id: str) -> Dict[str, Any]:
    """Build complete prerequisite dependency graph with cycle detection"""

def validate_content_integrity(self, node_id: str) -> Dict[str, Any]:
    """Validate content integrity including references and prerequisites"""

def generate_learning_path_recommendations(self, user_profile: Dict[str, Any]) -> List[str]:
    """Generate personalized learning path recommendations"""

def export_knowledge_in_format(self, format_type: str, content_filter: Dict[str, Any]) -> Any:
    """Export knowledge content in various formats (JSON, XML, RDF, etc.)"""

def create_content_backup(self) -> Path:
    """Create comprehensive content backup with metadata and relationships"""
```

### Foundations System Implementation
**Theoretical foundations management and organization**
- Implement information theory foundation concepts and relationships
- Create Bayesian inference fundamentals with proper mathematical grounding
- Develop Free Energy Principle theoretical framework implementation
- Implement Active Inference conceptual framework with validation

**Key Methods to Implement:**
```python
def implement_information_theory_foundation(self) -> None:
    """Implement complete information theory foundation with mathematical derivations"""

def implement_bayesian_inference_foundation(self) -> None:
    """Implement Bayesian inference foundation with computational examples"""

def implement_free_energy_principle(self) -> None:
    """Implement Free Energy Principle foundation with rigorous derivations"""

def implement_active_inference_framework(self) -> None:
    """Implement Active Inference framework with complete conceptual structure"""

def validate_theoretical_content(self, node_id: str) -> Dict[str, Any]:
    """Validate theoretical content for mathematical and conceptual accuracy"""

def create_foundation_cross_references(self) -> Dict[str, List[str]]:
    """Create comprehensive cross-references between all foundation concepts"""

def generate_foundation_progression_path(self, start_concept: str, target_level: str) -> List[str]:
    """Generate optimal progression path through foundation concepts"""

def create_foundation_assessment_system(self) -> Dict[str, Any]:
    """Create comprehensive assessment system for foundation concepts"""

def implement_concept_dependency_graph(self) -> Dict[str, Any]:
    """Implement complete concept dependency graph with validation"""

def validate_mathematical_consistency(self, concept_chain: List[str]) -> Dict[str, Any]:
    """Validate mathematical consistency across concept chains"""
```

### Mathematics System Implementation
**Mathematical formulations and computational implementations**
- Implement probability theory and Bayesian mathematics with validation
- Create information theory mathematical foundations with derivations
- Develop variational methods and free energy calculus systems
- Implement dynamical systems and stochastic processes for Active Inference

**Key Methods to Implement:**
```python
def implement_mathematical_derivation_engine(self) -> None:
    """Implement mathematical derivation engine with step-by-step validation"""

def create_computational_mathematics_library(self) -> None:
    """Create comprehensive computational mathematics library for Active Inference"""

def implement_information_theory_computations(self) -> None:
    """Implement complete information theory computation system"""

def create_variational_methods_system(self) -> None:
    """Implement variational methods and free energy calculus system"""

def implement_stochastic_processes(self) -> None:
    """Implement stochastic processes and dynamical systems for Active Inference"""

def validate_mathematical_implementations(self) -> Dict[str, Any]:
    """Validate all mathematical implementations for accuracy and efficiency"""

def create_mathematical_visualization_tools(self) -> Dict[str, Any]:
    """Create tools for mathematical concept visualization and exploration"""

def implement_symbolic_mathematics_engine(self) -> Dict[str, Any]:
    """Implement symbolic mathematics engine for derivations and proofs"""

def create_numerical_stability_system(self) -> Dict[str, Any]:
    """Create numerical stability system for robust mathematical computations"""

def implement_mathematical_testing_framework(self) -> Dict[str, Any]:
    """Implement comprehensive testing framework for mathematical functions"""
```

## Development Workflows

### Content Development Workflow
1. **Concept Analysis**: Analyze educational requirements and learning objectives
2. **Content Design**: Design structured content following established schemas
3. **Implementation**: Implement content with comprehensive validation
4. **Testing**: Create extensive test suites for content functionality
5. **Integration**: Ensure integration with learning path systems
6. **Validation**: Validate content against educational standards
7. **Documentation**: Generate comprehensive documentation and examples
8. **Review**: Submit for peer review and validation

### Mathematical Content Development
1. **Mathematical Analysis**: Analyze mathematical requirements and dependencies
2. **Derivation Design**: Design mathematical derivations with proper sequencing
3. **Implementation**: Implement mathematical functions with numerical stability
4. **Testing**: Create comprehensive test suites including edge cases
5. **Validation**: Validate mathematical accuracy and computational efficiency
6. **Documentation**: Generate detailed mathematical documentation
7. **Performance**: Optimize for performance and numerical stability

## Quality Assurance Standards

### Content Quality Requirements
- **Educational Accuracy**: All content must be mathematically and conceptually accurate
- **Learning Progression**: Content must follow proper learning progression
- **Prerequisite Validation**: All prerequisites must be properly validated
- **Assessment Integration**: Content must support assessment and validation
- **Accessibility**: Content must be accessible to target audience levels
- **Currency**: Content must reflect current research and understanding

### Technical Quality Requirements
- **Code Quality**: Follow established coding standards and patterns
- **Test Coverage**: Maintain >95% test coverage for all components
- **Performance**: Implement efficient algorithms and data structures
- **Error Handling**: Comprehensive error handling with informative messages
- **Documentation**: Complete documentation with examples and usage
- **Validation**: Built-in validation for all content and computations

## Testing Implementation

### Comprehensive Testing Framework
```python
class TestKnowledgeRepositoryImplementation(unittest.TestCase):
    """Test knowledge repository implementation and functionality"""

    def setUp(self):
        """Set up test environment with knowledge repository"""
        self.config = KnowledgeRepositoryConfig(
            root_path=test_knowledge_path,
            auto_index=True,
            cache_enabled=True
        )
        self.repository = KnowledgeRepository(self.config)

    def test_knowledge_node_lifecycle(self):
        """Test complete knowledge node lifecycle"""
        # Create node data
        node_data = {
            "id": "test_bayesian_inference",
            "title": "Bayesian Inference Fundamentals",
            "content_type": ContentType.FOUNDATION,
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "description": "Comprehensive introduction to Bayesian inference",
            "prerequisites": ["probability_basics"],
            "tags": ["bayesian", "inference", "probability"],
            "learning_objectives": [
                "Understand Bayesian probability",
                "Apply Bayes' theorem",
                "Update beliefs with evidence"
            ],
            "content": {
                "sections": [
                    {"title": "Introduction", "content": "Bayesian inference content..."},
                    {"title": "Mathematical Foundation", "content": "Mathematical derivation..."}
                ]
            }
        }

        # Test node creation
        success = self.repository.add_knowledge_node(node_data)
        self.assertTrue(success)

        # Test node retrieval
        node = self.repository.get_node("test_bayesian_inference")
        self.assertIsNotNone(node)
        self.assertEqual(node.title, "Bayesian Inference Fundamentals")

        # Test node update
        update_data = {"description": "Updated description"}
        update_success = self.repository.update_knowledge_node("test_bayesian_inference", update_data)
        self.assertTrue(update_success)

        # Test node deletion
        delete_success = self.repository.delete_knowledge_node("test_bayesian_inference")
        self.assertTrue(delete_success)

        # Verify deletion
        deleted_node = self.repository.get_node("test_bayesian_inference")
        self.assertIsNone(deleted_node)

    def test_search_and_filtering(self):
        """Test search functionality with various filters"""
        # Create test nodes
        test_nodes = [
            {
                "id": "entropy_info_theory",
                "title": "Entropy in Information Theory",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "Mathematical treatment of entropy",
                "tags": ["entropy", "information_theory", "mathematics"]
            },
            {
                "id": "bayesian_basics",
                "title": "Bayesian Basics",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.BEGINNER,
                "description": "Introduction to Bayesian probability",
                "tags": ["bayesian", "probability", "beginner"]
            }
        ]

        for node_data in test_nodes:
            self.repository.add_knowledge_node(node_data)

        # Test basic search
        results = self.repository.search("entropy")
        self.assertGreater(len(results), 0)

        # Test filtered search
        math_results = self.repository.search(
            "entropy",
            content_types=[ContentType.MATHEMATICS]
        )
        self.assertEqual(len(math_results), 1)
        self.assertEqual(math_results[0].id, "entropy_info_theory")

        # Test difficulty filtering
        beginner_results = self.repository.search(
            "bayesian",
            difficulty=[DifficultyLevel.BEGINNER]
        )
        self.assertEqual(len(beginner_results), 1)
        self.assertEqual(beginner_results[0].difficulty, DifficultyLevel.BEGINNER)

    def test_learning_path_validation(self):
        """Test learning path validation and prerequisite checking"""
        # Create prerequisite chain
        nodes_data = [
            {
                "id": "probability_basics",
                "title": "Probability Basics",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.BEGINNER,
                "prerequisites": [],
                "tags": ["probability"]
            },
            {
                "id": "bayesian_fundamentals",
                "title": "Bayesian Fundamentals",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "prerequisites": ["probability_basics"],
                "tags": ["bayesian", "fundamentals"]
            },
            {
                "id": "active_inference_intro",
                "title": "Active Inference Introduction",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.ADVANCED,
                "prerequisites": ["bayesian_fundamentals"],
                "tags": ["active_inference", "introduction"]
            }
        ]

        for node_data in nodes_data:
            self.repository.add_knowledge_node(node_data)

        # Create learning path
        path_data = {
            "id": "test_learning_path",
            "name": "Test Learning Path",
            "description": "Test path for validation",
            "nodes": ["probability_basics", "bayesian_fundamentals", "active_inference_intro"],
            "estimated_hours": 15,
            "difficulty": DifficultyLevel.INTERMEDIATE
        }

        path = LearningPath(**path_data)
        self.repository.add_learning_path(path)

        # Validate path
        validation = self.repository.validate_learning_path("test_learning_path")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["path_length"], 3)
        self.assertEqual(len(validation["issues"]), 0)
```

### Mathematical Testing Framework
```python
class TestMathematicalImplementations(unittest.TestCase):
    """Test mathematical implementations and derivations"""

    def setUp(self):
        """Set up mathematical testing environment"""
        self.math_system = Mathematics(test_repository)

    def test_kl_divergence_computation(self):
        """Test KL divergence computation accuracy"""
        # Test with known values
        p = np.array([0.1, 0.3, 0.4, 0.2])
        q = np.array([0.2, 0.2, 0.3, 0.3])

        kl_div = self.math_system.compute_kl_divergence(p, q)

        # Verify KL divergence is non-negative
        self.assertGreaterEqual(kl_div, 0)

        # Test symmetry (KL(p||q) != KL(q||p) in general)
        kl_div_reverse = self.math_system.compute_kl_divergence(q, p)
        self.assertNotEqual(kl_div, kl_div_reverse)

        # Test with identical distributions (should be 0)
        p_identity = np.array([0.25, 0.25, 0.25, 0.25])
        kl_div_identity = self.math_system.compute_kl_divergence(p_identity, p_identity)
        self.assertAlmostEqual(kl_div_identity, 0.0, places=6)

    def test_free_energy_derivation(self):
        """Test free energy mathematical derivation"""
        derivation = self.math_system.derive_free_energy_expression()

        # Validate derivation structure
        self.assertIn("expression", derivation)
        self.assertIn("components", derivation)
        self.assertIn("interpretation", derivation)
        self.assertIn("proof_sketch", derivation)

        # Validate mathematical expression format
        expression = derivation["expression"]
        self.assertIn("∫", expression)  # Integral notation
        self.assertIn("q(θ)", expression)  # Variational distribution
        self.assertIn("p(x,θ)", expression)  # Joint distribution

        # Validate components
        components = derivation["components"]
        self.assertIn("cross_entropy", components)
        self.assertIn("entropy", components)

    def test_expected_free_energy_computation(self):
        """Test expected free energy computation for policy selection"""
        # Create test policies and observations
        n_states = 4
        n_actions = 3
        time_horizon = 10

        policies = [np.random.dirichlet(np.ones(n_actions)) for _ in range(5)]
        observations = np.random.dirichlet(np.ones(n_states))

        efe_values = self.math_system.expected_free_energy(policies, observations)

        # Validate output structure
        self.assertEqual(len(efe_values), len(policies))
        self.assertTrue(all(isinstance(efe, (int, float)) for efe in efe_values))

        # Expected free energy should be finite
        self.assertTrue(all(np.isfinite(efe) for efe in efe_values))
```

## Performance Optimization

### Search and Indexing Performance
- **Index Optimization**: Optimize search indices for fast retrieval
- **Query Processing**: Efficient query parsing and execution
- **Caching Strategy**: Intelligent caching for frequently accessed content
- **Memory Management**: Efficient memory usage for large knowledge bases

### Mathematical Computation Performance
- **Numerical Stability**: Implement numerically stable algorithms
- **Vectorization**: Use vectorized operations for efficiency
- **Parallel Processing**: Parallel processing for intensive computations
- **Memory Efficiency**: Efficient memory usage for large mathematical operations

## Content Management Implementation

### Content Validation System
```python
class ContentValidationSystem:
    """Comprehensive content validation system"""

    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository
        self.validation_rules = self.initialize_validation_rules()

    def validate_knowledge_node(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate knowledge node against comprehensive criteria"""

        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "suggestions": []
        }

        # Schema validation
        schema_issues = self.validate_schema(node_data)
        if schema_issues:
            validation_results["issues"].extend(schema_issues)
            validation_results["valid"] = False

        # Prerequisite validation
        prereq_issues = self.validate_prerequisites(node_data)
        if prereq_issues:
            validation_results["issues"].extend(prereq_issues)

        # Content quality validation
        quality_issues = self.validate_content_quality(node_data)
        if quality_issues:
            validation_results["warnings"].extend(quality_issues)

        # Cross-reference validation
        xref_issues = self.validate_cross_references(node_data)
        if xref_issues:
            validation_results["issues"].extend(xref_issues)

        # Generate suggestions
        suggestions = self.generate_content_suggestions(node_data)
        validation_results["suggestions"] = suggestions

        return validation_results

    def validate_schema(self, node_data: Dict[str, Any]) -> List[str]:
        """Validate node data against JSON schema"""

        issues = []

        required_fields = ["id", "title", "content_type", "difficulty", "description"]
        for field in required_fields:
            if field not in node_data or node_data[field] is None:
                issues.append(f"Missing required field: {field}")

        # Validate field types
        if "id" in node_data and not isinstance(node_data["id"], str):
            issues.append("ID must be a string")

        if "title" in node_data and not isinstance(node_data["title"], str):
            issues.append("Title must be a string")

        if "content_type" in node_data and node_data["content_type"] not in ContentType:
            issues.append(f"Invalid content type: {node_data['content_type']}")

        if "difficulty" in node_data and node_data["difficulty"] not in DifficultyLevel:
            issues.append(f"Invalid difficulty level: {node_data['difficulty']}")

        return issues

    def validate_prerequisites(self, node_data: Dict[str, Any]) -> List[str]:
        """Validate prerequisite relationships"""

        issues = []
        prerequisites = node_data.get("prerequisites", [])

        for prereq_id in prerequisites:
            # Check if prerequisite exists
            prereq_node = self.repository.get_node(prereq_id)
            if prereq_node is None:
                issues.append(f"Prerequisite node not found: {prereq_id}")
                continue

            # Check for circular dependencies
            if self.has_circular_dependency(node_data["id"], prereq_id):
                issues.append(f"Circular dependency detected: {prereq_id}")

        return issues

    def validate_content_quality(self, node_data: Dict[str, Any]) -> List[str]:
        """Validate content quality metrics"""

        warnings = []

        # Check description length
        description = node_data.get("description", "")
        if len(description) < 50:
            warnings.append("Description may be too short for clarity")
        elif len(description) > 500:
            warnings.append("Description may be too long for overview")

        # Check learning objectives
        objectives = node_data.get("learning_objectives", [])
        if len(objectives) == 0:
            warnings.append("No learning objectives specified")
        elif len(objectives) > 10:
            warnings.append("Too many learning objectives may be overwhelming")

        # Check tags
        tags = node_data.get("tags", [])
        if len(tags) == 0:
            warnings.append("No tags specified for searchability")
        elif len(tags) > 20:
            warnings.append("Too many tags may reduce search effectiveness")

        return warnings
```

## Getting Started as an Agent

### Development Setup
1. **Explore Content Structure**: Review existing knowledge organization patterns
2. **Study Validation Systems**: Understand content validation and quality assurance
3. **Run Content Tests**: Ensure all content validation tests pass
4. **Performance Testing**: Validate search and mathematical computation performance
5. **Documentation**: Update README and AGENTS files for new features

### Implementation Process
1. **Design Phase**: Design new content structures and learning paths
2. **Implementation**: Implement following established patterns and validation
3. **Testing**: Create comprehensive tests including content validation
4. **Integration**: Ensure integration with existing knowledge systems
5. **Review**: Submit for educational and technical review

### Quality Assurance Checklist
- [ ] Content follows established educational patterns and schemas
- [ ] Mathematical content is accurate and properly validated
- [ ] Learning paths have proper prerequisite validation
- [ ] Search functionality works correctly with new content
- [ ] Content integrates properly with visualization systems
- [ ] Comprehensive documentation and examples included
- [ ] Performance requirements met for search and computation

## Related Documentation

- **[Main AGENTS.md](../AGENTS.md)**: Project-wide agent guidelines
- **[Knowledge README](README.md)**: Knowledge module overview
- **[Applications AGENTS.md](../applications/AGENTS.md)**: Application development guidelines
- **[Research AGENTS.md](../research/AGENTS.md)**: Research tool development guidelines
- **[Visualization AGENTS.md](../visualization/AGENTS.md)**: Visualization system guidelines
- **[Platform AGENTS.md](../platform/AGENTS.md)**: Platform infrastructure guidelines

---

*"Active Inference for, with, by Generative AI"* - Building comprehensive knowledge management through collaborative intelligence and structured educational frameworks.
