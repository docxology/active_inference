# Knowledge Repository - Source Code Implementation

This directory contains the source code implementation of the Active Inference knowledge repository system, providing structured educational content management, learning path organization, and semantic knowledge representation.

## Overview

The knowledge module provides the core functionality for managing educational content, organizing learning paths, and maintaining semantic relationships between Active Inference concepts. This module implements the knowledge graph, content validation, and educational workflow management systems.

## Module Structure

```
src/active_inference/knowledge/
├── __init__.py          # Module initialization and public API exports
├── repository.py        # Core knowledge repository implementation
├── foundations.py       # Theoretical foundations management system
├── mathematics.py       # Mathematical formulations and derivations
└── [subdirectories]     # Knowledge organization and assets
    ├── foundations/     # Foundation content implementations
    ├── mathematics/     # Mathematical content implementations
    ├── implementations/ # Implementation examples
    └── applications/    # Application-specific knowledge
```

## Core Components

### 🏗️ Knowledge Repository (`repository.py`)
**Central knowledge management and organization system**
- Knowledge node creation, validation, and management
- Learning path generation and prerequisite validation
- Search and indexing functionality
- Content organization and metadata management

**Key Methods to Implement:**
```python
def search(self, query: str = "", content_types: Optional[List[ContentType]] = None,
           difficulty: Optional[List[DifficultyLevel]] = None, tags: Optional[List[str]] = None,
           limit: int = 50) -> List[KnowledgeNode]:
    """Search knowledge nodes with comprehensive filtering and ranking"""

def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
    """Retrieve specific knowledge node by identifier"""

def get_learning_path(self, path_id: str) -> Optional[LearningPath]:
    """Get structured learning path with prerequisite validation"""

def get_prerequisites_graph(self, node_id: str) -> Dict[str, Any]:
    """Generate prerequisite dependency graph for knowledge node"""

def validate_learning_path(self, path_id: str) -> Dict[str, Any]:
    """Validate learning path completeness and prerequisite satisfaction"""

def export_knowledge_graph(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
    """Export knowledge repository as structured graph data"""

def get_statistics(self) -> Dict[str, Any]:
    """Get comprehensive statistics about knowledge repository content"""

def create_knowledge_node(self, node_data: Dict[str, Any]) -> KnowledgeNode:
    """Create and validate new knowledge node"""

def update_knowledge_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
    """Update existing knowledge node with validation"""

def delete_knowledge_node(self, node_id: str) -> bool:
    """Delete knowledge node and update dependent relationships"""
```

### 📚 Foundations System (`foundations.py`)
**Theoretical foundations management and organization**
- Information theory foundation concepts
- Bayesian inference fundamentals
- Free Energy Principle theoretical framework
- Active Inference conceptual framework

**Key Methods to Implement:**
```python
def get_foundation_tracks(self) -> Dict[str, List[str]]:
    """Get organized foundation learning tracks by topic"""

def get_complete_foundation_path(self) -> List[str]:
    """Get comprehensive foundation learning path with proper sequencing"""

def create_information_theory_nodes(self) -> None:
    """Create and initialize information theory foundation knowledge nodes"""

def create_bayesian_inference_nodes(self) -> None:
    """Create and initialize Bayesian inference foundation knowledge nodes"""

def create_free_energy_nodes(self) -> None:
    """Create and initialize Free Energy Principle foundation knowledge nodes"""

def create_active_inference_nodes(self) -> None:
    """Create and initialize Active Inference foundation knowledge nodes"""

def validate_foundation_content(self, node_id: str) -> Dict[str, Any]:
    """Validate foundation content for accuracy and completeness"""

def generate_foundation_assessment(self, track: str) -> Dict[str, Any]:
    """Generate assessment questions for foundation track"""

def create_foundation_cross_references(self) -> Dict[str, List[str]]:
    """Create cross-references between foundation concepts"""
```

### 📐 Mathematics System (`mathematics.py`)
**Mathematical formulations and computational implementations**
- Probability theory and Bayesian mathematics
- Information theory mathematical foundations
- Variational methods and free energy calculus
- Dynamical systems and stochastic processes

**Key Methods to Implement:**
```python
def derive_free_energy_expression(self) -> Dict[str, Any]:
    """Derive mathematical expression for variational free energy"""

def compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence between discrete probability distributions"""

def expected_free_energy(self, policies: List[np.ndarray], observations: np.ndarray) -> np.ndarray:
    """Compute expected free energy for policy selection in Active Inference"""

def get_mathematical_prerequisites(self) -> Dict[str, List[str]]:
    """Get mathematical prerequisite chains for Active Inference topics"""

def create_mathematical_learning_path(self) -> List[str]:
    """Create comprehensive mathematical learning path with proper sequencing"""

def validate_mathematical_derivation(self, derivation: Dict[str, Any]) -> Dict[str, Any]:
    """Validate mathematical derivation for correctness and completeness"""

def generate_mathematical_examples(self, concept: str) -> List[Dict[str, Any]]:
    """Generate computational examples for mathematical concepts"""

def create_interactive_mathematical_tutorial(self, concept: str) -> Dict[str, Any]:
    """Create interactive tutorial for mathematical concept with validation"""
```

## Implementation Architecture

### Knowledge Node System
The knowledge repository implements a structured node system with:
- **Content Types**: Foundation, Mathematics, Implementation, Application, Tutorial
- **Difficulty Levels**: Beginner, Intermediate, Advanced, Expert
- **Prerequisites**: Automatic prerequisite validation and graph generation
- **Metadata**: Comprehensive metadata for search, filtering, and organization
- **Validation**: Content validation and quality assurance

### Learning Path Management
Learning paths are implemented with:
- **Sequential Organization**: Proper prerequisite ordering and validation
- **Progress Tracking**: Learning progress and completion monitoring
- **Adaptive Recommendations**: Intelligent content recommendation systems
- **Assessment Integration**: Built-in assessment and validation systems

## Development Guidelines

### Content Management
- **Structured Content**: All content follows structured JSON schema
- **Metadata Standards**: Consistent metadata for search and organization
- **Validation Rules**: Comprehensive validation for content integrity
- **Version Control**: Content versioning and change management
- **Cross-References**: Automatic cross-reference generation and validation

### Quality Standards
- **Content Accuracy**: Peer-reviewed mathematical and theoretical content
- **Educational Value**: Progressive disclosure and scaffolding
- **Accessibility**: Multiple learning styles and accessibility support
- **Completeness**: Comprehensive coverage of Active Inference topics
- **Currency**: Updated with latest research developments

## Usage Examples

### Knowledge Repository Usage
```python
from active_inference.knowledge import KnowledgeRepository, KnowledgeRepositoryConfig

# Initialize repository
config = KnowledgeRepositoryConfig(
    root_path=Path("./knowledge"),
    auto_index=True,
    cache_enabled=True
)
repository = KnowledgeRepository(config)

# Search for specific concepts
entropy_nodes = repository.search(
    query="entropy",
    content_types=[ContentType.FOUNDATION, ContentType.MATHEMATICS],
    difficulty=[DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE],
    limit=20
)

print(f"Found {len(entropy_nodes)} entropy-related concepts")

# Get learning path
foundations_path = repository.get_learning_path("foundations_complete")
if foundations_path:
    print(f"Foundations path: {foundations_path.name}")
    print(f"Estimated time: {foundations_path.estimated_hours} hours")

    # Validate path integrity
    validation = repository.validate_learning_path("foundations_complete")
    if validation["valid"]:
        print("Learning path is valid")
    else:
        print(f"Issues found: {validation['issues']}")
```

### Mathematical Content Usage
```python
from active_inference.knowledge import Mathematics

# Initialize mathematics system
math_system = Mathematics(repository)

# Compute information-theoretic measures
p = np.array([0.1, 0.3, 0.4, 0.2])  # Probability distribution
q = np.array([0.2, 0.2, 0.3, 0.3])  # Another distribution

kl_div = math_system.compute_kl_divergence(p, q)
print(f"KL divergence: {kl_div".4f"}")

# Derive free energy expression
free_energy_derivation = math_system.derive_free_energy_expression()
print(f"Free energy expression: {free_energy_derivation['expression']}")

# Get mathematical prerequisites
prerequisites = math_system.get_mathematical_prerequisites()
print(f"Active Inference prerequisites: {prerequisites['active_inference_dynamics']}")
```

## Testing Framework

### Unit Testing Requirements
- **Content Validation**: Test knowledge node creation and validation
- **Search Functionality**: Test search algorithms and filtering
- **Learning Path Logic**: Test prerequisite validation and path generation
- **Mathematical Computations**: Test mathematical function accuracy
- **Integration Testing**: Test integration with other platform components

### Test Structure
```python
class TestKnowledgeRepository(unittest.TestCase):
    """Test knowledge repository functionality"""

    def setUp(self):
        """Set up test environment"""
        self.config = KnowledgeRepositoryConfig(
            root_path=test_knowledge_path,
            auto_index=True
        )
        self.repository = KnowledgeRepository(self.config)

    def test_knowledge_node_creation(self):
        """Test knowledge node creation and validation"""
        node_data = {
            "id": "test_entropy",
            "title": "Test Entropy Concept",
            "content_type": ContentType.FOUNDATION,
            "difficulty": DifficultyLevel.BEGINNER,
            "description": "Test description of entropy",
            "prerequisites": [],
            "tags": ["information_theory", "entropy"],
            "learning_objectives": ["Understand entropy", "Calculate entropy"]
        }

        node = KnowledgeNode(**node_data)
        result = self.repository.add_node(node_data)

        self.assertTrue(result)
        retrieved_node = self.repository.get_node("test_entropy")
        self.assertIsNotNone(retrieved_node)
        self.assertEqual(retrieved_node.title, "Test Entropy Concept")

    def test_search_functionality(self):
        """Test search functionality with various filters"""
        results = self.repository.search(
            query="entropy",
            content_types=[ContentType.FOUNDATION],
            difficulty=[DifficultyLevel.BEGINNER],
            limit=5
        )

        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, KnowledgeNode)
            self.assertIn("entropy", result.title.lower() or result.description.lower())

    def test_learning_path_validation(self):
        """Test learning path validation and prerequisite checking"""
        path_id = "test_path"
        path_data = {
            "id": path_id,
            "name": "Test Learning Path",
            "description": "Test path description",
            "nodes": ["node1", "node2", "node3"],
            "estimated_hours": 5,
            "difficulty": DifficultyLevel.BEGINNER
        }

        # Create test nodes
        for i in range(1, 4):
            node_data = {
                "id": f"node{i}",
                "title": f"Test Node {i}",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.BEGINNER,
                "description": f"Test node {i}",
                "prerequisites": [] if i == 1 else [f"node{i-1}"],
                "tags": ["test"],
                "learning_objectives": [f"Objective {i}"]
            }
            self.repository.add_node(node_data)

        # Add learning path
        path = LearningPath(**path_data)
        self.repository.add_learning_path(path)

        # Validate path
        validation = self.repository.validate_learning_path(path_id)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["path_length"], 3)
```

## Performance Considerations

### Search Performance
- **Indexing Strategy**: Efficient indexing for fast search operations
- **Query Optimization**: Optimized query processing and filtering
- **Caching**: Intelligent caching for frequently accessed content
- **Database Design**: Efficient data structures for knowledge storage

### Mathematical Computation Performance
- **Numerical Stability**: Implement numerically stable algorithms
- **Vectorization**: Use vectorized operations for efficiency
- **Memory Management**: Efficient memory usage for large computations
- **Parallel Processing**: Parallel processing for intensive calculations

## Content Management

### Content Validation
- **Schema Validation**: JSON schema validation for all content
- **Prerequisite Validation**: Automatic prerequisite graph validation
- **Cross-Reference Validation**: Validate all internal references
- **Quality Assurance**: Automated quality checks and validation

### Content Organization
- **Hierarchical Structure**: Clear hierarchical content organization
- **Tagging System**: Comprehensive tagging for search and filtering
- **Version Management**: Content versioning and change tracking
- **Metadata Management**: Rich metadata for search and discovery

## Contributing Guidelines

When contributing to the knowledge module:

1. **Content Creation**: Create structured content following established schemas
2. **Prerequisite Management**: Ensure proper prerequisite relationships
3. **Validation**: Include comprehensive validation and testing
4. **Documentation**: Update README and AGENTS files
5. **Quality Assurance**: Ensure content meets educational standards
6. **Review Process**: Submit for peer review and validation

## Related Documentation

- **[Main README](../README.md)**: Main package documentation
- **[AGENTS.md](AGENTS.md)**: Agent development guidelines for this module
- **[Repository Documentation](repository.py)**: Core repository system details
- **[Foundations Documentation](foundations.py)**: Theoretical foundations details
- **[Mathematics Documentation](mathematics.py)**: Mathematical formulations details
- **[Knowledge Content](../../knowledge/)**: Educational content organization

---

*"Active Inference for, with, by Generative AI"* - Building comprehensive knowledge management through collaborative intelligence and structured educational frameworks.
