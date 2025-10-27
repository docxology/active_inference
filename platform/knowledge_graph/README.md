# Knowledge Graph Engine

**Semantic knowledge representation and reasoning for the Active Inference Knowledge Environment.**

## Overview

The knowledge graph engine provides semantic representation, reasoning, and intelligent navigation of Active Inference concepts, relationships, and knowledge structures.

### Core Features

- **Semantic Representation**: Rich knowledge modeling with ontologies
- **Relationship Inference**: Automatic discovery of concept relationships
- **Intelligent Navigation**: Context-aware concept exploration
- **Query Engine**: Natural language and structured queries
- **Graph Analytics**: Network analysis and visualization
- **Integration**: Seamless integration with knowledge repository

## Architecture

### Knowledge Graph Components

```
┌─────────────────┐
│   Query Layer   │ ← Natural language, SPARQL, GraphQL
├─────────────────┤
│  Reasoning      │ ← Inference, recommendation, validation
│     Engine      │
├─────────────────┤
│   Graph Storage │ ← RDF triples, property graphs
├─────────────────┤
│ Knowledge        │ ← Concept extraction, relationship mining
│   Ingestion     │
└─────────────────┘
```

### Data Model

#### Concept Nodes
```python
concept = {
    "id": "active_inference",
    "type": "framework",
    "properties": {
        "name": "Active Inference",
        "description": "Theoretical framework for behavior",
        "mathematical_basis": ["bayesian_inference", "free_energy_principle"],
        "applications": ["neuroscience", "artificial_intelligence", "psychology"]
    }
}
```

#### Relationships
```python
relationship = {
    "source": "active_inference",
    "target": "free_energy_principle",
    "type": "based_on",
    "strength": 1.0,
    "bidirectional": True
}
```

### Integration Points

- **Knowledge Repository**: Source of structured content
- **Search Engine**: Enhanced search with semantic understanding
- **Visualization**: Graph visualization and exploration
- **Learning Paths**: Intelligent path recommendation

## Usage

### Basic Setup

```python
from platform.knowledge_graph import KnowledgeGraphEngine

# Initialize knowledge graph
config = {
    "storage_backend": "neo4j",  # or "rdf", "in_memory"
    "reasoning_enabled": True,
    "auto_indexing": True
}

kg = KnowledgeGraphEngine(config)
```

### Knowledge Ingestion

```python
# Ingest knowledge content
await kg.ingest_content("knowledge/foundations/active_inference_introduction.json")

# Extract concepts and relationships
concepts = kg.extract_concepts(content)
relationships = kg.infer_relationships(concepts)

# Add to graph
kg.add_concepts(concepts)
kg.add_relationships(relationships)
```

### Semantic Queries

```python
# Natural language query
results = await kg.query("What is the relationship between entropy and active inference?")

# Structured query
query = {
    "start_concept": "entropy",
    "relationship": "related_to",
    "end_concept": "active_inference",
    "max_depth": 3
}

path = kg.find_path(query)

# Concept recommendation
similar_concepts = kg.find_similar("bayesian_inference", limit=5)
```

### Graph Analytics

```python
# Get graph statistics
stats = kg.get_statistics()

print(f"Total concepts: {stats['node_count']}")
print(f"Total relationships: {stats['edge_count']}")
print(f"Graph density: {stats['density']}")

# Find central concepts
central_concepts = kg.find_central_concepts()

# Community detection
communities = kg.detect_communities()

# Path analysis
shortest_paths = kg.find_shortest_paths("entropy", "active_inference")
```

## Configuration

### Storage Configuration

```python
storage_config = {
    "backend": "neo4j",
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "password",
    "database": "active_inference"
}
```

### Reasoning Configuration

```python
reasoning_config = {
    "enabled": True,
    "algorithms": [
        "transitivity",
        "symmetry",
        "owl_reasoning"
    ],
    "confidence_threshold": 0.8,
    "max_inference_depth": 5
}
```

### Indexing Configuration

```python
indexing_config = {
    "auto_index": True,
    "index_types": [
        "fulltext",
        "vector",
        "semantic"
    ],
    "embedding_model": "sentence_transformers",
    "vector_dimensions": 768
}
```

## API Reference

### KnowledgeGraphEngine

Main interface for knowledge graph operations.

#### Core Methods

- `ingest_content(content_path: str) -> bool`: Ingest structured content
- `query(nl_query: str) -> QueryResult`: Natural language query
- `find_path(query: Dict) -> List[Path]`: Find concept paths
- `get_statistics() -> Dict`: Get graph statistics
- `recommend_concepts(concept_id: str, limit: int) -> List[str]`: Concept recommendations

### ConceptExtractor

Extracts concepts and relationships from content.

#### Methods

- `extract_concepts(content: Dict) -> List[Concept]`: Extract concepts from JSON
- `infer_relationships(concepts: List[Concept]) -> List[Relationship]`: Infer relationships
- `validate_concepts(concepts: List[Concept]) -> List[ValidationError]`: Validate concepts

### QueryProcessor

Processes natural language and structured queries.

#### Methods

- `parse_query(query: str) -> ParsedQuery`: Parse natural language query
- `execute_query(parsed: ParsedQuery) -> QueryResult`: Execute parsed query
- `rank_results(results: List, query: str) -> List`: Rank and filter results

## Advanced Features

### Semantic Reasoning

```python
# Enable reasoning
kg.enable_reasoning()

# Infer new relationships
new_relationships = kg.infer_new_relationships()

# Validate consistency
consistency = kg.validate_consistency()
```

### Graph Embeddings

```python
# Generate embeddings
embeddings = kg.generate_embeddings()

# Find similar concepts
similar = kg.find_similar_by_embedding("active_inference", threshold=0.8)

# Visualize embedding space
kg.visualize_embeddings()
```

### Knowledge Validation

```python
# Validate knowledge consistency
validation_results = kg.validate_knowledge()

# Check for circular dependencies
cycles = kg.detect_cycles()

# Validate prerequisite chains
prerequisite_issues = kg.validate_prerequisites()
```

## Performance

### Optimization

```python
# Enable caching
kg.enable_caching(cache_size=10000)

# Optimize queries
kg.optimize_queries()

# Precompute common paths
kg.precompute_paths()
```

### Metrics

```python
# Performance monitoring
metrics = kg.get_performance_metrics()

print(f"Query time: {metrics['avg_query_time']}ms")
print(f"Memory usage: {metrics['memory_usage']}MB")
print(f"Cache hit rate: {metrics['cache_hit_rate']}%")
```

## Testing

### Running Tests

```bash
# Run knowledge graph tests
make test-knowledge-graph

# Or run specific tests
pytest platform/knowledge_graph/tests/ -v

# Performance tests
pytest platform/knowledge_graph/tests/test_performance.py -v
```

### Test Coverage

- **Unit Tests**: Core graph operations
- **Integration Tests**: Knowledge ingestion and querying
- **Performance Tests**: Large graph operations
- **Validation Tests**: Knowledge consistency checks

## Monitoring

### Health Checks

```python
# Graph health check
health = kg.health_check()

print(f"Nodes: {health['node_count']}")
print(f"Relationships: {health['relationship_count']}")
print(f"Last update: {health['last_update']}")
```

### Metrics Dashboard

```bash
# Start metrics dashboard
make kg-metrics

# View real-time metrics
curl http://localhost:8080/kg/metrics
```

## Contributing

See [AGENTS.md](AGENTS.md) for development guidelines and [.cursorrules](../../../.cursorrules) for comprehensive development standards.

### Development Process

1. **Environment Setup**:
   ```bash
   cd platform/knowledge_graph
   make setup
   ```

2. **Testing**:
   ```bash
   make test
   make test-integration
   ```

3. **Documentation**:
   - Update README.md for new features
   - Update AGENTS.md for development patterns
   - Add comprehensive examples

## Security

### Access Control

```python
# Configure graph access
access_config = {
    "public_read": True,
    "authenticated_write": True,
    "admin_full_access": True,
    "audit_logging": True
}
```

### Data Protection

- Encrypted graph storage
- Access logging and audit trails
- GDPR compliance for personal data
- Secure query interfaces

## Troubleshooting

### Common Issues

#### Performance Issues
```bash
# Check query performance
kg.analyze_query_performance()

# Optimize graph structure
kg.optimize_structure()

# Clear caches
kg.clear_caches()
```

#### Data Issues
```bash
# Validate graph integrity
kg.validate_integrity()

# Check for orphaned nodes
orphaned = kg.find_orphaned_nodes()

# Repair broken relationships
kg.repair_relationships()
```

---

**Component Version**: 1.0.0 | **Development Status**: Active Development

*"Active Inference for, with, by Generative AI"* - Semantic understanding through intelligent knowledge representation.
