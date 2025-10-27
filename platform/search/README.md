# Intelligent Search Engine

**Multi-modal intelligent search and retrieval for the Active Inference Knowledge Environment.**

## Overview

The search engine provides intelligent, multi-modal search capabilities across all Active Inference knowledge content, research tools, and platform resources with semantic understanding and personalized results.

### Core Features

- **Multi-Modal Search**: Text, semantic, and structured queries
- **Intelligent Ranking**: Context-aware result ranking and filtering
- **Personalization**: User-specific search results and recommendations
- **Real-time Indexing**: Automatic content indexing and updates
- **Advanced Filters**: Multi-faceted filtering and refinement
- **Search Analytics**: Query analysis and optimization insights

## Architecture

### Search Components

```
┌─────────────────┐
│   Query Layer   │ ← Natural language, semantic, structured
├─────────────────┤
│   Ranking       │ ← Relevance scoring, personalization
│     Engine      │
├─────────────────┤
│   Index Storage │ ← Inverted index, embeddings, metadata
├─────────────────┤
│ Content          │ ← Crawling, extraction, indexing
│   Ingestion     │
└─────────────────┘
```

### Search Pipeline

1. **Query Processing**: Parse and understand user queries
2. **Content Retrieval**: Find relevant documents and content
3. **Relevance Scoring**: Rank results by relevance and context
4. **Personalization**: Adapt results to user preferences
5. **Result Formatting**: Format and highlight results

## Usage

### Basic Setup

```python
from platform.search import SearchEngine

# Initialize search engine
config = {
    "backend": "elasticsearch",  # or "solr", "opensearch"
    "indexing": "auto",
    "personalization": True
}

search = SearchEngine(config)
```

### Basic Search

```python
# Text search
results = search.search("active inference fundamentals")

# Semantic search
semantic_results = search.semantic_search("how does the brain make decisions?")

# Advanced search with filters
advanced_results = search.advanced_search({
    "query": "bayesian inference",
    "filters": {
        "content_type": "foundation",
        "difficulty": "beginner",
        "tags": ["probability"]
    },
    "sort": "relevance",
    "limit": 20
})
```

### Content Indexing

```python
# Index new content
await search.index_content("knowledge/foundations/active_inference_introduction.json")

# Index directory
await search.index_directory("knowledge/foundations/")

# Reindex all content
await search.reindex_all()
```

### Search Suggestions

```python
# Get search suggestions
suggestions = search.get_suggestions("active inf")

# Auto-complete queries
completions = search.autocomplete("bayesian")

# Related concepts
related = search.get_related_concepts("entropy")
```

## Configuration

### Backend Configuration

```python
backend_config = {
    "type": "elasticsearch",
    "hosts": ["localhost:9200"],
    "index_name": "active_inference",
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 1
    }
}
```

### Indexing Configuration

```python
indexing_config = {
    "auto_index": True,
    "batch_size": 1000,
    "refresh_interval": "30s",
    "embedding_model": "sentence_transformers",
    "fields_to_index": [
        "title",
        "description",
        "content",
        "tags",
        "metadata"
    ]
}
```

### Personalization Configuration

```python
personalization_config = {
    "enabled": True,
    "user_history_weight": 0.3,
    "collaborative_filtering": True,
    "learning_rate": 0.1,
    "decay_rate": 0.95
}
```

## API Reference

### SearchEngine

Main interface for search functionality.

#### Core Methods

- `search(query: str, options: Dict = None) -> SearchResult`: Text-based search
- `semantic_search(query: str, limit: int = 20) -> SearchResult`: Semantic search
- `advanced_search(params: Dict) -> SearchResult`: Advanced search with filters
- `index_content(content_path: str) -> bool`: Index new content
- `get_suggestions(query: str) -> List[str]`: Get search suggestions

### QueryProcessor

Processes and optimizes search queries.

#### Methods

- `parse_query(query: str) -> ParsedQuery`: Parse natural language query
- `expand_query(parsed: ParsedQuery) -> ExpandedQuery`: Expand with synonyms
- `optimize_query(query: str) -> OptimizedQuery`: Optimize for performance
- `extract_filters(query: str) -> Dict`: Extract search filters

### ResultRanker

Ranks and filters search results.

#### Methods

- `rank_results(results: List, query: str) -> List`: Rank by relevance
- `apply_filters(results: List, filters: Dict) -> List`: Apply search filters
- `personalize_results(results: List, user_id: str) -> List`: Personalize results
- `highlight_snippets(results: List) -> List`: Highlight relevant snippets

## Advanced Features

### Semantic Search

```python
# Configure semantic search
semantic_config = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384,
    "similarity_threshold": 0.7
}

# Semantic similarity search
similar_docs = search.find_similar_documents(
    source_doc_id="active_inference_basics",
    threshold=0.8,
    limit=10
)

# Concept-based search
concept_results = search.search_by_concept("free_energy_principle")
```

### Personalized Search

```python
# Track user behavior
search.track_user_interaction(user_id, doc_id, interaction_type="view")

# Get personalized recommendations
recommendations = search.get_personalized_recommendations(user_id, limit=10)

# Update user preferences
search.update_user_preferences(user_id, preferences)
```

### Search Analytics

```python
# Query analytics
analytics = search.get_query_analytics()

print(f"Top queries: {analytics['top_queries']}")
print(f"Zero results rate: {analytics['zero_results_rate']}")
print(f"Average query length: {analytics['avg_query_length']}")

# User behavior analytics
behavior = search.get_user_behavior_analytics()
```

## Performance

### Optimization

```python
# Enable search caching
search.enable_caching(cache_size=1000, ttl=300)

# Optimize index structure
search.optimize_index()

# Pre-warm popular queries
search.prewarm_queries(popular_queries)
```

### Performance Metrics

```python
# Monitor performance
metrics = search.get_performance_metrics()

print(f"Query time: {metrics['avg_query_time']}ms")
print(f"Index size: {metrics['index_size']}GB")
print(f"Cache hit rate: {metrics['cache_hit_rate']}%")
print(f"Throughput: {metrics['queries_per_second']} QPS")
```

## Testing

### Running Tests

```bash
# Run search tests
make test-search

# Or run specific tests
pytest platform/search/tests/ -v

# Load testing
pytest platform/search/tests/test_load.py -v

# Integration tests
pytest platform/search/tests/test_integration.py -v
```

### Test Coverage

- **Unit Tests**: Query processing and ranking
- **Integration Tests**: End-to-end search workflows
- **Performance Tests**: Search performance under load
- **Analytics Tests**: Search analytics and reporting

## Monitoring

### Health Checks

```python
# Search health check
health = search.health_check()

print(f"Index status: {health['index_status']}")
print(f"Query performance: {health['query_performance']}")
print(f"Disk usage: {health['disk_usage']}GB")
print(f"Last update: {health['last_update']}")
```

### Search Analytics Dashboard

```bash
# Start analytics dashboard
make search-analytics

# View real-time metrics
curl http://localhost:8080/search/analytics
```

## Integration

### Knowledge Repository Integration

```python
# Automatic indexing of new content
search.watch_content_directory("knowledge/")

# Index learning paths
search.index_learning_paths()

# Index research experiments
search.index_experiments()
```

### Platform Integration

```python
# Integrate with user management
search.set_user_service(user_service)

# Integrate with recommendation engine
search.set_recommendation_service(recommendation_service)

# Integrate with analytics
search.set_analytics_service(analytics_service)
```

## Contributing

See [AGENTS.md](AGENTS.md) for development guidelines and [.cursorrules](../../../.cursorrules) for comprehensive development standards.

### Development Process

1. **Environment Setup**:
   ```bash
   cd platform/search
   make setup
   ```

2. **Testing**:
   ```bash
   make test
   make test-performance
   ```

3. **Documentation**:
   - Update README.md for new features
   - Update AGENTS.md for development patterns
   - Add comprehensive examples

## Security

### Query Security

```python
# Enable query sanitization
search.enable_query_sanitization()

# Configure access control
search.set_access_control(permissions)

# Enable audit logging
search.enable_audit_logging()
```

### Data Protection

- Encrypted search indexes
- Secure query processing
- GDPR compliance for personal data
- Query log anonymization

## Troubleshooting

### Common Issues

#### Search Performance
```bash
# Check index performance
search.analyze_index_performance()

# Optimize search queries
search.optimize_queries()

# Reindex content
search.reindex_all()
```

#### No Results
```bash
# Check content indexing
search.check_indexing_status()

# Validate search configuration
search.validate_configuration()

# Test with simple queries
search.test_basic_queries()
```

#### Index Issues
```bash
# Check index health
search.check_index_health()

# Repair corrupted index
search.repair_index()

# Rebuild index
search.rebuild_index()
```

---

**Component Version**: 1.0.0 | **Development Status**: Active Development

*"Active Inference for, with, by Generative AI"* - Intelligent discovery through semantic understanding.
