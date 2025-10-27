# Platform Search Engine - Agent Development Guide

**Guidelines for AI agents working with search and discovery systems in the Active Inference Knowledge Environment.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with platform search systems:**

### Primary Responsibilities
- **Search Implementation**: Develop and maintain search and discovery capabilities
- **Information Retrieval**: Implement advanced information retrieval and ranking algorithms
- **Semantic Understanding**: Create systems for semantic search and concept understanding
- **Recommendation Systems**: Build intelligent recommendation and personalization engines
- **Search Analytics**: Implement search analytics and performance optimization

### Development Focus Areas
1. **Search Engine Development**: Implement core search functionality and indexing
2. **Semantic Search**: Develop semantic understanding and natural language processing
3. **Knowledge Graph Integration**: Build graph-based search and navigation
4. **Recommendation Systems**: Create personalized recommendation algorithms
5. **Search Analytics**: Implement search behavior analysis and optimization

## 🏗️ Architecture & Integration

### Search System Architecture

**Understanding how search systems fit into the larger platform ecosystem:**

```
Platform Services Layer
├── Search & Discovery (Text, semantic, graph, recommendations)
├── Content Management (Indexing, processing, metadata)
├── Knowledge Organization (Graph navigation, concept mapping)
└── User Experience (Interface, personalization, analytics)
```

### Integration Points

**Key integration points and dependencies:**

#### Upstream Components
- **Content Systems**: All platform content that needs to be indexed and searchable
- **Knowledge Graph**: Semantic relationships and concept organization
- **User Management**: User profiles and personalization data
- **Analytics Systems**: Search behavior and performance data

#### Downstream Components
- **User Interfaces**: Search interface integration across all platform components
- **Educational Systems**: Learning path discovery and content recommendation
- **Research Tools**: Research discovery and literature search
- **Community Features**: Social search and collaborative discovery

#### External Systems
- **Search Technologies**: Elasticsearch, Solr, OpenSearch, specialized search engines
- **NLP Libraries**: spaCy, NLTK, Hugging Face transformers for natural language processing
- **Vector Databases**: FAISS, Pinecone, Weaviate for semantic vector search
- **Machine Learning**: TensorFlow, PyTorch for recommendation systems

### Search Flow Patterns

```python
# Typical search system workflow
content → indexing → search → ranking → personalization → presentation → feedback
```

## 💻 Development Patterns

### Required Implementation Patterns

**All search development must follow these patterns:**

#### 1. Search Engine Factory Pattern (PREFERRED)

```python
def create_search_engine(search_type: str, config: Dict[str, Any]) -> BaseSearchEngine:
    """Create search engine using factory pattern with validation"""

    search_factories = {
        'full_text': create_full_text_search_engine,
        'semantic': create_semantic_search_engine,
        'graph': create_graph_search_engine,
        'recommendation': create_recommendation_engine,
        'hybrid': create_hybrid_search_engine,
        'multimodal': create_multimodal_search_engine
    }

    if search_type not in search_factories:
        raise ValueError(f"Unknown search type: {search_type}")

    # Validate search configuration
    validate_search_config(config)

    # Create search engine with error handling
    try:
        search_engine = search_factories[search_type](config)
        validate_search_functionality(search_engine)
        return search_engine
    except Exception as e:
        logger.error(f"Search engine creation failed: {e}")
        raise SearchEngineError(f"Failed to create search engine: {search_type}") from e

def validate_search_config(config: Dict[str, Any]) -> None:
    """Validate search engine configuration"""

    required_fields = ['search_type', 'content_sources', 'indexing_strategy']

    for field in required_fields:
        if field not in config:
            raise SearchConfigurationError(f"Missing required field: {field}")

    # Type-specific validation
    if config['search_type'] == 'semantic':
        validate_semantic_config(config)
    elif config['search_type'] == 'graph':
        validate_graph_config(config)
    elif config['search_type'] == 'recommendation':
        validate_recommendation_config(config)
```

#### 2. Search Configuration Pattern (MANDATORY)

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class SearchEngineConfig:
    """Configuration for search engine implementations"""

    # Core search settings
    search_type: str
    content_sources: List[str]
    indexing_strategy: str
    ranking_algorithm: str

    # Performance settings
    max_results: int = 100
    search_timeout: int = 30
    caching_enabled: bool = True
    parallel_processing: bool = True

    # Quality settings
    relevance_threshold: float = 0.7
    diversity_factor: float = 0.3
    personalization_enabled: bool = True
    explanation_enabled: bool = True

    # Integration settings
    knowledge_graph_integration: bool = True
    real_time_indexing: bool = False
    cross_platform_search: bool = False

    def validate_configuration(self) -> List[str]:
        """Validate search configuration"""
        errors = []

        # Check required fields
        required_fields = ['search_type', 'content_sources', 'indexing_strategy']
        for field in required_fields:
            if not getattr(self, field):
                errors.append(f"Missing required field: {field}")

        # Validate search type
        valid_types = ['full_text', 'semantic', 'graph', 'recommendation', 'hybrid', 'multimodal']
        if self.search_type not in valid_types:
            errors.append(f"Invalid search type: {self.search_type}")

        # Validate performance parameters
        if self.max_results <= 0:
            errors.append("max_results must be positive")

        if self.search_timeout <= 0:
            errors.append("search_timeout must be positive")

        return errors

    def to_search_parameters(self) -> Dict[str, Any]:
        """Convert configuration to search parameters"""
        return {
            'search_type': self.search_type,
            'content_sources': self.content_sources,
            'indexing_strategy': self.indexing_strategy,
            'ranking_algorithm': self.ranking_algorithm,
            'max_results': self.max_results,
            'search_timeout': self.search_timeout,
            'caching_enabled': self.caching_enabled,
            'parallel_processing': self.parallel_processing,
            'relevance_threshold': self.relevance_threshold,
            'diversity_factor': self.diversity_factor,
            'personalization_enabled': self.personalization_enabled,
            'explanation_enabled': self.explanation_enabled,
            'knowledge_graph_integration': self.knowledge_graph_integration,
            'real_time_indexing': self.real_time_indexing,
            'cross_platform_search': self.cross_platform_search
        }
```

#### 3. Search Result Pattern (MANDATORY)

```python
def create_search_result(query: str, results: List[Any], config: Dict[str, Any]) -> SearchResult:
    """Create comprehensive search result with explanations"""

    # Validate search results
    result_validation = validate_search_results(results, config)
    if not result_validation['valid']:
        raise SearchResultError(f"Invalid search results: {result_validation['errors']}")

    # Rank results by relevance
    ranked_results = rank_search_results(results, query, config)

    # Add explanations for each result
    explained_results = add_result_explanations(ranked_results, query, config)

    # Add personalization if enabled
    if config.get('personalization_enabled', False):
        personalized_results = personalize_results(explained_results, config)
    else:
        personalized_results = explained_results

    # Add diversity if required
    if config.get('diversity_factor', 0) > 0:
        diversified_results = add_result_diversity(personalized_results, config)
    else:
        diversified_results = personalized_results

    return SearchResult(
        query=query,
        results=diversified_results,
        ranking_config=config,
        validation=result_validation,
        timestamp=datetime.now()
    )

def add_result_explanations(results: List[Any], query: str, config: Dict[str, Any]) -> List[ExplainedResult]:
    """Add explanations for why each result was selected"""

    explained_results = []

    for result in results:
        # Generate explanation
        explanation = generate_result_explanation(result, query, config)

        # Add confidence score
        confidence = calculate_result_confidence(result, query, config)

        # Add related concepts
        related_concepts = find_related_concepts(result, config)

        explained_result = ExplainedResult(
            result=result,
            explanation=explanation,
            confidence=confidence,
            related_concepts=related_concepts
        )

        explained_results.append(explained_result)

    return explained_results
```

## 🧪 Search Testing Standards

### Search Testing Categories (MANDATORY)

#### 1. Search Accuracy Testing
**Test search accuracy and relevance:**

```python
def test_search_accuracy():
    """Test search accuracy and relevance"""
    # Test full-text search accuracy
    text_queries = [
        "active inference",
        "belief updating",
        "free energy principle",
        "variational inference",
        "predictive coding"
    ]

    for query in text_queries:
        results = perform_text_search(query)
        accuracy = measure_search_accuracy(results, query)

        assert accuracy['precision'] > 0.8, f"Low precision for query: {query}"
        assert accuracy['recall'] > 0.7, f"Low recall for query: {query}"

def test_semantic_search():
    """Test semantic search understanding"""
    # Test semantic understanding
    semantic_queries = [
        ("decision making under uncertainty", "active inference decision theory"),
        ("neural prediction", "predictive coding neuroscience"),
        ("robot learning", "active inference robotics")
    ]

    for query, expected_domain in semantic_queries:
        results = perform_semantic_search(query)
        relevance = measure_semantic_relevance(results, expected_domain)

        assert relevance['semantic_score'] > 0.75, f"Low semantic relevance for: {query}"
```

#### 2. Search Performance Testing
**Test search performance and scalability:**

```python
def test_search_performance():
    """Test search performance characteristics"""
    # Test response time
    queries = generate_performance_test_queries()

    response_times = []
    for query in queries:
        start_time = time.perf_counter()
        results = perform_search(query)
        end_time = time.perf_counter()

        response_time = end_time - start_time
        response_times.append(response_time)

        # Validate performance requirements
        assert response_time < 2.0, f"Search too slow: {response_time}s"

    # Test throughput
    concurrent_queries = 10
    throughput_test = test_concurrent_search_performance(concurrent_queries)
    assert throughput_test['queries_per_second'] > 5, "Insufficient search throughput"

def test_search_scalability():
    """Test search scalability with content size"""
    # Test with different content sizes
    content_sizes = [1000, 10000, 100000, 1000000]  # documents

    for size in content_sizes:
        # Index content of specified size
        test_content = generate_test_content(size)
        index_content(test_content)

        # Test search performance
        performance = measure_search_performance_with_size(size)

        # Validate scalability
        if size > 10000:
            assert performance['response_time'] < 5.0, f"Performance degraded at size {size}"
            assert performance['memory_usage'] < expected_memory_for_size(size)
```

#### 3. Recommendation System Testing
**Test recommendation accuracy and personalization:**

```python
def test_recommendation_accuracy():
    """Test recommendation system accuracy"""
    # Test user behavior simulation
    test_users = create_test_user_profiles()

    for user in test_users:
        # Generate recommendations
        recommendations = generate_user_recommendations(user)

        # Validate recommendation quality
        quality_metrics = evaluate_recommendation_quality(recommendations, user)
        assert quality_metrics['relevance_score'] > 0.8, f"Low recommendation relevance for user {user['id']}"

        # Test personalization
        personalization_test = test_recommendation_personalization(recommendations, user)
        assert personalization_test['personalized'], f"Recommendations not personalized for user {user['id']}"

def test_recommendation_diversity():
    """Test recommendation diversity and novelty"""
    # Test recommendation diversity
    user_recommendations = generate_user_recommendations(test_user)

    diversity_metrics = calculate_recommendation_diversity(user_recommendations)
    assert diversity_metrics['diversity_score'] > 0.6, "Recommendations not diverse enough"

    novelty_metrics = calculate_recommendation_novelty(user_recommendations, user)
    assert novelty_metrics['novelty_score'] > 0.4, "Recommendations not novel enough"
```

### Search Coverage Requirements

- **Query Coverage**: Support for all types of search queries
- **Content Coverage**: Search across all platform content types
- **Integration Coverage**: Integration with all platform components
- **Performance Coverage**: All search operations meet performance requirements
- **Analytics Coverage**: Complete search behavior tracking and analysis

### Search Testing Commands

```bash
# Test all search functionality
make test-search-engine

# Test search accuracy
pytest platform/search/tests/test_accuracy.py -v

# Test search performance
pytest platform/search/tests/test_performance.py -v

# Test recommendation systems
pytest platform/search/tests/test_recommendations.py -v

# Test search integration
pytest platform/search/tests/test_integration.py -v

# Validate search quality
python platform/search/validate_search_quality.py
```

## 📖 Search Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. Search Interface Documentation
**All search interfaces must have comprehensive documentation:**

```python
def document_search_interface(search_engine: BaseSearchEngine, config: Dict[str, Any]) -> str:
    """Document search interface with comprehensive guidance"""

    interface_documentation = {
        "api_overview": document_search_api_overview(search_engine, config),
        "query_syntax": document_query_syntax(search_engine, config),
        "filtering_options": document_filtering_options(search_engine, config),
        "sorting_options": document_sorting_options(search_engine, config),
        "result_format": document_result_format(search_engine, config),
        "error_handling": document_error_handling(search_engine, config)
    }

    return format_interface_documentation(interface_documentation)

def document_query_syntax(engine: BaseSearchEngine, config: Dict[str, Any]) -> str:
    """Document query syntax and capabilities"""

    # Document basic query syntax
    basic_syntax = """
# Basic Query Syntax

## Simple Queries
- Single terms: `active inference`
- Phrases: `"free energy principle"`
- Boolean: `active AND inference`, `belief OR policy`

## Field Queries
- Title: `title:active inference`
- Author: `author:karl friston`
- Type: `type:research_paper`
- Date: `date:2024`

## Advanced Queries
- Wildcards: `inferen*` (matches inference, inferential, etc.)
- Fuzzy: `inferense~2` (matches inference with 2 character difference)
- Proximity: `"active inference"~5` (words within 5 positions)
"""

    # Document semantic query syntax
    semantic_syntax = """
# Semantic Queries
- Concept search: `concept:active inference decision making`
- Related concepts: `related to:active inference`
- Similar content: `similar:active inference tutorial`
- Learning path: `learning path:active inference`
"""

    return basic_syntax + semantic_syntax
```

#### 2. Search Algorithm Documentation
**All search algorithms must be thoroughly documented:**

```python
def document_search_algorithms(engine: BaseSearchEngine, config: Dict[str, Any]) -> str:
    """Document search algorithms and ranking methods"""

    algorithm_documentation = {
        "ranking_algorithms": document_ranking_algorithms(engine, config),
        "indexing_algorithms": document_indexing_algorithms(engine, config),
        "semantic_algorithms": document_semantic_algorithms(engine, config),
        "recommendation_algorithms": document_recommendation_algorithms(engine, config),
        "performance_algorithms": document_performance_algorithms(engine, config)
    }

    return format_algorithm_documentation(algorithm_documentation)
```

#### 3. Integration Documentation
**Search integration must be comprehensively documented:**

```python
def document_search_integration(engine: BaseSearchEngine, config: Dict[str, Any]) -> str:
    """Document search integration with platform systems"""

    integration_documentation = {
        "content_integration": document_content_integration(engine, config),
        "user_integration": document_user_integration(engine, config),
        "knowledge_graph_integration": document_knowledge_graph_integration(engine, config),
        "analytics_integration": document_analytics_integration(engine, config),
        "performance_integration": document_performance_integration(engine, config)
    }

    return format_integration_documentation(integration_documentation)
```

## 🚀 Performance Optimization

### Search Performance Requirements

**Search systems must meet these performance standards:**

- **Query Response Time**: <2 seconds for typical queries
- **Indexing Speed**: Efficient indexing of new content
- **Scalability**: Handle growing content and user base
- **Memory Efficiency**: Efficient memory usage for large indices
- **Concurrent Users**: Support multiple simultaneous users

### Optimization Techniques

#### 1. Search Index Optimization

```python
def optimize_search_indexing(index_config: Dict[str, Any]) -> OptimizedIndex:
    """Optimize search indexing for performance and efficiency"""

    # Choose optimal indexing strategy
    optimal_strategy = select_indexing_strategy(index_config)

    # Optimize index structure
    optimized_structure = optimize_index_structure(optimal_strategy)

    # Implement incremental indexing
    incremental_indexing = implement_incremental_indexing(optimized_structure)

    # Add index caching
    index_caching = implement_index_caching(incremental_indexing)

    # Validate optimization
    optimization_validation = validate_indexing_optimization(index_caching, index_config)

    return OptimizedIndex(
        strategy=optimal_strategy,
        structure=optimized_structure,
        incremental=incremental_indexing,
        caching=index_caching,
        validation=optimization_validation
    )
```

#### 2. Query Processing Optimization

```python
def optimize_query_processing(query_config: Dict[str, Any]) -> OptimizedQueryProcessor:
    """Optimize query processing for speed and relevance"""

    # Optimize query parsing
    optimized_parsing = optimize_query_parsing(query_config)

    # Optimize search execution
    optimized_execution = optimize_search_execution(optimized_parsing)

    # Optimize result ranking
    optimized_ranking = optimize_result_ranking(optimized_execution)

    # Implement query caching
    query_caching = implement_query_caching(optimized_ranking)

    # Add parallel processing
    parallel_processing = implement_parallel_query_processing(query_caching)

    return OptimizedQueryProcessor(
        parsing=optimized_parsing,
        execution=optimized_execution,
        ranking=optimized_ranking,
        caching=query_caching,
        parallel=parallel_processing
    )
```

## 🔒 Search Security Standards

### Search Security Requirements (MANDATORY)

#### 1. Query Security

```python
def validate_query_security(query: str, security_config: Dict[str, Any]) -> SecurityResult:
    """Validate security of search queries"""

    security_checks = {
        "injection_prevention": prevent_query_injection(query),
        "input_validation": validate_query_input(query),
        "resource_limits": enforce_query_resource_limits(query),
        "content_filtering": apply_content_filtering(query)
    }

    return {
        "secure": all(security_checks.values()),
        "checks": security_checks,
        "violations": [k for k, v in security_checks.items() if not v]
    }

def sanitize_search_query(raw_query: str) -> str:
    """Sanitize search query for security"""

    # Remove potentially dangerous content
    sanitized = remove_dangerous_query_content(raw_query)

    # Validate query structure
    sanitized = validate_query_structure(sanitized)

    # Apply length limits
    sanitized = apply_query_length_limits(sanitized)

    # Add security logging
    log_security_check(raw_query, sanitized)

    return sanitized
```

#### 2. Result Security

```python
def validate_search_result_security(results: List[Any], security_config: Dict[str, Any]) -> SecurityResult:
    """Validate security of search results"""

    security_validation = {
        "content_safety": validate_result_content_safety(results),
        "access_control": validate_result_access_control(results),
        "data_privacy": validate_result_data_privacy(results),
        "information_disclosure": check_result_information_disclosure(results)
    }

    return {
        "secure": all(security_validation.values()),
        "validation": security_validation,
        "risks": [k for k, v in security_validation.items() if not v]
    }
```

## 🐛 Search Debugging & Troubleshooting

### Debug Configuration

```python
# Enable search debugging
debug_config = {
    "debug": True,
    "query_debugging": True,
    "indexing_debugging": True,
    "ranking_debugging": True,
    "performance_monitoring": True
}

# Debug search development
debug_search_workflow(debug_config)
```

### Common Debugging Patterns

#### 1. Search Accuracy Debugging

```python
def debug_search_accuracy(engine: BaseSearchEngine, test_queries: List[str]) -> DebugResult:
    """Debug search accuracy issues"""

    # Test query processing
    query_processing_debug = debug_query_processing(engine, test_queries)
    if not query_processing_debug['correct']:
        return {"type": "query_processing", "issues": query_processing_debug['issues']}

    # Test result ranking
    ranking_debug = debug_result_ranking(engine, test_queries)
    if not ranking_debug['accurate']:
        return {"type": "ranking", "issues": ranking_debug['issues']}

    # Test relevance scoring
    relevance_debug = debug_relevance_scoring(engine, test_queries)
    if not relevance_debug['relevant']:
        return {"type": "relevance", "issues": relevance_debug['issues']}

    return {"status": "search_accurate"}

def debug_relevance_scoring(engine: BaseSearchEngine, queries: List[str]) -> Dict[str, Any]:
    """Debug relevance scoring issues"""

    relevance_issues = []

    for query in queries:
        # Get search results
        results = engine.search(query)

        # Check result relevance
        relevance_scores = []
        for result in results[:10]:  # Check top 10 results
            relevance = calculate_manual_relevance(result, query)
            relevance_scores.append(relevance)

        # Check if relevant results are ranked highly
        if np.mean(relevance_scores[:3]) < 0.7:  # Top 3 should be relevant
            relevance_issues.append({
                "query": query,
                "top_results_relevance": relevance_scores[:3],
                "issue": "Top results not relevant"
            })

    return {
        "relevant": len(relevance_issues) == 0,
        "issues": relevance_issues,
        "recommendations": generate_relevance_improvements(relevance_issues)
    }
```

#### 2. Performance Debugging

```python
def debug_search_performance(engine: BaseSearchEngine, performance_config: Dict[str, Any]) -> DebugResult:
    """Debug search performance issues"""

    # Profile query processing time
    query_time_profile = profile_query_processing_time(engine)

    if query_time_profile['too_slow']:
        return {"type": "query_time", "profile": query_time_profile}

    # Profile indexing performance
    indexing_profile = profile_indexing_performance(engine)

    if indexing_profile['indexing_issues']:
        return {"type": "indexing", "profile": indexing_profile}

    # Profile memory usage
    memory_profile = profile_search_memory_usage(engine)

    if memory_profile['memory_issues']:
        return {"type": "memory", "profile": memory_profile}

    return {"status": "performance_ok"}
```

## 🔄 Development Workflow

### Agent Development Process

1. **Search System Assessment**
   - Understand current search capabilities and limitations
   - Identify search gaps and improvement opportunities
   - Review existing search performance and user experience

2. **Search Architecture Planning**
   - Design comprehensive search system architecture
   - Plan integration with platform content systems
   - Consider performance and scalability requirements

3. **Search Implementation Development**
   - Implement robust search algorithms and indexing
   - Create comprehensive semantic understanding
   - Develop intelligent recommendation systems

4. **Search Quality Assurance**
   - Test search accuracy and relevance
   - Validate performance and scalability
   - Ensure security and privacy compliance

5. **Integration and Optimization**
   - Test integration with platform systems
   - Optimize search performance and user experience
   - Update documentation and user guidance

### Code Review Checklist

**Before submitting search code for review:**

- [ ] **Search Accuracy**: Search results are accurate and relevant
- [ ] **Performance**: Search operations meet performance requirements
- [ ] **Integration**: Search integrates properly with platform systems
- [ ] **Security**: Search operations are secure and protect user privacy
- [ ] **Documentation**: Comprehensive documentation for all search features
- [ ] **Testing**: Thorough testing including edge cases and performance tests
- [ ] **Standards Compliance**: Follows all development and quality standards

## 📚 Learning Resources

### Search Development Resources

- **[Search Engine AGENTS.md](AGENTS.md)**: Search development guidelines
- **[Information Retrieval](https://example.com)**: Information retrieval fundamentals
- **[Search Engine Design](https://example.com)**: Search system architecture
- **[Natural Language Processing](https://example.com)**: NLP for search applications

### Technical References

- **[Elasticsearch Documentation](https://example.com)**: Elasticsearch search engine
- **[Vector Search](https://example.com)**: Semantic vector search techniques
- **[Recommendation Systems](https://example.com)**: Recommendation algorithm design
- **[Search Analytics](https://example.com)**: Search behavior analysis

### Related Components

Study these related components for integration patterns:

- **[Knowledge Graph](../../../platform/knowledge_graph/)**: Semantic knowledge organization
- **[Content Management](../../../platform/)**: Content indexing and management
- **[User Analytics](../../../platform/)**: User behavior and personalization
- **[Platform Services](../../../platform/)**: Platform infrastructure integration

## 🎯 Success Metrics

### Search Quality Metrics

- **Search Accuracy**: >90% relevance for typical queries
- **Response Time**: <2 seconds for 95% of queries
- **User Satisfaction**: >85% user satisfaction with search results
- **Content Coverage**: 100% of platform content searchable
- **Recommendation Quality**: >80% recommendation acceptance rate

### Development Metrics

- **Implementation Speed**: Search features implemented within 2 months
- **Quality Score**: Consistent high-quality search implementations
- **Integration Success**: Seamless integration with platform systems
- **Performance Achievement**: All performance requirements met
- **Maintenance Efficiency**: Easy to update and optimize search systems

---

**Platform Search Engine**: Agent Guide | **Version**: 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Enabling intelligent discovery through advanced search, semantic understanding, and personalized knowledge navigation in the Active Inference ecosystem.