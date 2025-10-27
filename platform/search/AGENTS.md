# Platform Search - Agent Development Guide

**Guidelines for AI agents working with intelligent search and retrieval systems.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with search systems:**

### Primary Responsibilities
- **Search Engine Development**: Build intelligent search capabilities
- **Query Processing**: Natural language and structured query understanding
- **Indexing Systems**: Content indexing and retrieval optimization
- **Ranking Algorithms**: Relevance scoring and result ranking
- **Personalization**: User-specific search result customization
- **Analytics**: Search performance and usage analytics

### Development Focus Areas
1. **Search Infrastructure**: Build scalable search systems
2. **Query Understanding**: Natural language processing for search
3. **Content Indexing**: Efficient content indexing and retrieval
4. **Result Ranking**: Machine learning for relevance ranking
5. **Search Analytics**: Usage patterns and performance monitoring

## 🏗️ Architecture & Integration

### Search Architecture

**Understanding the search system structure:**

```
Search Infrastructure
├── Query Processing (NLP, parsing)
├── Index Management (storage, updates)
├── Ranking Engine (relevance, personalization)
├── Result Formatting (highlighting, snippets)
└── Analytics Engine (metrics, optimization)
```

### Integration Points

**Key integration points for search:**

#### Platform Integration
- **Knowledge Repository**: Content indexing and search
- **User Management**: Personalized search results
- **Content Management**: Real-time content indexing
- **Analytics**: Search usage and performance tracking

#### External Systems
- **Search Engines**: Elasticsearch, Solr, OpenSearch
- **Vector Databases**: Pinecone, Weaviate, Chroma
- **NLP Services**: Natural language processing APIs
- **Analytics Platforms**: Search analytics and optimization tools

### Search Pipeline

```python
# Search processing pipeline
user_query → query_parsing → content_retrieval → relevance_ranking → result_formatting → user_results
```

## 💻 Development Patterns

### Required Implementation Patterns

**All search development must follow these patterns:**

#### 1. Query Processing Pattern
```python
class QueryProcessor:
    """Process and understand search queries"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp_engine = NLPEngine(config)
        self.query_parser = QueryParser(config)

    async def process_query(self, query: str, context: Dict = None) -> ProcessedQuery:
        """Process natural language query"""
        # Parse query text
        parsed_query = await self.parse_query_text(query)

        # Extract search intent
        intent = await self.extract_search_intent(parsed_query)

        # Identify entities
        entities = await self.identify_entities(parsed_query)

        # Determine query type
        query_type = await self.classify_query_type(parsed_query)

        # Apply context
        contextualized = await self.apply_context(parsed_query, context)

        return ProcessedQuery(
            original=query,
            parsed=parsed_query,
            intent=intent,
            entities=entities,
            query_type=query_type,
            context=contextualized
        )

    async def expand_query(self, processed_query: ProcessedQuery) -> ExpandedQuery:
        """Expand query with synonyms and related terms"""
        # Find synonyms
        synonyms = await self.find_synonyms(processed_query.entities)

        # Add related concepts
        related_concepts = await self.find_related_concepts(processed_query)

        # Create expanded query
        expanded_terms = set(processed_query.parsed.terms)
        expanded_terms.update(synonyms)
        expanded_terms.update(related_concepts)

        return ExpandedQuery(
            original=processed_query,
            expanded_terms=list(expanded_terms),
            expansion_confidence=self.calculate_expansion_confidence(synonyms, related_concepts)
        )
```

#### 2. Search Engine Pattern
```python
class SearchEngine:
    """Core search engine implementation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index_manager = IndexManager(config)
        self.ranker = SearchRanker(config)
        self.query_processor = QueryProcessor(config)

    async def search(self, query: str, options: Dict = None) -> SearchResult:
        """Execute search query"""
        # Process query
        processed_query = await self.query_processor.process_query(query)

        # Expand query if enabled
        if options.get("expand_query", True):
            processed_query = await self.query_processor.expand_query(processed_query)

        # Retrieve candidates
        candidates = await self.index_manager.retrieve_candidates(processed_query)

        # Rank results
        ranked_results = await self.ranker.rank_results(candidates, processed_query)

        # Apply filters
        if options.get("filters"):
            ranked_results = self.apply_filters(ranked_results, options["filters"])

        # Format results
        formatted_results = await self.format_search_results(ranked_results, query)

        return SearchResult(
            query=query,
            results=formatted_results,
            total_count=len(formatted_results),
            search_time=self.measure_search_time(),
            metadata=self.generate_search_metadata()
        )

    async def index_content(self, content: Dict, content_type: str = "document") -> bool:
        """Index content for search"""
        # Validate content
        validation = await self.validate_content(content)
        if not validation["valid"]:
            raise ValueError(f"Invalid content: {validation['errors']}")

        # Extract searchable fields
        searchable_fields = await self.extract_searchable_fields(content)

        # Generate embeddings
        embeddings = await self.generate_content_embeddings(content)

        # Index in search backend
        index_result = await self.index_manager.index_document({
            "id": content["id"],
            "type": content_type,
            "fields": searchable_fields,
            "embeddings": embeddings,
            "metadata": content.get("metadata", {})
        })

        return index_result["success"]
```

#### 3. Ranking Pattern
```python
class SearchRanker:
    """Rank search results by relevance"""

    async def rank_results(self, candidates: List[Document], query: ProcessedQuery) -> List[ScoredDocument]:
        """Rank search results"""
        # Calculate relevance scores
        scored_candidates = []
        for candidate in candidates:
            score = await self.calculate_relevance_score(candidate, query)
            scored_candidates.append(ScoredDocument(candidate, score))

        # Apply ranking algorithm
        if query.query_type == "semantic":
            ranked = await self.semantic_ranking(scored_candidates, query)
        elif query.query_type == "keyword":
            ranked = await self.keyword_ranking(scored_candidates, query)
        else:
            ranked = await self.hybrid_ranking(scored_candidates, query)

        # Apply personalization
        if query.context.get("user_id"):
            ranked = await self.personalize_ranking(ranked, query.context["user_id"])

        return ranked

    async def calculate_relevance_score(self, document: Document, query: ProcessedQuery) -> float:
        """Calculate relevance score for document"""
        # Text similarity
        text_score = self.calculate_text_similarity(document, query)

        # Semantic similarity
        semantic_score = await self.calculate_semantic_similarity(document, query)

        # Contextual relevance
        context_score = self.calculate_contextual_relevance(document, query)

        # User preference score
        preference_score = await self.calculate_user_preference_score(document, query)

        # Combine scores
        combined_score = self.combine_relevance_scores([
            text_score, semantic_score, context_score, preference_score
        ])

        return combined_score
```

## 🧪 Testing Standards

### Test Categories (MANDATORY)

#### 1. Query Processing Testing
```python
class TestQueryProcessing:
    """Test query processing functionality"""

    def test_natural_language_parsing(self):
        """Test natural language query parsing"""
        # Test query parsing
        query = "What is active inference?"
        parsed = self.query_processor.parse_query(query)

        assert parsed["entities"] == ["active inference"]
        assert parsed["intent"] == "definition"
        assert parsed["query_type"] == "informational"

    def test_query_expansion(self):
        """Test query expansion with synonyms"""
        query = ProcessedQuery("active inference", entities=["active inference"])
        expanded = self.query_processor.expand_query(query)

        # Check expansion
        assert "active inference" in expanded.expanded_terms
        assert "active inference framework" in expanded.expanded_terms
        assert expanded.expansion_confidence > 0.7

    def test_intent_classification(self):
        """Test search intent classification"""
        queries_and_intents = [
            ("What is active inference?", "informational"),
            ("How to implement active inference?", "procedural"),
            ("Show me examples of active inference", "navigational"),
            ("Compare active inference vs reinforcement learning", "comparative")
        ]

        for query_text, expected_intent in queries_and_intents:
            intent = self.query_processor.classify_intent(query_text)
            assert intent == expected_intent
```

#### 2. Search Engine Testing
```python
class TestSearchEngine:
    """Test search engine functionality"""

    async def test_basic_search(self):
        """Test basic search functionality"""
        # Index test content
        await self.index_test_content()

        # Execute search
        results = await self.search_engine.search("active inference")

        # Validate results
        assert len(results) > 0
        assert results[0]["relevance_score"] > 0.5
        assert "active inference" in results[0]["title"].lower()

    async def test_semantic_search(self):
        """Test semantic search capabilities"""
        # Index semantically related content
        await self.index_semantic_content()

        # Semantic search
        results = await self.search_engine.semantic_search("brain decision making")

        # Validate semantic understanding
        assert len(results) > 0
        assert any("neuroscience" in result["tags"] for result in results)

    async def test_performance_under_load(self):
        """Test search performance under concurrent load"""
        # Generate load
        load_tester = SearchLoadTester()

        results = await load_tester.run_concurrent_search_test(
            queries=["active inference", "bayesian inference", "free energy"],
            concurrent_users=100,
            duration=60
        )

        # Validate performance
        assert results["avg_response_time"] < 100  # ms
        assert results["throughput"] > 1000  # queries per second
        assert results["error_rate"] < 0.01  # 1%
```

#### 3. Ranking Testing
```python
class TestRanking:
    """Test search result ranking"""

    async def test_relevance_ranking(self):
        """Test relevance-based ranking"""
        # Create test documents with varying relevance
        documents = await self.create_test_documents_with_relevance()

        # Execute search
        results = await self.search_engine.search("test query")

        # Validate ranking
        assert results[0]["relevance_score"] > results[-1]["relevance_score"]
        assert all(results[i]["score"] >= results[i+1]["score"] for i in range(len(results)-1))

    async def test_personalization_ranking(self):
        """Test personalized ranking"""
        # Set up user preferences
        user_preferences = {
            "preferred_topics": ["neuroscience", "psychology"],
            "difficulty_level": "advanced",
            "content_types": ["research", "implementation"]
        }

        await self.set_user_preferences("test_user", user_preferences)

        # Search with personalization
        results = await self.search_engine.search("active inference", user_id="test_user")

        # Validate personalization
        assert all(result["topic"] in user_preferences["preferred_topics"] for result in results[:5])
```

## 📖 Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. Search API Documentation
**All search APIs must be documented:**

```python
async def search(query: str, options: Dict[str, Any] = None) -> SearchResult:
    """
    Execute intelligent search across knowledge repository.

    This function provides comprehensive search capabilities including
    text search, semantic search, and personalized results.

    Args:
        query: Search query string (natural language or structured)
        options: Search options including filters, sorting, and pagination
            - filters: Dictionary of field filters
            - sort: Sorting criteria ("relevance", "date", "popularity")
            - limit: Maximum number of results (default: 20)
            - offset: Result offset for pagination (default: 0)
            - include_snippets: Include highlighted snippets (default: True)

    Returns:
        SearchResult object containing results, metadata, and analytics

    Raises:
        SearchError: If search execution fails
        ValidationError: If query or options are invalid
        PermissionError: If user lacks search permissions

    Examples:
        >>> results = await search("active inference fundamentals")
        >>> print(f"Found {len(results)} results")

        >>> results = await search("bayesian inference", {
        ...     "filters": {"difficulty": "beginner"},
        ...     "sort": "relevance",
        ...     "limit": 10
        ... })
        >>> for result in results:
        ...     print(f"{result.title}: {result.relevance_score}")
    """
    pass
```

#### 2. Configuration Documentation
**Search configuration must be documented:**

```python
# Search engine configuration schema
search_config_schema = {
    "backend": {
        "type": "string",
        "enum": ["elasticsearch", "solr", "opensearch", "local"],
        "default": "elasticsearch"
    },
    "indexing": {
        "auto_index": {"type": "boolean", "default": True},
        "batch_size": {"type": "integer", "default": 1000},
        "refresh_interval": {"type": "string", "default": "30s"},
        "embedding_model": {"type": "string", "default": "sentence_transformers"}
    },
    "ranking": {
        "algorithm": {"type": "string", "default": "hybrid"},
        "personalization": {"type": "boolean", "default": True},
        "semantic_weight": {"type": "float", "default": 0.7},
        "text_weight": {"type": "float", "default": 0.3}
    }
}
```

## 🚀 Performance Optimization

### Performance Requirements

**Search system must meet these performance standards:**

- **Query Response Time**: <100ms for typical queries
- **Indexing Throughput**: 1000+ documents per second
- **Search Throughput**: 1000+ queries per second
- **Index Size**: Support 10M+ documents efficiently

### Optimization Techniques

#### 1. Index Optimization
```python
class IndexOptimizer:
    """Optimize search indexes"""

    async def optimize_index_structure(self, index: SearchIndex) -> OptimizedIndex:
        """Optimize index structure for performance"""
        # Analyze query patterns
        query_patterns = await self.analyze_query_patterns(index)

        # Optimize field mappings
        optimized_mappings = self.optimize_field_mappings(query_patterns)

        # Create compound indexes
        compound_indexes = self.create_compound_indexes(query_patterns)

        # Optimize analyzers
        optimized_analyzers = self.optimize_text_analyzers(query_patterns)

        return OptimizedIndex(index, optimized_mappings, compound_indexes, optimized_analyzers)

    async def implement_caching_strategy(self, index: SearchIndex) -> CacheStrategy:
        """Implement intelligent caching"""
        # Analyze access patterns
        access_patterns = await self.analyze_access_patterns(index)

        # Create multi-level cache
        cache_strategy = self.create_multi_level_cache_strategy(access_patterns)

        # Pre-warm popular queries
        await self.prewarm_popular_queries(cache_strategy)

        return cache_strategy
```

#### 2. Query Optimization
```python
class QueryOptimizer:
    """Optimize search queries"""

    async def optimize_query_execution(self, query: ProcessedQuery) -> OptimizedQuery:
        """Optimize query execution"""
        # Select optimal search strategy
        strategy = self.select_search_strategy(query)

        # Optimize field selection
        optimized_fields = self.optimize_field_selection(query, strategy)

        # Create query plan
        query_plan = self.create_query_execution_plan(query, optimized_fields)

        return OptimizedQuery(query, strategy, optimized_fields, query_plan)

    async def batch_optimize_queries(self, queries: List[ProcessedQuery]) -> List[OptimizedQuery]:
        """Batch optimize multiple queries"""
        # Group similar queries
        query_groups = self.group_similar_queries(queries)

        # Optimize each group
        optimized_groups = []
        for group in query_groups:
            optimized_group = await self.optimize_query_group(group)
            optimized_groups.append(optimized_group)

        return [query for group in optimized_groups for query in group]
```

## 🔒 Security Standards

### Security Requirements (MANDATORY)

#### 1. Query Security
```python
class SearchSecurity:
    """Secure search operations"""

    async def validate_query_security(self, query: str, user: User) -> SecurityValidation:
        """Validate query for security threats"""
        # SQL injection prevention
        sql_check = self.prevent_sql_injection(query)

        # Command injection prevention
        cmd_check = self.prevent_command_injection(query)

        # Access control validation
        access_check = await self.validate_query_access(query, user)

        # Content filtering
        content_check = self.validate_content_appropriateness(query)

        return {
            "sql_injection_safe": sql_check,
            "command_injection_safe": cmd_check,
            "access_authorized": access_check,
            "content_appropriate": content_check
        }

    async def sanitize_search_results(self, results: List[Document], user: User) -> List[Document]:
        """Sanitize search results for user"""
        # Filter sensitive content
        filtered_results = self.filter_sensitive_content(results, user)

        # Apply access control
        authorized_results = await self.apply_access_control(filtered_results, user)

        # Sanitize content
        sanitized_results = self.sanitize_content_data(authorized_results)

        return sanitized_results
```

#### 2. Index Security
```python
class IndexSecurity:
    """Secure search index"""

    def secure_index_access(self, index: SearchIndex) -> SecureIndex:
        """Secure index access"""
        # Encrypt sensitive fields
        encrypted_index = self.encrypt_sensitive_fields(index)

        # Set up access controls
        self.setup_index_access_controls(encrypted_index)

        # Enable audit logging
        self.enable_index_audit_logging(encrypted_index)

        return encrypted_index

    def validate_index_integrity(self, index: SearchIndex) -> IntegrityReport:
        """Validate index integrity"""
        # Check data consistency
        consistency_check = self.check_index_consistency(index)

        # Validate indexes
        index_validation = self.validate_search_indexes(index)

        # Check permissions
        permission_check = self.validate_index_permissions(index)

        return {
            "consistent": consistency_check,
            "indexes_valid": index_validation,
            "permissions_valid": permission_check
        }
```

## 🐛 Debugging & Troubleshooting

### Debug Configuration

```python
# Enable search debugging
debug_config = {
    "debug_mode": True,
    "log_level": "DEBUG",
    "query_debug": True,
    "index_debug": True,
    "ranking_debug": True
}
```

### Common Debugging Patterns

#### 1. Query Debugging
```python
class QueryDebugger:
    """Debug search queries"""

    async def trace_query_execution(self, query: str) -> QueryTrace:
        """Trace query execution through system"""
        # Process query with tracing
        trace = await self.query_processor.process_with_trace(query)

        # Execute with execution tracing
        execution_trace = await self.search_engine.execute_with_trace(trace)

        # Analyze performance
        performance_analysis = await self.analyze_query_performance(execution_trace)

        return {
            "query": query,
            "processing_trace": trace,
            "execution_trace": execution_trace,
            "performance": performance_analysis
        }

    async def debug_no_results(self, query: str) -> DebugReport:
        """Debug queries returning no results"""
        # Check query processing
        processed = await self.query_processor.process_query(query)

        # Check index status
        index_status = await self.check_index_status()

        # Test query components
        component_tests = await self.test_query_components(processed)

        # Suggest improvements
        suggestions = self.suggest_query_improvements(processed, index_status)

        return {
            "processed_query": processed,
            "index_status": index_status,
            "component_tests": component_tests,
            "suggestions": suggestions
        }
```

## 🔄 Development Workflow

### Agent Development Process

1. **Task Assessment**
   - Understand search requirements
   - Analyze query patterns
   - Consider performance constraints

2. **Architecture Planning**
   - Design search architecture
   - Plan indexing strategy
   - Consider scalability requirements

3. **Test-Driven Development**
   - Write search tests first
   - Test query processing
   - Validate ranking algorithms

4. **Implementation**
   - Implement search engine
   - Add query processing
   - Implement ranking system

5. **Quality Assurance**
   - Test with realistic queries
   - Validate performance requirements
   - Security compliance testing

6. **Integration**
   - Test with knowledge platform
   - Validate user experience
   - Performance optimization

### Code Review Checklist

**Before submitting search code for review:**

- [ ] **Query Tests**: Comprehensive query processing tests
- [ ] **Search Tests**: Search engine functionality tests
- [ ] **Ranking Tests**: Result ranking and personalization tests
- [ ] **Performance Tests**: Search performance under load
- [ ] **Integration Tests**: Platform integration validation
- [ ] **Documentation**: Complete API and configuration documentation

## 📚 Learning Resources

### Search Resources

- **[Search Engine Best Practices](https://example.com/search-engines)**: Search system design
- **[Information Retrieval](https://example.com/ir)**: IR fundamentals
- **[Natural Language Processing](https://example.com/nlp)**: Query understanding
- **[Vector Search](https://example.com/vector-search)**: Semantic search

### Platform Integration

- **[Knowledge Platform](../../../knowledge/README.md)**: Knowledge architecture
- **[Platform Services](../../platform/README.md)**: Platform integration
- **[User Management](../../../src/active_inference/platform/README.md)**: User systems

## 🎯 Success Metrics

### Quality Metrics

- **Search Relevance**: >90% query result relevance
- **Query Understanding**: >85% intent classification accuracy
- **Performance**: <100ms query response time
- **User Experience**: Intuitive search interface

### Development Metrics

- **Search Engine**: Efficient and scalable search implementation
- **Query Processing**: Accurate natural language understanding
- **Ranking System**: Effective relevance ranking
- **Integration**: Seamless platform integration

---

**Component**: Platform Search | **Version**: 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Intelligent discovery through semantic search.
