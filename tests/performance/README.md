# Performance Testing Framework

This directory contains performance tests for the Active Inference Knowledge Environment, ensuring that all components meet performance requirements and scale appropriately under various loads.

## Overview

Performance testing validates that the platform maintains acceptable response times, memory usage, and scalability characteristics across different usage scenarios. These tests ensure the system performs well under both normal and high-load conditions.

## Test Categories

### ⚡ Response Time Testing
Tests that measure and validate response times for critical operations:
- **Search Operations**: Knowledge search and filtering performance
- **Content Loading**: Repository loading and content access times
- **Graph Operations**: Prerequisite graph generation and traversal
- **Statistics Calculation**: Repository statistics computation time

### 💾 Memory Usage Testing
Tests that monitor and validate memory consumption:
- **Repository Loading**: Memory usage during content loading
- **Search Operations**: Memory overhead during search operations
- **Large Content Handling**: Memory management for large datasets
- **Memory Leak Detection**: Ensuring no memory leaks over time

### 📈 Scalability Testing
Tests that validate system behavior under increasing loads:
- **Content Volume Scaling**: Performance with increasing content volumes
- **Concurrent Access**: Multi-user and concurrent operation handling
- **Resource Scaling**: Performance scaling with available resources
- **Load Testing**: System behavior under sustained high loads

## Getting Started

### Running Performance Tests

```bash
# Run all performance tests
pytest tests/performance/ -v

# Run specific performance test categories
pytest tests/performance/test_knowledge_repository_performance.py -v
pytest tests/performance/test_llm_performance.py -v

# Run performance tests with profiling
pytest tests/performance/ --profile

# Run performance benchmarks
pytest tests/performance/ -m benchmark
```

### Performance Test Configuration

Performance tests can be configured through environment variables and pytest markers:

```bash
# Set performance thresholds
export PERF_MAX_SEARCH_TIME=1.0
export PERF_MAX_MEMORY_INCREASE=100  # MB
export PERF_MIN_THROUGHPUT=100     # operations/second

# Run with specific performance requirements
pytest tests/performance/ -m "performance and not slow"
pytest tests/performance/ -m "benchmark"
```

## Performance Benchmarks

### Target Performance Metrics

#### Knowledge Repository Performance
- **Search Response Time**: < 1 second for typical queries
- **Content Loading Time**: < 5 seconds for complete repository
- **Memory Usage**: < 500MB for typical repository sizes
- **Graph Generation Time**: < 2 seconds for prerequisite graphs

#### LLM Integration Performance
- **Response Time**: < 5 seconds for typical requests
- **Conversation Context**: Support for 100+ message conversations
- **Model Loading Time**: < 30 seconds for model initialization
- **Throughput**: > 10 requests per minute sustained

#### Platform Performance
- **API Response Time**: < 500ms for REST API calls
- **Concurrent Users**: Support for 100+ concurrent users
- **Database Query Time**: < 100ms for typical queries
- **File Upload Time**: < 10 seconds for typical files

## Test Implementation

### Performance Test Patterns

#### Response Time Testing
```python
def test_search_response_time(self, knowledge_repo):
    """Test search response time meets requirements"""
    start_time = time.time()

    results = knowledge_repo.search("active_inference")

    end_time = time.time()
    response_time = end_time - start_time

    assert response_time < 1.0, f"Search too slow: {response_time".3f"}s"
    assert len(results) > 0, "Search should return results"
```

#### Memory Usage Testing
```python
def test_memory_usage_limits(self, knowledge_repo):
    """Test memory usage stays within acceptable limits"""
    import psutil

    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # Perform memory-intensive operations
    for i in range(1000):
        results = knowledge_repo.search(f"query_{i}")

    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    assert memory_increase < 100, f"Memory usage too high: {memory_increase".1f"}MB"
```

#### Scalability Testing
```python
@pytest.mark.parametrize("content_size", [100, 1000, 10000])
def test_scalability_with_content_size(self, content_size):
    """Test performance scaling with content size"""
    # Create repository with specified content size
    repo = self.create_test_repository(content_size)

    # Measure performance metrics
    search_time = self.measure_search_time(repo)
    memory_usage = self.measure_memory_usage(repo)

    # Validate scaling characteristics
    if content_size == 100:
        assert search_time < 0.1
        assert memory_usage < 50
    elif content_size == 1000:
        assert search_time < 0.5
        assert memory_usage < 200
    elif content_size == 10000:
        assert search_time < 2.0
        assert memory_usage < 1000
```

## Performance Monitoring

### Continuous Performance Monitoring
- **Automated Benchmarks**: Regular performance benchmark execution
- **Performance Regression Detection**: Alert on performance degradation
- **Resource Usage Monitoring**: Track memory and CPU usage trends
- **Performance Dashboards**: Visual performance metrics and trends

### Performance Optimization
- **Profiling Integration**: Built-in profiling for performance analysis
- **Performance Regression Testing**: Detect performance regressions
- **Optimization Recommendations**: Automated performance optimization suggestions
- **Resource Allocation**: Dynamic resource allocation based on load

## Contributing

### Writing Performance Tests
1. **Identify Critical Paths**: Focus on most frequently used operations
2. **Establish Baselines**: Set realistic performance baselines
3. **Use Appropriate Tools**: Leverage profiling and monitoring tools
4. **Consider Real-World Scenarios**: Test with realistic data and loads
5. **Document Performance Requirements**: Clearly specify performance expectations

### Performance Test Best Practices
- **Realistic Test Data**: Use data that reflects actual usage patterns
- **Proper Benchmarking**: Use statistical methods for reliable measurements
- **Resource Cleanup**: Ensure proper cleanup between tests
- **Error Handling**: Handle performance test failures gracefully
- **Documentation**: Document performance requirements and expectations

## Related Documentation

- **[Testing README](../README.md)**: Main testing framework documentation
- **[Performance Guidelines](../../applications/best_practices/)**: Performance best practices
- **[System Requirements](../../README.md)**: System performance requirements
- **[Monitoring Guide](../../platform/)**: Platform monitoring and performance

---

*"Active Inference for, with, by Generative AI"* - Ensuring performance through comprehensive testing, monitoring, and optimization.
