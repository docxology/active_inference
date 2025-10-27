# Testing Tools - Agent Development Guide

**Guidelines for AI agents working with testing frameworks and validation systems.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with testing systems:**

### Primary Responsibilities
- **Test Framework Development**: Build comprehensive testing infrastructure
- **Knowledge Validation**: Create automated knowledge content validation
- **Integration Testing**: Develop cross-component testing utilities
- **Performance Testing**: Implement load testing and performance validation
- **Mock Services**: Create mock implementations for testing
- **Test Data Management**: Automated test data generation and management

### Development Focus Areas
1. **Test Infrastructure**: Build robust testing frameworks
2. **Validation Systems**: Knowledge and content validation
3. **Mock Services**: Create realistic test doubles
4. **Performance Testing**: Load testing and benchmarking
5. **Test Automation**: Automated testing pipelines

## 🏗️ Architecture & Integration

### Testing Architecture

**Understanding the testing system structure:**

```
Testing Infrastructure
├── Test Framework (pytest, custom)
├── Mock Services (test doubles)
├── Test Data Generation (fixtures)
├── Performance Testing (load testing)
├── Knowledge Validation (content testing)
└── Integration Testing (cross-component)
```

### Integration Points

**Key integration points for testing:**

#### Platform Integration
- **All Components**: Test all platform components
- **CI/CD Pipeline**: Automated testing in deployment
- **Quality Gates**: Testing requirements for releases
- **Development Workflow**: Testing in development process

#### External Systems
- **Test Runners**: pytest, unittest, hypothesis
- **Coverage Tools**: coverage.py, pytest-cov
- **Mock Libraries**: unittest.mock, responses
- **Performance Tools**: locust, pytest-benchmark

### Test Categories

```python
# Test organization pattern
tests/
├── unit/                    # Individual component tests
├── integration/            # Component interaction tests
├── knowledge/              # Knowledge content validation
├── performance/            # Performance and load tests
├── security/               # Security and vulnerability tests
└── fixtures/               # Test data and configurations
```

## 💻 Development Patterns

### Required Implementation Patterns

**All testing development must follow these patterns:**

#### 1. Test Framework Pattern
```python
class TestFramework:
    """Comprehensive testing framework"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_runner = TestRunner(config)
        self.mock_manager = MockServiceManager(config)
        self.data_generator = TestDataGenerator(config)

    async def run_comprehensive_tests(self, components: List[str] = None) -> TestResults:
        """Run comprehensive test suite"""
        # Set up test environment
        await self.setup_test_environment()

        # Run unit tests
        unit_results = await self.run_unit_tests(components)

        # Run integration tests
        integration_results = await self.run_integration_tests(components)

        # Run knowledge validation
        knowledge_results = await self.run_knowledge_tests()

        # Run performance tests
        performance_results = await self.run_performance_tests()

        # Aggregate results
        results = self.aggregate_results([
            unit_results, integration_results,
            knowledge_results, performance_results
        ])

        return results

    async def validate_knowledge_content(self, content_path: str) -> ValidationResults:
        """Validate knowledge content"""
        # Load content
        content = await self.load_knowledge_content(content_path)

        # Validate schema
        schema_validation = await self.validate_content_schema(content)

        # Validate prerequisites
        prerequisite_validation = await self.validate_prerequisites(content)

        # Validate cross-references
        reference_validation = await self.validate_cross_references(content)

        # Validate learning objectives
        objective_validation = await self.validate_learning_objectives(content)

        return {
            "schema": schema_validation,
            "prerequisites": prerequisite_validation,
            "references": reference_validation,
            "objectives": objective_validation
        }
```

#### 2. Mock Service Pattern
```python
class MockServiceManager:
    """Manage mock services for testing"""

    def create_mock_service(self, service_type: str) -> MockService:
        """Create mock service implementation"""
        # Service type validation
        self.validate_service_type(service_type)

        # Create mock implementation
        mock_service = self.instantiate_mock_service(service_type)

        # Configure mock behavior
        self.configure_mock_behavior(mock_service, service_type)

        # Set up response patterns
        self.setup_response_patterns(mock_service)

        return mock_service

    def configure_mock_responses(self, mock_service: MockService, responses: Dict) -> None:
        """Configure mock service responses"""
        for endpoint, response in responses.items():
            mock_service.when(endpoint).then_return(response)

        # Set up error scenarios
        self.setup_error_scenarios(mock_service)

        # Configure performance characteristics
        self.configure_performance_characteristics(mock_service)
```

#### 3. Test Data Generation Pattern
```python
class TestDataGenerator:
    """Generate realistic test data"""

    def generate_knowledge_content(self, content_type: str = "foundation") -> Dict:
        """Generate test knowledge content"""
        # Generate base structure
        content = self.create_content_structure(content_type)

        # Add realistic content
        content = self.populate_content_fields(content)

        # Add relationships
        content = self.add_content_relationships(content)

        # Add metadata
        content = self.add_content_metadata(content)

        return content

    def generate_test_users(self, count: int = 10) -> List[TestUser]:
        """Generate test users with realistic profiles"""
        users = []

        for i in range(count):
            user = self.create_test_user(
                user_id=f"user_{i}",
                roles=self.generate_user_roles(),
                preferences=self.generate_user_preferences(),
                activity_level=self.generate_activity_level()
            )
            users.append(user)

        return users

    def generate_test_scenarios(self, scenario_type: str) -> List[TestScenario]:
        """Generate realistic test scenarios"""
        # Analyze scenario requirements
        requirements = self.analyze_scenario_requirements(scenario_type)

        # Generate scenario components
        scenarios = []

        for i in range(requirements["count"]):
            scenario = self.create_scenario(
                scenario_type=scenario_type,
                complexity=requirements["complexity"],
                components=requirements["components"]
            )
            scenarios.append(scenario)

        return scenarios
```

## 🧪 Testing Standards

### Test Categories (MANDATORY)

#### 1. Component Testing
```python
class TestComponentFramework:
    """Test component testing framework"""

    def test_component_initialization(self):
        """Test component initialization"""
        # Create test component
        component = self.create_test_component()

        # Validate initialization
        assert component.initialized == True
        assert component.config is not None
        assert hasattr(component, 'main_method')

    def test_component_error_handling(self):
        """Test error handling"""
        component = self.create_test_component()

        # Test invalid input
        with pytest.raises(ValidationError):
            component.process_data(None)

        # Test missing dependencies
        with pytest.raises(DependencyError):
            component.process_data(valid_data)
```

#### 2. Knowledge Validation Testing
```python
class TestKnowledgeValidation:
    """Test knowledge content validation"""

    def test_content_schema_validation(self):
        """Test JSON schema compliance"""
        # Create test content
        content = self.generate_test_content()

        # Validate schema
        validator = KnowledgeValidator()
        result = validator.validate_schema(content)

        assert result["valid"] == True
        assert len(result["errors"]) == 0

    def test_prerequisite_validation(self):
        """Test prerequisite chain validation"""
        # Create content with prerequisites
        content_graph = self.create_content_with_prerequisites()

        # Validate prerequisites
        validator = PrerequisiteValidator()
        validation_result = validator.validate_chains(content_graph)

        # Check for cycles
        assert len(validation_result["cycles"]) == 0

        # Check prerequisite satisfaction
        assert all(chain["valid"] for chain in validation_result["chains"])

    def test_cross_reference_validation(self):
        """Test cross-reference validation"""
        # Create content with references
        content = self.create_content_with_references()

        # Validate references
        validator = ReferenceValidator()
        validation_result = validator.validate_references(content)

        assert validation_result["all_resolved"] == True
        assert len(validation_result["broken_references"]) == 0
```

#### 3. Integration Testing
```python
class TestPlatformIntegration:
    """Test platform component integration"""

    async def test_knowledge_search_integration(self):
        """Test knowledge and search integration"""
        # Set up test environment
        test_env = await self.setup_test_environment()

        # Create knowledge content
        knowledge_service = test_env.get_service("knowledge")
        content = self.create_test_content()
        content_id = await knowledge_service.create_content(content)

        # Index content
        search_service = test_env.get_service("search")
        await search_service.index_content(content_id)

        # Test search functionality
        results = await search_service.search("test content")

        assert len(results) > 0
        assert results[0]["id"] == content_id

    async def test_full_user_workflow(self):
        """Test complete user workflow"""
        # Set up complete platform
        platform = await self.setup_full_platform()

        # Simulate user journey
        user_actions = [
            "search_content",
            "view_content",
            "rate_content",
            "get_recommendations"
        ]

        for action in user_actions:
            result = await platform.execute_user_action(action)
            assert result["success"] == True
```

## 📖 Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. Test Documentation
**All tests must be documented with clear purpose:**

```python
def test_component_functionality_with_edge_cases():
    """
    Test component functionality under various edge case conditions.

    This test validates that the component handles edge cases gracefully
    and maintains expected behavior under unusual circumstances.

    Edge cases tested:
    - Empty input data
    - Maximum size input
    - Invalid data types
    - Network timeouts
    - Resource constraints

    Expected behavior:
    - Graceful degradation for recoverable errors
    - Clear error messages for validation failures
    - Resource cleanup for system errors
    """
    # Test implementation
    pass
```

#### 2. Test Data Documentation
**Test data generation must be documented:**

```python
# Test data generation patterns
test_data_patterns = {
    "knowledge_content": {
        "structure": {
            "id": "string",
            "title": "string",
            "content_type": "enum[foundation, mathematics, implementation]",
            "difficulty": "enum[beginner, intermediate, advanced, expert]",
            "content": "object",
            "metadata": "object"
        },
        "generation_rules": {
            "id": "unique_identifier_generation",
            "title": "meaningful_title_generation",
            "content": "realistic_content_generation"
        }
    },
    "user_profiles": {
        "structure": {
            "user_id": "string",
            "preferences": "object",
            "activity_level": "enum[low, medium, high]",
            "role": "enum[student, researcher, educator]"
        }
    }
}
```

## 🚀 Performance Optimization

### Performance Requirements

**Testing system must meet these performance standards:**

- **Test Execution Time**: <5 minutes for full test suite
- **Test Data Generation**: <1 second for typical datasets
- **Mock Service Response**: <10ms for mock responses
- **Performance Test Duration**: Configurable load test duration

### Optimization Techniques

#### 1. Parallel Test Execution
```python
class ParallelTestExecutor:
    """Execute tests in parallel"""

    async def execute_parallel_tests(self, test_suites: List[TestSuite]) -> TestResults:
        """Execute multiple test suites in parallel"""
        # Configure parallel execution
        executor = ProcessPoolExecutor(max_workers=self.config["max_workers"])

        # Execute test suites
        tasks = [
            self.execute_test_suite(suite, executor)
            for suite in test_suites
        ]

        # Gather results
        results = await asyncio.gather(*tasks)

        # Aggregate results
        return self.aggregate_parallel_results(results)
```

#### 2. Test Data Caching
```python
class TestDataCache:
    """Cache test data for performance"""

    def __init__(self, config: Dict[str, Any]):
        self.cache = {}
        self.config = config

    def get_cached_data(self, data_type: str, params: Dict) -> Any:
        """Get cached test data"""
        cache_key = self.generate_cache_key(data_type, params)

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate and cache data
        data = self.generate_test_data(data_type, params)
        self.cache[cache_key] = data

        return data

    def generate_cache_key(self, data_type: str, params: Dict) -> str:
        """Generate cache key for test data"""
        key_components = [data_type]
        for k, v in sorted(params.items()):
            key_components.append(f"{k}={v}")

        return "_".join(key_components)
```

## 🔒 Security Standards

### Security Requirements (MANDATORY)

#### 1. Test Environment Security
```python
class TestSecurityManager:
    """Secure test environment"""

    def setup_secure_test_environment(self, environment: str) -> None:
        """Set up secure test environment"""
        # Isolate test environment
        self.isolate_environment(environment)

        # Secure data handling
        self.setup_secure_data_handling()

        # Access control
        self.setup_test_access_control()

    def validate_test_data_security(self, test_data: Dict) -> SecurityReport:
        """Validate test data for security issues"""
        # Check for sensitive data
        sensitive_data_check = self.check_sensitive_data(test_data)

        # Validate data sanitization
        sanitization_check = self.check_data_sanitization(test_data)

        # Check access permissions
        permission_check = self.check_data_permissions(test_data)

        return {
            "sensitive_data": sensitive_data_check,
            "sanitization": sanitization_check,
            "permissions": permission_check
        }
```

## 🐛 Debugging & Troubleshooting

### Debug Configuration

```python
# Enable testing debugging
debug_config = {
    "debug_mode": True,
    "log_level": "DEBUG",
    "test_debug": True,
    "mock_debug": True,
    "performance_debug": True
}
```

### Common Debugging Patterns

#### 1. Test Failure Debugging
```python
class TestFailureDebugger:
    """Debug test failures"""

    def debug_test_failure(self, test_result: TestResult) -> DebugReport:
        """Debug failed test"""
        # Analyze failure
        failure_analysis = self.analyze_failure(test_result)

        # Generate reproduction steps
        reproduction = self.generate_reproduction_steps(test_result)

        # Suggest fixes
        fixes = self.suggest_fixes(failure_analysis)

        return {
            "analysis": failure_analysis,
            "reproduction": reproduction,
            "fixes": fixes
        }
```

## 🔄 Development Workflow

### Agent Development Process

1. **Task Assessment**
   - Understand testing requirements
   - Analyze component interfaces
   - Consider edge cases and scenarios

2. **Architecture Planning**
   - Design test framework structure
   - Plan mock service architecture
   - Consider performance testing needs

3. **Test-Driven Development**
   - Write tests before implementation
   - Create comprehensive test coverage
   - Validate testing infrastructure

4. **Implementation**
   - Implement test framework
   - Create mock services
   - Add validation systems

5. **Quality Assurance**
   - Test testing framework itself
   - Validate coverage requirements
   - Performance optimization

6. **Integration**
   - Test with platform components
   - Validate CI/CD integration
   - Performance optimization

### Code Review Checklist

**Before submitting testing code for review:**

- [ ] **Test Coverage**: Comprehensive test coverage for testing framework
- [ ] **Mock Validation**: Realistic and comprehensive mock implementations
- [ ] **Data Generation**: Robust test data generation
- [ ] **Performance Tests**: Testing framework performance validation
- [ ] **Integration Tests**: Testing framework integration validation
- [ ] **Documentation**: Complete testing documentation

## 📚 Learning Resources

### Testing Resources

- **[Testing Best Practices](https://example.com/testing)**: Software testing methodologies
- **[Mock Testing](https://example.com/mocking)**: Mock and test double patterns
- **[Performance Testing](https://example.com/perf-testing)**: Load testing techniques
- **[Test Automation](https://example.com/test-automation)**: Automated testing

### Platform Integration

- **[Platform Architecture](../../platform/README.md)**: Platform structure
- **[Component Testing](../../../src/active_inference/README.md)**: Component patterns
- **[Quality Standards](../../../.cursorrules)**: Development standards

## 🎯 Success Metrics

### Quality Metrics

- **Test Coverage**: >95% coverage for all components
- **Test Reliability**: <1% flaky tests
- **Mock Accuracy**: Realistic mock implementations
- **Performance**: Efficient test execution

### Development Metrics

- **Testing Framework**: Comprehensive testing infrastructure
- **Validation Systems**: Accurate content and data validation
- **Mock Services**: Realistic test doubles
- **Integration**: Seamless platform integration

---

**Component**: Testing Tools | **Version**: 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Rigorous testing for reliable intelligence.
