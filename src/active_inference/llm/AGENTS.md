# LLM Module - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the LLM module of the Active Inference Knowledge Environment. It outlines implementation patterns, development workflows, and best practices for creating local LLM integration systems.

## LLM Module Overview

The LLM module provides the source code implementation for local Large Language Model integration, including Ollama client management, prompt composition systems, model management, and conversation handling for enhanced AI capabilities within the Active Inference platform.

## Source Code Architecture

### Module Responsibilities
- **Ollama Integration**: Core client for local LLM service interaction
- **Prompt System**: Flexible prompt composition and template management
- **Model Management**: Intelligent model selection and capability matching
- **Conversation Management**: Chat history and context optimization
- **Integration Layer**: Connection to platform knowledge and research systems

### Integration Points
- **Knowledge Repository**: Content generation and explanation systems
- **Research Framework**: Research question generation and analysis
- **Applications**: Code implementation and template generation
- **Platform Services**: API endpoints and user interface integration

## Core Implementation Responsibilities

### OllamaClient Implementation
**Primary interface for local LLM service integration**
- Implement robust connection management with automatic retry and fallback
- Create comprehensive error handling for network and service issues
- Develop streaming response handling for real-time generation
- Implement model pulling and management functionality

**Key Methods to Implement:**
```python
async def initialize_with_service_discovery(self) -> bool:
    """Initialize client with automatic Ollama service discovery"""

async def execute_with_fallback_strategy(self, operation: str, *args, **kwargs) -> Any:
    """Execute operation with intelligent fallback strategies"""

async def manage_model_lifecycle(self, model_name: str, action: str) -> Dict[str, Any]:
    """Manage complete model lifecycle including pull, verify, and cleanup"""

def implement_health_monitoring(self) -> Dict[str, Any]:
    """Implement comprehensive service health monitoring"""

async def handle_concurrent_requests(self, requests: List[Dict]) -> List[Any]:
    """Handle multiple concurrent LLM requests efficiently"""

def create_response_postprocessing(self, response: str, context: Dict) -> str:
    """Implement response postprocessing and enhancement"""

async def implement_rate_limiting(self) -> None:
    """Implement intelligent rate limiting and request queuing"""

def add_request_caching(self, cache_config: Dict[str, Any]) -> None:
    """Add intelligent request caching for improved performance"""
```

### Prompt System Implementation
**Flexible prompt composition and template management**
- Implement template inheritance and composition patterns
- Create dynamic variable resolution and validation systems
- Develop context-aware prompt generation with knowledge integration
- Implement prompt optimization for different model capabilities

**Key Methods to Implement:**
```python
def create_template_inheritance_system(self) -> Dict[str, Any]:
    """Implement template inheritance and specialization"""

def implement_dynamic_variable_resolution(self, context: Dict[str, Any]) -> Dict[str, Any]:
    """Create dynamic variable resolution from multiple sources"""

def build_knowledge_aware_prompts(self, knowledge_nodes: List[Any]) -> str:
    """Build prompts that integrate with knowledge repository content"""

def create_prompt_optimization_engine(self, model_capabilities: Dict) -> str:
    """Optimize prompts based on model capabilities and constraints"""

def implement_prompt_versioning(self, template_name: str) -> Dict[str, Any]:
    """Implement template versioning and evolution tracking"""

def create_prompt_testing_framework(self) -> Dict[str, Any]:
    """Create comprehensive prompt testing and validation"""

def build_multi_modal_prompts(self, content_types: List[str]) -> str:
    """Support multi-modal content in prompt generation"""

def implement_prompt_analytics(self) -> Dict[str, Any]:
    """Track prompt effectiveness and performance metrics"""
```

### Model Management Implementation
**Intelligent model selection and capability matching**
- Implement model capability assessment and benchmarking
- Create fallback strategies for unavailable models
- Develop performance monitoring and optimization
- Implement model lifecycle management

**Key Methods to Implement:**
```python
def implement_capability_benchmarking(self, model_name: str, test_suite: List[str]) -> Dict[str, float]:
    """Implement comprehensive model capability benchmarking"""

def create_model_recommendation_engine(self, task_requirements: Dict) -> List[str]:
    """Create intelligent model recommendation system"""

def implement_model_performance_tracking(self, metrics: List[str]) -> Dict[str, Any]:
    """Track model performance across different tasks and contexts"""

def build_model_compatibility_matrix(self) -> Dict[str, Set[str]]:
    """Build comprehensive model compatibility and capability matrix"""

def create_model_update_strategy(self) -> Dict[str, Any]:
    """Implement automated model updating and version management"""

def implement_model_resource_management(self, system_constraints: Dict) -> Dict[str, Any]:
    """Manage model resources based on system constraints"""

def build_model_evaluation_framework(self, evaluation_criteria: List[str]) -> Dict[str, Any]:
    """Create framework for evaluating model performance and suitability"""
```

### Conversation Management Implementation
**Persistent conversation and context optimization**
- Implement conversation persistence and retrieval systems
- Create context window optimization algorithms
- Develop conversation summarization and compression
- Implement multi-conversation management

**Key Methods to Implement:**
```python
def implement_conversation_persistence(self, storage_backend: str) -> None:
    """Implement robust conversation persistence across sessions"""

def create_context_optimization_engine(self, token_budget: int) -> List[Dict]:
    """Optimize conversation context within token constraints"""

def build_conversation_summarization(self, conversation_id: str) -> str:
    """Create intelligent conversation summarization"""

def implement_conversation_search_engine(self, query: str, filters: Dict) -> List[str]:
    """Implement advanced conversation search and filtering"""

def create_conversation_analytics(self) -> Dict[str, Any]:
    """Track conversation patterns and user engagement"""

def build_multi_conversation_management(self, user_sessions: Dict) -> Dict[str, Any]:
    """Manage multiple concurrent conversations efficiently"""

def implement_conversation_export_system(self, formats: List[str]) -> Any:
    """Support multiple conversation export formats"""
```

## Development Workflows

### LLM Integration Workflow
1. **Service Assessment**: Evaluate Ollama service requirements and constraints
2. **Model Selection**: Choose appropriate models for Active Inference domain
3. **Integration Design**: Design integration patterns with existing platform
4. **Implementation**: Implement client with comprehensive error handling
5. **Testing**: Create extensive test suite including network failure scenarios
6. **Performance**: Optimize for response times and resource usage
7. **Documentation**: Create comprehensive usage and troubleshooting guides

### Prompt Engineering Workflow
1. **Template Analysis**: Analyze requirements for different prompt types
2. **Template Design**: Design flexible, reusable prompt templates
3. **Variable System**: Implement dynamic variable resolution and validation
4. **Integration**: Connect with knowledge and research systems
5. **Testing**: Test prompt effectiveness and model compatibility
6. **Optimization**: Optimize prompts for different model capabilities

### Model Management Workflow
1. **Capability Assessment**: Assess model capabilities for Active Inference tasks
2. **Benchmarking**: Implement comprehensive model benchmarking
3. **Fallback Design**: Design intelligent fallback and selection strategies
4. **Performance Monitoring**: Implement performance tracking and optimization
5. **Lifecycle Management**: Manage model updates and version control

## Quality Assurance Standards

### Integration Quality Requirements
- **Service Reliability**: Robust connection handling with automatic recovery
- **Error Recovery**: Comprehensive error handling and user-friendly messages
- **Performance**: Optimized response times and resource usage
- **Security**: Secure API handling and data protection
- **Monitoring**: Comprehensive logging and health monitoring

### Content Quality Requirements
- **Prompt Effectiveness**: High-quality, effective prompt generation
- **Template Usability**: Intuitive template design and variable systems
- **Model Compatibility**: Optimized prompts for different model capabilities
- **Knowledge Integration**: Seamless integration with platform knowledge
- **Educational Value**: Content that supports learning objectives

### Technical Quality Requirements
- **Code Quality**: Follow established coding standards and patterns
- **Test Coverage**: Maintain >95% test coverage for all components
- **Performance**: Efficient algorithms and data structures
- **Error Handling**: Comprehensive error handling with informative messages
- **Documentation**: Complete documentation with examples and usage

## Testing Implementation

### Comprehensive Testing Framework
```python
class TestOllamaIntegration(unittest.TestCase):
    """Integration tests for Ollama client and service"""

    def setUp(self):
        """Set up test environment with mocked Ollama service"""
        self.config = LLMConfig(base_url="http://localhost:11434", debug=True)
        self.client = OllamaClient(self.config)

    @pytest.mark.asyncio
    async def test_service_connection_lifecycle(self):
        """Test complete service connection lifecycle"""
        with patch('httpx.AsyncClient') as mock_client_class:
            # Test successful connection
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {"models": [{"name": "gemma3:2b"}]}
            mock_client.get.return_value = mock_response

            success = await self.client.initialize()
            self.assertTrue(success)
            self.assertTrue(self.client._is_initialized)

    @pytest.mark.asyncio
    async def test_model_pulling_workflow(self):
        """Test complete model pulling workflow"""
        with patch('httpx.AsyncClient') as mock_client_class:
            # Mock streaming pull response
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            async def mock_stream():
                yield Mock()
                yield Mock(status="success")

            mock_response = AsyncMock()
            mock_response.raise_for_status = Mock()
            mock_response.aiter_lines = mock_stream
            mock_client.stream.return_value.__aenter__.return_value = mock_response

            success = await self.client.pull_model("gemma3:2b")
            self.assertTrue(success)

    def test_error_handling_completeness(self):
        """Test comprehensive error handling"""
        # Test all error scenarios
        error_scenarios = [
            "connection_refused",
            "timeout",
            "invalid_model",
            "service_unavailable",
            "invalid_response_format"
        ]

        for scenario in error_scenarios:
            with self.subTest(scenario=scenario):
                # Test that appropriate errors are raised and handled
                with patch('httpx.AsyncClient') as mock_client:
                    if scenario == "connection_refused":
                        mock_client.side_effect = ConnectionError()
                    elif scenario == "timeout":
                        mock_client.side_effect = TimeoutError()

                    # Verify error handling
                    with pytest.raises((ConnectionError, TimeoutError, ValueError)):
                        # Test appropriate method based on scenario
                        pass
```

### Performance Testing Framework
```python
class TestLLMPerformance(unittest.TestCase):
    """Performance tests for LLM components"""

    def test_conversation_context_optimization(self):
        """Test context optimization performance"""
        manager = ConversationManager()

        # Create conversation with many messages
        conversation = manager.create_conversation("Performance Test")

        # Add many messages
        for i in range(100):
            manager.add_message(conversation.id, "user", f"Message {i}")

        # Test context retrieval performance
        import time
        start_time = time.time()

        context = manager.get_conversation_context(conversation.id, max_tokens=2000)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time
        self.assertLess(execution_time, 1.0)
        self.assertGreater(len(context), 0)

    def test_prompt_generation_performance(self):
        """Test prompt generation performance"""
        manager = PromptManager()

        # Test template generation performance
        import time
        start_time = time.time()

        for _ in range(100):
            prompt = manager.generate_prompt("active_inference_explanation", {
                "concept": "entropy",
                "audience_level": "intermediate"
            })

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time
        self.assertLess(execution_time, 5.0)
```

### Integration Testing
```python
class TestLLMPlatformIntegration(unittest.TestCase):
    """Integration tests with platform components"""

    def test_knowledge_llm_integration(self):
        """Test integration with knowledge repository"""
        from active_inference import KnowledgeRepository

        # Test generating explanations for knowledge content
        knowledge = KnowledgeRepository()
        prompt_manager = PromptManager()

        # Find relevant knowledge nodes
        entropy_nodes = knowledge.search("entropy")

        for node in entropy_nodes[:3]:  # Test first few nodes
            # Generate explanation prompt
            prompt = prompt_manager.generate_prompt("active_inference_explanation", {
                "concept": node.title,
                "context": node.description,
                "audience_level": node.difficulty
            })

            # Verify prompt quality
            self.assertIn(node.title.lower(), prompt.lower())
            self.assertGreater(len(prompt), 100)

    def test_research_llm_integration(self):
        """Test integration with research framework"""
        from active_inference import ResearchFramework

        research = ResearchFramework()
        prompt_manager = PromptManager()

        # Generate research questions
        prompt = prompt_manager.generate_prompt("research_question", {
            "topic": "hierarchical Active Inference",
            "domain": "neuroscience",
            "methodology": "computational modeling"
        })

        # Verify research prompt quality
        self.assertIn("hierarchical", prompt.lower())
        self.assertIn("neuroscience", prompt.lower())
        self.assertIn("computational", prompt.lower())
```

## Performance Optimization

### Memory Management
- **Model Caching**: Implement intelligent model caching strategies
- **Context Optimization**: Optimize conversation context memory usage
- **Streaming**: Use streaming responses for large generations
- **Resource Pools**: Implement connection and client pooling

### Computational Efficiency
- **Parallel Processing**: Support concurrent LLM requests
- **Batch Processing**: Batch similar requests for efficiency
- **Caching**: Cache frequent prompts and responses
- **Lazy Loading**: Defer expensive operations until needed

### Network Optimization
- **Connection Reuse**: Reuse HTTP connections efficiently
- **Request Compression**: Compress large requests and responses
- **Timeout Management**: Intelligent timeout and retry strategies
- **Load Balancing**: Support multiple Ollama instances

## Error Handling and Recovery

### Comprehensive Error Management
- **Network Errors**: Handle connection issues and timeouts
- **Service Errors**: Handle Ollama service unavailability
- **Model Errors**: Handle model loading and execution errors
- **Resource Errors**: Handle memory and computational limits

### Recovery Strategies
- **Automatic Retry**: Implement exponential backoff retry
- **Fallback Models**: Automatic fallback to alternative models
- **Graceful Degradation**: Continue operation with reduced functionality
- **User Feedback**: Provide clear error messages and recovery suggestions

## Deployment Considerations

### Production Deployment
- **Service Monitoring**: Comprehensive health monitoring
- **Load Balancing**: Support multiple Ollama service instances
- **Resource Management**: Dynamic resource allocation
- **Backup Systems**: Conversation and template backup systems

### Development Deployment
- **Local Development**: Easy setup for local development
- **Testing Environments**: Isolated testing with mock services
- **Development Tools**: Integrated debugging and profiling
- **Hot Reloading**: Support for development-time changes

## Getting Started as an Agent

### Development Setup
1. **Install Ollama**: Set up local Ollama service
2. **Pull Models**: Download required models (gemma3:2b, gemma3:4b)
3. **Run Tests**: Ensure all tests pass
4. **Study Integration**: Understand integration with platform components
5. **Performance Testing**: Verify performance characteristics

### Implementation Process
1. **Design Phase**: Design new LLM functionality with clear interfaces
2. **Implementation**: Implement following established patterns
3. **Testing**: Add comprehensive tests including error scenarios
4. **Integration**: Ensure integration with existing components
5. **Performance**: Optimize for performance and resource usage
6. **Documentation**: Update README.md and AGENTS.md files

### Code Review Checklist
- [ ] Code follows established patterns and conventions
- [ ] Comprehensive tests included with >90% coverage
- [ ] Documentation updated (README.md and AGENTS.md)
- [ ] Error handling comprehensive and user-friendly
- [ ] Performance considerations addressed
- [ ] Integration with existing components verified
- [ ] No breaking changes to existing interfaces
- [ ] Security considerations addressed

## Common Implementation Patterns

### Async Operation Pattern
```python
async def execute_llm_operation_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
    """Execute LLM operation with comprehensive retry logic"""

    max_retries = self.config.max_retries
    retry_delay = self.config.retry_delay

    for attempt in range(max_retries + 1):
        try:
            # Log operation attempt
            self.logger.info(f"Attempting LLM operation: {operation.__name__} (attempt {attempt + 1})")

            # Execute operation
            result = await operation(*args, **kwargs)

            # Log success
            self.logger.info(f"LLM operation completed successfully: {operation.__name__}")

            return result

        except (ConnectionError, TimeoutError) as e:
            if attempt < max_retries:
                self.logger.warning(f"LLM operation failed (attempt {attempt + 1}): {e}")
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                self.logger.error(f"LLM operation failed after {max_retries + 1} attempts: {e}")
                raise

        except Exception as e:
            # Non-retryable errors
            self.logger.error(f"LLM operation failed with non-retryable error: {e}")
            raise
```

### Model Selection Pattern
```python
def select_optimal_model_with_fallback(self, task_requirements: Dict[str, Any]) -> Optional[ModelInfo]:
    """Select optimal model with intelligent fallback"""

    # Primary selection based on task requirements
    primary_model = self.model_manager.find_best_model(
        task=task_requirements.get("task", ""),
        max_memory_gb=task_requirements.get("max_memory"),
        preferred_family=task_requirements.get("preferred_family")
    )

    if primary_model and self._verify_model_availability(primary_model.name):
        return primary_model

    # Fallback to alternative models
    fallback_models = self.model_manager.get_models_by_capability(
        task_requirements.get("required_capability", "text_generation")
    )

    # Filter by availability and constraints
    available_fallbacks = [
        model for model in fallback_models
        if self._verify_model_availability(model.name) and
           model.get_memory_requirement() <= task_requirements.get("max_memory", float('inf'))
    ]

    if available_fallbacks:
        # Sort by preference and efficiency
        available_fallbacks.sort(key=lambda m: (
            self.model_manager.registry.preferred_models.index(m.name)
            if m.name in self.model_manager.registry.preferred_models else len(self.model_manager.registry.preferred_models),
            model.get_memory_requirement()
        ))
        return available_fallbacks[0]

    return None
```

## Related Documentation

- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Knowledge AGENTS.md](../knowledge/AGENTS.md)**: Knowledge management guidelines
- **[Research AGENTS.md](../research/AGENTS.md)**: Research tool development guidelines
- **[Applications AGENTS.md](../applications/AGENTS.md)**: Application development guidelines
- **[Platform AGENTS.md](../platform/AGENTS.md)**: Platform infrastructure guidelines
- **[Tools AGENTS.md](../tools/AGENTS.md)**: Development tools guidelines

---

*"Active Inference for, with, by Generative AI"* - Building intelligent LLM integration through comprehensive local AI capabilities and seamless platform integration.
