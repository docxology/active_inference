# Large Language Model Integration

**LLM integration and conversational AI for the Active Inference Knowledge Environment.**

## Overview

This module provides integration with large language models to enhance the Active Inference platform with conversational AI, natural language understanding, and intelligent assistance capabilities.

### Core Features

- **Conversational Interface**: Natural language interaction with the platform
- **Knowledge Enhancement**: LLM-powered knowledge discovery and explanation
- **Code Generation**: AI-assisted code generation and completion
- **Content Generation**: Automated content creation and summarization
- **Query Processing**: Intelligent query understanding and routing
- **Personalization**: User-specific conversational experiences

## Architecture

### LLM Integration Components

```
┌─────────────────┐
│   Conversational│ ← Natural language interface
│     Interface   │
├─────────────────┤
│   Knowledge     │ ← LLM-enhanced knowledge access
│   Enhancement   │
├─────────────────┤
│   Model         │ ← LLM API integration, prompt engineering
│   Integration   │
├─────────────────┤
│ Content          │ ← Generation, summarization, analysis
│   Processing    │
└─────────────────┘
```

### Integration Points

- **Knowledge Repository**: Enhanced search and explanation
- **Platform Services**: Conversational API endpoints
- **Visualization**: AI-assisted visualization generation
- **Applications**: LLM-enhanced application development

## Usage

### Basic Setup

```python
from active_inference.llm import LLMClient

# Initialize LLM client
config = {
    "provider": "openai",  # or "anthropic", "local"
    "model": "gpt-4",
    "api_key": "your_api_key",
    "max_tokens": 2000
}

llm_client = LLMClient(config)
```

### Conversational Interface

```python
# Start conversation
conversation = llm_client.create_conversation()

# Ask questions
response = await conversation.ask(
    "What is the relationship between active inference and the free energy principle?"
)

# Get explanation
explanation = await conversation.explain_concept("bayesian inference")

# Generate code
code = await conversation.generate_code(
    "Create a simple active inference agent in Python"
)
```

### Knowledge Enhancement

```python
# Enhanced knowledge search
enhanced_results = await llm_client.enhanced_search(
    "active inference applications in neuroscience",
    context="research"
)

# Concept explanation
explanation = await llm_client.explain_concept(
    "variational free energy",
    difficulty="intermediate",
    context="mathematical"
)

# Learning path generation
learning_path = await llm_client.generate_learning_path(
    "active inference for beginners",
    user_level="novice"
)
```

### Content Generation

```python
# Generate tutorial
tutorial = await llm_client.generate_tutorial(
    topic="active inference basics",
    format="jupyter_notebook",
    difficulty="beginner"
)

# Create example code
example_code = await llm_client.generate_example(
    concept="policy selection",
    language="python",
    complexity="intermediate"
)

# Summarize content
summary = await llm_client.summarize_content(
    content_path="knowledge/foundations/active_inference_introduction.json",
    length="short"
)
```

## Configuration

### Provider Configuration

```python
# OpenAI configuration
openai_config = {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.7,
    "max_tokens": 2000,
    "streaming": True
}

# Anthropic configuration
anthropic_config = {
    "provider": "anthropic",
    "model": "claude-3-opus",
    "api_key": os.getenv("ANTHROPIC_API_KEY"),
    "temperature": 0.5
}

# Local model configuration
local_config = {
    "provider": "local",
    "model_path": "/path/to/model",
    "device": "cuda",
    "max_tokens": 1000
}
```

### Conversation Configuration

```python
conversation_config = {
    "system_prompt": "You are an expert in Active Inference and the Free Energy Principle...",
    "max_conversation_length": 50,
    "memory_enabled": True,
    "context_window": 4000,
    "response_format": "structured"
}
```

### Enhancement Configuration

```python
enhancement_config = {
    "knowledge_context": True,
    "cross_references": True,
    "difficulty_adaptation": True,
    "multilingual_support": True,
    "code_generation": True
}
```

## API Reference

### LLMClient

Main interface for LLM operations.

#### Core Methods

- `create_conversation(config: Dict = None) -> Conversation`: Create new conversation
- `enhanced_search(query: str, context: str = None) -> SearchResult`: Enhanced search
- `explain_concept(concept: str, difficulty: str = "intermediate") -> str`: Explain concept
- `generate_code(specification: str, language: str = "python") -> str`: Generate code
- `summarize_content(content_path: str, length: str = "medium") -> str`: Summarize content

### Conversation

Manages conversational interactions.

#### Methods

- `ask(question: str, context: Dict = None) -> str`: Ask question
- `explain_concept(concept: str, options: Dict = None) -> str`: Explain concept
- `generate_code(spec: str, options: Dict = None) -> str`: Generate code
- `continue_conversation(message: str) -> str`: Continue conversation
- `get_conversation_history() -> List[Message]`: Get conversation history

### ContentGenerator

Generates educational content and examples.

#### Methods

- `generate_tutorial(topic: str, format: str = "markdown") -> str`: Generate tutorial
- `create_example(concept: str, type: str = "code") -> str`: Create example
- `summarize_knowledge(content: Dict, style: str = "academic") -> str`: Summarize knowledge
- `generate_quiz(concepts: List[str], difficulty: str = "intermediate") -> Quiz`: Generate quiz

## Advanced Features

### Personalized Learning

```python
# Create personalized learning experience
personalizer = LLMPersonalizer(config)

# Adapt to user level
user_profile = personalizer.analyze_user_level(conversation_history)

# Generate personalized content
personalized_content = personalizer.generate_personalized_content(
    topic="active inference",
    user_level=user_profile["level"],
    learning_style=user_profile["style"]
)
```

### Multi-Modal Integration

```python
# Generate explanations with visuals
multimodal_response = await llm_client.generate_multimodal_explanation(
    concept="free_energy_principle",
    include_diagram=True,
    include_code=True,
    include_examples=True
)

# Create interactive tutorials
interactive_tutorial = await llm_client.create_interactive_tutorial(
    topic="bayesian_inference",
    format="jupyter",
    interactivity_level="high"
)
```

### Research Assistance

```python
# Literature analysis
analysis = await llm_client.analyze_literature(
    papers=["paper1.pdf", "paper2.pdf"],
    focus="active_inference_applications"
)

# Hypothesis generation
hypotheses = await llm_client.generate_hypotheses(
    research_question="How does active inference apply to social cognition?",
    constraints=["empirical_testability", "neuroscientific_plausibility"]
)
```

## Performance

### Optimization

```python
# Enable streaming responses
llm_client.enable_streaming()

# Cache common responses
llm_client.enable_response_caching()

# Optimize prompts
llm_client.optimize_prompts()
```

### Metrics

```python
# Monitor performance
metrics = llm_client.get_performance_metrics()

print(f"Response time: {metrics['avg_response_time']}ms")
print(f"Token usage: {metrics['total_tokens']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']}%")
```

## Testing

### Running Tests

```bash
# Run LLM integration tests
make test-llm

# Or run specific tests
pytest src/active_inference/llm/tests/ -v

# Integration tests
pytest src/active_inference/llm/tests/test_integration.py -v

# Performance tests
pytest src/active_inference/llm/tests/test_performance.py -v
```

### Test Coverage

- **Unit Tests**: LLM API integration
- **Integration Tests**: End-to-end conversational workflows
- **Performance Tests**: Response time and throughput
- **Quality Tests**: Generated content quality validation

## Security

### API Security

```python
# Secure API configuration
secure_config = {
    "api_key_encrypted": True,
    "rate_limiting": True,
    "request_validation": True,
    "audit_logging": True
}
```

### Content Safety

- Content filtering and validation
- User input sanitization
- Generated content review
- Ethical guidelines enforcement

## Monitoring

### Health Checks

```python
# LLM service health check
health = llm_client.health_check()

print(f"API status: {health['api_status']}")
print(f"Response time: {health['response_time']}ms")
print(f"Rate limit: {health['rate_limit_status']}")
```

### Usage Analytics

```bash
# Start usage analytics
make llm-analytics

# View usage patterns
curl http://localhost:8080/llm/analytics
```

## Contributing

See [AGENTS.md](AGENTS.md) for development guidelines and [.cursorrules](../../../.cursorrules) for comprehensive development standards.

### Development Process

1. **Environment Setup**:
   ```bash
   cd src/active_inference/llm
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

## Ethical Considerations

### Responsible AI Usage

- **Accuracy**: Ensure generated content accuracy
- **Transparency**: Clear indication of AI-generated content
- **Privacy**: Protect user data and conversations
- **Bias Mitigation**: Address potential biases in responses
- **Educational Value**: Enhance rather than replace learning

### Content Guidelines

```python
# Configure content guidelines
guidelines_config = {
    "accuracy_threshold": 0.9,
    "bias_detection": True,
    "fact_checking": True,
    "citation_requirements": True,
    "educational_focus": True
}
```

## Troubleshooting

### Common Issues

#### API Connection Issues
```bash
# Check API connectivity
llm_client.test_connection()

# Validate API configuration
llm_client.validate_config()

# Check rate limits
llm_client.check_rate_limits()
```

#### Performance Issues
```bash
# Monitor response times
llm_client.monitor_performance()

# Optimize configuration
llm_client.optimize_for_performance()

# Clear caches
llm_client.clear_caches()
```

#### Content Quality Issues
```bash
# Validate generated content
llm_client.validate_content_quality()

# Adjust generation parameters
llm_client.adjust_generation_parameters()

# Enable content review
llm_client.enable_content_review()
```

---

**Component Version**: 1.0.0 | **Development Status**: Active Development

*"Active Inference for, with, by Generative AI"* - Enhanced understanding through conversational intelligence.