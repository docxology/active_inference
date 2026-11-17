# LLM Module - Quick Start Guide

## ✅ Setup Verification

The LLM module is fully functional! Here's how to use it.

### Prerequisites

1. **Ollama installed and running**:
   ```bash
   # Check Ollama version
   ollama --version
   
   # Check if Ollama is running
   curl http://localhost:11434/api/version
   
   # List available models
   ollama list
   ```

2. **Pull recommended models** (if not already available):
   ```bash
   ollama pull gemma2:2b    # Fast, efficient model
   ollama pull gemma3:4b    # More capable model
   ```

## 🚀 Quick Start

### 1. Simple CLI Usage

Use the included CLI tool for quick interactions:

```bash
# List available models
python3 llm_cli.py --list-models

# Simple generation
python3 llm_cli.py "Explain entropy in one sentence"

# Use specific model
python3 llm_cli.py "What is Active Inference?" --model gemma3:4b

# Use explanation template
python3 llm_cli.py "variational free energy" --template explanation

# Interactive chat session
python3 llm_cli.py --chat
```

### 2. Python API Usage

```python
import asyncio
from active_inference.llm import OllamaClient, LLMConfig

async def example():
    # Configure and initialize client
    config = LLMConfig(default_model='gemma2:2b')
    
    async with OllamaClient(config) as client:
        # Simple generation
        response = await client.generate("What is the Free Energy Principle?")
        print(response)
        
        # Chat with history
        messages = [
            {'role': 'user', 'content': 'Explain active inference'},
        ]
        response = await client.chat(messages)
        print(response)

asyncio.run(example())
```

### 3. Using Prompt Templates

```python
from active_inference.llm import PromptManager, OllamaClient, LLMConfig
import asyncio

async def example():
    prompt_manager = PromptManager()
    
    # Generate prompt from template
    prompt = prompt_manager.generate_prompt('active_inference_explanation', {
        'concept': 'expected free energy',
        'context': 'decision making',
        'audience_level': 'intermediate',
        'key_points': 'policy selection, exploration, exploitation',
        'response_type': 'detailed explanation'
    })
    
    # Use with LLM
    config = LLMConfig(default_model='gemma2:2b')
    async with OllamaClient(config) as client:
        response = await client.generate(prompt)
        print(response)

asyncio.run(example())
```

### 4. Conversation Management

```python
from active_inference.llm import ConversationManager, OllamaClient, LLMConfig
import asyncio

async def example():
    # Create conversation manager
    conv_manager = ConversationManager()
    
    # Create conversation
    conversation = conv_manager.create_conversation("Learning FEP")
    
    # Add messages
    conv_manager.add_message(conversation.id, 'user', 'What is the FEP?')
    
    # Get context and continue with LLM
    context = conv_manager.get_conversation_context(conversation.id)
    
    config = LLMConfig(default_model='gemma2:2b')
    async with OllamaClient(config) as client:
        response = await client.chat(context)
        
    # Save response
    conv_manager.add_message(conversation.id, 'assistant', response)
    
    # Export conversation
    markdown = conv_manager.export_conversation(conversation.id, 'markdown')
    print(markdown)

asyncio.run(example())
```

## 🧪 Run Tests

Run the comprehensive demo:

```bash
python3 demo_llm.py
```

Run unit tests:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH}" python3 -m pytest tests/unit/test_llm_*.py -v
```

## ✨ Features Verified

### ✅ Core Functionality
- [x] Ollama client initialization and health checks
- [x] Simple text generation
- [x] Chat with message history  
- [x] Streaming responses
- [x] Model pulling and management

### ✅ Model Management
- [x] Model registry with 5 pre-configured models
- [x] Capability-based model selection
- [x] Model information and benchmarking
- [x] Intelligent fallback strategies

### ✅ Prompt System
- [x] 3 built-in templates (explanation, research, code)
- [x] Template variable validation
- [x] Prompt builder pattern
- [x] Active Inference specialized prompts

### ✅ Conversation Management
- [x] Conversation creation and persistence
- [x] Message history management
- [x] Context window optimization
- [x] Conversation search
- [x] Export (JSON, Markdown)
- [x] Conversation templates

### ✅ Integration
- [x] Async/await support
- [x] Context managers for resource management
- [x] Comprehensive error handling
- [x] Logging and debugging

## 🎯 Example Use Cases

### 1. Generate Educational Content

```python
from active_inference.llm import OllamaClient, LLMConfig, PromptManager
import asyncio

async def generate_tutorial():
    prompt_manager = PromptManager()
    config = LLMConfig(default_model='gemma2:2b')
    
    prompt = prompt_manager.generate_prompt('active_inference_explanation', {
        'concept': 'predictive coding',
        'context': 'beginner tutorial',
        'audience_level': 'beginner',
        'key_points': 'prediction errors, hierarchical processing, learning',
        'response_type': 'step-by-step tutorial'
    })
    
    async with OllamaClient(config) as client:
        tutorial = await client.generate(prompt, max_tokens=500)
        print(tutorial)

asyncio.run(generate_tutorial())
```

### 2. Research Question Generation

```python
from active_inference.llm import OllamaClient, LLMConfig, PromptManager
import asyncio

async def generate_research_questions():
    prompt_manager = PromptManager()
    config = LLMConfig(default_model='gemma3:4b')
    
    prompt = prompt_manager.generate_prompt('research_question', {
        'topic': 'multi-agent active inference',
        'domain': 'artificial intelligence',
        'current_knowledge': 'single agent models established',
        'research_gap': 'coordination and communication',
        'methodology': 'computational simulation',
        'num_questions': '5'
    })
    
    async with OllamaClient(config) as client:
        questions = await client.generate(prompt, max_tokens=800)
        print(questions)

asyncio.run(generate_research_questions())
```

### 3. Code Implementation Assistant

```python
from active_inference.llm import OllamaClient, LLMConfig, PromptManager
import asyncio

async def generate_code():
    prompt_manager = PromptManager()
    config = LLMConfig(default_model='gemma2:2b')
    
    prompt = prompt_manager.generate_prompt('code_implementation', {
        'algorithm': 'variational inference',
        'language': 'Python',
        'problem_description': 'Implement basic variational inference',
        'requirements': 'numpy, scipy, type hints',
        'constraints': 'numerical stability',
        'codebase_context': 'Active Inference framework'
    })
    
    async with OllamaClient(config) as client:
        code = await client.generate(prompt, max_tokens=600)
        print(code)

asyncio.run(generate_code())
```

## 📊 Performance Notes

- **gemma2:2b**: ~50 tokens/sec, 1.6GB model, 4GB RAM
- **gemma3:4b**: ~30 tokens/sec, 3.3GB model, 8GB RAM
- **Response times**: 2-5 seconds for typical queries
- **Context window**: 4096-8192 tokens depending on model

## 🐛 Troubleshooting

### Ollama not running
```bash
# Start Ollama service
ollama serve
```

### Model not available
```bash
# Pull the model
ollama pull gemma2:2b
```

### Connection refused
```bash
# Check if Ollama is running on correct port
curl http://localhost:11434/api/version

# If different port, update LLMConfig:
config = LLMConfig(base_url="http://localhost:YOUR_PORT")
```

### Out of memory
```bash
# Use smaller model
ollama pull gemma2:2b

# Or configure max tokens
config = LLMConfig(max_tokens=100)
```

## 📚 Additional Resources

- **README.md**: Complete module documentation
- **AGENTS.md**: Development guidelines for AI agents
- **demo_llm.py**: Comprehensive demonstration script
- **llm_cli.py**: Simple command-line interface

## ✅ Status: 100% Working!

All LLM module features are implemented, tested, and working correctly with Ollama.

