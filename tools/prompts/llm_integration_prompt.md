# LLM Integration and Management Prompt

**"Active Inference for, with, by Generative AI"**

## 🎯 Mission: Develop Comprehensive LLM Integration and Management

You are tasked with developing comprehensive Large Language Model integration and management systems for the Active Inference Knowledge Environment. This involves creating model registries, conversation management, prompt engineering frameworks, performance monitoring, and seamless integration with the platform's knowledge and research capabilities.

## 📋 LLM Integration Requirements

### Core LLM Standards (MANDATORY)
1. **Model Registry**: Centralized management of multiple LLM models and configurations
2. **Conversation Management**: Persistent, contextual conversation handling across sessions
3. **Prompt Engineering**: Structured prompt templates with version control and optimization
4. **Performance Monitoring**: Real-time monitoring of model performance and costs
5. **Safety Integration**: Content filtering, bias detection, and responsible AI practices
6. **Knowledge Integration**: Seamless integration with platform knowledge base and research tools

### LLM Integration Architecture
```
llm/
├── client.py                    # Unified LLM client interface
├── models.py                    # Model management and registry
├── conversations.py             # Conversation handling and storage
├── prompts.py                   # Prompt engineering and templates
├── monitoring.py                # Performance monitoring and analytics
├── safety.py                    # Safety filters and content moderation
├── knowledge_integration.py     # Integration with knowledge base
└── api/                         # REST API for LLM services
    ├── endpoints.py            # API endpoints
    ├── authentication.py       # API authentication
    ├── rate_limiting.py        # Rate limiting
    └── documentation.py        # API documentation
```

## 🏗️ LLM Client and Model Management

### Phase 1: Unified LLM Client Interface

#### 1.1 Model Registry System
```python
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from abc import ABC, abstractmethod
import asyncio
import aiohttp

@dataclass
class ModelConfiguration:
    """Configuration for an LLM model"""
    model_id: str
    provider: str  # 'openai', 'anthropic', 'google', 'local'
    model_name: str
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    context_window: int = 4096
    pricing: Dict[str, float] = field(default_factory=dict)  # per-token pricing
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None

@dataclass
class ModelPerformance:
    """Performance metrics for a model"""
    model_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class ModelRegistry:
    """Centralized registry for LLM models"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize model registry"""
        self.config = config
        self.logger = logging.getLogger('ModelRegistry')
        self.models: Dict[str, ModelConfiguration] = {}
        self.performance: Dict[str, ModelPerformance] = {}
        self.load_models()

    def load_models(self) -> None:
        """Load model configurations"""
        # Load from configuration or database
        default_models = self.get_default_models()
        for model_config in default_models:
            self.register_model(model_config)

    def get_default_models(self) -> List[ModelConfiguration]:
        """Get default model configurations"""
        return [
            ModelConfiguration(
                model_id='gpt-4',
                provider='openai',
                model_name='gpt-4',
                max_tokens=8192,
                context_window=8192,
                pricing={'input': 0.03, 'output': 0.06},
                capabilities=['text_generation', 'code_generation', 'analysis', 'reasoning'],
                limitations=['context_window_8192', 'no_image_generation']
            ),
            ModelConfiguration(
                model_id='claude-3-opus',
                provider='anthropic',
                model_name='claude-3-opus-20240229',
                max_tokens=4096,
                context_window=200000,
                pricing={'input': 0.015, 'output': 0.075},
                capabilities=['text_generation', 'analysis', 'reasoning', 'long_context'],
                limitations=['max_tokens_4096']
            ),
            ModelConfiguration(
                model_id='gemini-pro',
                provider='google',
                model_name='gemini-pro',
                max_tokens=30720,
                context_window=30720,
                pricing={'input': 0.00025, 'output': 0.0005},
                capabilities=['text_generation', 'multimodal', 'code_generation'],
                limitations=['beta_features']
            )
        ]

    def register_model(self, model_config: ModelConfiguration) -> None:
        """Register a new model"""
        self.models[model_config.model_id] = model_config
        self.performance[model_config.model_id] = ModelPerformance(model_id=model_config.model_id)
        self.logger.info(f"Registered model: {model_config.model_id}")

    def get_model(self, model_id: str) -> Optional[ModelConfiguration]:
        """Get model configuration by ID"""
        return self.models.get(model_id)

    def list_active_models(self) -> List[ModelConfiguration]:
        """List all active models"""
        return [model for model in self.models.values() if model.is_active]

    def update_model_performance(self, model_id: str, request_metrics: Dict[str, Any]) -> None:
        """Update performance metrics for a model"""
        if model_id not in self.performance:
            self.performance[model_id] = ModelPerformance(model_id=model_id)

        perf = self.performance[model_id]
        perf.total_requests += 1

        if request_metrics.get('success', False):
            perf.successful_requests += 1
        else:
            perf.failed_requests += 1

        if 'tokens' in request_metrics:
            perf.total_tokens += request_metrics['tokens']

        if 'cost' in request_metrics:
            perf.total_cost += request_metrics['cost']

        if 'response_time' in request_metrics:
            response_time = request_metrics['response_time']
            perf.response_times.append(response_time)
            # Keep only last 1000 response times
            if len(perf.response_times) > 1000:
                perf.response_times = perf.response_times[-1000:]

            perf.average_response_time = sum(perf.response_times) / len(perf.response_times)

        perf.error_rate = perf.failed_requests / perf.total_requests if perf.total_requests > 0 else 0
        perf.last_updated = datetime.now()

    def select_model_for_task(self, task_requirements: Dict[str, Any]) -> Optional[str]:
        """Select best model for a specific task"""
        candidates = self.list_active_models()

        # Filter by capabilities
        required_capabilities = task_requirements.get('required_capabilities', [])
        if required_capabilities:
            candidates = [m for m in candidates if all(cap in m.capabilities for cap in required_capabilities)]

        if not candidates:
            return None

        # Score models based on multiple criteria
        scored_models = []
        for model in candidates:
            score = self.score_model_for_task(model, task_requirements)
            scored_models.append((model.model_id, score))

        # Return highest scoring model
        return max(scored_models, key=lambda x: x[1])[0]

    def score_model_for_task(self, model: ModelConfiguration, task_requirements: Dict[str, Any]) -> float:
        """Score how well a model fits task requirements"""
        score = 0.0

        # Performance score (based on recent performance)
        perf = self.performance.get(model.model_id)
        if perf and perf.total_requests > 0:
            success_rate = perf.successful_requests / perf.total_requests
            score += success_rate * 0.4

            # Prefer faster models for speed-critical tasks
            if task_requirements.get('priority') == 'speed':
                avg_time = perf.average_response_time
                speed_score = max(0, 1.0 - (avg_time / 30.0))  # Prefer < 30s responses
                score += speed_score * 0.3

        # Capability match score
        required_caps = task_requirements.get('required_capabilities', [])
        if required_caps:
            capability_score = sum(1 for cap in required_caps if cap in model.capabilities) / len(required_caps)
            score += capability_score * 0.3

        # Cost efficiency score
        if 'max_budget' in task_requirements:
            # Estimate cost for typical task
            estimated_cost = self.estimate_task_cost(model, task_requirements)
            if estimated_cost <= task_requirements['max_budget']:
                cost_score = 1.0 - (estimated_cost / task_requirements['max_budget'])
                score += cost_score * 0.2

        return score

    def estimate_task_cost(self, model: ModelConfiguration, task_requirements: Dict[str, Any]) -> float:
        """Estimate cost for a task with this model"""
        # Estimate token usage
        estimated_input_tokens = task_requirements.get('estimated_input_tokens', 1000)
        estimated_output_tokens = task_requirements.get('estimated_output_tokens', 500)

        input_cost = estimated_input_tokens * model.pricing.get('input', 0) / 1000
        output_cost = estimated_output_tokens * model.pricing.get('output', 0) / 1000

        return input_cost + output_cost

    def get_model_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """Get performance summary for a model"""
        perf = self.performance.get(model_id)
        if not perf:
            return {'error': 'Model not found or no performance data'}

        return {
            'model_id': model_id,
            'total_requests': perf.total_requests,
            'success_rate': perf.successful_requests / perf.total_requests if perf.total_requests > 0 else 0,
            'error_rate': perf.error_rate,
            'average_response_time': perf.average_response_time,
            'total_tokens': perf.total_tokens,
            'total_cost': perf.total_cost,
            'cost_per_token': perf.total_cost / perf.total_tokens if perf.total_tokens > 0 else 0,
            'last_updated': perf.last_updated.isoformat()
        }

    def deactivate_model(self, model_id: str) -> bool:
        """Deactivate a model"""
        if model_id in self.models:
            self.models[model_id].is_active = False
            self.logger.info(f"Deactivated model: {model_id}")
            return True
        return False

    def export_registry(self, filepath: str) -> None:
        """Export model registry to file"""
        registry_data = {
            'models': [model.__dict__ for model in self.models.values()],
            'performance': [perf.__dict__ for perf in self.performance.values()],
            'exported_at': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2, default=str)
```

#### 1.2 Unified LLM Client
```python
class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    def __init__(self, model_config: ModelConfiguration):
        """Initialize LLM client"""
        self.model_config = model_config
        self.logger = logging.getLogger(f'LLMClient.{model_config.provider}')

    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using the LLM"""
        pass

    @abstractmethod
    async def generate_chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate chat response using the LLM"""
        pass

    def validate_request(self, **kwargs) -> None:
        """Validate request parameters"""
        max_tokens = kwargs.get('max_tokens', 0)
        if max_tokens > self.model_config.max_tokens:
            raise ValueError(f"max_tokens ({max_tokens}) exceeds model limit ({self.model_config.max_tokens})")

        temperature = kwargs.get('temperature', 0.7)
        if not 0 <= temperature <= 2:
            raise ValueError(f"temperature ({temperature}) must be between 0 and 2")

class OpenAIClient(LLMClient):
    """OpenAI API client"""

    def __init__(self, model_config: ModelConfiguration):
        """Initialize OpenAI client"""
        super().__init__(model_config)
        import openai
        self.client = openai.AsyncOpenAI(
            api_key=model_config.api_key,
            base_url=model_config.api_base_url
        )

    async def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using OpenAI"""
        try:
            response = await self.client.completions.create(
                model=self.model_config.model_name,
                prompt=prompt,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', self.model_config.temperature),
                top_p=kwargs.get('top_p', self.model_config.top_p),
                frequency_penalty=kwargs.get('frequency_penalty', self.model_config.frequency_penalty),
                presence_penalty=kwargs.get('presence_penalty', self.model_config.presence_penalty)
            )

            return {
                'success': True,
                'text': response.choices[0].text,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def generate_chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate chat response using OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_config.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', self.model_config.temperature),
                top_p=kwargs.get('top_p', self.model_config.top_p),
                frequency_penalty=kwargs.get('frequency_penalty', self.model_config.frequency_penalty),
                presence_penalty=kwargs.get('presence_penalty', self.model_config.presence_penalty)
            )

            return {
                'success': True,
                'text': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }

        except Exception as e:
            self.logger.error(f"OpenAI chat API error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class AnthropicClient(LLMClient):
    """Anthropic Claude API client"""

    def __init__(self, model_config: ModelConfiguration):
        """Initialize Anthropic client"""
        super().__init__(model_config)
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=model_config.api_key)

    async def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using Anthropic"""
        try:
            response = await self.client.messages.create(
                model=self.model_config.model_name,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', self.model_config.temperature),
                system="You are a helpful AI assistant.",
                messages=[{"role": "user", "content": prompt}]
            )

            return {
                'success': True,
                'text': response.content[0].text,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                },
                'model': response.model,
                'stop_reason': response.stop_reason
            }

        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def generate_chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate chat response using Anthropic"""
        try:
            # Convert messages format
            anthropic_messages = []
            system_message = None

            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    anthropic_messages.append(msg)

            response = await self.client.messages.create(
                model=self.model_config.model_name,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', self.model_config.temperature),
                system=system_message,
                messages=anthropic_messages
            )

            return {
                'success': True,
                'text': response.content[0].text,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                },
                'model': response.model,
                'stop_reason': response.stop_reason
            }

        except Exception as e:
            self.logger.error(f"Anthropic chat API error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class LLMClientFactory:
    """Factory for creating LLM clients"""

    @staticmethod
    def create_client(model_config: ModelConfiguration) -> LLMClient:
        """Create appropriate client for model provider"""

        if model_config.provider == 'openai':
            return OpenAIClient(model_config)
        elif model_config.provider == 'anthropic':
            return AnthropicClient(model_config)
        elif model_config.provider == 'google':
            return GoogleClient(model_config)
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")

class UnifiedLLMClient:
    """Unified interface for all LLM operations"""

    def __init__(self, registry: ModelRegistry, config: Dict[str, Any]):
        """Initialize unified LLM client"""
        self.registry = registry
        self.config = config
        self.logger = logging.getLogger('UnifiedLLMClient')
        self.clients: Dict[str, LLMClient] = {}
        self.request_history: List[Dict[str, Any]] = []

    async def generate_text(self, prompt: str, model_id: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """Generate text using specified or best available model"""
        start_time = datetime.now()

        # Select model
        if not model_id:
            task_reqs = {'required_capabilities': ['text_generation']}
            model_id = self.registry.select_model_for_task(task_reqs)

        if not model_id:
            return {'success': False, 'error': 'No suitable model available'}

        model_config = self.registry.get_model(model_id)
        if not model_config:
            return {'success': False, 'error': f'Model {model_id} not found'}

        # Get or create client
        client = self.get_client(model_config)

        # Generate text
        response = await client.generate_text(prompt, **kwargs)
        response_time = (datetime.now() - start_time).total_seconds()

        # Record request
        request_record = {
            'timestamp': start_time.isoformat(),
            'model_id': model_id,
            'request_type': 'text_generation',
            'prompt_length': len(prompt),
            'response_time': response_time,
            'success': response.get('success', False),
            'tokens_used': response.get('usage', {}).get('total_tokens', 0)
        }
        self.request_history.append(request_record)

        # Update performance metrics
        self.registry.update_model_performance(model_id, {
            'success': response.get('success', False),
            'response_time': response_time,
            'tokens': response.get('usage', {}).get('total_tokens', 0),
            'cost': self.calculate_cost(model_config, response.get('usage', {}))
        })

        return response

    async def generate_chat(self, messages: List[Dict[str, Any]],
                          model_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate chat response"""
        start_time = datetime.now()

        # Select model
        if not model_id:
            task_reqs = {'required_capabilities': ['text_generation']}
            model_id = self.registry.select_model_for_task(task_reqs)

        if not model_id:
            return {'success': False, 'error': 'No suitable model available'}

        model_config = self.registry.get_model(model_id)
        if not model_config:
            return {'success': False, 'error': f'Model {model_id} not found'}

        # Get or create client
        client = self.get_client(model_config)

        # Generate chat response
        response = await client.generate_chat(messages, **kwargs)
        response_time = (datetime.now() - start_time).total_seconds()

        # Record request
        request_record = {
            'timestamp': start_time.isoformat(),
            'model_id': model_id,
            'request_type': 'chat',
            'message_count': len(messages),
            'response_time': response_time,
            'success': response.get('success', False),
            'tokens_used': response.get('usage', {}).get('total_tokens', 0)
        }
        self.request_history.append(request_record)

        # Update performance metrics
        self.registry.update_model_performance(model_id, {
            'success': response.get('success', False),
            'response_time': response_time,
            'tokens': response.get('usage', {}).get('total_tokens', 0),
            'cost': self.calculate_cost(model_config, response.get('usage', {}))
        })

        return response

    def get_client(self, model_config: ModelConfiguration) -> LLMClient:
        """Get or create client for model"""
        if model_config.model_id not in self.clients:
            self.clients[model_config.model_id] = LLMClientFactory.create_client(model_config)

        return self.clients[model_config.model_id]

    def calculate_cost(self, model_config: ModelConfiguration, usage: Dict[str, Any]) -> float:
        """Calculate cost for API usage"""
        input_tokens = usage.get('input_tokens', usage.get('prompt_tokens', 0))
        output_tokens = usage.get('output_tokens', usage.get('completion_tokens', 0))

        input_cost = input_tokens * model_config.pricing.get('input', 0) / 1000
        output_cost = output_tokens * model_config.pricing.get('output', 0) / 1000

        return input_cost + output_cost

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics across all models"""
        total_requests = len(self.request_history)
        successful_requests = sum(1 for r in self.request_history if r['success'])
        total_tokens = sum(r.get('tokens_used', 0) for r in self.request_history)
        total_cost = sum(self.calculate_cost(
            self.registry.get_model(r['model_id']),
            {'input_tokens': r.get('tokens_used', 0) // 2, 'output_tokens': r.get('tokens_used', 0) // 2}
        ) for r in self.request_history)

        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'average_cost_per_request': total_cost / total_requests if total_requests > 0 else 0,
            'requests_per_model': self.get_requests_per_model()
        }

    def get_requests_per_model(self) -> Dict[str, int]:
        """Get request count per model"""
        model_counts = {}
        for request in self.request_history:
            model_id = request['model_id']
            model_counts[model_id] = model_counts.get(model_id, 0) + 1
        return model_counts
```

### Phase 2: Conversation Management System

#### 2.1 Conversation Storage and Retrieval
```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
import logging

@dataclass
class ConversationMessage:
    """Individual message in a conversation"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class Conversation:
    """Complete conversation session"""
    conversation_id: str
    title: str = ""
    messages: List[ConversationMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    model_used: Optional[str] = None
    total_tokens: int = 0
    total_cost: float = 0.0
    tags: List[str] = field(default_factory=list)

    def add_message(self, message: ConversationMessage) -> None:
        """Add message to conversation"""
        self.messages.append(message)
        self.updated_at = datetime.now()

        # Update token count if available
        if 'tokens' in message.metadata:
            self.total_tokens += message.metadata['tokens']

        # Update cost if available
        if 'cost' in message.metadata:
            self.total_cost += message.metadata['cost']

    def get_messages_for_api(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages formatted for LLM API calls"""
        messages = []

        for msg in self.messages:
            api_message = {
                'role': msg.role,
                'content': msg.content
            }
            messages.append(api_message)

        # If max_tokens specified, truncate from the beginning to fit
        if max_tokens:
            # Simple token estimation (rough approximation)
            total_tokens = sum(len(msg['content'].split()) for msg in messages)

            while total_tokens > max_tokens and len(messages) > 1:
                # Remove oldest messages (but keep system message if present)
                if messages[0]['role'] == 'system':
                    messages.pop(1)  # Remove after system message
                else:
                    messages.pop(0)
                total_tokens = sum(len(msg['content'].split()) for msg in messages)

        return messages

    def get_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        return {
            'conversation_id': self.conversation_id,
            'title': self.title or f"Conversation {self.conversation_id[:8]}",
            'message_count': len(self.messages),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'model_used': self.model_used,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'tags': self.tags,
            'last_message_preview': self.messages[-1].content[:100] if self.messages else ""
        }

class ConversationManager:
    """Manages conversation storage and retrieval"""

    def __init__(self, storage_config: Dict[str, Any]):
        """Initialize conversation manager"""
        self.storage_config = storage_config
        self.logger = logging.getLogger('ConversationManager')
        self.conversations: Dict[str, Conversation] = {}

        # Load existing conversations
        self.load_conversations()

    def create_conversation(self, title: str = "", metadata: Dict[str, Any] = None) -> str:
        """Create new conversation"""
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            conversation_id=conversation_id,
            title=title,
            metadata=metadata or {}
        )

        self.conversations[conversation_id] = conversation
        self.logger.info(f"Created conversation: {conversation_id}")

        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        return self.conversations.get(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str,
                   metadata: Dict[str, Any] = None) -> bool:
        """Add message to conversation"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False

        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )

        conversation.add_message(message)

        # Auto-save if configured
        if self.storage_config.get('auto_save', True):
            self.save_conversation(conversation_id)

        return True

    def list_conversations(self, limit: int = 50, offset: int = 0,
                          tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List conversations with optional filtering"""
        conversations = list(self.conversations.values())

        # Filter by tags if specified
        if tags:
            conversations = [c for c in conversations if any(tag in c.tags for tag in tags)]

        # Sort by updated time (most recent first)
        conversations.sort(key=lambda c: c.updated_at, reverse=True)

        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        paginated = conversations[start_idx:end_idx]

        return [c.get_summary() for c in paginated]

    def search_conversations(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search conversations by content"""
        query_lower = query.lower()
        matching_conversations = []

        for conversation in self.conversations.values():
            # Search in title
            if query_lower in conversation.title.lower():
                matching_conversations.append(conversation)
                continue

            # Search in message content
            for message in conversation.messages:
                if query_lower in message.content.lower():
                    matching_conversations.append(conversation)
                    break

        # Remove duplicates and sort by relevance
        unique_conversations = list(set(matching_conversations))
        unique_conversations.sort(key=lambda c: c.updated_at, reverse=True)

        return [c.get_summary() for c in unique_conversations[:limit]]

    def update_conversation_metadata(self, conversation_id: str,
                                   metadata: Dict[str, Any]) -> bool:
        """Update conversation metadata"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False

        conversation.metadata.update(metadata)
        conversation.updated_at = datetime.now()

        if self.storage_config.get('auto_save', True):
            self.save_conversation(conversation_id)

        return True

    def add_tags(self, conversation_id: str, tags: List[str]) -> bool:
        """Add tags to conversation"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False

        for tag in tags:
            if tag not in conversation.tags:
                conversation.tags.append(tag)

        conversation.updated_at = datetime.now()

        if self.storage_config.get('auto_save', True):
            self.save_conversation(conversation_id)

        return True

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation"""
        if conversation_id not in self.conversations:
            return False

        # Remove from memory
        del self.conversations[conversation_id]

        # Remove from storage if applicable
        storage_path = self.storage_config.get('storage_path', './conversations')
        import os
        file_path = os.path.join(storage_path, f"{conversation_id}.json")

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            self.logger.warning(f"Failed to delete conversation file: {e}")

        self.logger.info(f"Deleted conversation: {conversation_id}")
        return True

    def save_conversation(self, conversation_id: str) -> bool:
        """Save conversation to storage"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False

        storage_path = self.storage_config.get('storage_path', './conversations')
        os.makedirs(storage_path, exist_ok=True)

        file_path = os.path.join(storage_path, f"{conversation_id}.json")

        try:
            # Convert to serializable format
            conversation_data = {
                'conversation_id': conversation.conversation_id,
                'title': conversation.title,
                'metadata': conversation.metadata,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat(),
                'model_used': conversation.model_used,
                'total_tokens': conversation.total_tokens,
                'total_cost': conversation.total_cost,
                'tags': conversation.tags,
                'messages': [
                    {
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp.isoformat(),
                        'metadata': msg.metadata,
                        'message_id': msg.message_id
                    }
                    for msg in conversation.messages
                ]
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            self.logger.error(f"Failed to save conversation {conversation_id}: {e}")
            return False

    def load_conversations(self) -> None:
        """Load conversations from storage"""
        storage_path = self.storage_config.get('storage_path', './conversations')

        if not os.path.exists(storage_path):
            return

        try:
            for filename in os.listdir(storage_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(storage_path, filename)

                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Reconstruct conversation
                    messages = []
                    for msg_data in data['messages']:
                        message = ConversationMessage(
                            role=msg_data['role'],
                            content=msg_data['content'],
                            timestamp=datetime.fromisoformat(msg_data['timestamp']),
                            metadata=msg_data.get('metadata', {}),
                            message_id=msg_data.get('message_id', str(uuid.uuid4()))
                        )
                        messages.append(message)

                    conversation = Conversation(
                        conversation_id=data['conversation_id'],
                        title=data.get('title', ''),
                        messages=messages,
                        metadata=data.get('metadata', {}),
                        created_at=datetime.fromisoformat(data['created_at']),
                        updated_at=datetime.fromisoformat(data['updated_at']),
                        model_used=data.get('model_used'),
                        total_tokens=data.get('total_tokens', 0),
                        total_cost=data.get('total_cost', 0.0),
                        tags=data.get('tags', [])
                    )

                    self.conversations[conversation.conversation_id] = conversation

            self.logger.info(f"Loaded {len(self.conversations)} conversations")

        except Exception as e:
            self.logger.error(f"Failed to load conversations: {e}")

    def get_conversation_statistics(self) -> Dict[str, Any]:
        """Get statistics about conversations"""
        if not self.conversations:
            return {'total_conversations': 0}

        conversations = list(self.conversations.values())
        total_messages = sum(len(c.messages) for c in conversations)
        total_tokens = sum(c.total_tokens for c in conversations)
        total_cost = sum(c.total_cost for c in conversations)

        # Tag statistics
        all_tags = []
        for c in conversations:
            all_tags.extend(c.tags)

        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            'total_conversations': len(conversations),
            'total_messages': total_messages,
            'average_messages_per_conversation': total_messages / len(conversations),
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'most_common_tags': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
```

### Phase 3: Prompt Engineering Framework

#### 3.1 Dynamic Prompt Templates
```python
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

@dataclass
class PromptTemplate:
    """Dynamic prompt template with variable substitution"""

    template_id: str
    name: str
    description: str
    template_text: str
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def render(self, variables: Dict[str, Any]) -> str:
        """Render template with variable substitution"""
        rendered = self.template_text

        # Substitute variables
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            rendered = rendered.replace(placeholder, str(var_value))

        return rendered

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate that all required variables are provided"""
        missing_vars = []

        for var_name in self.variables:
            if var_name not in variables:
                missing_vars.append(var_name)

        return missing_vars

    def get_template_info(self) -> Dict[str, Any]:
        """Get template information"""
        return {
            'template_id': self.template_id,
            'name': self.name,
            'description': self.description,
            'variables': self.variables,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class PromptEngineeringFramework:
    """Framework for prompt engineering and optimization"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize prompt engineering framework"""
        self.config = config
        self.logger = logging.getLogger('PromptEngineeringFramework')
        self.templates: Dict[str, PromptTemplate] = {}
        self.template_performance: Dict[str, List[Dict[str, Any]]] = {}

        # Load default templates
        self.load_default_templates()

    def load_default_templates(self) -> None:
        """Load default prompt templates"""
        default_templates = [
            PromptTemplate(
                template_id='knowledge_query',
                name='Knowledge Base Query',
                description='Query the Active Inference knowledge base',
                template_text="""You are an expert in Active Inference. Use the following knowledge base to answer the query.

Knowledge Base Context:
{knowledge_context}

Query: {user_query}

Instructions:
1. Base your answer on the provided knowledge base context
2. Be accurate and cite specific concepts when possible
3. Explain complex ideas in accessible terms
4. If the context doesn't contain enough information, say so clearly

Answer:""",
                variables=['knowledge_context', 'user_query']
            ),
            PromptTemplate(
                template_id='code_generation',
                name='Code Generation',
                description='Generate code for Active Inference implementations',
                template_text="""Generate Python code for the following Active Inference task:

Task: {task_description}
Requirements: {requirements}
Context: {context}

Code should:
- Follow Active Inference principles
- Be well-documented with comments
- Include error handling
- Be efficient and scalable
- Follow PEP 8 style guidelines

Generate the complete, runnable code:

```python
{generated_code}
```""",
                variables=['task_description', 'requirements', 'context', 'generated_code']
            ),
            PromptTemplate(
                template_id='research_analysis',
                name='Research Analysis',
                description='Analyze research papers and findings',
                template_text="""Analyze the following research in the context of Active Inference:

Paper Title: {paper_title}
Authors: {authors}
Abstract: {abstract}
Key Findings: {key_findings}

Analysis Framework:
1. **Relevance to Active Inference**: How does this research relate to Active Inference principles?
2. **Theoretical Contributions**: What theoretical insights does it provide?
3. **Methodological Advances**: What new methods or approaches are introduced?
4. **Practical Implications**: How might this impact Active Inference applications?
5. **Future Research Directions**: What questions does this raise?

Provide a structured analysis:""",
                variables=['paper_title', 'authors', 'abstract', 'key_findings']
            )
        ]

        for template in default_templates:
            self.templates[template.template_id] = template

    def create_template(self, template_id: str, name: str, description: str,
                       template_text: str, variables: List[str]) -> PromptTemplate:
        """Create new prompt template"""
        template = PromptTemplate(
            template_id=template_id,
            name=name,
            description=description,
            template_text=template_text,
            variables=variables
        )

        self.templates[template_id] = template
        self.logger.info(f"Created prompt template: {template_id}")

        return template

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)

    def render_prompt(self, template_id: str, variables: Dict[str, Any]) -> Optional[str]:
        """Render prompt using template"""
        template = self.get_template(template_id)
        if not template:
            return None

        # Validate variables
        missing_vars = template.validate_variables(variables)
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        return template.render(variables)

    def optimize_prompt(self, template_id: str, optimization_criteria: Dict[str, Any]) -> Optional[PromptTemplate]:
        """Optimize prompt template based on performance data"""
        template = self.get_template(template_id)
        if not template:
            return None

        performance_data = self.template_performance.get(template_id, [])
        if not performance_data:
            return template

        # Analyze performance patterns
        optimization_suggestions = self.analyze_performance(performance_data)

        # Apply optimizations
        optimized_template = self.apply_optimizations(template, optimization_suggestions)

        return optimized_template

    def record_template_performance(self, template_id: str, performance_data: Dict[str, Any]) -> None:
        """Record template performance for optimization"""
        if template_id not in self.template_performance:
            self.template_performance[template_id] = []

        performance_record = {
            'timestamp': datetime.now().isoformat(),
            **performance_data
        }

        self.template_performance[template_id].append(performance_record)

        # Keep only recent performance data
        if len(self.template_performance[template_id]) > 100:
            self.template_performance[template_id] = self.template_performance[template_id][-100:]

    def analyze_performance(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze template performance patterns"""
        if not performance_data:
            return {}

        # Extract metrics
        response_qualities = [d.get('response_quality', 0) for d in performance_data]
        response_times = [d.get('response_time', 0) for d in performance_data]
        token_usages = [d.get('tokens_used', 0) for d in performance_data]

        suggestions = {}

        # Quality optimization
        avg_quality = sum(response_qualities) / len(response_qualities)
        if avg_quality < 0.7:
            suggestions['add_examples'] = True
            suggestions['clarify_instructions'] = True

        # Efficiency optimization
        avg_response_time = sum(response_times) / len(response_times)
        if avg_response_time > 30:  # More than 30 seconds
            suggestions['shorten_prompt'] = True
            suggestions['simplify_structure'] = True

        # Cost optimization
        avg_tokens = sum(token_usages) / len(token_usages)
        if avg_tokens > 2000:
            suggestions['reduce_context'] = True
            suggestions['be_more_specific'] = True

        return suggestions

    def apply_optimizations(self, template: PromptTemplate, suggestions: Dict[str, Any]) -> PromptTemplate:
        """Apply optimization suggestions to template"""
        optimized_text = template.template_text

        if suggestions.get('add_examples'):
            if 'Examples:' not in optimized_text:
                optimized_text += "\n\nExamples:\n- Example 1: [Add specific example]"

        if suggestions.get('clarify_instructions'):
            if 'Be specific about:' not in optimized_text:
                optimized_text += "\n- Be specific about requirements and constraints"

        if suggestions.get('shorten_prompt'):
            # Remove redundant sections (simplified)
            pass

        if suggestions.get('simplify_structure'):
            # Simplify complex structures (simplified)
            pass

        # Create optimized template
        optimized_template = PromptTemplate(
            template_id=f"{template.template_id}_optimized",
            name=f"{template.name} (Optimized)",
            description=template.description,
            template_text=optimized_text,
            variables=template.variables,
            metadata={**template.metadata, 'optimized': True},
            version=f"{template.version}-opt"
        )

        return optimized_template

    def export_templates(self, filepath: str) -> None:
        """Export templates to file"""
        templates_data = {
            'templates': [template.get_template_info() for template in self.templates.values()],
            'performance': self.template_performance,
            'exported_at': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(templates_data, f, indent=2, default=str)

    def import_templates(self, filepath: str) -> int:
        """Import templates from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        imported_count = 0
        for template_data in data.get('templates', []):
            # Recreate template
            template = PromptTemplate(
                template_id=template_data['template_id'],
                name=template_data['name'],
                description=template_data['description'],
                template_text="",  # Would need to be stored separately
                variables=template_data['variables'],
                version=template_data['version']
            )
            self.templates[template.template_id] = template
            imported_count += 1

        if 'performance' in data:
            self.template_performance.update(data['performance'])

        return imported_count

    def get_template_recommendations(self, task_description: str) -> List[Dict[str, Any]]:
        """Get template recommendations for a task"""
        # Simple keyword matching (could be improved with ML)
        task_lower = task_description.lower()
        recommendations = []

        for template in self.templates.values():
            score = 0

            # Keyword matching
            if 'knowledge' in task_lower and 'knowledge' in template.name.lower():
                score += 1
            if 'code' in task_lower and 'code' in template.name.lower():
                score += 1
            if 'research' in task_lower and 'research' in template.name.lower():
                score += 1
            if 'analysis' in task_lower and 'analysis' in template.name.lower():
                score += 1

            if score > 0:
                recommendations.append({
                    'template_id': template.template_id,
                    'name': template.name,
                    'description': template.description,
                    'score': score
                })

        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations[:5]  # Top 5 recommendations

class TestLLMIntegrationFramework:
    """Tests for LLM integration framework"""

    @pytest.fixture
    def model_registry(self):
        """Create model registry for testing"""
        config = {'test_mode': True}
        return ModelRegistry(config)

    @pytest.fixture
    def llm_client(self, model_registry):
        """Create LLM client for testing"""
        config = {'test_mode': True}
        return UnifiedLLMClient(model_registry, config)

    def test_model_selection(self, model_registry):
        """Test model selection for tasks"""
        task_reqs = {'required_capabilities': ['text_generation']}
        selected_model = model_registry.select_model_for_task(task_reqs)

        assert selected_model is not None
        assert selected_model in model_registry.models

    @pytest.mark.asyncio
    async def test_text_generation(self, llm_client):
        """Test text generation (mock)"""
        # This would need mocking of the actual API calls
        # For now, just test the interface
        prompt = "Test prompt"
        # response = await llm_client.generate_text(prompt)

        # assert 'success' in response
        # In real test, would assert response structure
        pass

    def test_conversation_management(self):
        """Test conversation management"""
        storage_config = {'storage_path': './test_conversations', 'auto_save': False}
        manager = ConversationManager(storage_config)

        # Create conversation
        conv_id = manager.create_conversation("Test Conversation")
        assert conv_id in manager.conversations

        # Add message
        success = manager.add_message(conv_id, 'user', 'Hello')
        assert success

        # Retrieve conversation
        conversation = manager.get_conversation(conv_id)
        assert conversation is not None
        assert len(conversation.messages) == 1
        assert conversation.messages[0].content == 'Hello'

    def test_prompt_template_rendering(self):
        """Test prompt template rendering"""
        framework = PromptEngineeringFramework({})

        template = framework.create_template(
            'test_template',
            'Test Template',
            'A test template',
            'Hello {name}, you are {age} years old.',
            ['name', 'age']
        )

        # Test rendering
        rendered = template.render({'name': 'Alice', 'age': '30'})
        assert rendered == 'Hello Alice, you are 30 years old.'

        # Test validation
        missing = template.validate_variables({'name': 'Alice'})
        assert 'age' in missing

    def test_template_recommendations(self):
        """Test template recommendations"""
        framework = PromptEngineeringFramework({})

        recommendations = framework.get_template_recommendations("analyze knowledge base")

        assert isinstance(recommendations, list)
        # Should find knowledge-related templates
        template_ids = [r['template_id'] for r in recommendations]
        assert 'knowledge_query' in template_ids
```

---

**"Active Inference for, with, by Generative AI"** - Building comprehensive LLM integration that enables intelligent, context-aware interactions with the Active Inference Knowledge Environment through advanced model management, conversation handling, and prompt engineering.
