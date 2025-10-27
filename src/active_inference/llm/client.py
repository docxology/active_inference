"""
Ollama LLM Client

Core client for interacting with Ollama local LLM service. Provides model management,
conversation handling, and integration with the Active Inference platform.

This module implements the primary interface for local LLM capabilities including:
- Model pulling and management
- Conversation and chat functionality
- Integration with platform services
- Error handling and fallback strategies
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, AsyncGenerator
from urllib.parse import urljoin

import httpx


@dataclass
class LLMConfig:
    """Configuration for LLM client and services"""

    base_url: str = "http://localhost:11434"
    default_model: str = "gemma3:2b"
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    chunk_size: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_tokens: int = 2048
    context_window: int = 4096
    system_prompt: str = "You are a helpful AI assistant specialized in Active Inference and the Free Energy Principle. Provide accurate, helpful responses based on scientific knowledge."
    enable_logging: bool = True
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'base_url': self.base_url,
            'default_model': self.default_model,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'chunk_size': self.chunk_size,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'repeat_penalty': self.repeat_penalty,
            'presence_penalty': self.presence_penalty,
            'frequency_penalty': self.frequency_penalty,
            'max_tokens': self.max_tokens,
            'context_window': self.context_window,
            'system_prompt': self.system_prompt,
            'enable_logging': self.enable_logging,
            'debug': self.debug,
        }


class OllamaClient:
    """Client for interacting with Ollama local LLM service"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.client: Optional[httpx.AsyncClient] = None
        self._available_models: List[str] = []
        self._model_info: Dict[str, Dict[str, Any]] = {}
        self._is_initialized = False

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the LLM client"""
        logger = logging.getLogger(f"active_inference.llm.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        level = logging.DEBUG if self.config.debug else logging.INFO
        logger.setLevel(level)
        return logger

    async def initialize(self) -> bool:
        """Initialize the Ollama client and check service availability"""
        try:
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )

            # Test connection and get available models
            refresh_success = await self._refresh_models()

            # Check if we successfully connected
            if not self.client or not refresh_success:
                self.logger.error("Failed to connect to Ollama service")
                return False

            # If we have no models and can't pull the default, consider it a failure
            if not self._available_models:
                self.logger.warning("No models available in Ollama. Run: ollama pull gemma3:2b")
                # Try to pull default model
                pull_success = await self.pull_model(self.config.default_model)
                if not pull_success:
                    self.logger.error("Failed to pull default model and no models available")
                    return False

            self._is_initialized = True
            self.logger.info(f"Ollama client initialized successfully. Available models: {self._available_models}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama client: {e}")
            self._is_initialized = False
            if self.client:
                await self.client.aclose()
                self.client = None
            return False

    async def _refresh_models(self) -> bool:
        """Refresh the list of available models"""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            self._available_models = [model['name'] for model in data.get('models', [])]
            self._model_info = {model['name']: model for model in data.get('models', [])}
            return True

        except Exception as e:
            self.logger.error(f"Failed to refresh models: {e}")
            self._available_models = []
            self._model_info = {}
            return False

    async def pull_model(self, model_name: str, progress_callback=None) -> bool:
        """Pull a model from Ollama registry"""
        try:
            self.logger.info(f"Pulling model: {model_name}")

            # Use the existing client if available, otherwise create a new one
            if self.client:
                client = self.client
                is_our_client = False
            else:
                client = httpx.AsyncClient(
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
                is_our_client = True

            try:
                # Make the pull request
                response = await client.post(
                    "/api/pull",
                    json={"name": model_name},
                    timeout=self.config.timeout
                )
                response.raise_for_status()

                # For simplicity, just check if the request succeeded
                # In a real implementation, this would handle streaming responses
                self.logger.info(f"Successfully pulled model: {model_name}")

                # Refresh models list
                await self._refresh_models()
                return model_name in self._available_models

            finally:
                if is_our_client:
                    await client.aclose()

        except Exception as e:
            self.logger.error(f"Failed to pull model {model_name}: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using the specified model"""
        if not self._is_initialized:
            raise RuntimeError("Ollama client not initialized. Call initialize() first.")

        model = model or self.config.default_model
        if model not in self._available_models:
            # Try to pull the model
            success = await self.pull_model(model)
            if not success:
                raise ValueError(f"Model {model} not available and could not be pulled")

        system_prompt = system_prompt or self.config.system_prompt

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "top_k": kwargs.get('top_k', self.config.top_k),
                "repeat_penalty": kwargs.get('repeat_penalty', self.config.repeat_penalty),
                "presence_penalty": kwargs.get('presence_penalty', self.config.presence_penalty),
                "frequency_penalty": kwargs.get('frequency_penalty', self.config.frequency_penalty),
                "num_predict": kwargs.get('max_tokens', self.config.max_tokens),
            }
        }

        try:
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()

            data = response.json()
            return data.get('response', '')

        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Chat with the model using message history"""
        if not self._is_initialized:
            raise RuntimeError("Ollama client not initialized. Call initialize() first.")

        model = model or self.config.default_model
        if model not in self._available_models:
            success = await self.pull_model(model)
            if not success:
                raise ValueError(f"Model {model} not available and could not be pulled")

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "top_k": kwargs.get('top_k', self.config.top_k),
                "repeat_penalty": kwargs.get('repeat_penalty', self.config.repeat_penalty),
                "presence_penalty": kwargs.get('presence_penalty', self.config.presence_penalty),
                "frequency_penalty": kwargs.get('frequency_penalty', self.config.frequency_penalty),
                "num_predict": kwargs.get('max_tokens', self.config.max_tokens),
            }
        }

        try:
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()

            data = response.json()
            return data.get('message', {}).get('content', '')

        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming response"""
        if not self._is_initialized:
            raise RuntimeError("Ollama client not initialized. Call initialize() first.")

        model = model or self.config.default_model
        system_prompt = system_prompt or self.config.system_prompt

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "top_k": kwargs.get('top_k', self.config.top_k),
                "repeat_penalty": kwargs.get('repeat_penalty', self.config.repeat_penalty),
                "presence_penalty": kwargs.get('presence_penalty', self.config.presence_penalty),
                "frequency_penalty": kwargs.get('frequency_penalty', self.config.frequency_penalty),
                "num_predict": kwargs.get('max_tokens', self.config.max_tokens),
            }
        }

        try:
            async with self.client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'response' in data:
                                yield data['response']
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            self.logger.error(f"Error in streaming generation: {e}")
            raise

    async def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        await self._refresh_models()
        return self._model_info.get(model)

    async def list_models(self) -> List[str]:
        """List all available models"""
        await self._refresh_models()
        return self._available_models.copy()

    async def delete_model(self, model: str) -> bool:
        """Delete a model from local storage"""
        try:
            response = await self.client.delete("/api/delete", json={"name": model})
            response.raise_for_status()

            await self._refresh_models()
            return model not in self._available_models

        except Exception as e:
            self.logger.error(f"Error deleting model {model}: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama service health"""
        try:
            response = await self.client.get("/api/version")
            response.raise_for_status()

            return {
                "status": "healthy",
                "version": response.json().get('version', 'unknown'),
                "available_models": await self.list_models(),
                "default_model": self.config.default_model,
                "initialized": self._is_initialized
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "available_models": [],
                "initialized": False
            }

    async def close(self) -> None:
        """Close the client connection"""
        if self.client:
            await self.client.aclose()
            self.client = None
        self._is_initialized = False

    def __enter__(self):
        """Context manager entry"""
        return self

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
