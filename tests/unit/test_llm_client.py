"""
Tests for LLM Client Module

Unit tests for the Ollama LLM client functionality, ensuring proper
operation of model management, conversation handling, and error recovery.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from active_inference.llm.client import OllamaClient, LLMConfig


class TestLLMConfig:
    """Test cases for LLMConfig class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = LLMConfig()

        assert config.base_url == "http://localhost:11434"
        assert config.default_model == "gemma3:2b"
        assert config.timeout == 300.0
        assert config.max_retries == 3
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    def test_custom_config(self):
        """Test custom configuration"""
        config = LLMConfig(
            base_url="http://custom:8080",
            default_model="llama2:7b",
            temperature=0.5,
            max_tokens=4096
        )

        assert config.base_url == "http://custom:8080"
        assert config.default_model == "llama2:7b"
        assert config.temperature == 0.5
        assert config.max_tokens == 4096

    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = LLMConfig(
            base_url="http://localhost:11434",
            default_model="gemma3:2b",
            debug=True
        )

        config_dict = config.to_dict()

        assert config_dict["base_url"] == "http://localhost:11434"
        assert config_dict["default_model"] == "gemma3:2b"
        assert config_dict["debug"] is True
        assert "timeout" in config_dict
        assert "temperature" in config_dict


class TestOllamaClient:
    """Test cases for OllamaClient class"""

    @pytest.fixture
    def client_config(self):
        """Create test client configuration"""
        return LLMConfig(
            base_url="http://localhost:11434",
            default_model="gemma3:2b",
            timeout=30.0,
            max_retries=1,
            debug=True
        )

    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama API response"""
        return {
            "response": "This is a test response from the model.",
            "done": True,
            "context": [1234, 5678],
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 500000000,
            "eval_count": 15,
            "eval_duration": 400000000
        }

    def test_client_initialization(self, client_config):
        """Test client initialization"""
        client = OllamaClient(client_config)

        assert client.config == client_config
        assert client.client is None
        assert not client._is_initialized
        assert client._available_models == []
        assert client._model_info == {}

    @pytest.mark.asyncio
    async def test_initialize_success(self, client_config):
        """Test successful client initialization"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock successful API calls
            mock_client.get.return_value = Mock()
            mock_client.get.return_value.raise_for_status = Mock()
            mock_client.get.return_value.json.return_value = {
                "models": [
                    {"name": "gemma3:2b", "size": 2000000000, "modified_at": "2024-01-01T00:00:00Z"},
                    {"name": "llama2:7b", "size": 7000000000, "modified_at": "2024-01-01T00:00:00Z"}
                ]
            }

            client = OllamaClient(client_config)
            success = await client.initialize()

            assert success is True
            assert client._is_initialized is True
            assert "gemma3:2b" in client._available_models
            assert "llama2:7b" in client._available_models

    @pytest.mark.asyncio
    async def test_initialize_failure(self, client_config):
        """Test client initialization failure"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock API failure
            mock_client.get.side_effect = Exception("Connection refused")

            client = OllamaClient(client_config)
            success = await client.initialize()

            assert success is False
            assert not client._is_initialized

    @pytest.mark.asyncio
    async def test_pull_model_success(self, client_config):
        """Test successful model pulling"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock pull response
            mock_pull_response = Mock()
            mock_pull_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_pull_response

            # Mock refresh models
            mock_refresh_response = Mock()
            mock_refresh_response.raise_for_status = Mock()
            mock_refresh_response.json.return_value = {
                "models": [
                    {"name": "gemma3:2b", "size": 2000000000}
                ]
            }
            mock_client.get.return_value = mock_refresh_response

            client = OllamaClient(client_config)
            client.client = mock_client
            client._is_initialized = True

            success = await client.pull_model("gemma3:2b")

            assert success is True
            mock_client.post.assert_called_once_with(
                "/api/pull",
                json={"name": "gemma3:2b"},
                timeout=client_config.timeout
            )

    @pytest.mark.asyncio
    async def test_generate_text(self, client_config, mock_ollama_response):
        """Test text generation"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock successful response
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_ollama_response
            mock_client.post.return_value = mock_response

            client = OllamaClient(client_config)
            client.client = mock_client
            client._is_initialized = True
            client._available_models = ["gemma3:2b"]

            result = await client.generate("Test prompt")

            assert result == "This is a test response from the model."
            mock_client.post.assert_called_once()

            # Check that the call included the right parameters
            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["model"] == "gemma3:2b"
            assert call_args[1]["json"]["prompt"] == "Test prompt"

    @pytest.mark.asyncio
    async def test_generate_with_custom_model(self, client_config, mock_ollama_response):
        """Test text generation with custom model"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_ollama_response
            mock_client.post.return_value = mock_response

            client = OllamaClient(client_config)
            client.client = mock_client
            client._is_initialized = True
            client._available_models = ["llama2:7b"]

            result = await client.generate("Test prompt", model="llama2:7b")

            assert result == "This is a test response from the model."

            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["model"] == "llama2:7b"

    @pytest.mark.skip(reason="Async mock issues with httpx - needs refactoring")
    @pytest.mark.asyncio
    async def test_generate_unavailable_model(self, client_config):
        """Test text generation with unavailable model"""
        # TODO: Fix async mocking for this test
        pass

    @pytest.mark.asyncio
    async def test_chat_functionality(self, client_config, mock_ollama_response):
        """Test chat functionality"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock chat response
            chat_response = {
                "message": {
                    "role": "assistant",
                    "content": "Chat response content"
                },
                "done": True
            }
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = chat_response
            mock_client.post.return_value = mock_response

            client = OllamaClient(client_config)
            client.client = mock_client
            client._is_initialized = True
            client._available_models = ["gemma3:2b"]

            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]

            result = await client.chat(messages)

            assert result == "Chat response content"

            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["model"] == "gemma3:2b"
            assert call_args[1]["json"]["messages"] == messages

    @pytest.mark.asyncio
    async def test_health_check(self, client_config):
        """Test health check functionality"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock version response (first call)
            mock_version_response = Mock()
            mock_version_response.raise_for_status = Mock()
            mock_version_response.json.return_value = {"version": "0.1.0"}

            # Mock models response (second call)
            mock_models_response = Mock()
            mock_models_response.raise_for_status = Mock()
            mock_models_response.json.return_value = {
                "models": [{"name": "gemma3:2b", "size": 2000000000}]
            }

            # Set up mock to return different responses for different endpoints
            mock_client.get.side_effect = [mock_version_response, mock_models_response]

            client = OllamaClient(client_config)
            client.client = mock_client
            client._is_initialized = True

            health = await client.health_check()

            assert health["status"] == "healthy"
            assert health["version"] == "0.1.0"
            assert health["initialized"] is True
            assert "gemma3:2b" in health["available_models"]

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client_config):
        """Test health check failure"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock API failure
            mock_client.get.side_effect = Exception("Connection error")

            client = OllamaClient(client_config)
            client.client = mock_client

            health = await client.health_check()

            assert health["status"] == "unhealthy"
            assert "Connection error" in health["error"]
            assert health["initialized"] is False

    def test_context_manager(self, client_config):
        """Test context manager functionality"""
        client = OllamaClient(client_config)

        # Test sync context manager
        assert hasattr(client, '__enter__')
        assert hasattr(client, '__aexit__')

    @pytest.mark.asyncio
    async def test_close_client(self, client_config):
        """Test client closing"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            client = OllamaClient(client_config)
            client.client = mock_client
            client._is_initialized = True

            await client.close()

            mock_client.aclose.assert_called_once()
            assert client.client is None
            assert not client._is_initialized

    @pytest.mark.asyncio
    async def test_list_models(self, client_config):
        """Test listing available models"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "models": [
                    {"name": "gemma3:2b", "size": 2000000000},
                    {"name": "llama2:7b", "size": 7000000000}
                ]
            }
            mock_client.get.return_value = mock_response

            client = OllamaClient(client_config)
            client.client = mock_client

            models = await client.list_models()

            assert "gemma3:2b" in models
            assert "llama2:7b" in models
            assert len(models) == 2

    @pytest.mark.asyncio
    async def test_get_model_info(self, client_config):
        """Test getting model information"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            model_info = {"name": "gemma3:2b", "size": 2000000000}
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {"models": [model_info]}
            mock_client.get.return_value = mock_response

            client = OllamaClient(client_config)
            client.client = mock_client

            info = await client.get_model_info("gemma3:2b")

            assert info == model_info

    @pytest.mark.asyncio
    async def test_delete_model(self, client_config):
        """Test model deletion"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_client.delete.return_value = mock_response

            # Mock refresh after deletion
            mock_get_response = Mock()
            mock_get_response.raise_for_status = Mock()
            mock_get_response.json.return_value = {"models": []}
            mock_client.get.return_value = mock_get_response

            client = OllamaClient(client_config)
            client.client = mock_client
            client._available_models = ["gemma3:2b"]

            success = await client.delete_model("gemma3:2b")

            assert success is True
            mock_client.delete.assert_called_once_with("/api/delete", json={"name": "gemma3:2b"})


if __name__ == "__main__":
    pytest.main([__file__])
