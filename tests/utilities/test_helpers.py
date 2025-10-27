"""
Test Helper Utilities

Comprehensive helper utilities for testing the Active Inference Knowledge Environment.
Provides mock objects, test data generators, and testing infrastructure helpers.
"""

import json
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Any, Generator
from unittest.mock import Mock, AsyncMock, MagicMock
import pytest


class MockResponse:
    """Mock HTTP response for testing"""

    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self.json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def create_mock_ollama_response(content: str = "Test response") -> Dict[str, Any]:
    """Create a mock Ollama API response"""
    return {
        "response": content,
        "done": True,
        "context": [1234, 5678, 9012],
        "total_duration": 1500000000,
        "load_duration": 50000000,
        "prompt_eval_count": 15,
        "prompt_eval_duration": 300000000,
        "eval_count": 25,
        "eval_duration": 1150000000
    }


def create_mock_chat_response(content: str = "Chat response") -> Dict[str, Any]:
    """Create a mock Ollama chat API response"""
    return {
        "message": {
            "role": "assistant",
            "content": content
        },
        "done": True
    }


def create_mock_models_response(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a mock models API response"""
    return {"models": models}


def generate_unique_test_id(prefix: str = "test") -> str:
    """Generate a unique test identifier"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def create_temp_file_with_content(content: str, suffix: str = ".json") -> Generator[Path, None, None]:
    """Create a temporary file with specified content"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        yield temp_path
    finally:
        if temp_path.exists():
            temp_path.unlink()


def create_test_knowledge_node(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a test knowledge node with optional overrides"""
    base_node = {
        "id": generate_unique_test_id("node"),
        "title": "Test Knowledge Node",
        "content_type": "foundation",
        "difficulty": "beginner",
        "description": "Test description",
        "prerequisites": [],
        "tags": ["test"],
        "learning_objectives": ["Learn something"],
        "content": {
            "overview": "Test overview",
            "examples": []
        },
        "metadata": {
            "author": "Test Author",
            "version": "1.0"
        }
    }

    if overrides:
        base_node.update(overrides)

    return base_node


def create_test_learning_path(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a test learning path with optional overrides"""
    base_path = {
        "id": generate_unique_test_id("path"),
        "name": "Test Learning Path",
        "description": "Test learning path description",
        "nodes": [generate_unique_test_id("node1"), generate_unique_test_id("node2")],
        "estimated_hours": 4,
        "difficulty": "beginner"
    }

    if overrides:
        base_path.update(overrides)

    return base_path


def assert_knowledge_node_structure(node_data: Dict[str, Any]) -> None:
    """Assert that knowledge node data has correct structure"""
    required_fields = [
        "id", "title", "content_type", "difficulty", "description",
        "prerequisites", "tags", "learning_objectives", "content", "metadata"
    ]

    for field in required_fields:
        assert field in node_data, f"Missing required field: {field}"

    # Check field types
    assert isinstance(node_data["id"], str)
    assert isinstance(node_data["title"], str)
    assert isinstance(node_data["prerequisites"], list)
    assert isinstance(node_data["tags"], list)
    assert isinstance(node_data["learning_objectives"], list)
    assert isinstance(node_data["content"], dict)
    assert isinstance(node_data["metadata"], dict)


def assert_learning_path_structure(path_data: Dict[str, Any]) -> None:
    """Assert that learning path data has correct structure"""
    required_fields = ["id", "name", "description", "nodes", "estimated_hours", "difficulty"]

    for field in required_fields:
        assert field in path_data, f"Missing required field: {field}"

    # Check field types
    assert isinstance(path_data["id"], str)
    assert isinstance(path_data["name"], str)
    assert isinstance(path_data["nodes"], list)
    assert isinstance(path_data["estimated_hours"], int)
    assert isinstance(path_data["difficulty"], str)


def create_async_mock_client() -> AsyncMock:
    """Create a mock HTTP client for async testing"""
    mock_client = AsyncMock()

    # Default successful responses
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = create_mock_ollama_response()

    mock_client.get.return_value = mock_response
    mock_client.post.return_value = mock_response

    return mock_client


def create_failing_mock_client(error_message: str = "Connection refused") -> AsyncMock:
    """Create a mock HTTP client that always fails"""
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception(error_message)
    mock_client.post.side_effect = Exception(error_message)
    return mock_client


def patch_llm_client_success() -> Generator[Mock, None, None]:
    """Context manager to patch LLM client with successful responses"""
    with pytest.mock.patch('httpx.AsyncClient') as mock_client_class:
        mock_client = create_async_mock_client()
        mock_client_class.return_value = mock_client
        yield mock_client


def patch_llm_client_failure(error_message: str = "Connection refused") -> Generator[Mock, None, None]:
    """Context manager to patch LLM client with failing responses"""
    with pytest.mock.patch('httpx.AsyncClient') as mock_client_class:
        mock_client = create_failing_mock_client(error_message)
        mock_client_class.return_value = mock_client
        yield mock_client


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate data against JSON schema and return validation errors"""
    errors = []

    # Check required fields
    required_fields = schema.get('required', [])
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Check field types
    properties = schema.get('properties', {})
    for field, field_schema in properties.items():
        if field in data:
            expected_type = field_schema.get('type')
            if expected_type == 'string' and not isinstance(data[field], str):
                errors.append(f"Field {field} must be string")
            elif expected_type == 'array' and not isinstance(data[field], list):
                errors.append(f"Field {field} must be array")
            elif expected_type == 'object' and not isinstance(data[field], dict):
                errors.append(f"Field {field} must be object")

    return errors
