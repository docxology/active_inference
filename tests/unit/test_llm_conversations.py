"""
Tests for LLM Conversation Management

Unit tests for the conversation and context management system, ensuring proper
operation of message handling, context optimization, and persistence.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from active_inference.llm.conversations import (
    Message,
    Conversation,
    ConversationManager,
    ConversationTemplate,
    ActiveInferenceTemplates
)


class TestMessage:
    """Test cases for Message class"""

    def test_message_creation(self):
        """Test creating a message"""
        message = Message(
            role="user",
            content="Hello, world!",
            metadata={"test": "value"}
        )

        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.metadata == {"test": "value"}
        assert isinstance(message.timestamp, datetime)

    def test_message_to_dict(self):
        """Test message serialization"""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        message = Message(
            role="assistant",
            content="Test response",
            timestamp=timestamp,
            metadata={"model": "gemma3:2b"}
        )

        data = message.to_dict()

        assert data["role"] == "assistant"
        assert data["content"] == "Test response"
        assert data["timestamp"] == "2024-01-01T12:00:00"
        assert data["metadata"]["model"] == "gemma3:2b"

    def test_message_from_dict(self):
        """Test message deserialization"""
        data = {
            "role": "system",
            "content": "System prompt",
            "timestamp": "2024-01-01T12:00:00",
            "metadata": {"type": "system"}
        }

        message = Message.from_dict(data)

        assert message.role == "system"
        assert message.content == "System prompt"
        assert message.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert message.metadata["type"] == "system"


class TestConversation:
    """Test cases for Conversation class"""

    def test_conversation_creation(self):
        """Test creating a conversation"""
        conversation = Conversation(
            id="test_id",
            title="Test Conversation",
            metadata={"type": "test"}
        )

        assert conversation.id == "test_id"
        assert conversation.title == "Test Conversation"
        assert len(conversation.messages) == 0
        assert conversation.metadata == {"type": "test"}
        assert isinstance(conversation.created_at, datetime)
        assert isinstance(conversation.updated_at, datetime)

    def test_add_message(self):
        """Test adding messages to conversation"""
        conversation = Conversation(id="test_id")

        # Add first message (should generate title)
        conversation.add_message("user", "Hello, how are you?")
        assert len(conversation.messages) == 1
        assert conversation.messages[0].role == "user"
        assert conversation.messages[0].content == "Hello, how are you?"
        assert conversation.title == "Hello, how are you?"

        # Add second message
        conversation.add_message("assistant", "I'm doing well, thank you!")
        assert len(conversation.messages) == 2
        assert conversation.messages[1].role == "assistant"

        # Check updated timestamp
        assert conversation.updated_at >= conversation.created_at

    def test_get_messages(self):
        """Test getting messages from conversation"""
        conversation = Conversation(id="test_id")

        # Add messages
        conversation.add_message("system", "System prompt")
        conversation.add_message("user", "User message")
        conversation.add_message("assistant", "Assistant response")

        # Get all messages
        all_messages = conversation.get_messages()
        assert len(all_messages) == 3

        # Get limited messages
        recent_messages = conversation.get_messages(limit=2)
        assert len(recent_messages) == 2
        assert recent_messages[0].role == "user"
        assert recent_messages[1].role == "assistant"

    def test_get_context_messages(self):
        """Test getting context messages within token limit"""
        conversation = Conversation(id="test_id")

        # Add messages with varying lengths
        conversation.add_message("system", "You are a helpful assistant.")
        conversation.add_message("user", "Short question")
        conversation.add_message("assistant", "This is a very long response that should exceed token limits when combined with other messages to test the context window optimization functionality.")

        # Get context with reasonable token limit
        context = conversation.get_context_messages(max_tokens=50)

        # Should include system message and recent messages within limit
        assert len(context) <= 3  # Should fit within reasonable limits
        assert any(msg.role == "system" for msg in context)

    def test_clear_messages(self):
        """Test clearing conversation messages"""
        conversation = Conversation(id="test_id")

        conversation.add_message("user", "Test message")
        assert len(conversation.messages) == 1

        conversation.clear_messages()
        assert len(conversation.messages) == 0
        assert conversation.updated_at >= conversation.created_at

    def test_conversation_to_dict(self):
        """Test conversation serialization"""
        conversation = Conversation(id="test_id", title="Test")
        conversation.add_message("user", "Test message")

        data = conversation.to_dict()

        assert data["id"] == "test_id"
        assert data["title"] == "Test"
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
        assert "created_at" in data
        assert "updated_at" in data

    def test_conversation_from_dict(self):
        """Test conversation deserialization"""
        data = {
            "id": "test_id",
            "title": "Test Conversation",
            "messages": [
                {
                    "role": "user",
                    "content": "Test message",
                    "timestamp": "2024-01-01T12:00:00",
                    "metadata": {}
                }
            ],
            "metadata": {"type": "test"},
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T12:01:00"
        }

        conversation = Conversation.from_dict(data)

        assert conversation.id == "test_id"
        assert conversation.title == "Test Conversation"
        assert len(conversation.messages) == 1
        assert conversation.messages[0].role == "user"
        assert conversation.metadata == {"type": "test"}


class TestConversationManager:
    """Test cases for ConversationManager class"""

    @pytest.fixture
    def temp_conversations_dir(self):
        """Create temporary conversations directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_manager_creation(self, temp_conversations_dir):
        """Test creating conversation manager"""
        manager = ConversationManager(temp_conversations_dir)

        assert manager.storage_path == temp_conversations_dir
        assert len(manager.conversations) == 0

    def test_create_conversation(self, temp_conversations_dir):
        """Test creating conversations"""
        manager = ConversationManager(temp_conversations_dir)

        conversation = manager.create_conversation(
            title="Test Conversation",
            metadata={"type": "test"}
        )

        assert conversation.id in manager.conversations
        assert conversation.title == "Test Conversation"
        assert conversation.metadata == {"type": "test"}

        # Check that conversation was saved to file
        conversation_file = temp_conversations_dir / f"{conversation.id}.json"
        assert conversation_file.exists()

    def test_get_conversation(self, temp_conversations_dir):
        """Test getting conversations"""
        manager = ConversationManager(temp_conversations_dir)

        # Create conversation
        conversation = manager.create_conversation("Test")
        conversation_id = conversation.id

        # Get conversation
        retrieved = manager.get_conversation(conversation_id)
        assert retrieved is not None
        assert retrieved.id == conversation_id
        assert retrieved.title == "Test"

        # Get non-existent conversation
        nonexistent = manager.get_conversation("nonexistent")
        assert nonexistent is None

    def test_list_conversations(self, temp_conversations_dir):
        """Test listing conversations"""
        manager = ConversationManager(temp_conversations_dir)

        # Create multiple conversations
        conv1 = manager.create_conversation("First")
        conv2 = manager.create_conversation("Second")

        conversations = manager.list_conversations()

        assert len(conversations) == 2
        conversation_ids = [c.id for c in conversations]
        assert conv1.id in conversation_ids
        assert conv2.id in conversation_ids

    def test_add_message_to_conversation(self, temp_conversations_dir):
        """Test adding messages to conversations"""
        manager = ConversationManager(temp_conversations_dir)

        conversation = manager.create_conversation("Test")
        conversation_id = conversation.id

        success = manager.add_message(
            conversation_id,
            "user",
            "Hello!",
            {"test": "metadata"}
        )

        assert success is True

        # Check message was added
        updated_conversation = manager.get_conversation(conversation_id)
        assert len(updated_conversation.messages) == 1
        assert updated_conversation.messages[0].role == "user"
        assert updated_conversation.messages[0].content == "Hello!"
        assert updated_conversation.messages[0].metadata == {"test": "metadata"}

    def test_search_conversations(self, temp_conversations_dir):
        """Test searching conversations"""
        manager = ConversationManager(temp_conversations_dir)

        # Create conversations with searchable content
        conv1 = manager.create_conversation("Machine Learning Discussion")
        manager.add_message(conv1.id, "user", "Explain neural networks")
        manager.add_message(conv1.id, "assistant", "Neural networks are...")

        conv2 = manager.create_conversation("Physics Discussion")
        manager.add_message(conv2.id, "user", "What is quantum mechanics?")
        manager.add_message(conv2.id, "assistant", "Quantum mechanics explains...")

        # Search for "neural"
        results = manager.search_conversations("neural")
        assert len(results) == 1
        assert results[0].title == "Machine Learning Discussion"

        # Search for "quantum"
        results = manager.search_conversations("quantum")
        assert len(results) == 1
        assert results[0].title == "Physics Discussion"

        # Search with limit
        results = manager.search_conversations("Discussion", limit=1)
        assert len(results) == 1

    def test_get_conversation_context(self, temp_conversations_dir):
        """Test getting conversation context for LLM"""
        manager = ConversationManager(temp_conversations_dir)

        conversation = manager.create_conversation("Test")
        conversation_id = conversation.id

        # Add messages
        manager.add_message(conversation_id, "system", "You are a helpful assistant.")
        manager.add_message(conversation_id, "user", "Hello")
        manager.add_message(conversation_id, "assistant", "Hi there!")

        context = manager.get_conversation_context(conversation_id, max_tokens=100)

        assert len(context) == 3
        assert context[0]["role"] == "system"
        assert context[1]["role"] == "user"
        assert context[2]["role"] == "assistant"

        # Check content format
        assert "content" in context[0]
        assert "role" in context[0]

    def test_export_conversation(self, temp_conversations_dir):
        """Test exporting conversations"""
        manager = ConversationManager(temp_conversations_dir)

        conversation = manager.create_conversation("Export Test")
        manager.add_message(conversation.id, "user", "Test message")
        manager.add_message(conversation.id, "assistant", "Test response")

        # Export as JSON
        json_export = manager.export_conversation(conversation.id, "json")
        assert json_export is not None

        data = json.loads(json_export)
        assert data["title"] == "Export Test"
        assert len(data["messages"]) == 2

        # Export as Markdown
        markdown_export = manager.export_conversation(conversation.id, "markdown")
        assert markdown_export is not None
        assert "# Export Test" in markdown_export
        assert "Test message" in markdown_export
        assert "Test response" in markdown_export

        # Test unsupported format
        with pytest.raises(ValueError, match="Unsupported export format"):
            manager.export_conversation(conversation.id, "unsupported")

    def test_cleanup_old_conversations(self, temp_conversations_dir):
        """Test cleaning up old conversations"""
        manager = ConversationManager(temp_conversations_dir)

        # Create conversations with different timestamps
        old_conversation = manager.create_conversation("Old")
        old_conversation.created_at = datetime.now() - timedelta(days=40)
        old_conversation.updated_at = datetime.now() - timedelta(days=40)

        recent_conversation = manager.create_conversation("Recent")
        recent_conversation.created_at = datetime.now() - timedelta(days=5)
        recent_conversation.updated_at = datetime.now() - timedelta(days=5)

        # Save conversations to trigger file creation
        manager._save_conversation(old_conversation)
        manager._save_conversation(recent_conversation)

        # Cleanup conversations older than 30 days
        deleted_count = manager.cleanup_old_conversations(days=30)

        assert deleted_count == 1
        assert old_conversation.id not in manager.conversations
        assert recent_conversation.id in manager.conversations

    def test_get_statistics(self, temp_conversations_dir):
        """Test getting conversation statistics"""
        manager = ConversationManager(temp_conversations_dir)

        # Create test conversations
        conv1 = manager.create_conversation("Test 1")
        manager.add_message(conv1.id, "user", "Message 1")
        manager.add_message(conv1.id, "assistant", "Response 1")

        conv2 = manager.create_conversation("Test 2")
        manager.add_message(conv2.id, "system", "System prompt")
        manager.add_message(conv2.id, "user", "Message 2")

        stats = manager.get_statistics()

        assert stats["total_conversations"] == 2
        assert stats["total_messages"] == 4
        assert "user" in stats["role_distribution"]
        assert "assistant" in stats["role_distribution"]
        assert "system" in stats["role_distribution"]


class TestConversationTemplate:
    """Test cases for ConversationTemplate class"""

    def test_template_creation(self):
        """Test creating conversation template"""
        template = ConversationTemplate(
            name="test_template",
            system_prompt="Test system prompt",
            initial_messages=[
                {"role": "user", "content": "Initial message"}
            ],
            metadata={"type": "test"}
        )

        assert template.name == "test_template"
        assert template.system_prompt == "Test system prompt"
        assert len(template.initial_messages) == 1
        assert template.metadata == {"type": "test"}

    def test_create_conversation_from_template(self):
        """Test creating conversation from template"""
        template = ConversationTemplate(
            name="test_template",
            system_prompt="System prompt",
            initial_messages=[
                {"role": "user", "content": "Hello"}
            ]
        )

        manager = ConversationManager()
        conversation = template.create_conversation("Test Chat", manager)

        assert conversation.title == "Test Chat"
        assert len(conversation.messages) == 2  # system + initial user message
        assert conversation.messages[0].role == "system"
        assert conversation.messages[1].role == "user"
        assert conversation.messages[1].content == "Hello"


class TestActiveInferenceTemplates:
    """Test cases for Active Inference conversation templates"""

    def test_explanation_template(self):
        """Test explanation template"""
        template = ActiveInferenceTemplates.get_explanation_template()

        assert template.name == "active_inference_explanation"
        assert "expert in Active Inference" in template.system_prompt
        assert template.metadata["type"] == "explanation"

    def test_research_template(self):
        """Test research template"""
        template = ActiveInferenceTemplates.get_research_template()

        assert template.name == "research_discussion"
        assert "research assistant" in template.system_prompt
        assert template.metadata["type"] == "research"

    def test_implementation_template(self):
        """Test implementation template"""
        template = ActiveInferenceTemplates.get_implementation_template()

        assert template.name == "implementation_help"
        assert "software engineer" in template.system_prompt
        assert template.metadata["type"] == "implementation"

    def test_education_template(self):
        """Test education template"""
        template = ActiveInferenceTemplates.get_education_template()

        assert template.name == "education_development"
        assert "educational content developer" in template.system_prompt
        assert template.metadata["type"] == "education"


if __name__ == "__main__":
    pytest.main([__file__])
