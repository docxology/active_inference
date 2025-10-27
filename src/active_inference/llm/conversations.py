"""
LLM Conversation Management

Conversation and context management system for maintaining chat history,
managing context windows, and providing conversational AI capabilities.

This module provides:
- Conversation history management
- Context window optimization
- Message role management
- Conversation persistence
- Integration with prompt system
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging


@dataclass
class Message:
    """A single message in a conversation"""

    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class Conversation:
    """A conversation with message history and context management"""

    id: str
    title: str = ""
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a message to the conversation"""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()

        # Auto-generate title if this is the first user message and no title is set
        if role == "user" and len(self.messages) == 1 and not self.title:
            self.title = self._generate_title(content)

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from the conversation"""
        if limit:
            return self.messages[-limit:]
        return self.messages

    def get_context_messages(self, max_tokens: int = 4000) -> List[Message]:
        """Get messages within token limit for context window"""
        # Simple token estimation (rough approximation)
        context_messages = []
        current_tokens = 0

        # Add messages in chronological order, prioritizing system messages
        for message in self.messages:
            estimated_tokens = len(message.content.split()) * 1.3  # Rough token estimate
            if current_tokens + estimated_tokens > max_tokens:
                break

            context_messages.append(message)
            current_tokens += estimated_tokens

        return context_messages

    def clear_messages(self) -> None:
        """Clear all messages from conversation"""
        self.messages.clear()
        self.updated_at = datetime.now()

    def get_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        return {
            "id": self.id,
            "title": self.title,
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }

    def _generate_title(self, first_message: str) -> str:
        """Generate a title from the first user message"""
        # Take first 50 characters and clean up
        title = first_message.strip()[:50]
        if len(first_message) > 50:
            title += "..."
        return title

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create conversation from dictionary"""
        return cls(
            id=data["id"],
            title=data["title"],
            messages=[Message.from_dict(msg_data) for msg_data in data["messages"]],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )


class ConversationManager:
    """Manager for multiple conversations and persistence"""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(__file__).parent / "conversations"
        self.conversations: Dict[str, Conversation] = {}
        self.logger = logging.getLogger(f"active_inference.llm.{self.__class__.__name__}")

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing conversations
        self._load_conversations()

    def create_conversation(self, title: str = "", metadata: Dict[str, Any] = None) -> Conversation:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            id=conversation_id,
            title=title,
            metadata=metadata or {}
        )

        self.conversations[conversation_id] = conversation
        self._save_conversation(conversation)

        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return self.conversations.get(conversation_id)

    def list_conversations(self) -> List[Conversation]:
        """List all conversations"""
        return list(self.conversations.values())

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            # Remove from storage
            conversation_file = self.storage_path / f"{conversation_id}.json"
            if conversation_file.exists():
                conversation_file.unlink()
            return True
        return False

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add a message to a conversation"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False

        conversation.add_message(role, content, metadata)
        self._save_conversation(conversation)
        return True

    def search_conversations(self, query: str, limit: int = 10) -> List[Conversation]:
        """Search conversations by content"""
        query_lower = query.lower()
        matching_conversations = []

        for conversation in self.conversations.values():
            # Search in title
            if query_lower in conversation.title.lower():
                matching_conversations.append(conversation)
                continue

            # Search in messages
            for message in conversation.messages:
                if query_lower in message.content.lower():
                    matching_conversations.append(conversation)
                    break

        # Sort by most recent first
        matching_conversations.sort(key=lambda c: c.updated_at, reverse=True)
        return matching_conversations[:limit]

    def get_conversation_context(
        self,
        conversation_id: str,
        max_tokens: int = 4000
    ) -> List[Dict[str, str]]:
        """Get conversation context for LLM API"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []

        context_messages = conversation.get_context_messages(max_tokens)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in context_messages
        ]

    def export_conversation(self, conversation_id: str, format: str = "json") -> Optional[str]:
        """Export conversation in specified format"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None

        if format.lower() == "json":
            return json.dumps(conversation.to_dict(), indent=2)
        elif format.lower() == "markdown":
            return self._export_as_markdown(conversation)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_as_markdown(self, conversation: Conversation) -> str:
        """Export conversation as Markdown"""
        lines = [f"# {conversation.title}", ""]

        for message in conversation.messages:
            role_display = message.role.upper()
            timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"## {role_display} - {timestamp}")
            lines.append("")
            lines.append(message.content)
            lines.append("")

        return "\n".join(lines)

    def _save_conversation(self, conversation: Conversation) -> None:
        """Save conversation to file"""
        try:
            file_path = self.storage_path / f"{conversation.id}.json"
            with open(file_path, 'w') as f:
                json.dump(conversation.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save conversation {conversation.id}: {e}")

    def _load_conversations(self) -> None:
        """Load conversations from storage"""
        try:
            for conversation_file in self.storage_path.glob("*.json"):
                try:
                    with open(conversation_file, 'r') as f:
                        data = json.load(f)

                    conversation = Conversation.from_dict(data)
                    self.conversations[conversation.id] = conversation

                except Exception as e:
                    self.logger.error(f"Failed to load conversation {conversation_file}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to load conversations: {e}")

    def cleanup_old_conversations(self, days: int = 30) -> int:
        """Clean up conversations older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0

        for conversation_id, conversation in list(self.conversations.items()):
            if conversation.updated_at < cutoff_date:
                self.delete_conversation(conversation_id)
                deleted_count += 1

        return deleted_count

    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        total_conversations = len(self.conversations)
        total_messages = sum(len(conv.messages) for conv in self.conversations.values())

        # Get message counts by role
        role_counts = {}
        for conversation in self.conversations.values():
            for message in conversation.messages:
                role_counts[message.role] = role_counts.get(message.role, 0) + 1

        # Get recent activity
        recent_conversations = [
            conv for conv in self.conversations.values()
            if (datetime.now() - conv.updated_at).days <= 7
        ]

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "role_distribution": role_counts,
            "recent_conversations": len(recent_conversations),
            "oldest_conversation": min((conv.created_at for conv in self.conversations.values()), default=None),
            "newest_conversation": max((conv.created_at for conv in self.conversations.values()), default=None)
        }


class ConversationTemplate:
    """Template for creating structured conversations"""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        initial_messages: List[Dict[str, str]] = None,
        metadata: Dict[str, Any] = None
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.initial_messages = initial_messages or []
        self.metadata = metadata or {}

    def create_conversation(
        self,
        title: str = "",
        manager: Optional[ConversationManager] = None
    ) -> Conversation:
        """Create a conversation from this template"""
        if not manager:
            manager = ConversationManager()

        conversation = manager.create_conversation(title or self.name, self.metadata)

        # Add system prompt
        if self.system_prompt:
            conversation.add_message("system", self.system_prompt)

        # Add initial messages
        for message in self.initial_messages:
            conversation.add_message(message["role"], message["content"])

        return conversation


# Predefined conversation templates for common use cases
class ActiveInferenceTemplates:
    """Predefined conversation templates for Active Inference"""

    @staticmethod
    def get_explanation_template() -> ConversationTemplate:
        """Template for Active Inference concept explanations"""
        return ConversationTemplate(
            name="active_inference_explanation",
            system_prompt="""You are an expert in Active Inference and the Free Energy Principle.
            Provide clear, accurate explanations of concepts with appropriate mathematical detail.
            Connect explanations to broader Active Inference framework.""",
            metadata={"type": "explanation", "domain": "active_inference"}
        )

    @staticmethod
    def get_research_template() -> ConversationTemplate:
        """Template for research discussions"""
        return ConversationTemplate(
            name="research_discussion",
            system_prompt="""You are a research assistant specializing in Active Inference methodology.
            Help develop research questions, analyze methods, and discuss theoretical implications.
            Provide critical analysis while maintaining scientific rigor.""",
            metadata={"type": "research", "domain": "active_inference"}
        )

    @staticmethod
    def get_implementation_template() -> ConversationTemplate:
        """Template for implementation discussions"""
        return ConversationTemplate(
            name="implementation_help",
            system_prompt="""You are a software engineer specializing in Active Inference implementations.
            Help with code design, algorithm implementation, and best practices.
            Provide working code examples and explain implementation choices.""",
            metadata={"type": "implementation", "domain": "active_inference"}
        )

    @staticmethod
    def get_education_template() -> ConversationTemplate:
        """Template for educational content development"""
        return ConversationTemplate(
            name="education_development",
            system_prompt="""You are an educational content developer specializing in Active Inference.
            Help create educational materials, learning paths, and assessment strategies.
            Focus on pedagogical effectiveness and accessibility.""",
            metadata={"type": "education", "domain": "active_inference"}
        )
