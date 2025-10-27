"""
LLM Module

Local Large Language Model integration for the Active Inference Knowledge Environment.
Provides Ollama integration with flexible prompt composition, model management, and
local LLM serving capabilities for enhanced AI capabilities and knowledge processing.

This module enables:
- Local LLM integration with Ollama
- Flexible prompt composition and templating
- Model management and fallback strategies
- Integration with knowledge and research systems
"""

from .client import OllamaClient, LLMConfig
from .prompts import PromptTemplate, PromptBuilder, PromptManager, ActiveInferencePromptBuilder
from .models import ModelManager, ModelInfo, ModelRegistry
from .conversations import ConversationManager, Conversation, ActiveInferenceTemplates

__all__ = [
    "OllamaClient",
    "LLMConfig",
    "PromptTemplate",
    "PromptBuilder",
    "PromptManager",
    "ActiveInferencePromptBuilder",
    "ModelManager",
    "ModelInfo",
    "ModelRegistry",
    "ConversationManager",
    "Conversation",
    "ActiveInferenceTemplates",
]
