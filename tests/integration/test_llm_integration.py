"""
Integration Tests for LLM Module

Integration tests for the complete LLM system, testing interactions between
components, end-to-end workflows, and real-world usage scenarios.
"""

import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from active_inference.llm import (
    OllamaClient,
    LLMConfig,
    PromptManager,
    ModelManager,
    ConversationManager,
    PromptBuilder,
    ActiveInferencePromptBuilder
)


class TestLLMIntegration:
    """Integration tests for LLM components"""

    @pytest.fixture
    def integration_config(self):
        """Create configuration for integration tests"""
        return LLMConfig(
            base_url="http://localhost:11434",
            default_model="gemma3:2b",
            timeout=30.0,
            debug=True
        )

    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama API responses for integration tests"""
        return {
            "response": "This is a comprehensive explanation of entropy in the context of Active Inference and the Free Energy Principle. Entropy measures the uncertainty or disorder in a probability distribution...",
            "done": True,
            "context": [1234, 5678, 9012],
            "total_duration": 1500000000,
            "load_duration": 50000000,
            "prompt_eval_count": 15,
            "prompt_eval_duration": 300000000,
            "eval_count": 25,
            "eval_duration": 1150000000
        }

    @pytest.mark.asyncio
    async def test_end_to_end_prompt_generation_and_execution(self, integration_config, mock_ollama_response):
        """Test complete workflow from prompt generation to LLM execution"""
        # Setup mock client
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock successful API responses
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_ollama_response
            mock_client.post.return_value = mock_response

            # Mock model availability
            mock_get_response = Mock()
            mock_get_response.raise_for_status = Mock()
            mock_get_response.json.return_value = {
                "models": [{"name": "gemma3:2b", "size": 2000000000}]
            }
            mock_client.get.return_value = mock_get_response

            # Initialize components
            client = OllamaClient(integration_config)
            await client.initialize()

            prompt_manager = PromptManager()
            model_manager = ModelManager()

            # Generate prompt using template
            variables = {
                "concept": "entropy",
                "context": "information theory and Active Inference",
                "audience_level": "intermediate",
                "key_points": "uncertainty, probability distributions, information content",
                "response_type": "comprehensive explanation"
            }

            prompt = prompt_manager.generate_prompt("active_inference_explanation", variables)

            # Find best model for the task
            best_model = model_manager.find_best_model("active inference explanation")
            assert best_model is not None

            # Execute prompt with LLM
            response = await client.generate(prompt, model=best_model.name)

            # Verify response
            assert len(response) > 0
            assert "entropy" in response.lower()
            assert "active inference" in response.lower()

            # Verify API call
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["model"] == best_model.name
            assert call_args[1]["json"]["prompt"] == prompt

    @pytest.mark.asyncio
    async def test_conversation_workflow(self, integration_config):
        """Test complete conversation workflow"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock chat API responses
            chat_response = {
                "message": {
                    "role": "assistant",
                    "content": "I understand you're asking about variational inference in the context of Active Inference. This is a key computational method for..."
                },
                "done": True
            }

            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = chat_response
            mock_client.post.return_value = mock_response

            # Mock model availability
            mock_get_response = Mock()
            mock_get_response.raise_for_status = Mock()
            mock_get_response.json.return_value = {
                "models": [{"name": "gemma3:2b", "size": 2000000000}]
            }
            mock_client.get.return_value = mock_get_response

            # Initialize components
            client = OllamaClient(integration_config)
            await client.initialize()

            conversation_manager = ConversationManager()

            # Create conversation
            conversation = conversation_manager.create_conversation(
                "Active Inference Discussion",
                {"domain": "active_inference"}
            )

            # Add initial messages
            conversation_manager.add_message(
                conversation.id,
                "system",
                "You are an expert in Active Inference and the Free Energy Principle."
            )

            conversation_manager.add_message(
                conversation.id,
                "user",
                "Explain variational inference in Active Inference."
            )

            # Get context for LLM
            context = conversation_manager.get_conversation_context(conversation.id)

            # Execute chat with context
            response = await client.chat(context)

            # Verify response
            assert len(response) > 0
            assert "variational inference" in response.lower()

            # Add assistant response to conversation
            conversation_manager.add_message(conversation.id, "assistant", response)

            # Verify conversation state
            updated_conversation = conversation_manager.get_conversation(conversation.id)
            assert len(updated_conversation.messages) == 3

    def test_prompt_builder_integration(self):
        """Test prompt builder integration with different components"""
        # Create specialized prompt builder
        builder = ActiveInferencePromptBuilder()

        # Build complex prompt
        prompt = (builder
            .add_concept_explanation(
                "variational free energy",
                "Bayesian inference and Active Inference",
                "advanced",
                include_math=True
            )
            .add_learning_objective([
                "Understand mathematical formulation of variational free energy",
                "Connect variational methods to Active Inference",
                "Apply variational inference in practice"
            ])
            .set_variables({
                "mathematical_rigor": "high",
                "include_code_examples": "true",
                "focus_areas": "theory, implementation, applications"
            })
            .build())

        # Verify prompt structure
        assert "variational free energy" in prompt
        assert "Bayesian inference" in prompt
        assert "advanced" in prompt
        assert "Learning Objectives" in prompt
        assert "mathematical formulation" in prompt

        # Test prompt manager integration
        prompt_manager = PromptManager()
        variables = {
            "concept": "variational free energy",
            "framework": "Active Inference",
            "difficulty": "advanced"
        }

        template_prompt = prompt_manager.generate_prompt("code_implementation", variables)

        assert "variational free energy" in template_prompt
        assert "Python" in template_prompt or "implementation" in template_prompt

    def test_model_fallback_integration(self):
        """Test model fallback and selection integration"""
        model_manager = ModelManager()

        # Test finding best model for different tasks
        tasks_and_expected_capabilities = [
            ("explain entropy", {"text_generation", "reasoning"}),
            ("mathematical derivation", {"mathematical_reasoning", "text_generation"}),
            ("code implementation", {"code_generation", "reasoning"}),
            ("research analysis", {"research_analysis", "reasoning"}),
            ("active inference tutorial", {"active_inference_explanation", "educational_content"})
        ]

        for task, expected_capabilities in tasks_and_expected_capabilities:
            best_model = model_manager.find_best_model(task)

            assert best_model is not None, f"No model found for task: {task}"
            assert expected_capabilities.issubset(best_model.capabilities), \
                f"Model {best_model.name} missing required capabilities for task: {task}"

        # Test model capability search
        reasoning_models = model_manager.get_models_by_capability("reasoning")
        assert len(reasoning_models) > 0
        assert all(model.has_capability("reasoning") for model in reasoning_models)

    def test_conversation_persistence_integration(self, temp_conversations_dir):
        """Test conversation persistence and loading"""
        # Create first manager instance
        manager1 = ConversationManager(temp_conversations_dir)

        # Create and populate conversation
        conversation = manager1.create_conversation("Persistent Test")
        manager1.add_message(conversation.id, "user", "First message")
        manager1.add_message(conversation.id, "assistant", "First response")

        conversation_id = conversation.id

        # Create second manager instance (simulates app restart)
        manager2 = ConversationManager(temp_conversations_dir)

        # Verify conversation was loaded
        loaded_conversation = manager2.get_conversation(conversation_id)
        assert loaded_conversation is not None
        assert loaded_conversation.title == "Persistent Test"
        assert len(loaded_conversation.messages) == 2

        # Add more messages with second manager
        manager2.add_message(conversation_id, "user", "Second message")

        # Verify persistence
        final_conversation = manager2.get_conversation(conversation_id)
        assert len(final_conversation.messages) == 3

    def test_template_system_integration(self, temp_templates_dir):
        """Test template system with file persistence"""
        # Create prompt manager with custom templates path
        prompt_manager = PromptManager(temp_templates_dir)

        # Create and save custom template
        custom_template = prompt_manager.create_custom_template(
            name="integration_test",
            description="Template for integration testing",
            template="Test template: {variable1} and {variable2}",
            variables=["variable1", "variable2"],
            examples=[{
                "name": "test_example",
                "variables": {"variable1": "value1", "variable2": "value2"}
            }]
        )

        # Save template
        prompt_manager.save_template(custom_template)

        # Create new manager instance and load template
        new_manager = PromptManager(temp_templates_dir)
        loaded_template = new_manager.load_template(temp_templates_dir / "integration_test.json")

        # Verify template was loaded correctly
        assert loaded_template.name == "integration_test"
        assert loaded_template.description == "Template for integration testing"
        assert loaded_template.variables == ["variable1", "variable2"]

        # Test using loaded template
        result = new_manager.generate_prompt("integration_test", {
            "variable1": "test1",
            "variable2": "test2"
        })

        assert result == "Test template: test1 and test2"

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, integration_config):
        """Test error handling across components"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock API failures
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client.post.side_effect = Exception("API error")

            # Test client error handling
            client = OllamaClient(integration_config)

            # Initialization should fail gracefully
            success = await client.initialize()
            assert success is False

            # Generation should raise appropriate error
            with pytest.raises(RuntimeError, match="not initialized"):
                await client.generate("test prompt")

        # Test prompt validation
        prompt_manager = PromptManager()

        # Should raise error for missing variables
        with pytest.raises(ValueError, match="Missing required variables"):
            prompt_manager.generate_prompt("active_inference_explanation", {
                "concept": "entropy"
                # Missing other required variables
            })

        # Test model manager error handling
        model_manager = ModelManager()

        # Should return None for unknown task
        unknown_model = model_manager.find_best_model("unknown_task_xyz")
        # This might return a default model or None depending on implementation

        # Should handle non-existent model gracefully
        nonexistent_info = model_manager.get_model_info("nonexistent_model")
        assert nonexistent_info is None


class TestLLMPerformance:
    """Performance and stress tests for LLM components"""

    def test_large_conversation_context(self):
        """Test handling of large conversation contexts"""
        conversation_manager = ConversationManager()

        conversation = conversation_manager.create_conversation("Large Context Test")

        # Add many messages to test context window management
        for i in range(50):
            conversation_manager.add_message(
                conversation.id,
                "user" if i % 2 == 0 else "assistant",
                f"This is message number {i} with some content to make it longer and test the context window optimization."
            )

        # Get context with reasonable token limit
        context = conversation_manager.get_conversation_context(conversation.id, max_tokens=1000)

        # Should return reasonable number of messages
        assert len(context) <= 50  # Should not exceed total messages
        assert len(context) > 0    # Should return some messages

        # Should maintain message order (system first, then chronological)
        roles = [msg["role"] for msg in context]
        if "system" in roles:
            assert roles[0] == "system"

    def test_prompt_builder_performance(self):
        """Test prompt builder performance with many sections"""
        builder = PromptBuilder()

        # Add many sections
        for i in range(100):
            builder.add_section(
                f"section_{i}",
                f"This is section {i} with content that should be included in the final prompt."
            )

        # Build prompt
        prompt = builder.build()

        # Should contain all sections
        for i in range(100):
            assert f"section_{i}" in prompt.upper()
            assert f"This is section {i}" in prompt

        # Should be properly formatted
        assert len(prompt) > 1000  # Should be substantial length

    def test_model_registry_performance(self):
        """Test model registry performance with many models"""
        model_manager = ModelManager()

        # Add many custom models
        for i in range(50):
            model_manager.add_custom_model(
                name=f"test_model_{i}",
                family="test",
                size=f"{i%10}b",
                capabilities=[f"capability_{j}" for j in range(i%5)]
            )

        # Test performance of model searches
        import time
        start_time = time.time()

        # Find models by capability
        for capability in ["capability_0", "capability_1", "capability_2"]:
            models = model_manager.get_models_by_capability(capability)
            assert len(models) > 0

        # Find best model for various tasks
        for task in ["reasoning", "mathematical", "coding"]:
            model = model_manager.find_best_model(task)
            assert model is not None

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 5.0, f"Performance test took too long: {execution_time}s"


if __name__ == "__main__":
    pytest.main([__file__])
