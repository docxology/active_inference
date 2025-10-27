"""
Tests for LLM Prompt System

Unit tests for the prompt composition and templating system, ensuring proper
operation of template management, variable substitution, and prompt building.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from active_inference.llm.prompts import (
    PromptTemplate,
    PromptBuilder,
    PromptManager,
    PromptSection,
    ActiveInferencePromptBuilder,
    EducationalPromptBuilder
)


class TestPromptTemplate:
    """Test cases for PromptTemplate class"""

    def test_template_creation(self):
        """Test creating a prompt template"""
        template = PromptTemplate(
            name="test_template",
            description="A test template",
            template="Hello {name}, you are {age} years old.",
            variables=["name", "age"]
        )

        assert template.name == "test_template"
        assert template.description == "A test template"
        assert template.template == "Hello {name}, you are {age} years old."
        assert template.variables == ["name", "age"]

    def test_template_format(self):
        """Test template formatting"""
        template = PromptTemplate(
            name="test_template",
            description="A test template",
            template="Hello {name}, you are {age} years old.",
            variables=["name", "age"]
        )

        result = template.format(name="Alice", age=30)

        assert result == "Hello Alice, you are 30 years old."

    def test_template_format_missing_variable(self):
        """Test template formatting with missing variable"""
        template = PromptTemplate(
            name="test_template",
            description="A test template",
            template="Hello {name}, you are {age} years old.",
            variables=["name", "age"]
        )

        with pytest.raises(ValueError, match="Missing required variable: age"):
            template.format(name="Alice")

    def test_validate_variables(self):
        """Test variable validation"""
        template = PromptTemplate(
            name="test_template",
            description="A test template",
            template="Hello {name}, you are {age} years old.",
            variables=["name", "age"]
        )

        # Valid variables
        missing = template.validate_variables({"name": "Alice", "age": 30})
        assert missing == []

        # Missing variables
        missing = template.validate_variables({"name": "Alice"})
        assert "age" in missing

        # Extra variables (should be OK)
        missing = template.validate_variables({"name": "Alice", "age": 30, "extra": "value"})
        assert missing == []

    def test_example_prompt(self):
        """Test getting example prompts"""
        template = PromptTemplate(
            name="test_template",
            description="A test template",
            template="Hello {name}, you are {age} years old.",
            variables=["name", "age"],
            examples=[
                {
                    "name": "example1",
                    "variables": {"name": "Bob", "age": 25}
                },
                {
                    "name": "example2",
                    "variables": {"name": "Charlie", "age": 35}
                }
            ]
        )

        # Get first example
        example = template.get_example_prompt()
        assert example == "Hello Bob, you are 25 years old."

        # Get specific example
        example = template.get_example_prompt("example2")
        assert example == "Hello Charlie, you are 35 years old."

        # Non-existent example
        example = template.get_example_prompt("nonexistent")
        assert example is None


class TestPromptBuilder:
    """Test cases for PromptBuilder class"""

    def test_builder_creation(self):
        """Test creating a prompt builder"""
        builder = PromptBuilder()

        assert len(builder.sections) == 0
        assert len(builder.variables) == 0
        assert len(builder.metadata) == 0

    def test_add_section(self):
        """Test adding sections to builder"""
        builder = PromptBuilder()

        builder.add_section("intro", "This is an introduction.")
        builder.add_section("main", "This is the main content.", separator="\n---\n")

        assert len(builder.sections) == 2
        assert builder.sections[0].name == "intro"
        assert builder.sections[0].content == "This is an introduction."
        assert builder.sections[1].name == "main"
        assert builder.sections[1].content == "This is the main content."
        assert builder.sections[1].separator == "\n---\n"

    def test_set_variables(self):
        """Test setting variables"""
        builder = PromptBuilder()

        builder.set_variable("name", "Alice")
        builder.set_variables({"age": 30, "city": "New York"})

        assert builder.variables["name"] == "Alice"
        assert builder.variables["age"] == 30
        assert builder.variables["city"] == "New York"

    def test_build_prompt(self):
        """Test building final prompt"""
        builder = PromptBuilder()

        builder.add_section("header", "Active Inference Explanation")
        builder.add_section("content", "Explain {concept} in the context of {framework}.")
        builder.set_variables({
            "concept": "entropy",
            "framework": "Free Energy Principle"
        })

        result = builder.build()

        expected = "Active Inference Explanation\n\nExplain entropy in the context of Free Energy Principle."
        assert result == expected

    def test_build_with_custom_separator(self):
        """Test building with custom section separators"""
        builder = PromptBuilder()

        builder.add_section("part1", "First part", separator=" | ")
        builder.add_section("part2", "Second part", separator=" | ")

        result = builder.build()

        assert result == "First part | Second part"

    def test_clear_builder(self):
        """Test clearing builder"""
        builder = PromptBuilder()

        builder.add_section("test", "content")
        builder.set_variable("test", "value")
        builder.set_metadata("key", "value")

        builder.clear()

        assert len(builder.sections) == 0
        assert len(builder.variables) == 0
        assert len(builder.metadata) == 0


class TestPromptManager:
    """Test cases for PromptManager class"""

    @pytest.fixture
    def temp_templates_dir(self):
        """Create temporary templates directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_manager_creation(self):
        """Test creating prompt manager"""
        manager = PromptManager()

        assert len(manager.templates) > 0  # Should have default templates
        assert "active_inference_explanation" in manager.templates
        assert "research_question" in manager.templates

    def test_add_template(self):
        """Test adding templates"""
        manager = PromptManager()

        template = PromptTemplate(
            name="custom_template",
            description="Custom template",
            template="Custom {variable}",
            variables=["variable"]
        )

        manager.add_template(template)

        assert "custom_template" in manager.templates
        assert manager.get_template("custom_template") == template

    def test_generate_prompt(self):
        """Test generating prompts from templates"""
        manager = PromptManager()

        variables = {
            "concept": "variational inference",
            "context": "Bayesian modeling",
            "audience_level": "advanced",
            "key_points": "mathematical formulation, computational methods",
            "response_type": "technical explanation"
        }

        result = manager.generate_prompt("active_inference_explanation", variables)

        assert "variational inference" in result
        assert "Bayesian modeling" in result
        assert "advanced" in result
        assert "mathematical formulation" in result

    def test_generate_prompt_missing_variables(self):
        """Test generating prompt with missing variables"""
        manager = PromptManager()

        variables = {
            "concept": "variational inference"
            # Missing other required variables
        }

        with pytest.raises(ValueError, match="Missing required variables"):
            manager.generate_prompt("active_inference_explanation", variables, validate=True)

    def test_list_templates(self):
        """Test listing available templates"""
        manager = PromptManager()

        templates = manager.list_templates()

        assert isinstance(templates, list)
        assert len(templates) > 0
        assert "active_inference_explanation" in templates

    def test_save_and_load_template(self, temp_templates_dir):
        """Test saving and loading templates"""
        manager = PromptManager(temp_templates_dir)

        template = PromptTemplate(
            name="test_template",
            description="Test template",
            template="Test {variable}",
            variables=["variable"],
            examples=[{"name": "test", "variables": {"variable": "value"}}]
        )

        # Save template
        manager.save_template(template)

        # Create new manager and load template
        new_manager = PromptManager(temp_templates_dir)
        loaded_template = new_manager.load_template(temp_templates_dir / "test_template.json")

        assert loaded_template.name == template.name
        assert loaded_template.description == template.description
        assert loaded_template.template == template.template
        assert loaded_template.variables == template.variables
        assert loaded_template.examples == template.examples

    def test_get_template_info(self):
        """Test getting template information"""
        manager = PromptManager()

        info = manager.get_template_info("active_inference_explanation")

        assert info is not None
        assert info["name"] == "active_inference_explanation"
        assert "description" in info
        assert "variables" in info
        assert "examples_count" in info


class TestSpecializedPromptBuilders:
    """Test cases for specialized prompt builders"""

    def test_active_inference_builder(self):
        """Test Active Inference prompt builder"""
        builder = ActiveInferencePromptBuilder()

        builder.add_concept_explanation(
            "entropy",
            "information theory",
            "intermediate",
            include_math=True
        )

        result = builder.build()

        assert "entropy" in result
        assert "information theory" in result
        assert "intermediate" in result
        assert "mathematical formulation" in result

    def test_educational_builder(self):
        """Test educational prompt builder"""
        builder = EducationalPromptBuilder()

        builder.add_learning_objective([
            "Understand entropy concept",
            "Calculate entropy for probability distributions",
            "Apply entropy in information theory"
        ], "intermediate")

        builder.add_prerequisite_check([
            "Basic probability theory",
            "Information theory fundamentals"
        ])

        result = builder.build()

        assert "Learning Objectives" in result
        assert "intermediate" in result
        assert "entropy concept" in result
        assert "Basic probability theory" in result


if __name__ == "__main__":
    pytest.main([__file__])
