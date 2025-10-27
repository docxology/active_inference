"""
Tests for Applications - Templates Module

Unit tests for the template system and code generation functionality,
ensuring proper template generation, configuration management, and
application framework operations.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from active_inference.applications.templates import (
    TemplateType,
    TemplateConfig,
    CodeGenerator,
    TemplateManager,
    ApplicationFramework
)


class TestTemplateType:
    """Test cases for TemplateType enum"""

    def test_template_type_values(self):
        """Test template type enum values"""
        assert TemplateType.BASIC_MODEL.value == "basic_model"
        assert TemplateType.RESEARCH_PIPELINE.value == "research_pipeline"
        assert TemplateType.SIMULATION_STUDY.value == "simulation_study"
        assert TemplateType.WEB_APPLICATION.value == "web_application"
        assert TemplateType.API_SERVICE.value == "api_service"
        assert TemplateType.EDUCATIONAL_TOOL.value == "educational_tool"

    def test_template_type_count(self):
        """Test correct number of template types"""
        types = list(TemplateType)
        assert len(types) == 11


class TestTemplateConfig:
    """Test cases for TemplateConfig dataclass"""

    def test_template_config_creation_basic(self):
        """Test basic template config creation"""
        config = TemplateConfig(
            name="test_model",
            template_type=TemplateType.BASIC_MODEL,
            description="Test model configuration"
        )

        assert config.name == "test_model"
        assert config.template_type == TemplateType.BASIC_MODEL
        assert config.description == "Test model configuration"
        assert config.parameters == {}
        assert config.output_directory is None

    def test_template_config_creation_with_parameters(self):
        """Test template config creation with parameters"""
        parameters = {"n_states": 4, "n_observations": 8}
        output_dir = Path("/tmp/test")

        config = TemplateConfig(
            name="test_model",
            template_type=TemplateType.BASIC_MODEL,
            description="Test model configuration",
            parameters=parameters,
            output_directory=output_dir
        )

        assert config.name == "test_model"
        assert config.parameters == parameters
        assert config.output_directory == output_dir

    def test_template_config_post_init(self):
        """Test template config post-initialization"""
        # Test with None parameters (should be converted to empty dict)
        config = TemplateConfig(
            name="test_model",
            template_type=TemplateType.BASIC_MODEL,
            description="Test model configuration",
            parameters=None
        )

        assert config.parameters == {}


class TestCodeGenerator:
    """Test cases for CodeGenerator class"""

    def test_initialization(self):
        """Test code generator initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = CodeGenerator(temp_dir)

            assert generator.templates_dir == Path(temp_dir)
            assert generator.templates_dir.exists()
            assert isinstance(generator.templates, dict)

    def test_generate_basic_model(self):
        """Test basic model generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = CodeGenerator(temp_dir)

            config = TemplateConfig(
                name="TestModel",
                template_type=TemplateType.BASIC_MODEL,
                description="Test model for unit testing",
                parameters={
                    "n_states": 4,
                    "n_observations": 8,
                    "time_horizon": 1000
                }
            )

            result = generator.generate_basic_model(config)

            # Check result structure
            assert "files" in result
            assert "requirements" in result
            assert "readme" in result

            # Check generated file
            assert "testmodel.py" in result["files"]
            generated_code = result["files"]["testmodel.py"]

            # Check code contains expected elements
            assert "TestModel" in generated_code
            assert "import numpy as np" in generated_code
            assert 'config.parameters.get("n_states", 4)' in generated_code
            assert 'config.parameters.get("n_observations", 8)' in generated_code
            assert "perceive" in generated_code
            assert "act" in generated_code
            assert "run_simulation" in generated_code

            # Check requirements
            assert "numpy" in result["requirements"]
            assert "typing" in result["requirements"]

            # Check readme
            assert "TestModel" in result["readme"]
            assert "Test model for unit testing" in result["readme"]

    def test_generate_basic_model_default_parameters(self):
        """Test basic model generation with default parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = CodeGenerator(temp_dir)

            config = TemplateConfig(
                name="DefaultModel",
                template_type=TemplateType.BASIC_MODEL,
                description="Default model parameters"
            )

            result = generator.generate_basic_model(config)
            generated_code = result["files"]["defaultmodel.py"]

            # Should use default values
            assert 'config.parameters.get("n_states", 4)' in generated_code  # default value
            assert 'config.parameters.get("n_observations", 8)' in generated_code  # default value
            assert 'config.parameters.get("time_horizon", 1000)' in generated_code  # default value

    def test_generate_research_pipeline(self):
        """Test research pipeline generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = CodeGenerator(temp_dir)

            config = TemplateConfig(
                name="ResearchPipeline",
                template_type=TemplateType.RESEARCH_PIPELINE,
                description="Test research pipeline"
            )

            result = generator.generate_research_pipeline(config)

            # Should return basic structure even if not fully implemented
            assert isinstance(result, dict)
            assert "files" in result
            assert "requirements" in result
            assert "readme" in result

    def test_code_generation_error_handling(self):
        """Test error handling in code generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = CodeGenerator(temp_dir)

            # Test with invalid config
            invalid_config = TemplateConfig(
                name="",  # Invalid empty name
                template_type=TemplateType.BASIC_MODEL,
                description="Invalid config"
            )

            # Should still generate code even with edge cases
            result = generator.generate_basic_model(invalid_config)
            assert "files" in result


class TestTemplateManager:
    """Test cases for TemplateManager class"""

    def test_initialization(self):
        """Test template manager initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemplateManager(temp_dir)

            assert manager.templates_dir == Path(temp_dir)
            assert hasattr(manager, 'code_generator')
            assert hasattr(manager, 'templates')
            assert isinstance(manager.templates, dict)

    def test_generate_application_basic_model(self):
        """Test application generation for basic model"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemplateManager(temp_dir)

            config = TemplateConfig(
                name="TestApp",
                template_type=TemplateType.BASIC_MODEL,
                description="Test application",
                parameters={"n_states": 2, "n_observations": 4}
            )

            result = manager.generate_application(config)

            # Check result structure
            assert "files" in result
            assert "requirements" in result
            assert "readme" in result

            # Check generated code
            assert "testapp.py" in result["files"]
            generated_code = result["files"]["testapp.py"]
            assert "TestApp" in generated_code
            assert 'config.parameters.get("n_states", 4)' in generated_code
            assert 'config.parameters.get("n_observations", 8)' in generated_code

    def test_generate_application_research_pipeline(self):
        """Test application generation for research pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemplateManager(temp_dir)

            config = TemplateConfig(
                name="ResearchApp",
                template_type=TemplateType.RESEARCH_PIPELINE,
                description="Research application"
            )

            result = manager.generate_application(config)

            # Should handle research pipeline template type
            assert isinstance(result, dict)

    def test_generate_application_unknown_type(self):
        """Test application generation for unknown template type"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemplateManager(temp_dir)

            # Create a mock template type that's not in the enum
            config = TemplateConfig(
                name="UnknownApp",
                template_type=TemplateType.BASIC_MODEL,  # Use valid type but modify after
                description="Unknown application"
            )

            # Modify the type to something not handled
            config.template_type = Mock()
            config.template_type.value = "unknown_type"

            result = manager.generate_application(config)

            # Should return empty dict for unknown types
            assert result == {}

    def test_list_available_templates(self):
        """Test listing available templates"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemplateManager(temp_dir)

            templates = manager.list_available_templates()

            assert isinstance(templates, list)
            assert len(templates) == len(TemplateType)

            # Check structure of template info
            for template in templates:
                assert "name" in template
                assert "type" in template
                assert "description" in template

                # Check that name matches enum value
                assert template["name"] in [t.value for t in TemplateType]

    def test_create_custom_template(self):
        """Test creating custom template"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemplateManager(temp_dir)

            template_data = {
                "name": "custom_template",
                "description": "Custom test template",
                "code": "print('Hello from custom template')",
                "requirements": ["numpy"]
            }

            success = manager.create_custom_template(template_data)

            assert success == True

            # Check that template file was created
            template_file = manager.templates_dir / "custom" / "custom_template.json"
            assert template_file.exists()

            # Check file content
            with open(template_file, 'r') as f:
                saved_data = json.load(f)

            assert saved_data == template_data

    def test_create_custom_template_no_name(self):
        """Test creating custom template without name"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemplateManager(temp_dir)

            template_data = {
                "description": "Template without name"
            }

            success = manager.create_custom_template(template_data)

            assert success == False

    def test_template_directory_creation(self):
        """Test template directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir) / "custom_templates"
            manager = TemplateManager(templates_dir)

            # Directory should be created automatically
            assert templates_dir.exists()
            assert (templates_dir / "generated").exists()


class TestApplicationFramework:
    """Test cases for ApplicationFramework class"""

    def test_initialization(self):
        """Test application framework initialization"""
        config = {"templates_dir": "/tmp/test_templates"}

        framework = ApplicationFramework(config)

        assert framework.config == config
        assert hasattr(framework, 'template_manager')
        assert framework.template_manager.templates_dir == Path("/tmp/test_templates")

    def test_create_application_basic_model(self):
        """Test creating application with basic model template"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"templates_dir": temp_dir}
            framework = ApplicationFramework(config)

            result = framework.create_application(
                template_type=TemplateType.BASIC_MODEL,
                name="FrameworkTestModel",
                parameters={"n_states": 3, "n_observations": 6}
            )

            # Check result structure
            assert "files" in result
            assert "requirements" in result
            assert "readme" in result

            # Check generated code
            assert "frameworktestmodel.py" in result["files"]
            generated_code = result["files"]["frameworktestmodel.py"]
            assert "FrameworkTestModel" in generated_code
            assert 'config.parameters.get("n_states", 4)' in generated_code
            assert 'config.parameters.get("n_observations", 8)' in generated_code

    def test_create_application_default_parameters(self):
        """Test creating application with default parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"templates_dir": temp_dir}
            framework = ApplicationFramework(config)

            result = framework.create_application(
                template_type=TemplateType.BASIC_MODEL,
                name="DefaultParamsModel"
            )

            # Should use default parameters
            assert "files" in result
            generated_code = result["files"]["defaultparamsmodel.py"]
            assert 'config.parameters.get("n_states", 4)' in generated_code  # default value

    def test_get_template_info(self):
        """Test getting template information"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"templates_dir": temp_dir}
            framework = ApplicationFramework(config)

            # Test existing template
            info = framework.get_template_info(TemplateType.BASIC_MODEL)
            assert isinstance(info, dict)
            assert info["name"] == "basic_model"
            assert "description" in info

            # Test non-existent template (should return empty dict)
            non_existent = framework.get_template_info(TemplateType.WEB_APPLICATION)
            assert isinstance(non_existent, dict)
            assert non_existent["name"] == "web_application"  # Still in the list

    def test_framework_configuration_handling(self):
        """Test framework configuration handling"""
        # Test with minimal config
        config = {}
        framework = ApplicationFramework(config)

        assert framework.config == {}
        assert framework.template_manager.templates_dir == Path("./templates")

        # Test with custom config
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"templates_dir": temp_dir}
            framework = ApplicationFramework(config)

            assert framework.config == config
            assert framework.template_manager.templates_dir == Path(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
