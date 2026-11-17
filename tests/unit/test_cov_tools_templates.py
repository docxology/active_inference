"""Behavioural tests for active_inference.applications.templates."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from active_inference.applications.templates import (
    ApplicationFramework,
    CodeGenerator,
    TemplateConfig,
    TemplateManager,
    TemplateType,
)


def _only_file(result: Dict[str, Any]) -> tuple:
    files = result["files"]
    assert len(files) == 1
    return next(iter(files.items()))


@pytest.fixture
def framework(tmp_path: Path) -> ApplicationFramework:
    return ApplicationFramework({"templates_dir": tmp_path / "templates"})


# -------------------------------------------------------------- code generator


def test_code_generator_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    generator = CodeGenerator(target)
    assert target.is_dir()
    assert generator.templates == {}


def test_template_config_defaults() -> None:
    config = TemplateConfig("n", TemplateType.API_SERVICE, "d")
    assert config.parameters == {}
    assert config.output_directory is None


def test_basic_model_generated_code_compiles_and_validates(
    tmp_path: Path, framework: ApplicationFramework
) -> None:
    generator = CodeGenerator(tmp_path)
    config = TemplateConfig("Agent Model", TemplateType.BASIC_MODEL, "A test agent")
    result = generator.generate_basic_model(config)

    filename, code = _only_file(result)
    assert filename == "agent_model.py"
    assert "class AgentModel:" in code
    assert "model = AgentModel(config)" in code
    assert result["requirements"] == ["numpy", "typing"]
    assert result["readme"].startswith("# Agent Model\n\nA test agent\n")

    # Regression: the generated module must be valid Python, and the basic
    # model validator must recognise its ``class ...Model`` declaration.
    compile(code, filename, "exec")
    validation = framework.validate_generated_code(code, "basic_model")
    assert validation == {
        "valid": True,
        "issues": [],
        "warnings": [],
        "suggestions": [],
    }


def test_basic_model_hyphenated_name(tmp_path: Path) -> None:
    config = TemplateConfig("my-Model", TemplateType.BASIC_MODEL, "d")
    _, code = _only_file(CodeGenerator(tmp_path).generate_basic_model(config))
    assert "class my_Model:" in code
    compile(code, "<generated>", "exec")


def test_research_pipeline_placeholder(tmp_path: Path) -> None:
    config = TemplateConfig("p", TemplateType.RESEARCH_PIPELINE, "d")
    assert CodeGenerator(tmp_path).generate_research_pipeline(config) == {
        "files": {},
        "requirements": [],
        "readme": "Research pipeline template",
    }


def test_robotics_control_template(tmp_path: Path) -> None:
    config = TemplateConfig("Arm Controller", TemplateType.ROBOTICS_CONTROL, "d")
    result = CodeGenerator(tmp_path).generate_robotics_control(config)
    filename, code = _only_file(result)
    assert filename == "arm_controller.py"
    assert '"""\nArm Controller - Robotics Control with Active Inference' in code
    assert "class RoboticsController:" in code
    assert "def check_safety" in code
    assert result["requirements"] == ["numpy", "scipy", "matplotlib"]
    assert result["readme"].startswith("# Arm Controller")
    compile(code, filename, "exec")


def test_neural_interface_template(tmp_path: Path) -> None:
    config = TemplateConfig("BCI", TemplateType.NEURAL_INTERFACE, "d")
    result = CodeGenerator(tmp_path).generate_neural_interface(config)
    filename, code = _only_file(result)
    assert filename == "bci.py"
    assert "class NeuralInterface:" in code
    assert "from scipy import signal" in code
    assert "mne" in result["requirements"]
    assert result["readme"].startswith("# BCI")
    compile(code, filename, "exec")


# ------------------------------------------------------------ template manager


def test_template_manager_layout(tmp_path: Path) -> None:
    manager = TemplateManager(tmp_path)
    assert manager.templates_dir == tmp_path
    assert manager.code_generator.templates_dir == tmp_path / "generated"
    assert (tmp_path / "generated").is_dir()


@pytest.mark.parametrize(
    "template_type, marker",
    [
        (TemplateType.BASIC_MODEL, "class Demo:"),
        (TemplateType.ROBOTICS_CONTROL, "class RoboticsController:"),
        (TemplateType.NEURAL_INTERFACE, "class NeuralInterface:"),
    ],
)
def test_generate_application_dispatch(
    tmp_path: Path, template_type: TemplateType, marker: str
) -> None:
    config = TemplateConfig("Demo", template_type, "d")
    _, code = _only_file(TemplateManager(tmp_path).generate_application(config))
    assert marker in code


def test_generate_application_pipeline_and_unimplemented(tmp_path: Path) -> None:
    manager = TemplateManager(tmp_path)
    pipeline = manager.generate_application(
        TemplateConfig("p", TemplateType.RESEARCH_PIPELINE, "d")
    )
    assert pipeline["readme"] == "Research pipeline template"
    assert (
        manager.generate_application(
            TemplateConfig("w", TemplateType.MULTI_AGENT_SYSTEM, "d")
        )
        == {}
    )


def test_list_available_templates(tmp_path: Path) -> None:
    templates = TemplateManager(tmp_path).list_available_templates()
    assert len(templates) == len(TemplateType)
    robotics = next(t for t in templates if t["name"] == "robotics_control")
    assert robotics == {
        "name": "robotics_control",
        "type": "ROBOTICS_CONTROL",
        "description": "Template for robotics control",
    }


def test_create_custom_template(tmp_path: Path) -> None:
    manager = TemplateManager(tmp_path)
    data = {"name": "mine", "fields": [1, 2]}
    assert manager.create_custom_template(data) is True
    saved = tmp_path / "custom" / "mine.json"
    assert json.loads(saved.read_text()) == data


@pytest.mark.parametrize("data", [{}, {"name": ""}])
def test_create_custom_template_requires_name(
    tmp_path: Path, data: Dict[str, Any]
) -> None:
    assert TemplateManager(tmp_path).create_custom_template(data) is False
    assert not (tmp_path / "custom").exists()


# ------------------------------------------------------- application framework


def test_framework_uses_configured_dir(
    tmp_path: Path, framework: ApplicationFramework
) -> None:
    assert framework.template_manager.templates_dir == tmp_path / "templates"
    assert (tmp_path / "templates" / "generated").is_dir()


def test_create_application(framework: ApplicationFramework) -> None:
    result = framework.create_application(
        TemplateType.BASIC_MODEL, "Mini Model", {"n_states": 2}
    )
    filename, code = _only_file(result)
    assert filename == "mini_model.py"
    assert "Application generated from basic_model template" in result["readme"]
    assert "with 2 states and 8 observations" in code


def test_create_application_default_parameters(
    framework: ApplicationFramework,
) -> None:
    result = framework.create_application(TemplateType.BASIC_MODEL, "Plain Model")
    _, code = _only_file(result)
    assert "with 4 states and 8 observations" in code


def test_get_template_info(framework: ApplicationFramework) -> None:
    assert framework.get_template_info(TemplateType.API_SERVICE) == {
        "name": "api_service",
        "type": "API_SERVICE",
        "description": "Template for api service",
    }


def test_framework_generate_basic_model(framework: ApplicationFramework) -> None:
    named = framework.generate_basic_model({"name": "Custom Model"})
    assert "custom_model.py" in named["files"]
    default = framework.generate_basic_model({})
    assert "basic_model.py" in default["files"]
    assert default["readme"].startswith(
        "# basic_model\n\nBasic Active Inference model implementation"
    )


def test_framework_generate_research_pipeline(
    framework: ApplicationFramework,
) -> None:
    assert framework.generate_research_pipeline({})["readme"] == (
        "Research pipeline template"
    )


@pytest.mark.parametrize(
    "method",
    ["generate_web_application", "generate_api_service", "generate_educational_tool"],
)
def test_framework_unimplemented_generators(
    framework: ApplicationFramework, method: str
) -> None:
    assert getattr(framework, method)({"name": "x"}) == {}


# --------------------------------------------------------------- validation


def test_validate_syntax_error(framework: ApplicationFramework) -> None:
    result = framework.validate_generated_code("def broken(:\n", "other")
    assert result["valid"] is False
    assert len(result["issues"]) == 1
    assert result["issues"][0].startswith("Syntax error:")


def test_validate_unknown_type_only_checks_syntax(
    framework: ApplicationFramework,
) -> None:
    assert framework.validate_generated_code("x = 1\n", "custom")["valid"] is True


def test_validate_basic_model_missing_elements(
    framework: ApplicationFramework,
) -> None:
    code = "class Helper:\n    def __init__(self):\n        pass\n"
    result = framework.validate_generated_code(code, "basic_model")
    assert result["valid"] is False
    assert result["issues"] == [
        "Missing required element: class.*Model",
        "Missing required element: def perceive",
        "Missing required element: def act",
    ]


def test_validate_research_pipeline(framework: ApplicationFramework) -> None:
    complete = (
        "def run_experiment():\n    pass\n"
        "def analyze_results():\n    pass\n"
        "def save_data():\n    pass\n"
    )
    assert framework.validate_generated_code(complete, "research_pipeline")["valid"]
    partial = framework.validate_generated_code(
        "def run_experiment():\n    pass\n", "research_pipeline"
    )
    assert partial["valid"] is False
    assert partial["issues"] == [
        "Missing required element: def analyze_results",
        "Missing required element: def save_data",
    ]


def test_validate_web_application(framework: ApplicationFramework) -> None:
    complete = (
        "from flask import Flask\n"
        "app = Flask(__name__)\n"
        "@app.route('/')\n"
        "def index():\n"
        "    return 'ok'\n"
    )
    assert framework.validate_generated_code(complete, "web_application") == {
        "valid": True,
        "issues": [],
        "warnings": [],
        "suggestions": [],
    }
    missing = framework.validate_generated_code("x = 1\n", "web_application")
    assert len(missing["issues"]) == 3
    assert missing["valid"] is False


# ------------------------------------------------------ template specification


def test_spec_with_string_type(framework: ApplicationFramework, tmp_path: Path) -> None:
    config = framework.create_template_from_specification(
        {
            "name": "spec",
            "type": "robotics_control",
            "description": "desc",
            "parameters": {"n_joints": 3},
            "output_directory": tmp_path,
        }
    )
    assert config.name == "spec"
    assert config.template_type is TemplateType.ROBOTICS_CONTROL
    assert config.description == "desc"
    assert config.parameters == {"n_joints": 3}
    assert config.output_directory == tmp_path


def test_spec_with_enum_type(framework: ApplicationFramework) -> None:
    config = framework.create_template_from_specification(
        {"name": "s", "type": TemplateType.API_SERVICE, "description": "d"}
    )
    assert config.template_type is TemplateType.API_SERVICE
    assert config.parameters == {}
    assert config.output_directory is None


def test_spec_unknown_string_type_falls_back(framework: ApplicationFramework) -> None:
    config = framework.create_template_from_specification(
        {"name": "s", "type": "quantum", "description": "d"}
    )
    assert config.template_type is TemplateType.BASIC_MODEL


@pytest.mark.parametrize("missing", ["name", "type", "description"])
def test_spec_missing_required_field(
    framework: ApplicationFramework, missing: str
) -> None:
    spec = {"name": "s", "type": "basic_model", "description": "d"}
    del spec[missing]
    with pytest.raises(ValueError, match=f"Missing required field.*{missing}"):
        framework.create_template_from_specification(spec)


def test_spec_invalid_type_value(framework: ApplicationFramework) -> None:
    with pytest.raises(ValueError, match="must be string or TemplateType"):
        framework.create_template_from_specification(
            {"name": "s", "type": 42, "description": "d"}
        )


# --------------------------------------------------------------- optimizers


def test_optimize_template_performance(framework: ApplicationFramework) -> None:
    code = (
        "import sys\n"
        "import os\n"
        "import sys\n"
        "for i in range(len(list)):\n"
        "    y = x * 2\n"
    )
    optimized = framework.optimize_template_performance(code)
    assert optimized == (
        "import os\n" "import sys\n" "\n" "for item in list:\n" "    y = x << 1\n"
    )


def test_optimize_continues_after_failure(
    framework: ApplicationFramework, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(code: str) -> str:
        raise RuntimeError("optimizer bug")

    monkeypatch.setattr(framework, "_optimize_loops", broken)
    optimized = framework.optimize_template_performance(
        "for i in range(len(list)):\n    z = x * 2"
    )
    # Loop rewrite skipped, later computation rewrite still applied.
    assert "for i in range(len(list)):" in optimized
    assert "x << 1" in optimized


def test_memory_optimization_is_identity(framework: ApplicationFramework) -> None:
    code = "values = [v for v in range(3)]"
    assert framework._optimize_memory_usage(code) == code
