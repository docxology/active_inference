"""Coverage-focused behaviour tests for ``llm.prompts``."""

import json
from pathlib import Path

import pytest

from active_inference.llm.prompts import (
    ActiveInferencePromptBuilder,
    EducationalPromptBuilder,
    PromptBuilder,
    PromptManager,
    PromptSection,
    PromptTemplate,
    PromptVariable,
)


class TestPromptTemplate:
    def test_format_and_missing_variable(self) -> None:
        tpl = PromptTemplate("t", "d", "Hello {name}", variables=["name"])
        assert tpl.format(name="Ada") == "Hello Ada"
        with pytest.raises(ValueError, match="Missing required variable: name"):
            tpl.format()

    def test_validate_variables(self) -> None:
        tpl = PromptTemplate("t", "d", "{a}{b}", variables=["a", "b"])
        assert tpl.validate_variables({"a": 1}) == ["b"]
        assert tpl.validate_variables({"a": 1, "b": 2}) == []

    def test_example_prompts(self) -> None:
        tpl = PromptTemplate(
            "t",
            "d",
            "Hi {who}",
            examples=[
                {"name": "first", "variables": {"who": "one"}},
                {"name": "second", "variables": {"who": "two"}},
            ],
        )
        assert tpl.get_example_prompt() == "Hi one"
        assert tpl.get_example_prompt("second") == "Hi two"
        assert tpl.get_example_prompt("absent") is None
        assert PromptTemplate("e", "d", "x").get_example_prompt() is None


class TestPromptVariable:
    def test_defaults_and_validator(self) -> None:
        plain = PromptVariable("x")
        assert (plain.var_type, plain.required, plain.default) == (
            "string",
            True,
            None,
        )
        assert plain.validate(object()) is True
        positive = PromptVariable("n", "int", "count", False, 1, lambda v: v > 0)
        assert positive.validate(3) is True
        assert positive.validate(-1) is False
        assert positive.default == 1


def test_section_format() -> None:
    assert PromptSection("intro", "body").format() == "INTRO:\nbody"


class TestPromptBuilder:
    def test_build_with_variables_and_separators(self) -> None:
        text = (
            PromptBuilder()
            .add_section("a", "Hello {name}")
            .add_section("b", "Age {age}", separator=" | ")
            .set_variable("name", "Ada")
            .set_variables({"age": 36})
            .set_metadata("k", "v")
            .build()
        )
        assert text == "Hello Ada | Age 36"

    def test_insert_at_front_is_built_first(self) -> None:
        builder = PromptBuilder().add_section("a", "A").add_section("b", "B")
        builder.add_section("c", "C", position=0)
        assert [s.position for s in builder.sections] == [0, 1, 2]
        assert builder.build() == "C\n\nA\n\nB"

    def test_insert_in_middle(self) -> None:
        builder = PromptBuilder().add_section("a", "A").add_section("b", "B")
        builder.add_section("c", "C", position=1)
        assert builder.build() == "A\n\nC\n\nB"

    def test_clear(self) -> None:
        builder = PromptBuilder().add_section("a", "A").set_variable("x", 1)
        builder.set_metadata("m", 2)
        assert builder.clear() is builder
        assert builder.sections == []
        assert builder.variables == {}
        assert builder.metadata == {}
        assert builder.build() == ""


class TestPromptManager:
    def test_defaults_and_generation(self, tmp_path: Path) -> None:
        mgr = PromptManager(templates_path=tmp_path)
        assert mgr.list_templates() == [
            "active_inference_explanation",
            "research_question",
            "code_implementation",
        ]
        with pytest.raises(ValueError, match="Template not found: nope"):
            mgr.generate_prompt("nope", {})
        with pytest.raises(ValueError, match="Missing required variables"):
            mgr.generate_prompt("active_inference_explanation", {"concept": "x"})
        tpl = mgr.get_template("active_inference_explanation")
        example_vars = tpl.examples[0]["variables"]
        prompt = mgr.generate_prompt("active_inference_explanation", example_vars)
        assert "Explain the concept of entropy" in prompt
        # Without validation, a missing variable surfaces from format().
        with pytest.raises(ValueError, match="Missing required variable: context"):
            mgr.generate_prompt(
                "active_inference_explanation", {"concept": "x"}, validate=False
            )

    def test_default_templates_path(self) -> None:
        mgr = PromptManager()
        assert mgr.templates_path.name == "templates"
        assert mgr.templates_path.parent.name == "llm"

    def test_custom_template_and_info(self, tmp_path: Path) -> None:
        mgr = PromptManager(templates_path=tmp_path)
        tpl = mgr.create_custom_template("c", "desc", "{q}?")
        assert (tpl.variables, tpl.examples) == ([], [])
        tpl2 = mgr.create_custom_template(
            "c2", "desc", "{q}?", variables=["q"], examples=[{"variables": {"q": 1}}]
        )
        mgr.add_template(tpl2)
        assert mgr.get_template("c") is None
        assert mgr.get_template_info("c2") == {
            "name": "c2",
            "description": "desc",
            "variables": ["q"],
            "examples_count": 1,
            "metadata": {},
        }
        assert mgr.get_template_info("missing") is None
        assert mgr.generate_prompt("c2", {"q": "why"}) == "why?"

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        mgr = PromptManager(templates_path=tmp_path / "tpl")
        tpl = PromptTemplate(
            "saved", "d", "Say {x}", ["x"], [{"variables": {"x": 1}}], {"v": 2}
        )
        mgr.save_template(tpl)
        default_file = tmp_path / "tpl" / "saved.json"
        assert json.loads(default_file.read_text()) == {
            "name": "saved",
            "description": "d",
            "template": "Say {x}",
            "variables": ["x"],
            "examples": [{"variables": {"x": 1}}],
            "metadata": {"v": 2},
        }
        explicit = tmp_path / "nested" / "dir" / "other.json"
        mgr.save_template(tpl, explicit)
        assert explicit.exists()

        fresh = PromptManager(templates_path=tmp_path / "unused")
        loaded = fresh.load_template(default_file)
        assert loaded == tpl
        assert fresh.get_template("saved") is loaded

    def test_load_minimal_file_uses_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "min.json"
        path.write_text(json.dumps({"name": "m", "description": "d", "template": "t"}))
        loaded = PromptManager(templates_path=tmp_path).load_template(path)
        assert (loaded.variables, loaded.examples, loaded.metadata) == ([], [], {})


class TestSpecialisedBuilders:
    def test_active_inference_builder_sections(self) -> None:
        builder = (
            ActiveInferencePromptBuilder()
            .add_concept_explanation("entropy", context="info theory")
            .add_learning_objective(["Define it", "Compute it"], "advanced")
            .add_research_question("planning", domain="robotics")
            .add_code_implementation("EFE", requirements=["numpy", "tests"])
        )
        names = [s.name for s in builder.sections]
        assert names == [
            "concept_explanation",
            "learning_objectives",
            "research_questions",
            "code_implementation",
        ]
        text = builder.build()
        assert "Explain entropy in Active Inference context" in text
        assert "Context: info theory" in text
        assert "Audience level: intermediate" in text
        assert "Include mathematical formulation where applicable." in text
        assert "Learning Objectives (Advanced level):\n1. Define it\n2. Compute it" in (
            text
        )
        assert (
            "Generate research questions about planning in robotics using"
            " theoretical_analysis approaches." in text
        )
        assert text.endswith("Implement EFE in Python\n\nRequirements: numpy, tests")

    def test_active_inference_builder_minimal_options(self) -> None:
        text = (
            ActiveInferencePromptBuilder()
            .add_concept_explanation("x", audience_level="", include_math=False)
            .add_code_implementation("algo", language="Julia")
            .build()
        )
        assert (
            text == "Explain x in Active Inference context\n\nImplement algo in Julia"
        )

    def test_educational_builder(self) -> None:
        text = (
            EducationalPromptBuilder()
            .add_learning_objective(["a", "b"], "beginner")
            .add_prerequisite_check(["probability"])
            .add_assessment_criteria(["recall", "apply"])
            .build()
        )
        assert text == (
            "Learning Objectives (beginner level):\n- a\n- b\n\n"
            "Ensure explanation builds on these prerequisite concepts:\n"
            "- probability\n\n"
            "Content should enable assessment of:\n- recall\n- apply"
        )
