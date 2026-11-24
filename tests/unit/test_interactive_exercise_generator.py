"""Unit tests for tools/interactive_exercise_generator.py."""

import ast
import importlib.util
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "interactive_exercise_generator",
        ROOT / "tools" / "interactive_exercise_generator.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator_mod = _load_generator_module()
InteractiveExerciseGenerator = generator_mod.InteractiveExerciseGenerator


def _run_generated_main(code: str, tmp_path: Path, filename: str) -> None:
    script_path = tmp_path / filename
    script_path.write_text(code, encoding="utf-8")
    runpy.run_path(str(script_path), run_name="__main__")


class TestInteractiveExerciseGeneratorCodeTemplates:
    def test_generate_starter_code_renders_valid_executable_python(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        generator = InteractiveExerciseGenerator()
        code = generator._generate_starter_code({"title": "Variational Free Energy"})

        assert "# Implement Variational Free Energy" in code
        assert "Implement the Variational Free Energy functionality" in code
        assert 'print(f"Result: {result}")' in code
        ast.parse(code)

        _run_generated_main(code, tmp_path, "starter_exercise.py")
        assert capsys.readouterr().out.strip() == "Result: None"

    def test_generate_solution_code_renders_valid_executable_python(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        generator = InteractiveExerciseGenerator()
        code = generator._generate_solution_code({"title": "Expected Free Energy"})

        assert "# Complete implementation of Expected Free Energy" in code
        assert 'print(f"Result: {result}")' in code
        ast.parse(code)

        _run_generated_main(code, tmp_path, "solution_exercise.py")
        assert capsys.readouterr().out.strip() == "Result: implementation_complete"

    def test_generate_code_templates_default_title(self) -> None:
        generator = InteractiveExerciseGenerator()

        starter = generator._generate_starter_code({})
        solution = generator._generate_solution_code({})

        assert "# Implement the concept" in starter
        assert "Implement the concept functionality" in starter
        assert "# Complete implementation of the concept" in solution
        ast.parse(starter)
        ast.parse(solution)

    def test_generate_exercises_for_implementation_node(self) -> None:
        generator = InteractiveExerciseGenerator()
        exercises = generator.generate_exercises_for_node(
            {
                "id": "impl-1",
                "title": "Active Inference Agent",
                "content_type": "implementation",
                "difficulty": "intermediate",
            }
        )
        coding = [ex for ex in exercises if ex["type"] == "coding"]
        assert len(coding) == 1
        ast.parse(coding[0]["starter_code"])
        ast.parse(coding[0]["solution"])
        assert 'print(f"Result: {result}")' in coding[0]["starter_code"]
        assert 'print(f"Result: {result}")' in coding[0]["solution"]
