"""Coverage-focused behaviour tests for ``knowledge.mathematics``."""

import importlib.util
import math
import random
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from active_inference.knowledge import mathematics as math_mod
from active_inference.knowledge.mathematics import Mathematics
from active_inference.knowledge.repository import KnowledgeRepository


@pytest.fixture
def maths() -> Mathematics:
    repo = Mock(spec=KnowledgeRepository)
    return Mathematics(repo)


@pytest.fixture
def no_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(math_mod, "HAS_NUMPY", False)


@pytest.fixture
def no_sympy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(math_mod, "HAS_SYMPY", False)


class _FakeExpr:
    def __init__(self, text: str) -> None:
        self.text = text

    def has(self, marker: object) -> bool:
        if marker == "log":
            return "log" in self.text
        if marker == 0:
            return "0" in self.text
        return False


def _fake_sympy() -> ModuleType:
    """Minimal stand-in for sympy (not installed in this environment)."""
    fake = ModuleType("sympy")

    def sympify(text: str) -> _FakeExpr:
        if text.count("(") != text.count(")"):
            raise SyntaxError("unbalanced parentheses")
        return _FakeExpr(text)

    fake.sympify = sympify  # type: ignore[attr-defined]

    def simplify(expr: _FakeExpr) -> str:
        return expr.text.replace("x + x", "2*x")

    fake.simplify = simplify  # type: ignore[attr-defined]
    fake.log = "log"  # type: ignore[attr-defined]
    fake.S = SimpleNamespace(Zero=0)  # type: ignore[attr-defined]
    return fake


def _load_isolated_copy(
    monkeypatch: pytest.MonkeyPatch, numpy: object, sympy: object
) -> ModuleType:
    """Import a private copy of the module with the given optional deps."""
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "sympy", sympy)
    name = "active_inference.knowledge._cov_mathematics_copy"
    spec = importlib.util.spec_from_file_location(name, math_mod.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_without_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_isolated_copy(monkeypatch, None, None)
    assert module.HAS_NUMPY is False
    assert module.HAS_SYMPY is False
    assert module.np is None
    assert module.sp is None
    inst = module.Mathematics(Mock(spec=KnowledgeRepository))
    assert inst.compute_entropy([0.5, 0.5]) == pytest.approx(1.0)


def test_import_with_optional_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_sympy()
    module = _load_isolated_copy(monkeypatch, np, fake)
    assert module.HAS_NUMPY is True
    assert module.HAS_SYMPY is True
    assert module.sp is fake


@pytest.fixture
def fake_sympy(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    fake = _fake_sympy()
    monkeypatch.setitem(sys.modules, "sympy", fake)
    monkeypatch.setattr(math_mod, "HAS_SYMPY", True)
    return fake


def test_save_nodes_placeholder_returns_none(maths: Mathematics) -> None:
    assert maths._save_nodes_to_repository([{"id": "x"}]) is None


def test_derive_free_energy_expression(maths: Mathematics) -> None:
    result = maths.derive_free_energy_expression()
    assert result["expression"] == "F = ∫ q(θ) log(q(θ)/p(x,θ)) dθ"
    assert set(result["components"]) == {"cross_entropy", "entropy"}
    assert len(result["proof_sketch"]) == 3


class TestKLDivergence:
    def test_numpy_matches_closed_form(self, maths: Mathematics) -> None:
        p, q = [0.5, 0.5], [0.25, 0.75]
        expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
        assert maths.compute_kl_divergence(p, q) == pytest.approx(expected)

    def test_normalises_inputs(self, maths: Mathematics) -> None:
        assert maths.compute_kl_divergence([2, 2], [1, 3]) == pytest.approx(
            maths.compute_kl_divergence([0.5, 0.5], [0.25, 0.75])
        )

    def test_fallback_matches_numpy(self, maths: Mathematics, no_numpy: None) -> None:
        p, q = [1.0, 3.0], [2.0, 2.0]
        expected = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
        assert maths.compute_kl_divergence(p, q) == pytest.approx(expected)
        assert maths.compute_kl_divergence([1, 1], [1, 1]) == 0.0
        # 0 * log(0 / q) is taken as 0.
        assert maths.compute_kl_divergence([0, 1], [0.5, 0.5]) == pytest.approx(
            math.log(2)
        )


class TestExpectedFreeEnergy:
    def test_numpy_branch_is_one_value_per_policy(self, maths: Mathematics) -> None:
        np.random.seed(0)
        values = maths.expected_free_energy([[0.5, 0.5], [0.1, 0.9]], [1.0])
        assert len(values) == 2
        assert all(isinstance(v, float) and 0.0 <= v < 1.0 for v in values)
        np.random.seed(0)
        assert maths.expected_free_energy([[1.0], [1.0]], []) == values

    def test_fallback_branch(self, maths: Mathematics, no_numpy: None) -> None:
        random.seed(3)
        values = maths.expected_free_energy([[1.0]] * 3, [0.0])
        random.seed(3)
        assert values == [random.random() for _ in range(3)]
        assert maths.expected_free_energy([], []) == []


class TestEntropyAndInformation:
    def test_entropy_numpy(self, maths: Mathematics) -> None:
        assert maths.compute_entropy([0.5, 0.5]) == pytest.approx(1.0)
        assert maths.compute_entropy([0.0, 0.0]) == 0.0
        assert maths.compute_entropy([1.0, 0.0]) == 0.0

    def test_entropy_fallback(self, maths: Mathematics, no_numpy: None) -> None:
        assert maths.compute_entropy([0.25] * 4) == pytest.approx(2.0)
        assert maths.compute_entropy([1.0, 0.0]) == 0.0
        assert maths.compute_entropy([]) == 0.0

    def test_mutual_information_independent_is_zero(self, maths: Mathematics) -> None:
        joint = [[0.12, 0.28], [0.18, 0.42]]
        assert maths.compute_mutual_information(
            joint, [0.4, 0.6], [0.3, 0.7]
        ) == pytest.approx(0.0, abs=1e-12)

    def test_mutual_information_perfect_correlation(self, maths: Mathematics) -> None:
        joint = [[0.5, 0.0], [0.0, 0.5]]
        assert maths.compute_mutual_information(
            joint, [0.5, 0.5], [0.5, 0.5]
        ) == pytest.approx(1.0)

    def test_mutual_information_fallback_matches_numpy(
        self, maths: Mathematics, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        joint = [[0.1, 0.2], [0.3, 0.4]]
        mx, my = [0.3, 0.7], [0.4, 0.6]
        with_numpy = maths.compute_mutual_information(joint, mx, my)
        monkeypatch.setattr(math_mod, "HAS_NUMPY", False)
        assert maths.compute_mutual_information(joint, mx, my) == pytest.approx(
            with_numpy
        )
        perfect = [[0.5, 0.0], [0.0, 0.5]]
        assert maths.compute_mutual_information(
            perfect, [0.5, 0.5], [0.5, 0.5]
        ) == pytest.approx(1.0)


class TestOptimisation:
    def test_simple_arithmetic(self, maths: Mathematics) -> None:
        assert maths.compute_variational_free_energy(-1.5, 0.5) == -2.0
        assert maths.compute_expected_free_energy_components(1.0, 2.0, 0.5) == {
            "risk": 1.0,
            "ambiguity": 2.0,
            "value": 0.5,
            "total_efe": 2.5,
        }

    def test_optimize_numpy(self, maths: Mathematics) -> None:
        result = maths.optimize_free_energy([1.5, 0.5, -0.5], num_iterations=10)
        decay = (1 - 0.001) ** 10
        assert result == pytest.approx([0.5 + decay, 0.5, 0.5 - decay])

    def test_optimize_fallback_matches(
        self, maths: Mathematics, no_numpy: None
    ) -> None:
        start = [1.5, 0.5, -0.5]
        result = maths.optimize_free_energy(start, num_iterations=10)
        decay = (1 - 0.001) ** 10
        assert result == pytest.approx([0.5 + decay, 0.5, 0.5 - decay])
        assert start == [1.5, 0.5, -0.5]
        assert maths.optimize_free_energy([2.0], num_iterations=0) == [2.0]


class TestSymbolicValidation:
    def test_sympy_simplifies_and_reports_failures(
        self, maths: Mathematics, fake_sympy: ModuleType
    ) -> None:
        result = maths.validate_mathematical_consistency(["x + x", "(", "log(x)"])
        assert result["valid"] is False
        assert result["simplified_expressions"] == ["2*x", "log(x)"]
        assert result["warnings"] == []
        assert result["issues"] == [
            "Failed to parse expression '(': unbalanced parentheses"
        ]

    def test_sympy_log_zero_warning(
        self, maths: Mathematics, fake_sympy: ModuleType
    ) -> None:
        result = maths.validate_mathematical_consistency(["log(0*x)"])
        assert result["warnings"] == ["Potential log(0) in: log(0*x)"]
        assert result["simplified_expressions"] == ["log(0*x)"]
        assert result["valid"] is True

    def test_fallback_warnings(self, maths: Mathematics, no_sympy: None) -> None:
        result = maths.validate_mathematical_consistency(
            ["log(0)", "divide x by 0", "x + y"]
        )
        assert result == {
            "valid": True,
            "issues": [],
            "warnings": [
                "Potential log(0) issue in: log(0)",
                "Potential division by zero in: divide x by 0",
            ],
        }


class TestDerivationsAndGeometry:
    def test_known_and_unknown_derivations(self, maths: Mathematics) -> None:
        assert maths.generate_mathematical_derivation("entropy").startswith(
            "Step-by-step derivation of entropy"
        )
        assert "Gibbs' inequality" in maths.generate_mathematical_derivation(
            "kl_divergence"
        )
        assert "Jensen's inequality" in maths.generate_mathematical_derivation(
            "variational_free_energy"
        )
        assert (
            maths.generate_mathematical_derivation("topology")
            == "Derivation for topology concept"
        )

    def test_fisher_metric_numpy(self, maths: Mathematics) -> None:
        assert maths.compute_information_geometry_metric([0.5, 0.25, 0.0]) == [
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
        ]

    def test_fisher_metric_fallback(self, maths: Mathematics, no_numpy: None) -> None:
        assert maths.compute_information_geometry_metric([0.5, 0.25]) == [
            [2.0, 0.0],
            [0.0, 4.0],
        ]

    def test_complexity_analysis(self, maths: Mathematics) -> None:
        metrics = maths.analyze_mathematical_complexity("log(exp(x+y))*∫z")
        assert metrics["num_operators"] == 2
        assert metrics["num_functions"] == 2
        assert metrics["num_parentheses"] == 4
        assert metrics["num_integrals"] == 1
        assert metrics["max_nesting_depth"] == 2
        assert metrics["expression_length"] == len("log(exp(x+y))*∫z")
        expected = (
            2 * 0.5 + 2 * 1.0 + metrics["num_variables"] * 0.3 + 4 * 0.2 + 1 * 2.0
        )
        assert metrics["total_complexity"] == pytest.approx(expected)


class TestPrerequisites:
    def test_valid_chain_reports_redundant(self, maths: Mathematics) -> None:
        result = maths.validate_mathematical_prerequisites(
            ["basic_probability", "bayesian_inference"]
        )
        assert result["valid"] is True
        assert result["missing_prerequisites"] == []
        assert result["circular_dependencies"] == []
        assert "probability_basics" in result["redundant_prerequisites"]

    def test_unknown_concepts_are_missing(self, maths: Mathematics) -> None:
        result = maths.validate_mathematical_prerequisites(["quantum_gravity"])
        assert result["missing_prerequisites"] == ["quantum_gravity"]
        assert result["valid"] is True

    def test_circular_dependency_detected(
        self, maths: Mathematics, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            maths,
            "get_mathematical_prerequisites",
            lambda: {"a": ["b"], "b": ["a"], "c": []},
        )
        result = maths.validate_mathematical_prerequisites(["a", "c"])
        assert result["circular_dependencies"] == ["a"]
        assert result["valid"] is False

    def test_summary_lists_sequence_and_network(
        self, maths: Mathematics, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            maths, "create_mathematical_learning_path", lambda: ["a_b", "c"]
        )
        monkeypatch.setattr(
            maths, "get_mathematical_prerequisites", lambda: {"c": ["a_b"], "d": []}
        )
        lines = maths.generate_mathematical_summary().splitlines()
        assert lines[0] == "Mathematical Foundations Summary"
        assert "Total Mathematical Concepts: 2" in lines
        assert "Total Prerequisite Relationships: 1" in lines
        assert " 1. A B" in lines
        assert "    Prerequisites: a_b" in lines
        assert lines[-1] == "c: a_b"
        assert "d: " not in "\n".join(lines)

    def test_default_summary(self, maths: Mathematics) -> None:
        summary = maths.generate_mathematical_summary()
        assert "Total Mathematical Concepts: 11" in summary
        assert (
            "free_energy_principle: bayesian_mathematics -> information_theory"
            in summary
        )
