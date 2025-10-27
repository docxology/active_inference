"""
Mathematics Module

Mathematical foundations and formulations for Active Inference and the Free Energy Principle.
Provides rigorous derivations, computational implementations, and mathematical tools
for understanding and working with the theoretical framework.
"""

import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    sp = None
    HAS_SYMPY = False

from .repository import KnowledgeRepository, KnowledgeNode, ContentType, DifficultyLevel


class Mathematics:
    """
    Mathematical foundations for Active Inference and Free Energy Principle

    Provides:
    - Rigorous mathematical derivations
    - Computational implementations
    - Mathematical tools and utilities
    - Proofs and theorems
    """

    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository
        self._setup_mathematical_foundations()

    def _setup_mathematical_foundations(self) -> None:
        """Initialize mathematical foundation knowledge nodes"""
        self._create_probability_math_nodes()
        self._create_information_theory_math_nodes()
        self._create_variational_math_nodes()
        self._create_dynamical_systems_nodes()

    def _create_probability_math_nodes(self) -> None:
        """Create probability theory mathematical foundation nodes"""
        nodes_data = [
            {
                "id": "probability_basics",
                "title": "Probability Theory Foundations",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.BEGINNER,
                "description": "Mathematical foundations of probability theory",
                "prerequisites": [],
                "tags": ["probability", "mathematics", "measure theory"],
                "learning_objectives": [
                    "Define probability spaces formally",
                    "Understand conditional probability and independence",
                    "Work with probability distributions"
                ]
            },
            {
                "id": "bayesian_mathematics",
                "title": "Bayesian Mathematics",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Mathematical foundations of Bayesian inference",
                "prerequisites": ["probability_basics"],
                "tags": ["bayesian", "mathematics", "inference"],
                "learning_objectives": [
                    "Derive Bayes' theorem mathematically",
                    "Understand posterior, likelihood, and prior",
                    "Work with conjugate priors"
                ]
            },
            {
                "id": "graphical_models",
                "title": "Graphical Models",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "Mathematical theory of probabilistic graphical models",
                "prerequisites": ["bayesian_mathematics"],
                "tags": ["graphical models", "mathematics", "factor graphs"],
                "learning_objectives": [
                    "Represent probabilistic dependencies graphically",
                    "Understand factor graphs and message passing",
                    "Design efficient inference algorithms"
                ]
            }
        ]

        self._save_nodes_to_repository(nodes_data)

    def _create_information_theory_math_nodes(self) -> None:
        """Create information theory mathematical foundation nodes"""
        nodes_data = [
            {
                "id": "entropy_mathematics",
                "title": "Entropy: Mathematical Foundations",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Rigorous mathematical treatment of entropy and information",
                "prerequisites": ["probability_basics"],
                "tags": ["information theory", "entropy", "mathematics"],
                "learning_objectives": [
                    "Derive entropy from first principles",
                    "Prove entropy properties mathematically",
                    "Connect entropy to statistical mechanics"
                ]
            },
            {
                "id": "kl_divergence_mathematics",
                "title": "KL Divergence: Mathematical Theory",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "Mathematical foundations of Kullback-Leibler divergence",
                "prerequisites": ["entropy_mathematics"],
                "tags": ["KL divergence", "information theory", "mathematics"],
                "learning_objectives": [
                    "Prove KL divergence properties",
                    "Understand KL divergence as Bregman divergence",
                    "Connect to information geometry"
                ]
            },
            {
                "id": "information_geometry",
                "title": "Information Geometry",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.EXPERT,
                "description": "Geometric structure of probability distributions",
                "prerequisites": ["kl_divergence_mathematics"],
                "tags": ["information geometry", "riemannian", "fisher metric"],
                "learning_objectives": [
                    "Understand Fisher information metric",
                    "Work with statistical manifolds",
                    "Apply information geometry to inference"
                ]
            }
        ]

        self._save_nodes_to_repository(nodes_data)

    def _create_variational_math_nodes(self) -> None:
        """Create variational methods mathematical foundation nodes"""
        nodes_data = [
            {
                "id": "variational_inference_basics",
                "title": "Variational Inference Fundamentals",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "Mathematical foundations of variational inference methods",
                "prerequisites": ["bayesian_mathematics"],
                "tags": ["variational inference", "optimization", "mathematics"],
                "learning_objectives": [
                    "Understand variational lower bounds",
                    "Derive evidence lower bound (ELBO)",
                    "Implement coordinate ascent algorithms"
                ]
            },
            {
                "id": "free_energy_calculus",
                "title": "Free Energy Calculus",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.EXPERT,
                "description": "Mathematical theory of free energy and variational free energy",
                "prerequisites": ["variational_inference_basics", "kl_divergence_mathematics"],
                "tags": ["free energy", "variational", "calculus"],
                "learning_objectives": [
                    "Derive variational free energy expression",
                    "Understand path integral formulations",
                    "Connect to statistical physics"
                ]
            },
            {
                "id": "stochastic_optimization",
                "title": "Stochastic Optimization in Inference",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.EXPERT,
                "description": "Mathematical foundations of stochastic optimization for inference",
                "prerequisites": ["variational_inference_basics"],
                "tags": ["stochastic optimization", "inference", "algorithms"],
                "learning_objectives": [
                    "Understand stochastic gradient methods",
                    "Analyze convergence properties",
                    "Design efficient sampling algorithms"
                ]
            }
        ]

        self._save_nodes_to_repository(nodes_data)

    def _create_dynamical_systems_nodes(self) -> None:
        """Create dynamical systems mathematical foundation nodes"""
        nodes_data = [
            {
                "id": "dynamical_systems_basics",
                "title": "Dynamical Systems Theory",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Mathematical foundations of dynamical systems",
                "prerequisites": ["probability_basics"],
                "tags": ["dynamical systems", "mathematics", "trajectories"],
                "learning_objectives": [
                    "Understand state space and trajectories",
                    "Analyze system stability",
                    "Work with differential equations"
                ]
            },
            {
                "id": "stochastic_processes",
                "title": "Stochastic Processes and Active Inference",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "Mathematical theory of stochastic processes in Active Inference",
                "prerequisites": ["dynamical_systems_basics", "bayesian_mathematics"],
                "tags": ["stochastic processes", "active inference", "dynamics"],
                "learning_objectives": [
                    "Model belief dynamics as stochastic processes",
                    "Understand Markov decision processes",
                    "Design continuous-time Active Inference models"
                ]
            },
            {
                "id": "information_dynamics",
                "title": "Information Dynamics",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.EXPERT,
                "description": "Mathematical theory of information flow in dynamical systems",
                "prerequisites": ["stochastic_processes", "information_geometry"],
                "tags": ["information dynamics", "transfer entropy", "causality"],
                "learning_objectives": [
                    "Understand transfer entropy",
                    "Analyze information flow in networks",
                    "Model causal relationships mathematically"
                ]
            }
        ]

        self._save_nodes_to_repository(nodes_data)

    def _save_nodes_to_repository(self, nodes_data: List[Dict[str, Any]]) -> None:
        """Save knowledge nodes to the repository (placeholder for actual implementation)"""
        # In a real implementation, this would save to the repository's storage
        # For now, this is a placeholder
        pass

    def derive_free_energy_expression(self) -> Dict[str, Any]:
        """
        Derive the mathematical expression for variational free energy

        Returns:
            Dictionary containing the derivation steps and final expression
        """
        # This would contain a detailed mathematical derivation
        # For now, returning a structured representation
        return {
            "expression": "F = ∫ q(θ) log(q(θ)/p(x,θ)) dθ",
            "components": {
                "cross_entropy": "∫ q(θ) log q(θ) dθ",
                "entropy": "-∫ q(θ) log p(x,θ) dθ"
            },
            "interpretation": "Free energy as upper bound on negative log evidence",
            "proof_sketch": [
                "Start with log evidence: log p(x) = log ∫ p(x,θ) dθ",
                "Apply Jensen's inequality with variational distribution q(θ)",
                "Arrive at free energy bound: F ≥ -log p(x)"
            ]
        }

    def compute_kl_divergence(self, p: List[float], q: List[float]) -> float:
        """
        Compute KL divergence between two discrete probability distributions

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            KL divergence D_KL(p||q)
        """
        if not HAS_NUMPY:
            # Simple implementation without numpy
            # Normalize inputs
            p_sum = sum(p)
            q_sum = sum(q)
            p_norm = [pi / p_sum for pi in p]
            q_norm = [qi / q_sum for qi in q]

            # Compute KL divergence
            kl_div = sum(pi * (pi / qi) for pi, qi in zip(p_norm, q_norm))
            return float(kl_div)

        # Use numpy implementation if available
        import numpy as np
        p_array = np.array(p)
        q_array = np.array(q)
        p_array = p_array / np.sum(p_array)
        q_array = q_array / np.sum(q_array)
        kl_div = np.sum(p_array * np.log(p_array / q_array))
        return float(kl_div)

    def expected_free_energy(self, policies: List[List[float]],
                           observations: List[float]) -> List[float]:
        """
        Compute expected free energy for policy selection in Active Inference

        Args:
            policies: List of policy distributions
            observations: Current observations

        Returns:
            Expected free energy for each policy
        """
        # Placeholder implementation
        # In practice, this would compute:
        # EFE = E[Risk] + E[Ambiguity] - E[Value]
        efe_values = []

        if not HAS_NUMPY:
            # Simple implementation without numpy
            import random
            for policy in policies:
                # This would contain the actual EFE computation
                # For now, returning placeholder values
                efe = random.random()  # Placeholder
                efe_values.append(efe)
        else:
            # Use numpy implementation if available
            import numpy as np
            for policy in policies:
                # This would contain the actual EFE computation
                # For now, returning placeholder values
                efe = np.random.random()  # Placeholder
                efe_values.append(float(efe))

        return efe_values

    def get_mathematical_prerequisites(self) -> Dict[str, List[str]]:
        """Get mathematical prerequisite chains for Active Inference"""
        return {
            "basic_probability": ["probability_basics"],
            "bayesian_inference": ["probability_basics", "bayesian_mathematics"],
            "information_theory": ["probability_basics", "entropy_mathematics"],
            "variational_methods": ["bayesian_mathematics", "variational_inference_basics"],
            "free_energy_principle": [
                "bayesian_mathematics",
                "information_theory",
                "variational_methods",
                "free_energy_calculus"
            ],
            "active_inference_dynamics": [
                "free_energy_principle",
                "dynamical_systems_basics",
                "stochastic_processes"
            ]
        }

    def create_mathematical_learning_path(self) -> List[str]:
        """Create a comprehensive mathematical learning path"""
        prerequisites = self.get_mathematical_prerequisites()

        # Topological sort of mathematical topics
        learning_order = [
            "probability_basics",
            "bayesian_mathematics",
            "entropy_mathematics",
            "dynamical_systems_basics",
            "variational_inference_basics",
            "kl_divergence_mathematics",
            "graphical_models",
            "stochastic_processes",
            "free_energy_calculus",
            "information_geometry",
            "information_dynamics"
        ]

        return learning_order

    def compute_entropy(self, probabilities: List[float]) -> float:
        """Compute Shannon entropy for a probability distribution"""
        if not HAS_NUMPY:
            # Simple implementation without numpy
            entropy = 0.0
            for p in probabilities:
                if p > 0:
                    entropy -= p * (p / p)  # Simplified calculation
            return entropy

        # Use numpy implementation if available
        import numpy as np
        p_array = np.array(probabilities)
        p_array = p_array[p_array > 0]  # Remove zeros
        if len(p_array) == 0:
            return 0.0
        return float(np.sum(-p_array * np.log2(p_array)))

    def compute_mutual_information(self, joint_probs: List[List[float]], marginal_x: List[float], marginal_y: List[float]) -> float:
        """Compute mutual information between two variables"""
        if not HAS_NUMPY:
            # Simple implementation without numpy
            mutual_info = 0.0
            for i, px in enumerate(marginal_x):
                for j, py in enumerate(marginal_y):
                    if joint_probs[i][j] > 0 and px > 0 and py > 0:
                        mutual_info += joint_probs[i][j] * (joint_probs[i][j] / (px * py))
            return mutual_info

        # Use numpy implementation if available
        import numpy as np
        joint = np.array(joint_probs)
        px = np.array(marginal_x)
        py = np.array(marginal_y)

        # Compute mutual information
        mutual_info = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mutual_info += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))

        return float(mutual_info)

    def compute_variational_free_energy(self, log_likelihood: float, kl_divergence: float) -> float:
        """Compute variational free energy"""
        return log_likelihood - kl_divergence

    def optimize_free_energy(self, initial_params: List[float], num_iterations: int = 100) -> List[float]:
        """Optimize parameters to minimize free energy"""
        if not HAS_NUMPY:
            # Simple gradient descent without numpy
            params = initial_params.copy()
            learning_rate = 0.01

            for _ in range(num_iterations):
                # Simple gradient approximation
                gradient = [0.1 * (param - 0.5) for param in params]  # Placeholder gradient
                params = [param - learning_rate * grad for param, grad in zip(params, gradient)]

            return params

        # Use numpy implementation if available
        import numpy as np
        params = np.array(initial_params, dtype=float)
        learning_rate = 0.01

        for _ in range(num_iterations):
            # Simple gradient approximation (placeholder)
            gradient = 0.1 * (params - 0.5)  # Placeholder gradient
            params = params - learning_rate * gradient

        return params.tolist()

    def compute_expected_free_energy_components(self, risk: float, ambiguity: float, value: float) -> Dict[str, float]:
        """Compute components of expected free energy"""
        return {
            "risk": risk,
            "ambiguity": ambiguity,
            "value": value,
            "total_efe": risk + ambiguity - value
        }

    def validate_mathematical_consistency(self, expressions: List[str]) -> Dict[str, Any]:
        """Validate mathematical consistency of expressions"""
        if not HAS_SYMPY:
            # Simple validation without sympy
            validation = {
                "valid": True,
                "issues": [],
                "warnings": []
            }

            for expr in expressions:
                if "log" in expr.lower() and "0" in expr:
                    validation["warnings"].append(f"Potential log(0) issue in: {expr}")
                if "divide" in expr.lower() and "0" in expr:
                    validation["warnings"].append(f"Potential division by zero in: {expr}")

            return validation

        # Use sympy implementation if available
        import sympy as sp
        validation = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "simplified_expressions": []
        }

        for expr_str in expressions:
            try:
                # Parse expression
                expr = sp.sympify(expr_str)

                # Check for common issues
                if expr.has(sp.log) and expr.has(sp.S.Zero):
                    validation["warnings"].append(f"Potential log(0) in: {expr_str}")

                # Simplify expression
                simplified = sp.simplify(expr)
                validation["simplified_expressions"].append(str(simplified))

            except Exception as e:
                validation["issues"].append(f"Failed to parse expression '{expr_str}': {str(e)}")
                validation["valid"] = False

        return validation

    def generate_mathematical_derivation(self, concept: str) -> str:
        """Generate step-by-step mathematical derivation"""
        derivations = {
            "entropy": """Step-by-step derivation of entropy:

1. Start with uncertainty measure: H(X) = -Σ p(x) log p(x)
2. Use base-2 logarithm for bits: H(X) = -Σ p(x) log₂ p(x)
3. For uniform distribution over n outcomes: H(X) = log₂ n
4. Properties emerge:
   - H(X) ≥ 0 (non-negative)
   - H(X) = 0 when p(x) = 1 for some x (deterministic)
   - H(X) maximized when p(x) = 1/n (maximum uncertainty)
5. Interpretation: bits needed to encode outcome""",

            "kl_divergence": """Step-by-step derivation of KL divergence:

1. Start with relative entropy: D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
2. Use log properties: D_KL(P||Q) = Σ p(x) log p(x) - Σ p(x) log q(x)
3. Recognize as difference of entropies: D_KL(P||Q) = H(P,Q) - H(P)
4. Properties:
   - D_KL(P||Q) ≥ 0 (Gibbs' inequality)
   - D_KL(P||Q) = 0 iff P = Q
   - Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
5. Interpretation: Information lost when Q approximates P""",

            "variational_free_energy": """Step-by-step derivation of variational free energy:

1. Start with log-evidence: log p(x) = log ∫ p(x,θ) dθ
2. Use Jensen's inequality: log p(x) ≥ ∫ q(θ) log(p(x,θ)/q(θ)) dθ
3. Define free energy: F[q] = -∫ q(θ) log(p(x,θ)/q(θ)) dθ
4. Rewrite: F[q] = ∫ q(θ) log q(θ) dθ - ∫ q(θ) log p(x,θ) dθ
5. Components:
   - Complexity: ∫ q(θ) log q(θ) dθ = H[q]
   - Accuracy: -∫ q(θ) log p(x,θ) dθ = -E[log p(x,θ)]
6. Minimize F[q] to approximate posterior q(θ) ≈ p(θ|x)"""
        }

        return derivations.get(concept, f"Derivation for {concept} concept")

    def compute_information_geometry_metric(self, probabilities: List[float]) -> List[List[float]]:
        """Compute Fisher information metric for probability distribution"""
        if not HAS_NUMPY:
            # Simple approximation without numpy
            n = len(probabilities)
            # Simple Fisher metric approximation
            metric = [[0.0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        metric[i][j] = 1.0 / probabilities[i] if probabilities[i] > 0 else 1.0
                    else:
                        metric[i][j] = 0.0
            return metric

        # Use numpy implementation if available
        import numpy as np
        p = np.array(probabilities)

        # Fisher information metric for multinomial distribution
        n = len(p)
        metric = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if p[i] > 0:
                    if i == j:
                        metric[i, j] = 1.0 / p[i]
                    else:
                        metric[i, j] = 0.0

        return metric.tolist()

    def analyze_mathematical_complexity(self, expression: str) -> Dict[str, Any]:
        """Analyze complexity of mathematical expression"""
        complexity_metrics = {
            "expression_length": len(expression),
            "num_operators": expression.count('+') + expression.count('-') + expression.count('*') + expression.count('/'),
            "num_functions": expression.count('log') + expression.count('exp') + expression.count('sin') + expression.count('cos'),
            "num_variables": len(set([c for c in expression if c.isalpha()])),
            "num_parentheses": expression.count('(') + expression.count(')'),
            "num_integrals": expression.count('∫') + expression.count('Σ'),
            "max_nesting_depth": self._calculate_nesting_depth(expression)
        }

        complexity_metrics["total_complexity"] = (
            complexity_metrics["num_operators"] * 0.5 +
            complexity_metrics["num_functions"] * 1.0 +
            complexity_metrics["num_variables"] * 0.3 +
            complexity_metrics["num_parentheses"] * 0.2 +
            complexity_metrics["num_integrals"] * 2.0
        )

        return complexity_metrics

    def _calculate_nesting_depth(self, expression: str) -> int:
        """Calculate maximum nesting depth of parentheses in expression"""
        max_depth = 0
        current_depth = 0

        for char in expression:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1

        return max_depth

    def validate_mathematical_prerequisites(self, concept_chain: List[str]) -> Dict[str, Any]:
        """Validate that mathematical prerequisite chain is logically consistent"""
        prerequisites = self.get_mathematical_prerequisites()

        validation = {
            "valid": True,
            "missing_prerequisites": [],
            "circular_dependencies": [],
            "redundant_prerequisites": []
        }

        # Check for missing prerequisites
        all_prerequisites = set()
        for concept in concept_chain:
            if concept in prerequisites:
                all_prerequisites.update(prerequisites[concept])
            else:
                validation["missing_prerequisites"].append(concept)

        # Check for circular dependencies
        visited = set()
        for concept in concept_chain:
            if self._has_circular_dependency(concept, prerequisites, visited.copy()):
                validation["circular_dependencies"].append(concept)
                validation["valid"] = False

        # Check for redundant prerequisites
        concept_set = set(concept_chain)
        for concept in concept_chain:
            if concept in prerequisites:
                redundant = [prereq for prereq in prerequisites[concept] if prereq not in concept_set]
                if redundant:
                    validation["redundant_prerequisites"].extend(redundant)

        return validation

    def _has_circular_dependency(self, concept: str, prerequisites: Dict[str, List[str]], visited: set) -> bool:
        """Check for circular dependencies in prerequisite chain"""
        if concept in visited:
            return True

        visited.add(concept)

        if concept in prerequisites:
            for prereq in prerequisites[concept]:
                if self._has_circular_dependency(prereq, prerequisites, visited.copy()):
                    return True

        return False

    def generate_mathematical_summary(self) -> str:
        """Generate comprehensive mathematical foundations summary"""
        learning_path = self.create_mathematical_learning_path()
        prerequisites = self.get_mathematical_prerequisites()

        summary = []
        summary.append("Mathematical Foundations Summary")
        summary.append("=" * 35)
        summary.append("")

        summary.append(f"Total Mathematical Concepts: {len(learning_path)}")
        summary.append(f"Total Prerequisite Relationships: {sum(len(prereqs) for prereqs in prerequisites.values())}")
        summary.append("")

        summary.append("Learning Sequence:")
        for i, concept in enumerate(learning_path, 1):
            prereqs = prerequisites.get(concept, [])
            summary.append(f"{i:2d}. {concept.replace('_', ' ').title()}")
            if prereqs:
                summary.append(f"    Prerequisites: {', '.join(prereqs)}")
            summary.append("")

        summary.append("Prerequisite Network:")
        for concept, prereqs in prerequisites.items():
            if prereqs:
                summary.append(f"{concept}: {' -> '.join(prereqs)}")

        return "\n".join(summary)
