"""
Foundations Module

Core theoretical foundations of Active Inference and the Free Energy Principle.
Provides structured educational content covering the fundamental concepts,
principles, and theoretical framework.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from .repository import KnowledgeRepository, KnowledgeNode, ContentType, DifficultyLevel


class Foundations:
    """
    Foundations of Active Inference and the Free Energy Principle

    Provides structured access to fundamental concepts including:
    - Information theory basics
    - Bayesian inference
    - Free Energy Principle
    - Active Inference framework
    """

    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository
        self._setup_foundations()

    def _setup_foundations(self) -> None:
        """Initialize foundation knowledge nodes"""
        self._create_information_theory_nodes()
        self._create_bayesian_inference_nodes()
        self._create_free_energy_nodes()
        self._create_active_inference_nodes()

    def _create_information_theory_nodes(self) -> None:
        """Create information theory foundation nodes"""
        nodes_data = [
            {
                "id": "info_theory_entropy",
                "title": "Entropy and Information",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.BEGINNER,
                "description": "Understanding entropy as a measure of uncertainty and information content",
                "prerequisites": [],
                "tags": ["information theory", "entropy", "uncertainty"],
                "learning_objectives": [
                    "Define entropy mathematically",
                    "Understand entropy as uncertainty measure",
                    "Calculate entropy for simple distributions"
                ]
            },
            {
                "id": "info_theory_kl_divergence",
                "title": "Kullback-Leibler Divergence",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Measuring the difference between probability distributions",
                "prerequisites": ["info_theory_entropy"],
                "tags": ["information theory", "KL divergence", "distance"],
                "learning_objectives": [
                    "Define KL divergence mathematically",
                    "Understand KL divergence as distribution distance",
                    "Apply KL divergence in model comparison"
                ]
            },
            {
                "id": "info_theory_mutual_information",
                "title": "Mutual Information",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Quantifying dependence between random variables",
                "prerequisites": ["info_theory_entropy"],
                "tags": ["information theory", "mutual information", "dependence"],
                "learning_objectives": [
                    "Define mutual information",
                    "Understand information sharing between variables",
                    "Calculate mutual information for discrete variables"
                ]
            }
        ]

        self._save_nodes_to_repository(nodes_data)

    def _create_bayesian_inference_nodes(self) -> None:
        """Create Bayesian inference foundation nodes"""
        nodes_data = [
            {
                "id": "bayesian_basics",
                "title": "Bayesian Probability",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.BEGINNER,
                "description": "Fundamental concepts of Bayesian probability and inference",
                "prerequisites": [],
                "tags": ["bayesian", "probability", "inference"],
                "learning_objectives": [
                    "Understand subjective probability interpretation",
                    "Apply Bayes' theorem",
                    "Update beliefs with new evidence"
                ]
            },
            {
                "id": "bayesian_models",
                "title": "Bayesian Models",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Building and working with Bayesian probabilistic models",
                "prerequisites": ["bayesian_basics"],
                "tags": ["bayesian", "models", "generative"],
                "learning_objectives": [
                    "Define generative models",
                    "Understand model parameters vs observations",
                    "Work with hierarchical Bayesian models"
                ]
            },
            {
                "id": "belief_updating",
                "title": "Belief Updating",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Mechanisms for updating probabilistic beliefs over time",
                "prerequisites": ["bayesian_basics"],
                "tags": ["bayesian", "updating", "dynamics"],
                "learning_objectives": [
                    "Implement belief updating algorithms",
                    "Understand convergence properties",
                    "Handle sequential data streams"
                ]
            }
        ]

        self._save_nodes_to_repository(nodes_data)

    def _create_free_energy_nodes(self) -> None:
        """Create Free Energy Principle foundation nodes"""
        nodes_data = [
            {
                "id": "fep_introduction",
                "title": "Free Energy Principle Overview",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Introduction to the Free Energy Principle as a unifying theory",
                "prerequisites": ["info_theory_kl_divergence", "bayesian_models"],
                "tags": ["free energy principle", "theory", "unification"],
                "learning_objectives": [
                    "State the Free Energy Principle",
                    "Understand free energy as prediction error",
                    "Connect to information theory and Bayesian inference"
                ]
            },
            {
                "id": "fep_mathematical_formulation",
                "title": "Mathematical Foundations of FEP",
                "content_type": ContentType.MATHEMATICS,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "Rigorous mathematical formulation of the Free Energy Principle",
                "prerequisites": ["fep_introduction", "info_theory_kl_divergence"],
                "tags": ["free energy principle", "mathematics", "variational"],
                "learning_objectives": [
                    "Derive free energy expression mathematically",
                    "Understand variational free energy",
                    "Connect to information geometry"
                ]
            },
            {
                "id": "fep_biological_systems",
                "title": "FEP in Biological Systems",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "How the Free Energy Principle explains biological self-organization",
                "prerequisites": ["fep_introduction"],
                "tags": ["free energy principle", "biology", "self-organization"],
                "learning_objectives": [
                    "Apply FEP to biological systems",
                    "Understand homeostasis as free energy minimization",
                    "Connect to evolutionary principles"
                ]
            }
        ]

        self._save_nodes_to_repository(nodes_data)

    def _create_active_inference_nodes(self) -> None:
        """Create Active Inference foundation nodes"""
        nodes_data = [
            {
                "id": "active_inference_introduction",
                "title": "Active Inference Overview",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Introduction to Active Inference as a framework for behavior",
                "prerequisites": ["fep_introduction", "belief_updating"],
                "tags": ["active inference", "behavior", "planning"],
                "learning_objectives": [
                    "Define Active Inference framework",
                    "Understand planning as inference",
                    "Connect to decision theory"
                ]
            },
            {
                "id": "ai_generative_models",
                "title": "Generative Models in Active Inference",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "Structure and role of generative models in Active Inference",
                "prerequisites": ["active_inference_introduction", "bayesian_models"],
                "tags": ["active inference", "generative models", "representation"],
                "learning_objectives": [
                    "Design generative models for Active Inference",
                    "Understand model structure requirements",
                    "Implement hierarchical generative models"
                ]
            },
            {
                "id": "ai_policy_selection",
                "title": "Policy Selection and Planning",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "How Active Inference agents select actions and plan behavior",
                "prerequisites": ["active_inference_introduction", "ai_generative_models"],
                "tags": ["active inference", "planning", "policies"],
                "learning_objectives": [
                    "Understand expected free energy",
                    "Implement policy selection mechanisms",
                    "Design planning algorithms"
                ]
            }
        ]

        self._save_nodes_to_repository(nodes_data)

    def _save_nodes_to_repository(self, nodes_data: List[Dict[str, Any]]) -> None:
        """Save knowledge nodes to the repository"""
        for node_data in nodes_data:
            # Add comprehensive content for each node
            node_data["content"] = self._generate_node_content(node_data)
            node_data["metadata"] = {
                "author": "Active Inference Community",
                "version": "1.0",
                "estimated_reading_time": 15,
                "last_updated": "2024-10-27"
            }

            # Create knowledge node and add to repository
            try:
                self.repository._nodes[node_data["id"]] = KnowledgeNode(**node_data)
            except Exception as e:
                print(f"Warning: Failed to create foundation node {node_data['id']}: {e}")

    def _generate_node_content(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive content for foundation nodes"""
        node_id = node_data["id"]

        # Generate content based on node type
        if node_id == "info_theory_entropy":
            return {
                "overview": "Entropy measures the uncertainty or disorder in a probability distribution. It quantifies how much information is needed to describe the outcome of a random variable.",
                "mathematical_definition": "For a discrete random variable X with probability mass function p(x), the entropy H(X) is defined as:\n\nH(X) = -Σ_x p(x) log₂ p(x)\n\nFor continuous variables, differential entropy is used:\n\nh(X) = -∫ p(x) log p(x) dx",
                "properties": [
                    "Entropy is always non-negative: H(X) ≥ 0",
                    "Entropy is maximized for uniform distributions",
                    "Entropy measures uncertainty, not complexity",
                    "Units are typically bits (log₂) or nats (logₑ)"
                ],
                "examples": [
                    {
                        "name": "Fair Coin",
                        "description": "A fair coin has maximum entropy",
                        "calculation": "H = -0.5×log₂(0.5) - 0.5×log₂(0.5) = 1 bit",
                        "interpretation": "1 bit of information is needed to describe the outcome"
                    },
                    {
                        "name": "Biased Coin",
                        "description": "A biased coin has lower entropy",
                        "calculation": "H = -0.9×log₂(0.9) - 0.1×log₂(0.1) ≈ 0.47 bits",
                        "interpretation": "Less information needed due to predictability"
                    }
                ],
                "applications": [
                    "Data compression (entropy coding)",
                    "Cryptography (key strength measurement)",
                    "Machine learning (feature selection)",
                    "Neuroscience (information processing)",
                    "Active Inference (uncertainty quantification)"
                ]
            }

        elif node_id == "info_theory_kl_divergence":
            return {
                "overview": "Kullback-Leibler divergence measures how much a probability distribution Q differs from a reference distribution P. It quantifies the information lost when Q is used to approximate P.",
                "mathematical_definition": "For discrete distributions:\n\nD_KL(P||Q) = Σ_x p(x) log(p(x)/q(x))\n\nFor continuous distributions:\n\nD_KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx",
                "properties": [
                    "D_KL(P||Q) ≥ 0 (Gibbs' inequality)",
                    "D_KL(P||Q) ≠ D_KL(Q||P) (asymmetric)",
                    "D_KL(P||P) = 0 (minimum at equality)",
                    "Measures relative entropy between distributions"
                ],
                "examples": [
                    {
                        "name": "Model Comparison",
                        "description": "Compare true distribution vs model approximation",
                        "interpretation": "KL divergence quantifies model accuracy"
                    },
                    {
                        "name": "Free Energy Principle",
                        "description": "KL divergence between posterior and prior beliefs",
                        "interpretation": "Measures surprise or prediction error"
                    }
                ],
                "applications": [
                    "Model selection and comparison",
                    "Variational inference",
                    "Statistical hypothesis testing",
                    "Information geometry",
                    "Active Inference (surprise minimization)"
                ]
            }

        elif node_id == "info_theory_mutual_information":
            return {
                "overview": "Mutual information measures the amount of information shared between two random variables. It quantifies how much knowing one variable reduces uncertainty about the other.",
                "mathematical_definition": "I(X;Y) = H(X) + H(Y) - H(X,Y)\n\nI(X;Y) = Σ_{x,y} p(x,y) log(p(x,y)/(p(x)p(y)))\n\nI(X;Y) = D_KL(p(x,y) || p(x)p(y))",
                "properties": [
                    "I(X;Y) ≥ 0 (non-negative)",
                    "I(X;Y) = I(Y;X) (symmetric)",
                    "I(X;Y) = 0 if X and Y are independent",
                    "I(X;X) = H(X) (self-information equals entropy)"
                ],
                "examples": [
                    {
                        "name": "Feature Selection",
                        "description": "Mutual information for selecting relevant features",
                        "interpretation": "Higher mutual information indicates better feature relevance"
                    },
                    {
                        "name": "Communication Channels",
                        "description": "Information transmission through noisy channels",
                        "interpretation": "Mutual information quantifies channel capacity"
                    }
                ],
                "applications": [
                    "Feature selection in machine learning",
                    "Information bottleneck method",
                    "Causal discovery",
                    "Network analysis",
                    "Active Inference (information flow)"
                ]
            }

        elif node_id == "bayesian_basics":
            return {
                "overview": "Bayesian probability interprets probability as a measure of belief or uncertainty about an event, rather than long-run frequency. This allows for coherent updating of beliefs as new evidence becomes available.",
                "mathematical_definition": "Bayesian inference updates beliefs using Bayes' theorem:\n\nP(θ|data) = P(data|θ) × P(θ) / P(data)\n\nPosterior = Likelihood × Prior / Evidence",
                "key_concepts": [
                    "Prior: Initial beliefs before seeing data",
                    "Likelihood: Probability of data given parameters",
                    "Posterior: Updated beliefs after seeing data",
                    "Evidence: Probability of data (normalizing constant)"
                ],
                "properties": [
                    "Beliefs are represented as probability distributions",
                    "All uncertainty is quantified probabilistically",
                    "Beliefs are updated coherently with new evidence",
                    "No distinction between parameters and data"
                ],
                "examples": [
                    {
                        "name": "Medical Diagnosis",
                        "description": "Update disease probability with test results",
                        "prior": "P(disease) = 0.01",
                        "likelihood": "P(positive test|disease) = 0.95",
                        "posterior": "P(disease|positive test) = 0.16"
                    }
                ],
                "applications": [
                    "Statistical inference",
                    "Machine learning (Bayesian methods)",
                    "Decision making under uncertainty",
                    "Scientific hypothesis testing",
                    "Active Inference framework"
                ]
            }

        elif node_id == "bayesian_models":
            return {
                "overview": "Bayesian models are generative models that specify how observed data is generated from underlying parameters. They provide a complete probabilistic description of the data generation process.",
                "mathematical_definition": "A Bayesian model consists of:\n\n1. Prior distribution: p(θ)\n2. Likelihood function: p(data|θ)\n3. Posterior distribution: p(θ|data) ∝ p(data|θ) × p(θ)",
                "model_components": [
                    "Parameters (θ): Unknown quantities we want to learn",
                    "Data (x): Observed evidence",
                    "Prior (p(θ)): Initial beliefs about parameters",
                    "Likelihood (p(x|θ)): Data generation process",
                    "Posterior (p(θ|x)): Updated beliefs"
                ],
                "properties": [
                    "Generative: Can generate synthetic data",
                    "Probabilistic: All uncertainty is quantified",
                    "Hierarchical: Can model complex relationships",
                    "Compositional: Can be combined modularly"
                ],
                "examples": [
                    {
                        "name": "Linear Regression",
                        "description": "Bayesian treatment of linear regression",
                        "model": "y = β₀ + β₁x + ε, ε ~ N(0, σ²)",
                        "interpretation": "Parameters have distributions, not point estimates"
                    }
                ],
                "applications": [
                    "Regression and classification",
                    "Time series modeling",
                    "Hierarchical modeling",
                    "Mixture models",
                    "Active Inference (generative models)"
                ]
            }

        elif node_id == "belief_updating":
            return {
                "overview": "Belief updating is the process of revising probabilistic beliefs as new evidence becomes available. In Bayesian inference, this is done coherently using Bayes' theorem.",
                "mathematical_definition": "Sequential updating:\n\np(θ|x₁, x₂, ..., xₜ) ∝ p(xₜ|θ, x₁, ..., xₜ₋₁) × p(θ|x₁, ..., xₜ₋₁)",
                "updating_mechanisms": [
                    "Batch updating: Update with all data at once",
                    "Sequential updating: Update with each new observation",
                    "Recursive filtering: Online belief updating",
                    "Variational updating: Approximate belief updating"
                ],
                "properties": [
                    "Convergence: Beliefs stabilize with sufficient data",
                    "Consistency: Multiple updates equivalent to batch update",
                    "Robustness: Handles missing or noisy data",
                    "Efficiency: Online algorithms for large datasets"
                ],
                "examples": [
                    {
                        "name": "Kalman Filtering",
                        "description": "Recursive belief updating for linear systems",
                        "interpretation": "Optimal estimation under Gaussian assumptions"
                    },
                    {
                        "name": "Particle Filtering",
                        "description": "Sequential Monte Carlo belief updating",
                        "interpretation": "Handles non-linear, non-Gaussian systems"
                    }
                ],
                "applications": [
                    "Signal processing",
                    "Control systems",
                    "Robotics (state estimation)",
                    "Finance (portfolio optimization)",
                    "Active Inference (belief dynamics)"
                ]
            }

        elif node_id == "fep_introduction":
            return {
                "overview": "The Free Energy Principle (FEP) proposes that all biological systems minimize free energy to maintain their structural and functional integrity. Free energy serves as a unified measure of surprise or prediction error.",
                "mathematical_definition": "The free energy F for a model with parameters θ and data x is:\n\nF = -log p(x|θ) + D_KL(q(θ) || p(θ))\n\nwhere:\n- -log p(x|θ) is accuracy (negative log-likelihood)\n- D_KL(q(θ) || p(θ)) is complexity (KL divergence)",
                "core_principle": "All biological systems act to minimize their free energy, which bounds the surprise or negative log-evidence of their sensory inputs.",
                "key_implications": [
                    "Perception: Inferring hidden causes of sensory data",
                    "Action: Selecting actions that fulfill predictions",
                    "Learning: Updating models to better predict sensory data",
                    "Self-organization: Maintaining system integrity"
                ],
                "properties": [
                    "Universal: Applies to all biological systems",
                    "Unifying: Connects perception, action, and learning",
                    "Information-theoretic: Based on information theory",
                    "Variational: Uses variational inference principles"
                ],
                "examples": [
                    {
                        "name": "Homeostasis",
                        "description": "Maintaining body temperature",
                        "interpretation": "Actions minimize prediction error about ideal temperature"
                    },
                    {
                        "name": "Visual Perception",
                        "description": "Recognizing objects in images",
                        "interpretation": "Brain infers object identities to explain visual input"
                    }
                ],
                "applications": [
                    "Neuroscience (brain function)",
                    "Psychology (behavior)",
                    "Artificial intelligence",
                    "Systems biology",
                    "Active Inference framework"
                ]
            }

        elif node_id == "fep_mathematical_formulation":
            return {
                "overview": "The mathematical foundations of the Free Energy Principle provide a rigorous framework for understanding how systems minimize surprise through variational inference.",
                "variational_free_energy": "The variational free energy F[q] for an approximate posterior q(θ) is:\n\nF[q] = E_q[log q(θ)] - E_q[log p(x,θ)]\n\n= D_KL(q(θ) || p(θ|x)) - log p(x)",
                "free_energy_bound": "The variational free energy provides an upper bound on surprise:\n\n-log p(x) ≤ F[q]\n\nMinimizing free energy minimizes surprise.",
                "gradient_descent": "Free energy can be minimized using gradient descent:\n\n∂F/∂θ = E_q[∂log p(x,θ)/∂θ] - ∂log q(θ)/∂θ",
                "information_geometry": "The free energy landscape defines an information geometry where:\n- Riemannian metric given by Fisher information\n- Geodesics correspond to optimal inference paths\n- Curvature reflects model complexity",
                "properties": [
                    "F ≥ -log p(x) (free energy bounds log-evidence)",
                    "F = 0 when q(θ) = p(θ|x) (perfect inference)",
                    "Gradient descent converges to posterior",
                    "Higher-order terms give information geometry"
                ],
                "examples": [
                    {
                        "name": "Variational Inference",
                        "description": "Approximating posterior distributions",
                        "interpretation": "Free energy minimization finds best approximation"
                    }
                ],
                "applications": [
                    "Variational Bayesian methods",
                    "Neural networks (variational autoencoders)",
                    "Reinforcement learning",
                    "Information geometry",
                    "Active Inference mathematics"
                ]
            }

        elif node_id == "fep_biological_systems":
            return {
                "overview": "The Free Energy Principle explains how biological systems maintain their integrity by minimizing free energy, leading to emergent behaviors like homeostasis, perception, and action.",
                "homeostasis": "Homeostasis can be understood as free energy minimization:\n\n- Set point: Predicted state of the system\n- Error signal: Difference between prediction and actual state\n- Action: Behavior that reduces prediction error",
                "self_organization": "Biological systems self-organize to minimize free energy:\n\n- Structure formation minimizes free energy\n- Function follows from structure that minimizes surprise\n- Evolution selects for free energy minimizing architectures",
                "perception_action_cycle": "The perception-action cycle implements free energy minimization:\n\n1. Generate predictions about sensory input\n2. Compare predictions with actual sensory data\n3. Update model to reduce prediction error\n4. Select actions that fulfill predictions",
                "hierarchical_structure": "Biological systems have hierarchical organization:\n\n- Lower levels: Fast, local control (reflexes)\n- Higher levels: Slow, global control (goals)\n- Top level: Ultimate goals (survival, reproduction)",
                "properties": [
                    "Self-organizing: Structure emerges from dynamics",
                    "Adaptive: Systems learn and adapt to environment",
                    "Robust: Multiple mechanisms for error correction",
                    "Hierarchical: Multi-scale organization"
                ],
                "examples": [
                    {
                        "name": "Thermoregulation",
                        "description": "Maintaining body temperature",
                        "mechanism": "Sweating, shivering, behavior changes"
                    },
                    {
                        "name": "Immune System",
                        "description": "Pathogen detection and elimination",
                        "mechanism": "Pattern recognition and response generation"
                    }
                ],
                "applications": [
                    "Physiology (organ systems)",
                    "Neuroscience (brain organization)",
                    "Ecology (ecosystem dynamics)",
                    "Developmental biology",
                    "Active Inference (biological implementation)"
                ]
            }

        elif node_id == "active_inference_introduction":
            return {
                "overview": "Active Inference is a theoretical framework that explains behavior as the process of minimizing expected free energy. Agents act to fulfill their predictions about the world.",
                "core_idea": "Active Inference posits that:\n\n1. Agents have generative models of their environment\n2. They select actions that minimize expected surprise\n3. Action is planning as inference\n4. Behavior maximizes evidence for the agent's model",
                "mathematical_framework": "Expected Free Energy (EFE) for policy π:\n\nG(π) = Σ_τ E[log q(o_τ|s_τ)] - E[log p(o_τ)]\n\nwhere:\n- First term: Risk (expected prediction error)\n- Second term: Ambiguity (uncertainty about outcomes)",
                "planning_as_inference": "Policy selection is cast as Bayesian inference:\n\np(π) ∝ exp(-G(π))\n\nOptimal policies minimize expected free energy.",
                "perception_action_loop": "The Active Inference cycle:\n\n1. Perceive: Update beliefs about hidden states\n2. Plan: Select policies that minimize expected free energy\n3. Act: Execute selected actions\n4. Learn: Update generative model based on outcomes",
                "properties": [
                    "Goal-directed: Actions serve to achieve preferred outcomes",
                    "Information-seeking: Agents explore to reduce uncertainty",
                    "Adaptive: Learning updates model based on experience",
                    "Predictive: Behavior is prediction-driven"
                ],
                "examples": [
                    {
                        "name": "Foraging",
                        "description": "Animal searching for food",
                        "behavior": "Go to locations where food is expected",
                        "inference": "Infer food locations from environmental cues"
                    },
                    {
                        "name": "Social Interaction",
                        "description": "Human social behavior",
                        "behavior": "Actions that fulfill social expectations",
                        "inference": "Infer others' intentions and preferences"
                    }
                ],
                "applications": [
                    "Robotics (autonomous agents)",
                    "Artificial intelligence",
                    "Psychology (decision making)",
                    "Neuroscience (brain function)",
                    "Clinical applications (mental health)"
                ]
            }

        elif node_id == "ai_generative_models":
            return {
                "overview": "In Active Inference, generative models specify how sensory data is generated from hidden causes. These models encode the agent's understanding of the world and guide both perception and action.",
                "model_structure": "A generative model typically includes:\n\n1. Likelihood: p(o|s) - how observations are generated\n2. Prior: p(s) - prior beliefs about hidden states\n3. Transition: p(s_t|s_t₋₁, a) - state dynamics\n4. Preferences: p(o*) - preferred observations",
                "hierarchical_models": "Active Inference often uses hierarchical models:\n\n- Lower levels: Sensory and motor representations\n- Higher levels: Abstract concepts and goals\n- Top level: Ultimate objectives (survival, well-being)",
                "model_learning": "Models are learned through:\n\n1. Variational inference for parameter estimation\n2. Structure learning for model selection\n3. Active learning for efficient data collection\n4. Meta-learning for learning to learn",
                "recognition_models": "Recognition models (variational posteriors) approximate:\n\nq(s|o) ≈ p(s|o)\n\nThese provide beliefs about hidden states given observations.",
                "properties": [
                    "Generative: Can synthesize data from the model",
                    "Probabilistic: All uncertainty is quantified",
                    "Hierarchical: Multi-level abstraction",
                    "Learnable: Models can be updated from experience"
                ],
                "examples": [
                    {
                        "name": "Visual Object Recognition",
                        "description": "Recognizing objects in images",
                        "model": "p(image|object) × p(object)",
                        "inference": "Infer object identity from image features"
                    },
                    {
                        "name": "Motor Control",
                        "description": "Controlling limb movements",
                        "model": "p(sensory feedback|motor command) × p(motor command)",
                        "inference": "Infer motor commands from desired sensory outcomes"
                    }
                ],
                "applications": [
                    "Computer vision",
                    "Robotics",
                    "Natural language processing",
                    "Cognitive modeling",
                    "Active Inference implementations"
                ]
            }

        elif node_id == "ai_policy_selection":
            return {
                "overview": "Policy selection in Active Inference chooses actions that minimize expected free energy. This framework casts planning as Bayesian inference over future courses of action.",
                "expected_free_energy": "The expected free energy G(π) for policy π is:\n\nG(π) = Σ_τ [E[log q(o_τ|s_τ)] - E[log p(o_τ)]] + complexity terms\n\n= Risk + Ambiguity - Value",
                "policy_components": [
                    "Risk: Expected prediction error (epistemic)",
                    "Ambiguity: Uncertainty about outcomes (aleatoric)",
                    "Value: Expected reward or utility",
                    "Complexity: Cost of maintaining the policy"
                ],
                "planning_horizon": "Planning considers future consequences:\n\n- Short-term: Immediate action selection\n- Long-term: Multi-step planning\n- Hierarchical: Planning at different time scales",
                "information_seeking": "Active Inference agents seek information:\n\n- Exploration: Reduce uncertainty about the world\n- Exploitation: Maximize expected reward\n- Balance: Trade-off between exploration and exploitation",
                "decision_making": "Decision making under uncertainty:\n\n1. Evaluate all possible policies\n2. Compute expected free energy for each\n3. Select policy with minimal expected free energy\n4. Execute action and update beliefs",
                "properties": [
                    "Goal-directed: Actions serve preferred outcomes",
                    "Information-seeking: Agents explore to learn",
                    "Risk-sensitive: Avoids high-variance outcomes",
                    "Adaptive: Updates based on experience"
                ],
                "examples": [
                    {
                        "name": "Optimal Foraging",
                        "description": "Animal foraging decisions",
                        "policy": "Choose patches with highest expected food value",
                        "considerations": "Travel cost, competition, uncertainty"
                    },
                    {
                        "name": "Clinical Decision Making",
                        "description": "Medical treatment selection",
                        "policy": "Balance treatment efficacy with side effects",
                        "considerations": "Patient preferences, uncertainty"
                    }
                ],
                "applications": [
                    "Reinforcement learning",
                    "Decision theory",
                    "Economics (rational choice)",
                    "Artificial intelligence",
                    "Active Inference implementations"
                ]
            }

        else:
            # Default content for unknown nodes
            return {
                "overview": f"Content for {node_data['title']}",
                "mathematical_definition": "Mathematical formulation to be provided",
                "examples": [],
                "applications": ["Active Inference", "Related fields"]
            }

    def get_foundation_tracks(self) -> Dict[str, List[str]]:
        """Get organized foundation learning tracks"""
        return {
            "information_theory": [
                "info_theory_entropy",
                "info_theory_kl_divergence",
                "info_theory_mutual_information"
            ],
            "bayesian_inference": [
                "bayesian_basics",
                "bayesian_models",
                "belief_updating"
            ],
            "free_energy_principle": [
                "fep_introduction",
                "fep_mathematical_formulation",
                "fep_biological_systems"
            ],
            "active_inference": [
                "active_inference_introduction",
                "ai_generative_models",
                "ai_policy_selection"
            ]
        }

    def get_complete_foundation_path(self) -> List[str]:
        """Get a comprehensive foundation learning path"""
        tracks = self.get_foundation_tracks()
        complete_path = []

        # Add tracks in logical order
        for track_name in ["information_theory", "bayesian_inference", "free_energy_principle", "active_inference"]:
            complete_path.extend(tracks[track_name])

        return complete_path

    def get_foundation_by_topic(self, topic: str) -> List[str]:
        """Get foundation nodes related to a specific topic"""
        tracks = self.get_foundation_tracks()
        related_nodes = []

        for track_name, nodes in tracks.items():
            related_nodes.extend(nodes)

        # Filter nodes that contain the topic in their ID or tags
        # In a real implementation, this would use the repository search
        return [node_id for node_id in related_nodes if topic.lower() in node_id.lower()]

    def validate_foundation_consistency(self) -> Dict[str, Any]:
        """Validate consistency of foundation knowledge"""
        tracks = self.get_foundation_tracks()
        validation_result = {
            "valid": True,
            "issues": [],
            "track_completeness": {},
            "prerequisite_satisfaction": {}
        }

        # Check track completeness
        for track_name, nodes in tracks.items():
            completeness = len(nodes)
            validation_result["track_completeness"][track_name] = completeness
            if completeness == 0:
                validation_result["issues"].append(f"Empty track: {track_name}")
                validation_result["valid"] = False

        # Check prerequisite chains
        complete_path = self.get_complete_foundation_path()
        for i, node_id in enumerate(complete_path):
            # In a real implementation, this would check if prerequisites are satisfied
            validation_result["prerequisite_satisfaction"][node_id] = True

        return validation_result

    def generate_foundation_summary(self) -> str:
        """Generate a comprehensive summary of all foundation concepts"""
        tracks = self.get_foundation_tracks()
        complete_path = self.get_complete_foundation_path()

        summary = []
        summary.append("Active Inference Foundations Summary")
        summary.append("=" * 40)
        summary.append("")

        summary.append(f"Total Learning Tracks: {len(tracks)}")
        summary.append(f"Total Foundation Nodes: {len(complete_path)}")
        summary.append("")

        for track_name, nodes in tracks.items():
            summary.append(f"{track_name.replace('_', ' ').title()}:")
            summary.append(f"  Nodes: {len(nodes)}")
            summary.append(f"  Path: {' -> '.join(nodes)}")
            summary.append("")

        summary.append("Complete Learning Sequence:")
        for i, node_id in enumerate(complete_path, 1):
            summary.append(f"{i:2d}. {node_id.replace('_', ' ').title()}")

        return "\n".join(summary)

    def search_foundation_concepts(self, query: str) -> List[Dict[str, Any]]:
        """Search for foundation concepts by query"""
        tracks = self.get_foundation_tracks()
        results = []

        for track_name, nodes in tracks.items():
            for node_id in nodes:
                if query.lower() in node_id.lower() or query.lower() in track_name.lower():
                    results.append({
                        "id": node_id,
                        "track": track_name,
                        "relevance_score": 1.0 if query.lower() in node_id.lower() else 0.5
                    })

        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

    def get_foundation_dependencies(self, node_id: str) -> Dict[str, Any]:
        """Get dependency information for a foundation node"""
        tracks = self.get_foundation_tracks()

        # Find which track contains this node
        containing_track = None
        for track_name, nodes in tracks.items():
            if node_id in nodes:
                containing_track = track_name
                break

        if not containing_track:
            return {"error": f"Node {node_id} not found in foundation tracks"}

        # Find prerequisites and dependents
        track_nodes = tracks[containing_track]
        node_index = track_nodes.index(node_id)

        prerequisites = track_nodes[:node_index]
        dependents = track_nodes[node_index + 1:]

        return {
            "node_id": node_id,
            "track": containing_track,
            "prerequisites": prerequisites,
            "dependents": dependents,
            "position": node_index,
            "total_in_track": len(track_nodes)
        }

    def export_foundation_graph(self, format: str = "json") -> str:
        """Export foundation concepts as a graph structure"""
        tracks = self.get_foundation_tracks()

        graph = {
            "nodes": [],
            "edges": [],
            "tracks": tracks,
            "metadata": {
                "total_tracks": len(tracks),
                "total_nodes": sum(len(nodes) for nodes in tracks.values()),
                "export_format": format,
                "generated_by": "Active Inference Foundations"
            }
        }

        # Add nodes
        node_id_to_track = {}
        for track_name, nodes in tracks.items():
            for node_id in nodes:
                graph["nodes"].append({
                    "id": node_id,
                    "track": track_name,
                    "label": node_id.replace("_", " ").title()
                })
                node_id_to_track[node_id] = track_name

        # Add edges (prerequisite relationships)
        for track_name, nodes in tracks.items():
            for i in range(len(nodes) - 1):
                from_node = nodes[i]
                to_node = nodes[i + 1]
                graph["edges"].append({
                    "source": from_node,
                    "target": to_node,
                    "type": "prerequisite",
                    "track": track_name
                })

        if format == "json":
            return json.dumps(graph, indent=2)
        else:
            return str(graph)
