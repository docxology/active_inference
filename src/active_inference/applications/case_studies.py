"""
Application Framework - Case Studies and Examples

Collection of real-world case studies and example applications demonstrating
Active Inference implementations across different domains. Provides practical
examples, best practices, and implementation patterns.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ApplicationDomain(Enum):
    """Application domains for case studies"""
    ROBOTICS = "robotics"
    DECISION_MAKING = "decision_making"
    PERCEPTION = "perception"
    MOTOR_CONTROL = "motor_control"
    SOCIAL_COGNITION = "social_cognition"
    CLINICAL = "clinical"
    EDUCATION = "education"


@dataclass
class CaseStudy:
    """Represents a case study or example application"""
    id: str
    title: str
    domain: ApplicationDomain
    description: str
    difficulty: str  # beginner, intermediate, advanced
    implementation_files: List[str]
    requirements: List[str]
    learning_objectives: List[str]
    key_concepts: List[str]


class ExampleApplications:
    """Collection of example applications"""

    def __init__(self, examples_dir: Path):
        self.examples_dir = Path(examples_dir)
        self.case_studies: Dict[str, CaseStudy] = {}

        self._load_case_studies()
        logger.info(f"ExampleApplications initialized with {len(self.case_studies)} case studies")

    def _load_case_studies(self) -> None:
        """Load available case studies"""
        # Define built-in case studies
        self.case_studies = {
            "perceptual_inference": CaseStudy(
                id="perceptual_inference",
                title="Perceptual Inference in Vision",
                domain=ApplicationDomain.PERCEPTION,
                description="Implementation of perceptual inference for visual recognition using Active Inference",
                difficulty="intermediate",
                implementation_files=["perceptual_model.py", "visual_dataset.py", "inference_engine.py"],
                requirements=["numpy", "scipy", "matplotlib", "pillow"],
                learning_objectives=[
                    "Understand perceptual inference in Active Inference",
                    "Implement generative models for visual data",
                    "Apply variational inference to vision tasks"
                ],
                key_concepts=["perceptual_inference", "generative_models", "variational_inference", "visual_cognition"]
            ),

            "decision_making": CaseStudy(
                id="decision_making",
                title="Decision Making Under Uncertainty",
                domain=ApplicationDomain.DECISION_MAKING,
                description="Active Inference model for decision making in uncertain environments",
                difficulty="advanced",
                implementation_files=["decision_model.py", "environment.py", "policy_selection.py"],
                requirements=["numpy", "scipy", "pandas"],
                learning_objectives=[
                    "Model decision making with Active Inference",
                    "Implement expected free energy minimization",
                    "Handle uncertainty in decision processes"
                ],
                key_concepts=["expected_free_energy", "policy_selection", "uncertainty", "planning"]
            ),

            "motor_control": CaseStudy(
                id="motor_control",
                title="Active Inference for Motor Control",
                domain=ApplicationDomain.MOTOR_CONTROL,
                description="Implementation of motor control using Active Inference principles",
                difficulty="intermediate",
                implementation_files=["motor_model.py", "plant_dynamics.py", "control_loop.py"],
                requirements=["numpy", "scipy", "control"],
                learning_objectives=[
                    "Apply Active Inference to motor control",
                    "Implement forward and inverse models",
                    "Understand active learning in control"
                ],
                key_concepts=["motor_control", "forward_models", "inverse_models", "active_learning"]
            )
        }

    def get_case_study(self, study_id: str) -> Optional[CaseStudy]:
        """Retrieve a specific case study"""
        return self.case_studies.get(study_id)

    def list_case_studies(self, domain: Optional[ApplicationDomain] = None,
                         difficulty: Optional[str] = None) -> List[CaseStudy]:
        """List case studies with optional filtering"""
        studies = list(self.case_studies.values())

        if domain:
            studies = [s for s in studies if s.domain == domain]

        if difficulty:
            studies = [s for s in studies if s.difficulty == difficulty]

        return sorted(studies, key=lambda x: x.title)

    def generate_example_code(self, study_id: str) -> str:
        """Generate example code for a case study"""
        study = self.get_case_study(study_id)
        if not study:
            return f"# Case study '{study_id}' not found - Implementation pending"

        if study_id == "perceptual_inference":
            return self._generate_perceptual_inference_code()
        elif study_id == "decision_making":
            return self._generate_decision_making_code()
        elif study_id == "motor_control":
            return self._generate_motor_control_code()
        else:
            return f"# Code for {study.title} - Implementation pending"

    def _generate_perceptual_inference_code(self) -> str:
        """Generate perceptual inference example code"""
        return '''
"""
Perceptual Inference Example - Visual Recognition

This example demonstrates how to implement perceptual inference
for visual recognition tasks using Active Inference principles.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


class PerceptualInferenceModel:
    """Active Inference model for visual perception"""

    def __init__(self, n_features: int = 784, n_categories: int = 10):
        self.n_features = n_features  # e.g., 28x28 pixels
        self.n_categories = n_categories

        # Generative model parameters
        self.A = np.random.dirichlet(np.ones(n_features), n_categories)  # Likelihood
        self.B = np.random.dirichlet(np.ones(n_categories), n_categories)  # Transitions
        self.C = np.ones(n_categories) / n_categories  # Prior preferences
        self.D = np.ones(n_categories) / n_categories  # Initial beliefs

    def infer(self, observation: np.ndarray, n_iterations: int = 10) -> Dict[str, Any]:
        """
        Perform perceptual inference on visual observation

        Args:
            observation: Binary feature vector (e.g., MNIST digit)
            n_iterations: Number of variational updates

        Returns:
            Inference results including posterior beliefs
        """
        # Initialize posterior with prior
        posterior = self.D.copy()

        # Variational inference loop
        for iteration in range(n_iterations):
            # Compute prediction
            prediction = self.A.T.dot(posterior)

            # Compute prediction error
            prediction_error = observation - prediction

            # Update posterior (simplified update)
            learning_rate = 0.1
            posterior = posterior + learning_rate * (self.A.dot(prediction_error))

            # Normalize
            posterior = np.maximum(posterior, 1e-6)  # Avoid log(0)
            posterior = posterior / np.sum(posterior)

        return {
            "posterior": posterior,
            "prediction": prediction,
            "prediction_error": prediction_error,
            "predicted_category": np.argmax(posterior)
        }

    def update_model(self, observation: np.ndarray, true_category: int) -> None:
        """Update generative model based on feedback"""
        # Simplified learning rule
        learning_rate = 0.01

        # Update likelihood matrix (A)
        self.A[true_category] = (1 - learning_rate) * self.A[true_category] + learning_rate * observation

        # Normalize
        for i in range(self.n_categories):
            self.A[i] = self.A[i] / np.sum(self.A[i])


# Example usage
if __name__ == "__main__":
    # Create model
    model = PerceptualInferenceModel(n_features=16, n_categories=4)

    # Example observation (simple pattern)
    observation = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

    # Perform inference
    results = model.infer(observation)

    print(f"Predicted category: {results['predicted_category']}")
    print(f"Posterior beliefs: {results['posterior']}")
    print(f"Prediction error: {np.sum(results['prediction_error']**2)}")
'''

    def _generate_decision_making_code(self) -> str:
        """Generate decision making example code"""
        return '''
"""
Decision Making Under Uncertainty Example

This example demonstrates decision making under uncertainty
using expected free energy minimization in Active Inference.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DecisionMakingModel:
    """Active Inference model for decision making"""

    def __init__(self, n_states: int = 4, n_actions: int = 3):
        self.n_states = n_states
        self.n_actions = n_actions

        # Generative model
        self.A = np.random.dirichlet(np.ones(n_states), n_actions)  # State likelihood
        self.B = np.random.dirichlet(np.ones(n_actions), n_actions)  # Action transitions
        self.C = np.random.uniform(0, 1, n_states)  # Prior preferences
        self.D = np.ones(n_states) / n_states  # Initial state beliefs

    def compute_expected_free_energy(self, policy: List[int]) -> float:
        """
        Compute expected free energy for a policy

        Args:
            policy: Sequence of actions

        Returns:
            Expected free energy value
        """
        expected_fe = 0.0

        # Current beliefs
        beliefs = self.D.copy()

        for action in policy:
            # Predict next state
            predicted_state = self.B[action].dot(beliefs)

            # Compute epistemic value (uncertainty reduction)
            epistemic_value = -np.sum(predicted_state * np.log(predicted_state + 1e-6))

            # Compute pragmatic value (preference satisfaction)
            pragmatic_value = -np.sum(predicted_state * self.C)

            # Expected free energy combines both
            expected_fe += epistemic_value + pragmatic_value

            # Update beliefs for next timestep
            beliefs = predicted_state

        return expected_fe

    def select_action(self) -> int:
        """
        Select action by minimizing expected free energy

        Returns:
            Selected action index
        """
        min_fe = float('inf')
        best_action = 0

        for action in range(self.n_actions):
            # Test single-step policy
            policy = [action]
            expected_fe = self.compute_expected_free_energy(policy)

            if expected_fe < min_fe:
                min_fe = expected_fe
                best_action = action

        logger.debug(f"Selected action {best_action} with expected FE {min_fe}")
        return best_action

    def simulate_decision_process(self, n_steps: int = 20) -> Dict[str, List]:
        """
        Simulate the decision making process

        Args:
            n_steps: Number of decision steps

        Returns:
            Simulation history
        """
        history = {
            "actions": [],
            "expected_free_energy": [],
            "beliefs": [],
            "outcomes": []
        }

        for step in range(n_steps):
            # Select action
            action = self.select_action()
            history["actions"].append(action)

            # Simulate outcome
            true_state = np.random.choice(self.n_states, p=self.D)
            outcome = np.random.multinomial(1, self.A[action, true_state])

            # Update beliefs based on outcome
            posterior = self.D * self.A[action, outcome.astype(bool)]
            posterior = posterior / np.sum(posterior)
            self.D = posterior

            history["beliefs"].append(posterior.copy())
            history["outcomes"].append(true_state)

            # Compute expected FE for logging
            policy = [action]
            fe = self.compute_expected_free_energy(policy)
            history["expected_free_energy"].append(fe)

        return history


# Example usage
if __name__ == "__main__":
    # Create decision model
    model = DecisionMakingModel(n_states=4, n_actions=3)

    # Run simulation
    results = model.simulate_decision_process(n_steps=20)

    print("Decision making simulation completed:")
    print(f"Actions taken: {results['actions']}")
    print(f"Final beliefs: {results['beliefs'][-1]}")
    print(f"Average expected FE: {np.mean(results['expected_free_energy'])".3f"}")
'''

    def _generate_motor_control_code(self) -> str:
        """Generate motor control example code"""
        return '''
"""
Active Inference for Motor Control Example

This example demonstrates motor control using Active Inference,
showing how agents can learn to control their movements through
minimizing free energy.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


class MotorControlModel:
    """Active Inference model for motor control"""

    def __init__(self, n_joints: int = 2, n_timepoints: int = 100):
        self.n_joints = n_joints
        self.n_timepoints = n_timepoints

        # Motor plant parameters
        self.motor_dynamics = np.random.uniform(-0.1, 0.1, (n_joints, n_joints))

        # Generative model parameters
        self.A = np.random.dirichlet(np.ones(n_joints), n_joints)  # Sensory mapping
        self.B = np.random.dirichlet(np.ones(n_joints), n_joints)  # Control transitions
        self.C = np.random.uniform(0, 1, n_joints)  # Desired states
        self.D = np.ones(n_joints) / n_joints  # Initial beliefs

    def forward_model(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Forward model predicting next state from current state and action

        Args:
            current_state: Current joint positions/velocities
            action: Motor commands

        Returns:
            Predicted next state
        """
        # Simple linear dynamics
        next_state = current_state + self.motor_dynamics.dot(action)
        return next_state

    def inverse_model(self, current_state: np.ndarray, desired_state: np.ndarray) -> np.ndarray:
        """
        Inverse model computing required action to reach desired state

        Args:
            current_state: Current state
            desired_state: Target state

        Returns:
            Required action
        """
        # Simplified inverse dynamics
        state_error = desired_state - current_state
        action = np.linalg.pinv(self.motor_dynamics).dot(state_error)
        return action

    def control_step(self, current_state: np.ndarray, target_state: np.ndarray) -> Dict[str, Any]:
        """
        Execute one control step using Active Inference

        Args:
            current_state: Current joint positions
            target_state: Desired joint positions

        Returns:
            Control results
        """
        # Compute prediction error
        predicted_state = self.A.dot(self.D)
        prediction_error = current_state - predicted_state

        # Compute free energy
        free_energy = 0.5 * np.sum(prediction_error**2)

        # Update beliefs using variational inference
        learning_rate = 0.1
        self.D = self.D + learning_rate * (self.A.T.dot(prediction_error))

        # Normalize beliefs
        self.D = np.maximum(self.D, 1e-6)
        self.D = self.D / np.sum(self.D)

        # Compute action using active inference
        action = self.inverse_model(current_state, target_state)

        # Add exploration noise (active learning)
        exploration_noise = 0.1 * np.random.normal(0, 1, self.n_joints)
        action = action + exploration_noise

        return {
            "action": action,
            "prediction_error": prediction_error,
            "free_energy": free_energy,
            "beliefs": self.D.copy(),
            "predicted_state": predicted_state
        }

    def simulate_reaching_task(self, start_position: np.ndarray,
                              target_position: np.ndarray) -> Dict[str, List]:
        """
        Simulate a reaching task

        Args:
            start_position: Starting joint configuration
            target_position: Target joint configuration

        Returns:
            Simulation results
        """
        current_state = start_position.copy()
        history = {
            "states": [current_state.copy()],
            "actions": [],
            "free_energy": [],
            "prediction_errors": [],
            "targets": [target_position.copy()]
        }

        for step in range(self.n_timepoints):
            # Control step
            result = self.control_step(current_state, target_position)

            # Apply action to get next state
            current_state = self.forward_model(current_state, result["action"])

            # Store results
            history["states"].append(current_state.copy())
            history["actions"].append(result["action"])
            history["free_energy"].append(result["free_energy"])
            history["prediction_errors"].append(result["prediction_error"])

            # Check if target reached
            distance_to_target = np.linalg.norm(current_state - target_position)
            if distance_to_target < 0.1:
                logger.info(f"Target reached at step {step}")
                break

        return history


# Example usage
if __name__ == "__main__":
    # Create motor control model
    model = MotorControlModel(n_joints=2, n_timepoints=100)

    # Define reaching task
    start_pos = np.array([0.0, 0.0])
    target_pos = np.array([1.0, 0.5])

    # Run simulation
    results = model.simulate_reaching_task(start_pos, target_pos)

    print("Motor control simulation completed:")
    print(f"Final position: {results['states'][-1]}")
    print(f"Target position: {target_pos}")
    print(f"Final error: {np.linalg.norm(results['states'][-1] - target_pos)".3f"}")
    print(f"Average free energy: {np.mean(results['free_energy'])".3f"}")
'''


class CaseStudyManager:
    """Manages case studies and examples"""

    def __init__(self, examples_dir: Path):
        self.examples_dir = Path(examples_dir)
        self.example_applications = ExampleApplications(self.examples_dir / "examples")

        logger.info("CaseStudyManager initialized")

    def get_case_study(self, study_id: str) -> Optional[CaseStudy]:
        """Get a specific case study"""
        return self.example_applications.get_case_study(study_id)

    def list_case_studies(self, **filters) -> List[CaseStudy]:
        """List case studies with optional filters"""
        return self.example_applications.list_case_studies(**filters)

    def run_example(self, study_id: str) -> Dict[str, Any]:
        """Run an example case study"""
        study = self.get_case_study(study_id)
        if not study:
            return {"error": f"Case study {study_id} not found"}

        # Generate and execute example code
        code = self.example_applications.generate_example_code(study_id)

        return {
            "study_id": study_id,
            "title": study.title,
            "code": code,
            "requirements": study.requirements,
            "learning_objectives": study.learning_objectives
        }
