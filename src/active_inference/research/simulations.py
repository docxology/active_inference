"""
Research Framework - Simulation Engine

Multi-scale modeling and simulation capabilities for Active Inference research.
Provides computational models, simulation runners, and analysis tools for
studying complex adaptive systems.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for simulation models"""
    name: str
    model_type: str = "active_inference"
    parameters: Dict[str, Any] = None
    time_horizon: int = 1000
    time_step: float = 0.1

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class SimulationEngine:
    """Core simulation engine for Active Inference models"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, ModelConfig] = {}
        self.running_simulations: Dict[str, Any] = {}

        logger.info("SimulationEngine initialized")

    def register_model(self, config: ModelConfig) -> bool:
        """Register a new simulation model"""
        if config.name in self.models:
            logger.warning(f"Model {config.name} already registered")
            return False

        self.models[config.name] = config
        logger.info(f"Registered model: {config.name}")
        return True

    def run_simulation(self, model_name: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a simulation with the specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        config = self.models[model_name]
        logger.info(f"Running simulation: {model_name}")

        # Placeholder simulation logic
        simulation_id = f"sim_{model_name}_{len(self.running_simulations)}"

        # Simulate model execution
        time_points = np.linspace(0, config.time_horizon * config.time_step, config.time_horizon)

        # Generate mock results based on model type
        if config.model_type == "active_inference":
            results = self._simulate_active_inference(time_points, config, inputs or {})
        else:
            results = self._simulate_generic_model(time_points, config, inputs or {})

        results["simulation_id"] = simulation_id
        results["model_name"] = model_name
        results["timestamp"] = "2024-10-27T12:00:00"

        self.running_simulations[simulation_id] = results
        logger.info(f"Simulation {simulation_id} completed")

        return results

    def _simulate_active_inference(self, time_points: np.ndarray, config: ModelConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a realistic Active Inference model"""
        n_states = inputs.get("n_states", 4)
        n_observations = inputs.get("n_observations", 8)
        n_actions = inputs.get("n_actions", 4)

        # Initialize generative model matrices
        np.random.seed(42)  # For reproducible results

        # Observation likelihood A: P(o|s)
        A = np.random.rand(n_observations, n_states)
        A = A / A.sum(axis=0)  # Normalize columns

        # Transition likelihood B: P(s'|s,a)
        B = np.random.rand(n_states, n_states, n_actions)
        B = B / B.sum(axis=0)  # Normalize

        # Prior preferences C: log P(o)
        C = np.random.normal(0, 0.1, n_observations)

        # Initialize beliefs (uniform prior)
        beliefs = np.ones(n_states) / n_states

        # Initialize action selection
        current_action = 0

        # Simulation data storage
        free_energy_trajectory = []
        belief_trajectory = []
        accuracy_trajectory = []
        action_trajectory = []
        observation_trajectory = []

        for t in range(len(time_points)):
            # Generate observation from current belief state
            observation_probs = A @ beliefs
            observation = np.random.choice(n_observations, p=observation_probs)
            observation_trajectory.append(int(observation))

            # Update beliefs using variational inference (simplified)
            likelihood = A[observation, :]
            posterior = likelihood * beliefs
            posterior = posterior / posterior.sum()
            beliefs = posterior
            belief_trajectory.append(beliefs.tolist())

            # Calculate expected free energy for action selection
            expected_free_energy = np.zeros(n_actions)

            for a in range(n_actions):
                # Predicted beliefs after action
                predicted_beliefs = B[:, :, a] @ beliefs

                # Expected observations under predicted beliefs
                expected_obs = A @ predicted_beliefs

                # Epistemic affordance (information gain)
                epistemic = 0.5 * np.sum(expected_obs * np.log(expected_obs + 1e-10))

                # Extrinsic affordance (preference satisfaction)
                extrinsic = np.sum(expected_obs * C)

                # Total expected free energy
                expected_free_energy[a] = extrinsic - epistemic

            # Select action (greedy policy)
            current_action = np.argmin(expected_free_energy)  # Minimize free energy
            action_trajectory.append(int(current_action))

            # Calculate variational free energy (simplified)
            # F = E_Q[log Q] - E_Q[log P]
            log_likelihood = np.log(np.maximum(likelihood, 1e-10))
            entropy = -np.sum(beliefs * np.log(np.maximum(beliefs, 1e-10)))
            free_energy = -np.mean(log_likelihood) - entropy
            free_energy_trajectory.append(float(free_energy))

            # Calculate accuracy (confidence in most likely state)
            accuracy = float(np.max(beliefs))
            accuracy_trajectory.append(accuracy)

        return {
            "time_points": time_points.tolist(),
            "free_energy": free_energy_trajectory,
            "accuracy": accuracy_trajectory,
            "state_beliefs": belief_trajectory,
            "actions": action_trajectory,
            "observations": observation_trajectory,
            "n_states": n_states,
            "n_observations": n_observations,
            "n_actions": n_actions,
            "model_parameters": {
                "A_shape": A.shape,
                "B_shape": B.shape,
                "C_shape": C.shape,
                "final_beliefs": beliefs.tolist(),
                "preferred_observations": np.argsort(C)[-3:].tolist()  # Top 3 preferred observations
            }
        }

    def _simulate_generic_model(self, time_points: np.ndarray, config: ModelConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generic model simulation"""
        # Simple generic simulation
        amplitude = config.parameters.get("amplitude", 1.0)
        frequency = config.parameters.get("frequency", 1.0)

        signal = amplitude * np.sin(2 * np.pi * frequency * time_points / len(time_points))

        return {
            "time_points": time_points.tolist(),
            "signal": signal.tolist(),
            "parameters": config.parameters
        }

    def get_simulation_results(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve simulation results"""
        return self.running_simulations.get(simulation_id)

    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())


class ModelRunner:
    """High-level interface for running multiple models"""

    def __init__(self, simulation_engine: SimulationEngine):
        self.engine = simulation_engine

    def run_comparison_study(self, model_configs: List[ModelConfig], inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run comparative study across multiple models"""
        logger.info(f"Starting comparison study with {len(model_configs)} models")

        results = {
            "study_id": f"comparison_{len(model_configs)}",
            "models": [],
            "comparison_metrics": {}
        }

        for config in model_configs:
            # Register model if not already registered
            self.engine.register_model(config)

            # Run simulation
            sim_results = self.engine.run_simulation(config.name, inputs)

            results["models"].append({
                "config": config,
                "results": sim_results
            })

        # Generate comparison metrics
        results["comparison_metrics"] = self._compute_comparison_metrics(results["models"])

        logger.info("Comparison study completed")
        return results

    def _compute_comparison_metrics(self, model_results: List[Dict]) -> Dict[str, Any]:
        """Compute comparison metrics across models"""
        metrics = {
            "n_models": len(model_results),
            "model_names": [m["config"].name for m in model_results]
        }

        # Add more sophisticated comparison metrics here
        return metrics


