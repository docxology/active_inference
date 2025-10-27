"""
Research Framework - Experiment Management

Core research framework for managing experiments, simulations, and analysis
in Active Inference research. Provides reproducible workflows and standardized
interfaces for scientific computing.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for research experiments"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    model_type: str = "active_inference"
    simulation_steps: int = 1000
    output_dir: Optional[Path] = None
    random_seed: Optional[int] = None


class ExperimentManager:
    """Manages research experiments and their execution"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, Dict] = {}
        logger.info(f"ExperimentManager initialized with base directory: {self.base_dir}")

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment"""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        experiment_data = {
            "id": experiment_id,
            "config": config,
            "status": ExperimentStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "results": None,
            "metadata": {}
        }

        self.experiments[experiment_id] = experiment_data

        # Save to disk
        experiment_dir = self.base_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)

        config_file = experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                "name": config.name,
                "description": config.description,
                "parameters": config.parameters,
                "model_type": config.model_type,
                "simulation_steps": config.simulation_steps,
                "output_dir": str(config.output_dir) if config.output_dir else None,
                "random_seed": config.random_seed
            }, f, indent=2)

        logger.info(f"Created experiment {experiment_id}: {config.name}")
        return experiment_id

    def run_experiment(self, experiment_id: str, real_execution: bool = False) -> bool:
        """Execute an experiment with optional real computation"""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False

        experiment = self.experiments[experiment_id]
        experiment["status"] = ExperimentStatus.RUNNING
        experiment["started_at"] = datetime.now().isoformat()

        try:
            config = experiment["config"]
            logger.info(f"Running experiment {experiment_id}: {config.name}")

            if real_execution:
                # Real Active Inference experiment execution
                results = self._run_active_inference_experiment(config)
            else:
                # Mock results for demonstration
                results = self._generate_mock_results(experiment_id, config)

            experiment["results"] = results
            experiment["status"] = ExperimentStatus.COMPLETED
            experiment["completed_at"] = datetime.now().isoformat()

            # Save results
            results_file = self.base_dir / experiment_id / "results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            # Save detailed logs
            self._save_experiment_logs(experiment_id, experiment)

            logger.info(f"Experiment {experiment_id} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            experiment["status"] = ExperimentStatus.FAILED
            experiment["error"] = str(e)
            experiment["failed_at"] = datetime.now().isoformat()
            return False

    def _run_active_inference_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a real Active Inference experiment"""
        import numpy as np

        # Initialize Active Inference agent
        n_states = config.parameters.get("n_states", 4)
        n_observations = config.parameters.get("n_observations", 8)
        time_steps = config.simulation_steps

        # Initialize beliefs (uniform prior)
        beliefs = np.ones(n_states) / n_states

        # Initialize generative model (random transition/observation matrices)
        A = np.random.rand(n_observations, n_states)  # observation likelihood
        A = A / A.sum(axis=0)  # normalize columns

        B = np.random.rand(n_states, n_states, n_states)  # transition likelihood
        B = B / B.sum(axis=0)  # normalize

        # Initialize precision parameters
        alpha = 1.0  # observation precision
        beta = 1.0   # transition precision

        # Simulation data
        free_energy_trajectory = []
        belief_trajectory = []
        accuracy_trajectory = []

        for t in range(time_steps):
            # Generate observation (simplified: random observation given current belief)
            observation_probs = A @ beliefs
            observation = np.random.choice(n_observations, p=observation_probs)

            # Update beliefs using variational inference
            # Simplified belief update (would use more sophisticated VI in practice)
            likelihood = A[observation, :]
            posterior = alpha * likelihood * beliefs
            posterior = posterior / posterior.sum()
            beliefs = posterior

            # Calculate free energy (simplified)
            # F = E[log Q(s|o)] - E[log P(o,s)]
            log_likelihood = np.log(np.maximum(likelihood, 1e-10))
            entropy = -np.sum(beliefs * np.log(np.maximum(beliefs, 1e-10)))
            free_energy = -log_likelihood.mean() - beta * entropy
            free_energy_trajectory.append(float(free_energy))

            # Store belief trajectory
            belief_trajectory.append(beliefs.tolist())

            # Calculate accuracy (simplified: max belief confidence)
            accuracy = float(np.max(beliefs))
            accuracy_trajectory.append(accuracy)

        return {
            "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "completed",
            "model_type": "active_inference",
            "parameters": {
                "n_states": n_states,
                "n_observations": n_observations,
                "time_steps": time_steps,
                "alpha": alpha,
                "beta": beta
            },
            "free_energy": free_energy_trajectory,
            "belief_trajectory": belief_trajectory,
            "accuracy": accuracy_trajectory,
            "final_beliefs": beliefs.tolist(),
            "convergence_time": len([f for f in free_energy_trajectory[-10:] if f < free_energy_trajectory[0] * 0.9]) * 10,
            "timestamp": datetime.now().isoformat()
        }

    def _generate_mock_results(self, experiment_id: str, config: ExperimentConfig) -> Dict[str, Any]:
        """Generate mock results for demonstration"""
        import numpy as np

        # Generate realistic-looking trajectories
        time_steps = config.simulation_steps
        base_free_energy = 1.0

        free_energy = [base_free_energy]
        for t in range(1, time_steps):
            # Free energy decreases over time (convergence)
            fe = max(0.1, free_energy[-1] * 0.98 + np.random.normal(0, 0.01))
            free_energy.append(fe)

        # Accuracy increases over time
        accuracy = [0.5]
        for t in range(1, time_steps):
            acc = min(0.95, accuracy[-1] + np.random.normal(0.005, 0.01))
            accuracy.append(max(0.5, acc))

        return {
            "experiment_id": experiment_id,
            "status": "completed",
            "model_type": config.model_type,
            "parameters": config.parameters,
            "free_energy": free_energy,
            "accuracy": accuracy,
            "convergence_time": len([f for f in free_energy[-10:] if f < base_free_energy * 0.8]) * 10,
            "timestamp": datetime.now().isoformat()
        }

    def _save_experiment_logs(self, experiment_id: str, experiment: Dict) -> None:
        """Save detailed experiment logs"""
        log_data = {
            "experiment_id": experiment_id,
            "config": experiment["config"].__dict__,
            "status": experiment["status"].value,
            "started_at": experiment.get("started_at"),
            "completed_at": experiment.get("completed_at"),
            "duration_seconds": None,
            "results_summary": None
        }

        if "started_at" in experiment and "completed_at" in experiment:
            start_time = datetime.fromisoformat(experiment["started_at"])
            end_time = datetime.fromisoformat(experiment["completed_at"])
            log_data["duration_seconds"] = (end_time - start_time).total_seconds()

        if "results" in experiment:
            results = experiment["results"]
            log_data["results_summary"] = {
                "final_free_energy": results.get("free_energy", [])[-1] if results.get("free_energy") else None,
                "final_accuracy": results.get("accuracy", [])[-1] if results.get("accuracy") else None,
                "convergence_time": results.get("convergence_time"),
                "total_steps": len(results.get("free_energy", []))
            }

        # Save log file
        log_file = self.base_dir / experiment_id / "experiment_log.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Retrieve experiment data"""
        return self.experiments.get(experiment_id)

    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Dict]:
        """List experiments, optionally filtered by status"""
        experiments = list(self.experiments.values())

        if status:
            experiments = [exp for exp in experiments if exp["status"] == status]

        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)


class ResearchFramework:
    """Main research framework coordinating experiments and analysis"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_manager = ExperimentManager(
            config.get("experiments_dir", Path("./experiments"))
        )
        self.simulation_engine = None  # Placeholder
        self.analysis_tools = None  # Placeholder

        logger.info("ResearchFramework initialized")

    def run_study(self, study_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a complete research study"""
        logger.info(f"Starting research study: {study_config.get('name', 'Unnamed')}")

        results = {
            "study_name": study_config.get("name"),
            "experiments": [],
            "summary": {}
        }

        # Create and run experiments
        for exp_config in study_config.get("experiments", []):
            config = ExperimentConfig(**exp_config)
            exp_id = self.experiment_manager.create_experiment(config)

            success = self.experiment_manager.run_experiment(exp_id)
            experiment_data = self.experiment_manager.get_experiment(exp_id)

            results["experiments"].append({
                "id": exp_id,
                "success": success,
                "data": experiment_data
            })

        # Generate summary
        completed_experiments = [exp for exp in results["experiments"] if exp["success"]]
        results["summary"] = {
            "total_experiments": len(results["experiments"]),
            "successful_experiments": len(completed_experiments),
            "success_rate": len(completed_experiments) / len(results["experiments"]) if results["experiments"] else 0
        }

        logger.info(f"Research study completed: {results['summary']}")
        return results


