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

    def run_experiment(self, experiment_id: str) -> bool:
        """Execute an experiment"""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False

        experiment = self.experiments[experiment_id]
        experiment["status"] = ExperimentStatus.RUNNING

        try:
            # Placeholder for actual experiment execution
            logger.info(f"Running experiment {experiment_id}")

            # Simulate experiment execution
            import time
            time.sleep(0.1)  # Simulate work

            # Store mock results
            results = {
                "experiment_id": experiment_id,
                "status": "completed",
                "free_energy": [1.0, 0.8, 0.6, 0.4, 0.2],
                "accuracy": [0.5, 0.7, 0.8, 0.9, 0.95],
                "timestamp": datetime.now().isoformat()
            }

            experiment["results"] = results
            experiment["status"] = ExperimentStatus.COMPLETED

            # Save results
            results_file = self.base_dir / experiment_id / "results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Experiment {experiment_id} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            experiment["status"] = ExperimentStatus.FAILED
            return False

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
