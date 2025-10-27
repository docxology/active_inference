"""
Tools - Utility Functions

Helper functions and utilities for Active Inference development and operations.
Provides data processing tools, mathematical utilities, configuration management,
and common operations used throughout the platform.
"""

import logging
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class DataProcessingTools:
    """Data processing utilities"""

    def __init__(self):
        logger.info("DataProcessingTools initialized")

    def normalize_data(self, data: List[float], method: str = "minmax") -> List[float]:
        """Normalize data using specified method"""
        if not data:
            return []

        data_array = np.array(data)

        if method == "minmax":
            min_val, max_val = np.min(data_array), np.max(data_array)
            if min_val == max_val:
                return [0.0] * len(data)
            return ((data_array - min_val) / (max_val - min_val)).tolist()

        elif method == "zscore":
            mean_val, std_val = np.mean(data_array), np.std(data_array)
            if std_val == 0:
                return [0.0] * len(data)
            return ((data_array - mean_val) / std_val).tolist()

        else:
            logger.warning(f"Unknown normalization method: {method}")
            return data

    def smooth_data(self, data: List[float], window_size: int = 5) -> List[float]:
        """Apply moving average smoothing"""
        if len(data) < window_size:
            return data

        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start:end]))

        return smoothed

    def compute_statistics(self, data: List[float]) -> Dict[str, float]:
        """Compute basic statistics for data"""
        if not data:
            return {}

        data_array = np.array(data)
        return {
            "mean": float(np.mean(data_array)),
            "median": float(np.median(data_array)),
            "std": float(np.std(data_array)),
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "count": len(data)
        }

    def resample_data(self, data: List[float], target_length: int) -> List[float]:
        """Resample data to target length"""
        if len(data) == target_length:
            return data

        if len(data) == 0:
            return [0.0] * target_length

        # Simple linear interpolation
        old_indices = np.linspace(0, len(data) - 1, len(data))
        new_indices = np.linspace(0, len(data) - 1, target_length)

        return np.interp(new_indices, old_indices, data).tolist()


class ConfigurationManager:
    """Configuration management utilities"""

    def __init__(self):
        self.config_cache: Dict[str, Any] = {}
        logger.info("ConfigurationManager initialized")

    def load_config(self, config_path: Path, config_type: str = "auto") -> Dict[str, Any]:
        """Load configuration from file"""
        if config_path in self.config_cache:
            return self.config_cache[config_path]

        config_path = Path(config_path)

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return {}

        # Determine format
        if config_type == "auto":
            if config_path.suffix in [".yml", ".yaml"]:
                config_type = "yaml"
            elif config_path.suffix == ".json":
                config_type = "json"
            else:
                logger.error(f"Unknown config format for {config_path}")
                return {}

        # Load configuration
        try:
            with open(config_path, 'r') as f:
                if config_type == "yaml":
                    config = yaml.safe_load(f)
                elif config_type == "json":
                    config = json.load(f)
                else:
                    config = {}

            self.config_cache[config_path] = config
            logger.info(f"Loaded config from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return {}

    def save_config(self, config: Dict[str, Any], config_path: Path, config_type: str = "json") -> bool:
        """Save configuration to file"""
        try:
            config_path = Path(config_path)

            with open(config_path, 'w') as f:
                if config_type == "yaml":
                    yaml.dump(config, f, default_flow_style=False)
                elif config_type == "json":
                    json.dump(config, f, indent=2)
                else:
                    logger.error(f"Unsupported config format: {config_type}")
                    return False

            # Update cache
            self.config_cache[config_path] = config
            logger.info(f"Saved config to {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config {config_path}: {e}")
            return False

    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations"""
        merged = base_config.copy()

        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged


class FileManager:
    """File and directory management utilities"""

    def __init__(self):
        logger.info("FileManager initialized")

    def ensure_directory(self, directory: Path) -> bool:
        """Ensure directory exists"""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False

    def safe_write_file(self, file_path: Path, content: str) -> bool:
        """Safely write content to file"""
        try:
            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first
            temp_path = file_path.with_suffix(file_path.suffix + '.tmp')

            with open(temp_path, 'w') as f:
                f.write(content)

            # Atomic move
            temp_path.replace(file_path)

            logger.debug(f"Wrote file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return False

    def load_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON file safely"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            return None

    def save_json_file(self, data: Dict[str, Any], file_path: Path) -> bool:
        """Save data as JSON file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to save JSON file {file_path}: {e}")
            return False


class MathematicalUtilities:
    """Mathematical utilities for Active Inference"""

    def __init__(self):
        logger.info("MathematicalUtilities initialized")

    def softmax(self, x: List[float], temperature: float = 1.0) -> List[float]:
        """Compute softmax with temperature"""
        x_array = np.array(x)

        if temperature != 1.0:
            x_array = x_array / temperature

        # Numerical stability
        x_max = np.max(x_array)
        exp_x = np.exp(x_array - x_max)

        return (exp_x / np.sum(exp_x)).tolist()

    def normalize_probabilities(self, probabilities: List[float]) -> List[float]:
        """Normalize values to probabilities"""
        prob_array = np.array(probabilities)

        # Handle negative values
        prob_array = prob_array - np.min(prob_array)
        prob_array = prob_array + 1e-6  # Avoid zeros

        return (prob_array / np.sum(prob_array)).tolist()

    def compute_entropy(self, probabilities: List[float]) -> float:
        """Compute Shannon entropy"""
        prob_array = np.array(probabilities)

        # Filter out zeros to avoid log(0)
        prob_array = prob_array[prob_array > 0]

        if len(prob_array) == 0:
            return 0.0

        return float(-np.sum(prob_array * np.log(prob_array)))

    def kl_divergence(self, p: List[float], q: List[float]) -> float:
        """Compute KL divergence between two distributions"""
        p_array = np.array(p)
        q_array = np.array(q)

        # Avoid division by zero and log of zero
        p_array = np.maximum(p_array, 1e-10)
        q_array = np.maximum(q_array, 1e-10)

        return float(np.sum(p_array * np.log(p_array / q_array)))


class HelperFunctions:
    """General helper functions"""

    def __init__(self):
        logger.info("HelperFunctions initialized")

    def format_timestamp(self, timestamp: Optional[datetime] = None) -> str:
        """Format timestamp as ISO string"""
        if timestamp is None:
            timestamp = datetime.now()
        return timestamp.isoformat()

    def generate_id(self, prefix: str = "id", length: int = 8) -> str:
        """Generate a unique ID"""
        import uuid
        return f"{prefix}_{str(uuid.uuid4())[:length]}"

    def flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary"""
        flattened = {}

        for key, value in d.items():
            new_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flattened.update(self.flatten_dict(value, new_key))
            else:
                flattened[new_key] = value

        return flattened

    def deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = dict1.copy()

        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value

        return result


class Utilities:
    """Main utilities class coordinating all utility functions"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_processing = DataProcessingTools()
        self.configuration = ConfigurationManager()
        self.file_manager = FileManager()
        self.mathematics = MathematicalUtilities()
        self.helpers = HelperFunctions()

        logger.info("Utilities initialized")

    def process_time_series(self, data: List[float], operations: List[str] = None) -> Dict[str, Any]:
        """Process time series data with specified operations"""
        if operations is None:
            operations = ["normalize", "smooth", "statistics"]

        result = {"original": data}

        if "normalize" in operations:
            result["normalized"] = self.data_processing.normalize_data(data)

        if "smooth" in operations:
            result["smoothed"] = self.data_processing.smooth_data(data)

        if "statistics" in operations:
            result["statistics"] = self.data_processing.compute_statistics(data)

        return result

    def load_and_merge_configs(self, base_config_path: Path, override_config_path: Path) -> Dict[str, Any]:
        """Load and merge configuration files"""
        base_config = self.configuration.load_config(base_config_path)
        override_config = self.configuration.load_config(override_config_path)

        return self.configuration.merge_configs(base_config, override_config)

    def backup_and_save(self, data: Dict[str, Any], file_path: Path, backup_suffix: str = ".backup") -> bool:
        """Backup existing file and save new data"""
        file_path = Path(file_path)

        # Create backup if file exists
        if file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
            try:
                backup_path.write_text(file_path.read_text())
                logger.info(f"Created backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

        # Save new data
        return self.file_manager.save_json_file(data, file_path)

