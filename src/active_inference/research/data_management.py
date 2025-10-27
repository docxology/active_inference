"""
Research Data Management Module

Comprehensive data management for Active Inference research including data collection,
preprocessing, storage, security, and validation. Provides robust data handling
for scientific research workflows.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import hashlib
import uuid

logger = logging.getLogger(__name__)


class DataSecurityLevel(Enum):
    """Data security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class DataFormat(Enum):
    """Supported data formats"""
    JSON = "json"
    CSV = "csv"
    HDF5 = "hdf5"
    NPZ = "npz"
    PICKLE = "pickle"
    CUSTOM = "custom"


@dataclass
class DataMetadata:
    """Metadata for research data"""
    dataset_id: str
    name: str
    description: str
    format: DataFormat
    security_level: DataSecurityLevel
    created_at: datetime
    created_by: str
    size_bytes: int
    checksum: str
    tags: List[str] = field(default_factory=list)
    schema: Dict[str, Any] = field(default_factory=dict)
    access_permissions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataCollectionConfig:
    """Configuration for data collection"""
    source_type: str
    collection_method: str
    sampling_rate: Optional[float] = None
    batch_size: Optional[int] = None
    preprocessing_steps: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class DataManager:
    """Central data management for research workflows"""

    def __init__(self, base_dir: Path, config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.config = config
        self.datasets: Dict[str, DataMetadata] = {}
        self._setup_data_directories()

    def _setup_data_directories(self) -> None:
        """Initialize data management directories"""
        directories = [
            "raw_data",
            "processed_data",
            "validated_data",
            "metadata",
            "backups"
        ]

        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"DataManager initialized with base directory: {self.base_dir}")

    def collect_data(self, config: DataCollectionConfig) -> str:
        """Collect data according to configuration"""
        dataset_id = str(uuid.uuid4())

        try:
            # Validate collection configuration
            self._validate_collection_config(config)

            # Perform data collection
            raw_data = self._execute_data_collection(config)

            # Apply preprocessing
            processed_data = self._apply_preprocessing(raw_data, config.preprocessing_steps)

            # Validate data
            validation_result = self._validate_data(processed_data, config.validation_rules)

            if not validation_result["valid"]:
                raise ValueError(f"Data validation failed: {validation_result['issues']}")

            # Store data
            self._store_dataset(dataset_id, processed_data, config)

            # Create metadata
            metadata = self._create_metadata(dataset_id, config, processed_data)
            self.datasets[dataset_id] = metadata

            # Save metadata
            self._save_metadata(metadata)

            logger.info(f"Successfully collected and stored dataset: {dataset_id}")
            return dataset_id

        except Exception as e:
            logger.error(f"Data collection failed for dataset {dataset_id}: {e}")
            raise

    def _validate_collection_config(self, config: DataCollectionConfig) -> None:
        """Validate data collection configuration"""
        required_fields = ["source_type", "collection_method"]
        for field in required_fields:
            if not getattr(config, field):
                raise ValueError(f"Missing required field: {field}")

    def _execute_data_collection(self, config: DataCollectionConfig) -> Any:
        """Execute data collection based on configuration"""
        # This is a placeholder - actual implementation would depend on source type
        if config.source_type == "simulation":
            return self._collect_simulation_data(config)
        elif config.source_type == "experiment":
            return self._collect_experiment_data(config)
        elif config.source_type == "api":
            return self._collect_api_data(config)
        else:
            raise NotImplementedError(f"Unsupported source type: {config.source_type}")

    def _collect_simulation_data(self, config: DataCollectionConfig) -> Dict[str, Any]:
        """Collect data from simulation sources"""
        # Placeholder implementation
        return {"simulation_data": "placeholder"}

    def _collect_experiment_data(self, config: DataCollectionConfig) -> Dict[str, Any]:
        """Collect data from experimental sources"""
        # Placeholder implementation
        return {"experiment_data": "placeholder"}

    def _collect_api_data(self, config: DataCollectionConfig) -> Dict[str, Any]:
        """Collect data from API sources"""
        # Placeholder implementation
        return {"api_data": "placeholder"}

    def _apply_preprocessing(self, data: Any, preprocessing_steps: List[str]) -> Any:
        """Apply preprocessing steps to raw data"""
        processed_data = data

        for step in preprocessing_steps:
            if step == "normalize":
                processed_data = self._normalize_data(processed_data)
            elif step == "filter":
                processed_data = self._filter_data(processed_data)
            elif step == "transform":
                processed_data = self._transform_data(processed_data)
            # Add more preprocessing steps as needed

        return processed_data

    def _normalize_data(self, data: Any) -> Any:
        """Normalize data values"""
        # Placeholder implementation
        return data

    def _filter_data(self, data: Any) -> Any:
        """Filter data based on criteria"""
        # Placeholder implementation
        return data

    def _transform_data(self, data: Any) -> Any:
        """Transform data format"""
        # Placeholder implementation
        return data

    def _validate_data(self, data: Any, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against specified rules"""
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": []
        }

        # Check data completeness
        if not data:
            validation_result["issues"].append("Empty data")
            validation_result["valid"] = False

        # Check data format
        if validation_rules.get("required_format"):
            if not self._check_data_format(data, validation_rules["required_format"]):
                validation_result["issues"].append("Invalid data format")
                validation_result["valid"] = False

        # Check data quality
        quality_issues = self._check_data_quality(data)
        if quality_issues:
            validation_result["warnings"].extend(quality_issues)

        return validation_result

    def _check_data_format(self, data: Any, required_format: str) -> bool:
        """Check if data matches required format"""
        # Placeholder implementation
        return True

    def _check_data_quality(self, data: Any) -> List[str]:
        """Check data quality and return issues"""
        issues = []

        # Check for missing values
        if self._has_missing_values(data):
            issues.append("Data contains missing values")

        # Check for outliers
        if self._has_outliers(data):
            issues.append("Data contains potential outliers")

        return issues

    def _has_missing_values(self, data: Any) -> bool:
        """Check if data contains missing values"""
        # Placeholder implementation
        return False

    def _has_outliers(self, data: Any) -> bool:
        """Check if data contains outliers"""
        # Placeholder implementation
        return False

    def _store_dataset(self, dataset_id: str, data: Any, config: DataCollectionConfig) -> None:
        """Store dataset in appropriate format"""
        # Determine storage format
        storage_format = self._determine_storage_format(data, config)

        # Store data
        if storage_format == DataFormat.JSON:
            self._store_json_data(dataset_id, data)
        elif storage_format == DataFormat.HDF5:
            self._store_hdf5_data(dataset_id, data)
        else:
            self._store_custom_data(dataset_id, data, storage_format)

    def _determine_storage_format(self, data: Any, config: DataCollectionConfig) -> DataFormat:
        """Determine appropriate storage format for data"""
        # Default to JSON for simplicity
        return DataFormat.JSON

    def _store_json_data(self, dataset_id: str, data: Any) -> None:
        """Store data as JSON"""
        file_path = self.base_dir / "processed_data" / f"{dataset_id}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _store_hdf5_data(self, dataset_id: str, data: Any) -> None:
        """Store data as HDF5"""
        # Placeholder implementation
        pass

    def _store_custom_data(self, dataset_id: str, data: Any, format_type: DataFormat) -> None:
        """Store data in custom format"""
        # Placeholder implementation
        pass

    def _create_metadata(self, dataset_id: str, config: DataCollectionConfig, data: Any) -> DataMetadata:
        """Create metadata for dataset"""
        # Calculate data size
        size_bytes = self._calculate_data_size(data)

        # Calculate checksum
        checksum = self._calculate_checksum(data)

        return DataMetadata(
            dataset_id=dataset_id,
            name=config.collection_method,
            description=f"Data collected via {config.collection_method}",
            format=self._determine_storage_format(data, config),
            security_level=DataSecurityLevel.INTERNAL,
            created_at=datetime.now(),
            created_by="system",
            size_bytes=size_bytes,
            checksum=checksum,
            tags=config.preprocessing_steps
        )

    def _calculate_data_size(self, data: Any) -> int:
        """Calculate data size in bytes"""
        # Placeholder implementation
        return 1024

    def _calculate_checksum(self, data: Any) -> str:
        """Calculate data checksum"""
        # Placeholder implementation
        return hashlib.md5(str(data).encode()).hexdigest()

    def _save_metadata(self, metadata: DataMetadata) -> None:
        """Save metadata to file"""
        file_path = self.base_dir / "metadata" / f"{metadata.dataset_id}.json"
        with open(file_path, 'w') as f:
            json.dump({
                "dataset_id": metadata.dataset_id,
                "name": metadata.name,
                "description": metadata.description,
                "format": metadata.format.value,
                "security_level": metadata.security_level.value,
                "created_at": metadata.created_at.isoformat(),
                "created_by": metadata.created_by,
                "size_bytes": metadata.size_bytes,
                "checksum": metadata.checksum,
                "tags": metadata.tags,
                "schema": metadata.schema,
                "access_permissions": metadata.access_permissions
            }, f, indent=2)

    def get_dataset(self, dataset_id: str) -> Optional[Any]:
        """Retrieve dataset by ID"""
        if dataset_id not in self.datasets:
            return None

        metadata = self.datasets[dataset_id]
        file_path = self.base_dir / "processed_data" / f"{dataset_id}.json"

        if not file_path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            return None

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Verify checksum
            if self._calculate_checksum(data) != metadata.checksum:
                logger.warning(f"Checksum mismatch for dataset {dataset_id}")

            return data

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {e}")
            return None

    def list_datasets(self) -> List[DataMetadata]:
        """List all available datasets"""
        return list(self.datasets.values())

    def validate_dataset_integrity(self, dataset_id: str) -> Dict[str, Any]:
        """Validate dataset integrity"""
        if dataset_id not in self.datasets:
            return {"valid": False, "error": "Dataset not found"}

        metadata = self.datasets[dataset_id]
        data = self.get_dataset(dataset_id)

        if data is None:
            return {"valid": False, "error": "Could not load dataset"}

        validation_result = {
            "valid": True,
            "issues": [],
            "metadata": {
                "size_match": True,
                "checksum_match": True,
                "format_valid": True
            }
        }

        # Check file size
        file_path = self.base_dir / "processed_data" / f"{dataset_id}.json"
        if file_path.exists():
            actual_size = file_path.stat().st_size
            if actual_size != metadata.size_bytes:
                validation_result["metadata"]["size_match"] = False
                validation_result["issues"].append("File size mismatch")

        # Check checksum
        actual_checksum = self._calculate_checksum(data)
        if actual_checksum != metadata.checksum:
            validation_result["metadata"]["checksum_match"] = False
            validation_result["issues"].append("Checksum mismatch")

        if not validation_result["issues"]:
            validation_result["valid"] = True
        else:
            validation_result["valid"] = False

        return validation_result

    def backup_dataset(self, dataset_id: str) -> Optional[Path]:
        """Create backup of dataset"""
        if dataset_id not in self.datasets:
            return None

        backup_dir = self.base_dir / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Create backup file
        data = self.get_dataset(dataset_id)
        if data is None:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{dataset_id}_{timestamp}.json"

        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Created backup of dataset {dataset_id}: {backup_file}")
        return backup_file
