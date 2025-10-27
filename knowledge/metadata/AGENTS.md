# Knowledge Metadata - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working with knowledge repository metadata in the Active Inference Knowledge Environment.

## Module Overview

The Knowledge Metadata module manages comprehensive metadata for the knowledge repository, including repository information, content statistics, quality metrics, and system configurations. This module ensures accurate tracking and management of knowledge base metadata.

## Core Responsibilities

### Metadata Management
- **Repository Metadata**: Maintain accurate repository information and statistics
- **Content Statistics**: Track and update content metrics and analytics
- **Quality Metrics**: Monitor and record quality scores and validation data
- **System Configuration**: Manage knowledge repository configuration

### Metadata Validation
- **Schema Validation**: Ensure metadata follows correct schema
- **Data Integrity**: Verify metadata accuracy and consistency
- **Quality Assurance**: Validate quality metrics and statistics
- **Version Management**: Track metadata version and changes

## Development Workflows

### Metadata Update Process
1. **Analyze Knowledge Base**: Review knowledge content and statistics
2. **Update Statistics**: Calculate and update content statistics
3. **Calculate Metrics**: Compute quality metrics and scores
4. **Validate Metadata**: Ensure metadata accuracy and completeness
5. **Version Update**: Update version and timestamp
6. **Document Changes**: Record metadata updates and changes

### Metadata Generation Workflow
```python
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
import json

class MetadataGenerator:
    """Generate and update repository metadata"""

    def __init__(self, knowledge_base_path: str):
        """Initialize metadata generator"""
        self.knowledge_base_path = Path(knowledge_base_path)
        self.metadata = self.load_current_metadata()

    def generate_comprehensive_metadata(self) -> Dict[str, Any]:
        """Generate comprehensive repository metadata"""
        
        # Scan knowledge base
        knowledge_content = self.scan_knowledge_base()
        
        # Calculate statistics
        statistics = self.calculate_statistics(knowledge_content)
        
        # Calculate quality metrics
        quality_metrics = self.calculate_quality_metrics(knowledge_content)
        
        # Update metadata
        metadata = {
            "repository": self.metadata["repository"],
            "statistics": statistics,
            "quality_metrics": quality_metrics,
            "generated_at": datetime.now().isoformat()
        }
        
        # Validate metadata
        self.validate_metadata(metadata)
        
        return metadata

    def calculate_statistics(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate content statistics"""
        
        statistics = {
            "total_nodes": len(content.get("nodes", [])),
            "content_types": self.count_content_types(content),
            "difficulty_distribution": self.count_difficulty_levels(content),
            "learning_paths": {
                "total_paths": len(content.get("learning_paths", [])),
                "total_tracks": sum(len(path.get("tracks", [])) for path in content.get("learning_paths", []))
            }
        }
        
        return statistics

    def calculate_quality_metrics(self, content: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for content"""
        
        quality_metrics = {
            "overall_score": self.calculate_overall_score(content),
            "completeness": self.calculate_completeness_score(content),
            "accuracy": self.calculate_accuracy_score(content),
            "clarity": self.calculate_clarity_score(content)
        }
        
        return quality_metrics
```

## Implementation Patterns

### Metadata Schema Pattern
```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RepositoryMetadata:
    """Structured representation of repository metadata"""
    
    repository_id: str
    name: str
    version: str
    description: str
    last_updated: str
    total_nodes: int
    content_type_counts: Dict[str, int]
    quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "repository": {
                "id": self.repository_id,
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "last_updated": self.last_updated
            },
            "statistics": {
                "total_nodes": self.total_nodes,
                "content_types": self.content_type_counts
            },
            "quality_metrics": {
                "overall_score": self.quality_score
            }
        }
```

### Metadata Validator Pattern
```python
class MetadataValidator:
    """Validate repository metadata"""

    def __init__(self, schema_path: str):
        """Initialize metadata validator"""
        self.schema = self.load_schema(schema_path)

    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata against schema"""
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate schema structure
        structure_validation = self.validate_structure(metadata)
        if not structure_validation["is_valid"]:
            validation_result["is_valid"] = False
            validation_result["errors"].extend(structure_validation["errors"])
        
        # Validate data integrity
        integrity_validation = self.validate_data_integrity(metadata)
        if not integrity_validation["is_valid"]:
            validation_result["is_valid"] = False
            validation_result["errors"].extend(integrity_validation["errors"])
        
        # Check for warnings
        warnings = self.check_warnings(metadata)
        validation_result["warnings"].extend(warnings)
        
        return validation_result
```

## Quality Standards

### Metadata Quality Requirements
- **Accuracy**: Metadata must accurately reflect repository state
- **Completeness**: All required metadata fields must be present
- **Consistency**: Metadata must be consistent across updates
- **Timeliness**: Metadata must be kept current and up-to-date

### Statistics Accuracy
- **Precision**: Statistics must be calculated precisely
- **Coverage**: Statistics must cover all relevant content
- **Validation**: Statistics must be validated for correctness
- **Documentation**: Statistics must be clearly documented

## Related Documentation

- **[Knowledge Metadata README](./README.md)**: Metadata overview and usage
- **[Knowledge Base](../README.md)**: Knowledge base documentation
- **[Knowledge AGENTS.md](../AGENTS.md)**: Agent guidelines for knowledge content

---

*"Active Inference for, with, by Generative AI"* - Supporting knowledge management through comprehensive metadata and repository information.

