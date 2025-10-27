# Knowledge Repository Metadata

**Comprehensive metadata management for the Active Inference Knowledge Environment repository.**

## 📖 Overview

This directory contains metadata definitions and configurations for the knowledge repository, including repository information, content statistics, quality metrics, and system configurations.

## 🎯 Purpose

The metadata directory provides:

- **Repository Information**: Core repository metadata and configuration
- **Content Statistics**: Knowledge base content metrics and analytics
- **Quality Metrics**: Content quality tracking and validation data
- **System Configuration**: Knowledge repository configuration and settings

## 📁 Directory Structure

```
knowledge/metadata/
├── repository.json          # Core repository metadata
└── README.md               # This file
```

## 📄 Core Files

### repository.json

Contains comprehensive metadata about the knowledge repository including:

- **Repository Identity**: Name, version, description
- **Content Statistics**: Total nodes, content types, difficulty distribution
- **Learning Paths**: Available learning paths and tracks
- **Quality Metrics**: Content quality scores and validation status
- **Metadata**: Author information, last updated, version history

#### Metadata Schema

```json
{
  "repository": {
    "id": "active_inference_knowledge",
    "name": "Active Inference Knowledge Base",
    "version": "0.2.0",
    "description": "Comprehensive knowledge base for Active Inference",
    "last_updated": "2024-10-27T00:00:00Z"
  },
  "statistics": {
    "total_nodes": 0,
    "content_types": {
      "foundation": 0,
      "mathematics": 0,
      "implementation": 0,
      "application": 0
    },
    "difficulty_distribution": {
      "beginner": 0,
      "intermediate": 0,
      "advanced": 0,
      "expert": 0
    }
  },
  "learning_paths": {
    "total_paths": 0,
    "total_tracks": 0
  },
  "quality_metrics": {
    "overall_score": 0.0,
    "completeness": 0.0,
    "accuracy": 0.0,
    "clarity": 0.0
  }
}
```

## 🚀 Usage

### Accessing Repository Metadata

```python
import json
from pathlib import Path

# Load repository metadata
metadata_path = Path("knowledge/metadata/repository.json")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Access repository information
repository_info = metadata["repository"]
print(f"Repository: {repository_info['name']}")
print(f"Version: {repository_info['version']}")

# Access content statistics
statistics = metadata["statistics"]
print(f"Total Nodes: {statistics['total_nodes']}")

# Access quality metrics
quality = metadata["quality_metrics"]
print(f"Overall analyzed quality: {quality['overall_score']}")
```

### Updating Metadata

```python
# Update repository metadata
metadata["repository"]["last_updated"] = datetime.now().isoformat()
metadata["statistics"]["total_nodes"] = calculate_total_nodes()

# Save updated metadata
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
```

## 🔧 Metadata Management

### Metadata Generation

Metadata can be automatically generated using the repository audit tools:

```bash
# Generate metadata from knowledge base
python tools/knowledge_validation.py --generate-metadata
```

### Metadata Validation

Validate metadata structure and content:

```bash
# Validate metadata
python tools/json_validation.py knowledge/metadata/repository.json
```

## 📊 Statistics and Analytics

### Content Statistics

The metadata file provides comprehensive statistics about knowledge content:

- **Total Content Nodes**: Count of all knowledge nodes
- **Content Type Distribution**: Distribution across content types
- **Difficulty Distribution**: Distribution of difficulty levels
- **Learning Path Statistics**: Available learning paths and tracks

### Quality Metrics

Quality metrics track content quality across multiple dimensions:

- **Overall Score**: Weighted average quality score
- **Completeness**: Percentage of required fields present
- **Accuracy**: Content accuracy and correctness
- **Clarity**: Content clarity and accessibility

## 🔄 Integration

### Platform Integration

Repository metadata integrates with:

- **Knowledge Graph Engine**: Metadata enhances knowledge graph construction
- **Search System**: Metadata improves search relevance
- **Learning Paths**: Metadata supports learning path generation
- **Quality Assurance**: Metadata tracks quality across content

### Tool Integration

Metadata supports various tools:

- **Validation Tools**: Metadata guides validation processes
- **Analysis Tools**: Metadata provides context for analysis
- **Reporting Tools**: Metadata enables comprehensive reporting
- **Maintenance Tools**: Metadata supports maintenance workflows

## 📚 Related Documentation

- **[Knowledge Base README](../README.md)**: Knowledge base overview
- **[Knowledge AGENTS.md](../AGENTS.md)**: Agent guidelines for knowledge content
- **[Validation Tools](../../tools/knowledge_validation.py)**: Knowledge validation tools
- **[Repository Configuration](../../tools/repository.json)**: Repository configuration

---

*"Active Inference for, with, by Generative AI"* - Supporting knowledge management through comprehensive metadata and repository information.

