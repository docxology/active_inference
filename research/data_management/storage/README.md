# Data Storage & Management Framework

This directory contains comprehensive storage solutions for research data, including file systems, databases, cloud storage, and specialized research data formats. The framework provides scalable, secure, and efficient data storage with comprehensive metadata management and access control.

## Overview

The Data Storage Framework provides a unified interface for storing, retrieving, and managing research data across multiple storage backends. It supports various data formats, provides versioning and backup capabilities, and ensures data integrity and security throughout the research lifecycle.

## Directory Structure

```
storage/
├── file_storage/         # File system-based storage solutions
├── database_storage/     # Database storage implementations
├── cloud_storage/        # Cloud storage integrations
├── archival_storage/     # Long-term archival and preservation
├── metadata_storage/     # Metadata and provenance tracking
└── README.md            # This file
```

## Core Components

### 📁 File System Storage
Hierarchical file organization with metadata management:

```python
from active_inference.research.data_management.storage.file_storage import FileStorageManager

# Configure file storage
file_config = {
    'base_path': './research_data/',
    'organization_scheme': 'hierarchical',  # flat, hierarchical, by_experiment
    'file_formats': {
        'primary': 'parquet',    # parquet, hdf5, json, csv
        'compression': 'snappy',
        'metadata_format': 'json'
    },
    'naming_convention': {
        'pattern': '{experiment_id}/{subject_id}/{session_id}/{data_type}.{extension}',
        'timestamp_format': '%Y%m%d_%H%M%S',
        'include_hash': True
    },
    'permissions': {
        'default_mode': 0o644,
        'backup_mode': 0o444,
        'sensitive_mode': 0o600
    }
}

# Initialize file storage manager
file_storage = FileStorageManager(file_config)

# Store research data with metadata
data_id = file_storage.store_data(
    data=experimental_results,
    metadata={
        'experiment_id': 'active_inference_001',
        'subject_id': 'subject_001',
        'session_id': 'session_2024_10_27_01',
        'data_type': 'behavioral',
        'timestamp': '2024-10-27T10:30:00Z',
        'quality_score': 0.95,
        'processing_pipeline': 'standard_preprocessing_v2.1',
        'researcher': 'dr_smith',
        'project': 'active_inference_study'
    },
    tags=['active_inference', 'behavioral', 'experiment_001']
)

# Retrieve data with metadata
retrieved_data, metadata = file_storage.retrieve_data(data_id)
print(f"📁 Retrieved data: {retrieved_data.shape}")
print(f"📋 Metadata: {metadata.keys()}")

# Search data by metadata
search_results = file_storage.search_data(
    metadata_filters={
        'experiment_id': 'active_inference_001',
        'data_type': 'behavioral'
    },
    date_range=['2024-10-01', '2024-10-31'],
    tags=['active_inference']
)
```

### 🗄️ Database Storage
SQL and NoSQL database integration for structured data:

```python
from active_inference.research.data_management.storage.database_storage import DatabaseStorageManager

# Configure database storage
db_config = {
    'primary_database': {
        'engine': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'research_data',
        'schema': 'active_inference',
        'connection_pool': {
            'min_connections': 5,
            'max_connections': 20,
            'max_overflow': 10
        }
    },
    'backup_database': {
        'engine': 'sqlite',
        'path': './backup/research_backup.db'
    },
    'schema_management': {
        'auto_migrate': True,
        'version_control': True,
        'validation': True
    },
    'indexing': {
        'full_text_search': True,
        'semantic_search': True,
        'temporal_indexes': True,
        'custom_indexes': [
            'experiment_id',
            'subject_id',
            'data_type',
            'quality_score'
        ]
    }
}

# Initialize database storage
db_storage = DatabaseStorageManager(db_config)

# Define data schema
experiment_schema = {
    'table_name': 'experiment_results',
    'columns': {
        'id': {'type': 'SERIAL', 'primary_key': True},
        'experiment_id': {'type': 'VARCHAR(50)', 'index': True},
        'subject_id': {'type': 'VARCHAR(50)', 'index': True},
        'session_id': {'type': 'VARCHAR(50)', 'index': True},
        'data_type': {'type': 'VARCHAR(30)', 'index': True},
        'timestamp': {'type': 'TIMESTAMP', 'index': True},
        'data': {'type': 'JSONB', 'compress': True},
        'metadata': {'type': 'JSONB'},
        'quality_score': {'type': 'FLOAT', 'index': True},
        'file_path': {'type': 'VARCHAR(500)'},
        'checksum': {'type': 'VARCHAR(64)'}
    }
}

# Create schema and store data
db_storage.create_schema(experiment_schema)

# Store experimental data
storage_id = db_storage.store_record(
    table='experiment_results',
    data={
        'experiment_id': 'ai_study_001',
        'subject_id': 'subject_001',
        'session_id': 'session_001',
        'data_type': 'behavioral',
        'timestamp': '2024-10-27 10:30:00',
        'data': experimental_results,
        'metadata': experiment_metadata,
        'quality_score': 0.95,
        'file_path': '/path/to/backup/file.parquet',
        'checksum': 'sha256_hash_here'
    }
)

# Query research data
query_results = db_storage.query_data(
    table='experiment_results',
    filters={
        'experiment_id': 'ai_study_001',
        'quality_score': {'>=': 0.9}
    },
    order_by='timestamp DESC',
    limit=100
)

# Advanced search with full-text and semantic capabilities
semantic_results = db_storage.semantic_search(
    query='active inference behavioral experiments high quality',
    table='experiment_results',
    limit=50,
    threshold=0.7
)
```

### ☁️ Cloud Storage Integration
Cloud storage solutions for scalability and accessibility:

```python
from active_inference.research.data_management.storage.cloud_storage import CloudStorageManager

# Configure cloud storage
cloud_config = {
    'providers': {
        'aws_s3': {
            'bucket': 'research-data-bucket',
            'region': 'us-east-1',
            'access_key': 'your_access_key',
            'secret_key': 'your_secret_key',
            'encryption': 'AES256',
            'versioning': True,
            'lifecycle_policy': {
                'transition_to_ia': 30,  # days
                'transition_to_glacier': 90,  # days
                'expiration': 2555  # days (7 years)
            }
        },
        'google_cloud': {
            'bucket': 'research-data-bucket',
            'project': 'research-project',
            'credentials_file': './credentials/gcp_credentials.json',
            'storage_class': 'STANDARD',
            'versioning': True
        }
    },
    'replication': {
        'cross_region': True,
        'cross_provider': True,
        'sync_frequency': 'daily'
    },
    'access_patterns': {
        'hot_data': {'provider': 'aws_s3', 'storage_class': 'STANDARD'},
        'warm_data': {'provider': 'aws_s3', 'storage_class': 'STANDARD_IA'},
        'cold_data': {'provider': 'aws_s3', 'storage_class': 'GLACIER'}
    }
}

# Initialize cloud storage manager
cloud_storage = CloudStorageManager(cloud_config)

# Multi-cloud storage strategy
def store_with_redundancy(data, metadata, redundancy_level='high'):
    """Store data with multi-cloud redundancy"""
    storage_locations = []

    # Primary storage (fast access)
    primary_location = cloud_storage.store_primary(
        data=data,
        metadata=metadata,
        provider='aws_s3',
        storage_class='STANDARD'
    )
    storage_locations.append(primary_location)

    if redundancy_level in ['medium', 'high']:
        # Secondary storage (warm backup)
        secondary_location = cloud_storage.store_backup(
            data=data,
            metadata=metadata,
            provider='google_cloud',
            storage_class='STANDARD'
        )
        storage_locations.append(secondary_location)

    if redundancy_level == 'high':
        # Tertiary storage (cold archive)
        archive_location = cloud_storage.store_archive(
            data=data,
            metadata=metadata,
            provider='aws_s3',
            storage_class='GLACIER'
        )
        storage_locations.append(archive_location)

    return {
        'primary': primary_location,
        'redundancy_level': redundancy_level,
        'locations': storage_locations,
        'access_urls': cloud_storage.get_access_urls(storage_locations)
    }

# Store research data with redundancy
storage_result = store_with_redundancy(
    data=large_experiment_dataset,
    metadata=experiment_metadata,
    redundancy_level='high'
)

# Access data from any location
data_retrieval = cloud_storage.retrieve_with_fallback(
    primary_url=storage_result['primary'],
    fallback_urls=storage_result['locations'][1:]
)
```

## Data Management Operations

### Version Control & Provenance

```python
from active_inference.research.data_management.storage.versioning import DataVersionManager

# Configure versioning system
versioning_config = {
    'versioning_strategy': 'semantic',  # semantic, timestamp, hash
    'auto_versioning': True,
    'major_version_triggers': [
        'schema_change',
        'significant_quality_improvement',
        'new_preprocessing_method'
    ],
    'provenance_tracking': {
        'capture_sources': True,
        'track_transformations': True,
        'record_access': True,
        'maintain_chain': True
    },
    'metadata_enrichment': {
        'auto_timestamp': True,
        'auto_checksum': True,
        'auto_size': True,
        'custom_fields': ['processing_pipeline', 'quality_metrics']
    }
}

# Initialize version manager
version_manager = DataVersionManager(versioning_config)

# Create new data version
def create_data_version(data, description, version_type='minor'):
    """Create a new version of research data"""
    # Calculate version number
    current_version = version_manager.get_current_version(data_id)
    new_version = version_manager.calculate_new_version(current_version, version_type)

    # Create version metadata
    version_metadata = {
        'version': new_version,
        'previous_version': current_version,
        'description': description,
        'changes': version_manager.analyze_changes(data, current_version),
        'timestamp': datetime.now(),
        'author': 'researcher_name',
        'quality_impact': version_manager.assess_quality_impact(data),
        'backward_compatibility': version_manager.check_compatibility(new_version)
    }

    # Store versioned data
    version_id = version_manager.store_version(
        data=data,
        metadata=version_metadata,
        data_id=data_id
    )

    # Update provenance chain
    version_manager.update_provenance_chain(data_id, version_id, version_metadata)

    return version_id, new_version

# Manage data versions
version_history = version_manager.get_version_history(data_id)
print(f"📋 Version history: {len(version_history)} versions")

# Compare versions
comparison = version_manager.compare_versions(data_id, '1.0.0', '1.1.0')
print(f"🔍 Changes: {comparison['summary']}")

# Access specific version
historical_data = version_manager.retrieve_version(data_id, '1.0.0')
```

### Backup & Recovery

```python
from active_inference.research.data_management.storage.backup import BackupManager

# Configure backup system
backup_config = {
    'backup_schedules': {
        'real_time': {
            'enabled': True,
            'interval_seconds': 300,  # 5 minutes
            'retention_days': 7
        },
        'hourly': {
            'enabled': True,
            'retention_days': 30
        },
        'daily': {
            'enabled': True,
            'retention_days': 90
        },
        'weekly': {
            'enabled': True,
            'retention_days': 365
        },
        'monthly': {
            'enabled': True,
            'retention_years': 7
        }
    },
    'backup_destinations': [
        {
            'type': 'local',
            'path': './backups/local/',
            'encryption': 'AES256',
            'compression': 'gzip'
        },
        {
            'type': 'cloud',
            'provider': 'aws_s3',
            'bucket': 'research-backups',
            'region': 'us-east-1'
        }
    ],
    'recovery_strategies': {
        'point_in_time': True,
        'version_rollback': True,
        'partial_recovery': True,
        'cross_location': True
    }
}

# Initialize backup manager
backup_manager = BackupManager(backup_config)

# Set up automated backup
backup_manager.setup_automated_backup()

# Monitor backup status
backup_status = backup_manager.get_backup_status()
print("💾 Backup Status")
print("=" * 20)
print(f"Last backup: {backup_status['last_backup']}")
print(f"Backup size: {backup_status['total_size_gb']:.1f} GB")
print(f"Success rate: {backup_status['success_rate']:.1%}")
print(f"Next backup: {backup_status['next_backup']}")

# Test backup and recovery
def test_backup_recovery():
    """Test backup and recovery procedures"""
    print("🧪 Testing Backup and Recovery")
    print("=" * 30)

    # Create test data
    test_data = generate_test_dataset(size_gb=1)

    # Store original data
    original_id = backup_manager.store_test_data(test_data)

    # Simulate data loss
    print("📁 Simulating data loss...")
    backup_manager.simulate_data_loss(original_id)

    # Test recovery
    print("🔄 Testing recovery...")
    recovered_data = backup_manager.recover_data(original_id)

    # Validate recovery
    validation = backup_manager.validate_recovery(test_data, recovered_data)
    print(f"✅ Recovery validation: {validation['success']}")

    # Test point-in-time recovery
    print("⏰ Testing point-in-time recovery...")
    point_in_time_data = backup_manager.point_in_time_recovery(
        data_id=original_id,
        target_time='2024-10-27T10:00:00Z'
    )

    return validation

# Run backup recovery test
recovery_test = test_backup_recovery()
```

## Search & Discovery

### Advanced Search Capabilities

```python
from active_inference.research.data_management.storage.search import ResearchDataSearch

# Configure search system
search_config = {
    'search_types': {
        'full_text': {
            'enabled': True,
            'fields': ['description', 'notes', 'tags', 'metadata'],
            'boost_factors': {'title': 2.0, 'tags': 1.5}
        },
        'semantic': {
            'enabled': True,
            'model': 'sentence_transformers',
            'dimensions': 768,
            'similarity_threshold': 0.7
        },
        'structured': {
            'enabled': True,
            'fields': [
                'experiment_id', 'subject_id', 'data_type',
                'quality_score', 'timestamp', 'researcher'
            ],
            'range_queries': True,
            'boolean_logic': True
        }
    },
    'indexing': {
        'auto_index': True,
        'incremental_updates': True,
        'relevance_scoring': True,
        'fuzzy_matching': True
    }
}

# Initialize search system
research_search = ResearchDataSearch(search_config)

# Multi-modal search
def comprehensive_data_search(query):
    """Perform comprehensive search across all data types"""
    print(f"🔍 Searching for: '{query}'")
    print("=" * 40)

    # Full-text search
    text_results = research_search.full_text_search(query, limit=20)

    # Semantic search
    semantic_results = research_search.semantic_search(query, limit=20)

    # Structured search
    structured_results = research_search.structured_search({
        'experiment_type': 'active_inference',
        'quality_score': {'>=': 0.9},
        'date_range': ['2024-01-01', '2024-12-31']
    })

    # Combine and rank results
    combined_results = research_search.combine_search_results(
        [text_results, semantic_results, structured_results]
    )

    # Display results
    print(f"📊 Found {len(combined_results)} results")
    for i, result in enumerate(combined_results[:10], 1):
        print(f"{i:2d}. {result['title']} (score: {result['score']:.3f})")
        print(f"    Type: {result['data_type']}, Quality: {result['quality_score']:.2f}")

    return combined_results

# Advanced search with filters
advanced_search = research_search.advanced_search({
    'query': 'active inference behavioral experiments',
    'filters': {
        'data_types': ['behavioral', 'physiological'],
        'quality_range': [0.85, 1.0],
        'date_range': ['2024-01-01', '2024-12-31'],
        'researchers': ['dr_smith', 'dr_johnson'],
        'tags': ['active_inference', 'experiment']
    },
    'sorting': {
        'primary': 'quality_score',
        'secondary': 'recency',
        'direction': 'descending'
    },
    'facets': ['data_type', 'researcher', 'experiment_type']
})

# Search result analysis
search_analytics = research_search.analyze_search_patterns()
print(f"📈 Popular search terms: {search_analytics['popular_terms']}")
print(f"🔄 Search success rate: {search_analytics['success_rate']:.1%}")
```

## Performance Optimization

### High-Performance Storage

```python
from active_inference.research.data_management.storage.performance import HighPerformanceStorage

# Configure high-performance storage
performance_config = {
    'optimization_targets': {
        'read_speed': 'high',
        'write_speed': 'high',
        'query_performance': 'high',
        'storage_efficiency': 'medium'
    },
    'caching_strategy': {
        'levels': ['memory', 'ssd', 'network'],
        'cache_sizes': {'memory': '4GB', 'ssd': '100GB'},
        'eviction_policy': 'lru',
        'preload_strategy': 'predictive'
    },
    'parallelization': {
        'max_workers': 8,
        'chunk_size': 10000,
        'async_processing': True,
        'load_balancing': True
    },
    'compression': {
        'algorithm': 'adaptive',  # none, gzip, snappy, lz4, adaptive
        'level': 'balanced',
        'chunking': True,
        'streaming': True
    }
}

# Initialize high-performance storage
hp_storage = HighPerformanceStorage(performance_config)

# Performance monitoring
def monitor_storage_performance():
    """Monitor storage system performance"""
    monitor = hp_storage.get_performance_monitor()

    print("📊 Storage Performance Dashboard")
    print("=" * 35)

    while True:
        metrics = monitor.get_current_metrics()

        print(f"Read Speed:  {metrics['read_speed_mbps']:.1f} MB/s")
        print(f"Write Speed: {metrics['write_speed_mbps']:.1f} MB/s")
        print(f"Query Time:  {metrics['avg_query_time_ms']:.1f} ms")
        print(f"Cache Hit Rate: {metrics['cache_hit_rate']:.1%}")
        print(f"Storage Usage:  {metrics['storage_usage_percent']:.1f}%")
        print(f"Active Connections: {metrics['active_connections']}")

        # Performance alerts
        alerts = monitor.get_performance_alerts()
        if alerts:
            print("\n🚨 Performance Alerts:")
            for alert in alerts:
                print(f"  - {alert['type']}: {alert['message']}")
        else:
            print("\n✅ All performance metrics normal")

        print(f"\nLast updated: {metrics['timestamp']}")
        time.sleep(30)  # Update every 30 seconds

# Start performance monitoring
monitor_storage_performance()
```

## Integration Examples

### Research Pipeline Integration

```python
from active_inference.research.experiments import ExperimentManager
from active_inference.research.data_management.storage import UnifiedStorageManager

# Set up unified storage for research pipeline
experiment = ExperimentManager()
storage_manager = UnifiedStorageManager(storage_config)

# Configure experiment data storage
storage_manager.configure_experiment_storage(
    experiment_id='active_inference_study_001',
    storage_requirements={
        'data_types': ['raw', 'processed', 'results', 'metadata'],
        'retention_policy': '7_years',
        'backup_strategy': '3-2-1',  # 3 copies, 2 media types, 1 off-site
        'access_patterns': {
            'hot_data': 'ssd',
            'warm_data': 'hdd',
            'cold_data': 'tape'
        }
    }
)

# Integrated experiment execution
def run_experiment_with_storage(experiment_config):
    """Run experiment with integrated storage management"""
    print("🚀 Experiment with Integrated Storage")
    print("=" * 40)

    # Pre-experiment setup
    storage_manager.initialize_experiment_storage(experiment_config['id'])

    # Run experiment phases
    for phase in experiment_config['phases']:
        print(f"\n📋 Phase: {phase['name']}")

        # Collect phase data
        phase_data = experiment.run_phase(phase)

        # Store phase data
        phase_storage = storage_manager.store_phase_data(
            experiment_id=experiment_config['id'],
            phase_name=phase['name'],
            data=phase_data,
            metadata=phase['metadata']
        )

        # Quality validation
        quality_check = storage_manager.validate_phase_storage(phase_storage)
        print(f"   Quality score: {quality_check['score']:.3f}")

    # Post-experiment processing
    final_results = experiment.finalize_experiment()

    # Store final results with archival
    archival_storage = storage_manager.archive_experiment_results(
        experiment_id=experiment_config['id'],
        results=final_results,
        archival_config={
            'retention': 'permanent',
            'access': 'restricted',
            'format': 'preservation_ready'
        }
    )

    # Generate storage report
    storage_report = storage_manager.generate_storage_report(experiment_config['id'])

    return final_results, storage_report

# Execute experiment with storage integration
results, report = run_experiment_with_storage(active_inference_experiment_config)
```

## Contributing

We welcome contributions to the storage framework! See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for detailed guidelines.

### Development Setup
```bash
# Install storage dependencies
pip install -e ".[data_storage,dev]"

# Set up test databases
python tools/setup_test_databases.py

# Run storage tests
pytest tests/data_management/storage/ -v

# Benchmark storage performance
python benchmarks/storage_performance.py
```

### Priority Areas
- **New Storage Backends**: Integration with emerging storage technologies
- **Performance Optimization**: High-throughput and low-latency improvements
- **Security Enhancement**: Advanced encryption and access control
- **Metadata Systems**: Enhanced metadata and provenance tracking
- **Cloud Integration**: Multi-cloud and hybrid storage solutions

## Learning Resources

### Storage Best Practices
- **Database Design**: Database architecture and optimization
- **File System Organization**: Hierarchical data organization strategies
- **Cloud Storage**: Distributed storage and scalability patterns
- **Data Preservation**: Long-term archival and preservation techniques

### Technical Documentation
- **Storage API Reference**: Complete storage system API documentation
- **Performance Tuning**: Optimization and performance tuning guides
- **Security Guide**: Data security and access control implementation
- **Migration Guide**: Data migration and system upgrade procedures

## Related Documentation

- **[Data Management README](../README.md)**: Data management module overview
- **[Storage AGENTS.md](./AGENTS.md)**: Storage development guidelines
- **[Main AGENTS.md](../../../AGENTS.md)**: Project-wide agent guidelines
- **[Research AGENTS.md](../../AGENTS.md)**: Research tools module guidelines
- **[Contributing Guide](../../../CONTRIBUTING.md)**: Contribution processes

## Storage Architecture

### Unified Storage Architecture

```mermaid
graph TB
    A[Data Sources] --> B[Ingestion Layer]
    B --> C[Validation Layer]
    C --> D[Storage Router]
    D --> E[File Storage]
    D --> F[Database Storage]
    D --> G[Cloud Storage]
    D --> H[Archive Storage]

    I[Access Layer] --> E
    I --> F
    I --> G
    I --> H

    J[Metadata Store] -.-> E
    J -.-> F
    J -.-> G
    J -.-> H

    K[Backup System] -.-> E
    K -.-> F
    K -.-> G

    L[Monitoring] -.-> B
    L -.-> C
    L -.-> D
    L -.-> I
    L -.-> K
```

### Data Lifecycle Management

```mermaid
graph LR
    A[Data Ingestion] --> B[Quality Validation]
    B --> C{High Quality?}
    C -->|Yes| D[Primary Storage]
    C -->|No| E[Improvement]
    E --> B
    D --> F[Active Use]
    F --> G{Retention Time?}
    G -->|Active| D
    G -->|Archive| H[Archive Storage]
    G -->|Delete| I[Secure Deletion]
    H --> J[Long-term Preservation]
    I --> K[Deletion Record]
```

## Performance Benchmarks

### Storage Performance Metrics
- **Write Speed**: 500+ MB/s for sequential writes
- **Read Speed**: 1,000+ MB/s for sequential reads
- **Query Performance**: <50ms for complex queries
- **Concurrent Users**: 100+ simultaneous users
- **Data Integrity**: 99.9999% accuracy guarantee

### System Reliability
- **Uptime**: 99.95% system availability
- **Data Durability**: 99.999999999% (11 9's) durability
- **Recovery Time**: <4 hours for full system recovery
- **Backup Success**: 99.9% backup completion rate
- **Security Incidents**: Zero data breach incidents

---

*"Active Inference for, with, by Generative AI"* - Advancing research through comprehensive data storage, performance optimization, and reliable data management.
