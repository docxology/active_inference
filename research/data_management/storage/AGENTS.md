# Research Data Storage - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Research Data Storage module of the Active Inference Knowledge Environment. It outlines data storage methodologies, implementation patterns, and best practices for robust, scalable, and secure data storage throughout the research lifecycle.

## Data Storage Module Overview

The Research Data Storage module provides a comprehensive framework for storing, retrieving, and managing research data across multiple storage backends. It ensures data integrity, security, accessibility, and compliance while supporting various data formats and access patterns.

## Core Responsibilities

### Data Storage & Retrieval
- **Multi-Backend Storage**: Support for files, databases, cloud storage, and archives
- **Version Management**: Data versioning and change tracking
- **Metadata Management**: Comprehensive metadata and provenance tracking
- **Access Optimization**: Efficient data access and query optimization
- **Backup & Recovery**: Automated backup and disaster recovery

### Data Organization & Search
- **Hierarchical Organization**: Flexible data organization schemes
- **Advanced Search**: Full-text, semantic, and structured search capabilities
- **Indexing**: Efficient indexing for fast data retrieval
- **Tagging & Categorization**: Flexible data tagging and categorization
- **Cross-Reference**: Data relationship and dependency tracking

### Performance & Scalability
- **High-Performance Storage**: Optimized storage for research workloads
- **Horizontal Scaling**: Support for growing data volumes and users
- **Caching**: Intelligent caching for frequently accessed data
- **Compression**: Data compression for storage efficiency
- **Load Balancing**: Distributed load across storage systems

## Development Workflows

### Data Storage Development Process
1. **Requirements Analysis**: Analyze storage requirements and data characteristics
2. **Architecture Design**: Design storage architecture and data organization
3. **Backend Selection**: Choose appropriate storage technologies and configurations
4. **Implementation**: Implement storage interfaces and optimization
5. **Testing**: Test storage systems with realistic data volumes
6. **Performance Optimization**: Optimize for performance and scalability
7. **Security Implementation**: Implement security measures and access controls
8. **Documentation**: Document storage systems and usage patterns
9. **Deployment**: Deploy with monitoring and maintenance plans
10. **Review**: Regular review and performance monitoring

### File Storage Implementation
1. **Organization Design**: Design hierarchical file organization schemes
2. **Format Selection**: Choose appropriate file formats and compression
3. **Naming Conventions**: Implement consistent file naming and metadata
4. **Access Patterns**: Optimize for common access patterns and queries
5. **Backup Strategy**: Design backup and archival strategies
6. **Performance Testing**: Test with large file volumes and concurrent access
7. **Documentation**: Document file organization and access procedures

### Database Storage Implementation
1. **Schema Design**: Design database schemas for research data types
2. **Index Optimization**: Optimize indexes for query performance
3. **Connection Management**: Implement efficient connection pooling
4. **Query Optimization**: Optimize complex research queries
5. **Transaction Management**: Handle concurrent access and data integrity
6. **Migration Planning**: Plan for schema evolution and data migration
7. **Documentation**: Document database schemas and query patterns

## Quality Standards

### Storage Quality Standards
- **Data Integrity**: 100% data integrity and consistency guarantees
- **Availability**: 99.9% storage system availability
- **Durability**: 99.999999999% data durability (11 9's)
- **Performance**: Meet performance requirements for research workloads
- **Security**: Comprehensive security and access control

### Performance Standards
- **Access Latency**: Sub-100ms access times for typical queries
- **Throughput**: Support for high-volume data operations
- **Concurrent Users**: Support for multiple simultaneous users
- **Scalability**: Linear scaling with data volume and user count
- **Efficiency**: Optimal resource utilization and storage efficiency

### Reliability Standards
- **Backup Success**: 99.9% successful backup operations
- **Recovery Time**: <4 hours for full system recovery
- **Data Consistency**: ACID compliance where applicable
- **Error Recovery**: Robust error handling and recovery
- **Monitoring Coverage**: 100% system monitoring and alerting

## Implementation Patterns

### Data Storage Framework
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import logging

@dataclass
class StorageConfig:
    """Configuration for data storage"""
    storage_type: str  # file, database, cloud, archive
    location: str
    format: str
    compression: Optional[str] = None
    encryption: Optional[Dict[str, Any]] = None
    backup_config: Optional[Dict[str, Any]] = None
    access_config: Optional[Dict[str, Any]] = None

@dataclass
class StorageResult:
    """Result of storage operation"""
    storage_id: str
    timestamp: datetime
    data_size: int
    metadata_size: int
    access_paths: List[str]
    checksum: str
    version: str
    status: str

class BaseDataStorage(ABC):
    """Base class for data storage implementations"""

    def __init__(self, config: StorageConfig):
        """Initialize data storage with configuration"""
        self.config = config
        self.storage_id = f"{config.storage_type}_{id(self)}"
        self.logger = logging.getLogger(f"storage.{self.__class__.__name__}")
        self.access_stats = {
            'total_stores': 0,
            'total_retrieves': 0,
            'total_queries': 0,
            'total_errors': 0,
            'last_access': None
        }

    @abstractmethod
    def store_data(self, data: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Store data with metadata"""
        pass

    @abstractmethod
    def retrieve_data(self, data_id: str) -> Dict[str, Any]:
        """Retrieve data by ID"""
        pass

    @abstractmethod
    def query_data(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query stored data"""
        pass

    @abstractmethod
    def delete_data(self, data_id: str) -> bool:
        """Delete data by ID"""
        pass

    @abstractmethod
    def list_data(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List stored data with optional filters"""
        pass

    def update_access_stats(self, operation: str) -> None:
        """Update access statistics"""
        self.access_stats[f'total_{operation}s'] += 1
        self.access_stats['last_access'] = datetime.now()

    def get_access_stats(self) -> Dict[str, Any]:
        """Get access statistics"""
        return self.access_stats.copy()

class FileStorage(BaseDataStorage):
    """File-based data storage implementation"""

    def __init__(self, config: StorageConfig):
        """Initialize file storage"""
        super().__init__(config)
        self.storage_path = Path(config.location)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.file_index = self._load_file_index()

    def _load_file_index(self) -> Dict[str, Dict[str, Any]]:
        """Load file index from disk"""
        index_file = self.storage_path / '.file_index.json'
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_file_index(self) -> None:
        """Save file index to disk"""
        index_file = self.storage_path / '.file_index.json'
        with open(index_file, 'w') as f:
            json.dump(self.file_index, f, indent=2)

    def store_data(self, data: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Store data to file"""
        import uuid
        import hashlib

        data_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Create storage record
        storage_record = {
            'id': data_id,
            'timestamp': timestamp,
            'data': data,
            'metadata': metadata or {},
            'checksum': self._calculate_checksum(data)
        }

        # Determine file path
        file_path = self._get_file_path(data_id, metadata)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save data
        with open(file_path, 'w') as f:
            json.dump(storage_record, f, indent=2)

        # Update index
        self.file_index[data_id] = {
            'file_path': str(file_path),
            'timestamp': timestamp,
            'size': file_path.stat().st_size,
            'metadata': metadata or {}
        }
        self._save_file_index()

        self.update_access_stats('store')
        self.logger.info(f"Stored data {data_id} to {file_path}")

        return data_id

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate data checksum"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _get_file_path(self, data_id: str, metadata: Dict[str, Any] = None) -> Path:
        """Get file path for data ID"""
        # Use hierarchical organization by default
        if metadata and 'experiment_id' in metadata:
            experiment_id = metadata['experiment_id']
            subject_id = metadata.get('subject_id', 'unknown')
            data_type = metadata.get('data_type', 'general')
            return self.storage_path / experiment_id / subject_id / data_type / f"{data_id}.json"

        # Flat organization as fallback
        return self.storage_path / f"{data_id}.json"

    def retrieve_data(self, data_id: str) -> Dict[str, Any]:
        """Retrieve data from file"""
        if data_id not in self.file_index:
            raise FileNotFoundError(f"Data {data_id} not found")

        file_path = Path(self.file_index[data_id]['file_path'])

        if not file_path.exists():
            raise FileNotFoundError(f"Data file {file_path} not found")

        with open(file_path, 'r') as f:
            data_record = json.load(f)

        self.update_access_stats('retrieve')
        return data_record

    def query_data(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query files based on parameters"""
        results = []

        for data_id, file_info in self.file_index.items():
            if self._matches_query(file_info, query_params):
                try:
                    data_record = self.retrieve_data(data_id)
                    results.append(data_record)
                except Exception as e:
                    self.logger.warning(f"Error retrieving {data_id}: {str(e)}")

        return results

    def _matches_query(self, file_info: Dict[str, Any], query_params: Dict[str, Any]) -> bool:
        """Check if file info matches query parameters"""
        for key, value in query_params.items():
            if key == 'metadata':
                for meta_key, meta_value in value.items():
                    if file_info.get('metadata', {}).get(meta_key) != meta_value:
                        return False
            elif key == 'date_range':
                file_date = datetime.fromisoformat(file_info['timestamp'])
                start_date, end_date = value
                if not (start_date <= file_date <= end_date):
                    return False
            else:
                if file_info.get(key) != value:
                    return False

        return True

    def delete_data(self, data_id: str) -> bool:
        """Delete data file"""
        if data_id not in self.file_index:
            return False

        file_path = Path(self.file_index[data_id]['file_path'])

        if file_path.exists():
            file_path.unlink()
            del self.file_index[data_id]
            self._save_file_index()
            self.logger.info(f"Deleted data {data_id}")
            return True

        return False

    def list_data(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List all stored data"""
        if filters:
            return self.query_data(filters)

        results = []
        for data_id in self.file_index:
            try:
                data_record = self.retrieve_data(data_id)
                results.append(data_record)
            except Exception as e:
                self.logger.warning(f"Error listing {data_id}: {str(e)}")

        return results

class DatabaseStorage(BaseDataStorage):
    """Database-based data storage implementation"""

    def __init__(self, config: StorageConfig):
        """Initialize database storage"""
        super().__init__(config)
        self.db_path = config.location
        self.connection = None
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema"""
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            # Create main data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_records (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL,
                    metadata TEXT,
                    checksum TEXT,
                    version TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create metadata index table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_metadata (
                    data_id TEXT,
                    key TEXT,
                    value TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (data_id, key)
                )
            ''')

            # Create full-text search table
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS data_search USING fts5(
                    data_id, content, metadata
                )
            ''')

            conn.commit()

    def store_data(self, data: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Store data in database"""
        import uuid
        import sqlite3
        import json

        data_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        checksum = self._calculate_checksum(data)

        with sqlite3.connect(self.db_path) as conn:
            # Store main data
            conn.execute(
                'INSERT INTO data_records (id, timestamp, data, metadata, checksum, version) VALUES (?, ?, ?, ?, ?, ?)',
                (data_id, timestamp, json.dumps(data), json.dumps(metadata or {}), checksum, '1.0')
            )

            # Store metadata for indexing
            if metadata:
                for key, value in metadata.items():
                    conn.execute(
                        'INSERT INTO data_metadata (data_id, key, value) VALUES (?, ?, ?)',
                        (data_id, key, str(value))
                    )

            # Store for full-text search
            searchable_content = json.dumps({**data, **(metadata or {})})
            conn.execute(
                'INSERT INTO data_search (data_id, content, metadata) VALUES (?, ?, ?)',
                (data_id, searchable_content, json.dumps(metadata or {}))
            )

            conn.commit()

        self.update_access_stats('store')
        self.logger.info(f"Stored data {data_id} in database")

        return data_id

    def retrieve_data(self, data_id: str) -> Dict[str, Any]:
        """Retrieve data from database"""
        import sqlite3
        import json

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT data, metadata FROM data_records WHERE id = ?', (data_id,))
            result = cursor.fetchone()

            if not result:
                raise ValueError(f"Data {data_id} not found")

            data, metadata = result
            return {
                'id': data_id,
                'data': json.loads(data),
                'metadata': json.loads(metadata) if metadata else {}
            }

    def query_data(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query database for data"""
        import sqlite3
        import json

        with sqlite3.connect(self.db_path) as conn:
            # Build query based on parameters
            where_clauses = []
            params = []

            for key, value in query_params.items():
                if key == 'metadata':
                    for meta_key, meta_value in value.items():
                        where_clauses.append('dm.key = ? AND dm.value = ?')
                        params.extend([meta_key, str(meta_value)])
                elif key == 'full_text':
                    # Use full-text search
                    where_clauses.append('content MATCH ?')
                    params.append(value)
                else:
                    where_clauses.append(f'dr.{key} = ?')
                    params.append(value)

            where_clause = ' AND '.join(where_clauses) if where_clauses else '1=1'

            query = f'''
                SELECT DISTINCT dr.id, dr.data, dr.metadata, dr.timestamp
                FROM data_records dr
                LEFT JOIN data_metadata dm ON dr.id = dm.data_id
                WHERE {where_clause}
                ORDER BY dr.timestamp DESC
            '''

            cursor = conn.execute(query, params)
            results = []

            for row in cursor.fetchall():
                data_id, data, metadata, timestamp = row
                results.append({
                    'id': data_id,
                    'data': json.loads(data),
                    'metadata': json.loads(metadata) if metadata else {},
                    'timestamp': timestamp
                })

            return results

    def delete_data(self, data_id: str) -> bool:
        """Delete data from database"""
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            # Delete from all tables
            cursor = conn.execute('DELETE FROM data_records WHERE id = ?', (data_id,))
            deleted_main = cursor.rowcount > 0

            conn.execute('DELETE FROM data_metadata WHERE data_id = ?', (data_id,))
            conn.execute('DELETE FROM data_search WHERE data_id = ?', (data_id,))

            conn.commit()

            if deleted_main:
                self.logger.info(f"Deleted data {data_id} from database")
                return True

            return False

    def list_data(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List all data in database"""
        if filters:
            return self.query_data(filters)

        import sqlite3
        import json

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT id, data, metadata, timestamp FROM data_records ORDER BY timestamp DESC')
            results = []

            for row in cursor.fetchall():
                data_id, data, metadata, timestamp = row
                results.append({
                    'id': data_id,
                    'data': json.loads(data),
                    'metadata': json.loads(metadata) if metadata else {},
                    'timestamp': timestamp
                })

            return results

class StorageManager:
    """Manager for multiple storage backends"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize storage manager"""
        self.config = config
        self.storages: Dict[str, BaseDataStorage] = {}
        self.default_storage = config.get('default_storage', 'file')
        self._initialize_storages()

    def _initialize_storages(self) -> None:
        """Initialize configured storage backends"""
        storage_configs = self.config.get('storages', {})

        for storage_name, storage_config in storage_configs.items():
            storage_type = storage_config.get('type', 'file')

            if storage_type == 'file':
                storage = FileStorage(StorageConfig(**storage_config))
            elif storage_type == 'database':
                storage = DatabaseStorage(StorageConfig(**storage_config))
            # Add other storage types as needed

            self.storages[storage_name] = storage
            self.logger.info(f"Initialized storage backend: {storage_name}")

    def store_data(self, data: Dict[str, Any], metadata: Dict[str, Any] = None,
                   storage_name: Optional[str] = None) -> str:
        """Store data using specified or default storage"""
        storage = self._get_storage(storage_name)
        return storage.store_data(data, metadata)

    def retrieve_data(self, data_id: str, storage_name: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve data from specified or default storage"""
        storage = self._get_storage(storage_name)
        return storage.retrieve_data(data_id)

    def query_data(self, query_params: Dict[str, Any], storage_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query data across specified or default storage"""
        storage = self._get_storage(storage_name)
        return storage.query_data(query_params)

    def _get_storage(self, storage_name: Optional[str] = None) -> BaseDataStorage:
        """Get storage backend by name or default"""
        if storage_name and storage_name in self.storages:
            return self.storages[storage_name]
        return self.storages[self.default_storage]

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics for all storage backends"""
        stats = {}
        for name, storage in self.storages.items():
            stats[name] = storage.get_access_stats()
        return stats
```

### Version Management Pattern
```python
class DataVersionManager:
    """Manager for data versioning and provenance"""

    def __init__(self, storage_manager: StorageManager):
        """Initialize version manager"""
        self.storage_manager = storage_manager
        self.version_history: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = logging.getLogger("version.manager")

    def create_version(self, data_id: str, data: Dict[str, Any],
                      metadata: Dict[str, Any], version_type: str = 'minor') -> str:
        """Create new version of data"""
        # Get current version
        current_version = self.get_current_version(data_id)

        # Calculate new version
        new_version = self._calculate_new_version(current_version, version_type)

        # Create version metadata
        version_metadata = {
            'version': new_version,
            'previous_version': current_version,
            'timestamp': datetime.now().isoformat(),
            'author': metadata.get('author', 'system'),
            'description': metadata.get('description', f'Version {version_type} update'),
            'changes': self._analyze_changes(data_id, current_version, data),
            'version_type': version_type
        }

        # Store versioned data
        version_id = self.storage_manager.store_data(data, {
            **metadata,
            'version': new_version,
            'version_metadata': version_metadata
        })

        # Update version history
        if data_id not in self.version_history:
            self.version_history[data_id] = []

        self.version_history[data_id].append({
            'version_id': version_id,
            'version': new_version,
            'timestamp': version_metadata['timestamp'],
            'metadata': version_metadata
        })

        self.logger.info(f"Created version {new_version} for data {data_id}")
        return version_id

    def get_version_history(self, data_id: str) -> List[Dict[str, Any]]:
        """Get version history for data"""
        return self.version_history.get(data_id, [])

    def get_current_version(self, data_id: str) -> str:
        """Get current version of data"""
        history = self.get_version_history(data_id)
        if not history:
            return '1.0.0'

        # Sort by timestamp and get latest
        latest = max(history, key=lambda x: x['timestamp'])
        return latest['version']

    def retrieve_version(self, data_id: str, version: str) -> Dict[str, Any]:
        """Retrieve specific version of data"""
        history = self.get_version_history(data_id)

        for version_record in history:
            if version_record['version'] == version:
                return self.storage_manager.retrieve_data(version_record['version_id'])

        raise ValueError(f"Version {version} not found for data {data_id}")

    def _calculate_new_version(self, current_version: str, version_type: str) -> str:
        """Calculate new version number"""
        major, minor, patch = map(int, current_version.split('.'))

        if version_type == 'major':
            return f"{major + 1}.0.0"
        elif version_type == 'minor':
            return f"{major}.{minor + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{patch + 1}"

    def _analyze_changes(self, data_id: str, old_version: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze changes between versions"""
        # This would implement detailed change analysis
        # For now, return simple summary
        return {
            'change_type': 'content_update',
            'summary': 'Data content updated',
            'timestamp': datetime.now().isoformat()
        }
```

## Testing Guidelines

### Storage Testing
- **Unit Tests**: Test individual storage operations and methods
- **Integration Tests**: Test storage with realistic data volumes and patterns
- **Performance Tests**: Test storage performance under load
- **Reliability Tests**: Test backup, recovery, and error handling
- **Security Tests**: Validate security measures and access controls

### Quality Assurance
- **Data Integrity Testing**: Ensure data integrity across operations
- **Performance Validation**: Validate performance with large datasets
- **Backup Testing**: Test backup and recovery procedures
- **Security Validation**: Validate security measures and compliance
- **Scalability Testing**: Test system scaling with data growth

## Performance Considerations

### Storage Performance
- **Access Optimization**: Optimize data access patterns and caching
- **Query Performance**: Optimize queries and indexing strategies
- **Compression**: Use appropriate compression for storage efficiency
- **Parallel Access**: Support concurrent read/write operations
- **Network Optimization**: Minimize network overhead for distributed storage

### Scalability
- **Data Volume Scaling**: Maintain performance with growing data volumes
- **User Scaling**: Support increasing numbers of concurrent users
- **Geographic Scaling**: Support distributed and multi-region deployments
- **Backup Scaling**: Scale backup operations with data growth
- **Monitoring Scaling**: Scale monitoring with system complexity

## Maintenance and Evolution

### Storage System Updates
- **Schema Evolution**: Manage data schema changes and migrations
- **Performance Tuning**: Continuous performance optimization
- **Security Updates**: Regular security patches and updates
- **Capacity Planning**: Plan for data growth and resource needs

### Technology Evolution
- **New Storage Technologies**: Evaluate and adopt new storage solutions
- **Performance Improvements**: Implement storage performance enhancements
- **Security Enhancements**: Update security measures and protocols
- **Compliance Updates**: Update for regulatory and compliance changes

## Common Challenges and Solutions

### Challenge: Data Volume Growth
**Solution**: Implement data lifecycle management, compression, archiving, and horizontal scaling strategies.

### Challenge: Performance Degradation
**Solution**: Implement comprehensive monitoring, performance profiling, query optimization, and caching strategies.

### Challenge: Data Consistency
**Solution**: Use appropriate consistency models, implement comprehensive validation, and provide rollback capabilities.

### Challenge: Security and Compliance
**Solution**: Implement comprehensive security measures, regular compliance audits, and automated security monitoring.

## Getting Started as an Agent

### Development Setup
1. **Study Storage Architecture**: Understand storage system design and patterns
2. **Learn Data Characteristics**: Study different data types and access patterns
3. **Practice Implementation**: Practice implementing storage interfaces and optimization
4. **Understand Security**: Learn storage security and access control requirements

### Contribution Process
1. **Identify Storage Needs**: Find gaps in current storage capabilities
2. **Research Storage Technologies**: Study relevant storage technologies and solutions
3. **Design Storage Solutions**: Create detailed storage system designs
4. **Implement and Test**: Follow security and performance implementation standards
5. **Validate Thoroughly**: Ensure storage reliability, performance, and security
6. **Document Completely**: Provide comprehensive storage documentation
7. **Security Review**: Submit for security and compliance review

### Learning Resources
- **Storage Systems**: Study data storage systems and architectures
- **Database Design**: Learn database design and optimization
- **File Systems**: Understand file system organization and performance
- **Cloud Storage**: Learn cloud storage and distributed systems
- **Security Standards**: Study data security and privacy protection

## Related Documentation

- **[Storage README](./README.md)**: Data storage module overview
- **[Data Management AGENTS.md](../AGENTS.md)**: Data management development guidelines
- **[Main AGENTS.md](../../../AGENTS.md)**: Project-wide agent guidelines
- **[Research AGENTS.md](../../AGENTS.md)**: Research tools module guidelines
- **[Contributing Guide](../../../CONTRIBUTING.md)**: Contribution processes

---

*"Active Inference for, with, by Generative AI"* - Advancing research through comprehensive data storage, performance optimization, and reliable data management.
