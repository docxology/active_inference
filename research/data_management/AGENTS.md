# Research Data Management - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Research Data Management module of the Active Inference Knowledge Environment. It outlines data management methodologies, implementation patterns, and best practices for robust, reproducible data handling throughout the research lifecycle.

## Data Management Module Overview

The Research Data Management module provides a comprehensive framework for managing research data from collection through analysis to publication. It ensures data integrity, reproducibility, security, and compliance with research standards across all stages of the research process.

## Core Responsibilities

### Data Collection & Acquisition
- **Automated Collection**: Implement automated data collection systems
- **Integration Tools**: Connect with external data sources and APIs
- **Real-time Streaming**: Support for real-time data acquisition
- **Quality Assurance**: Validate data at point of collection
- **Metadata Management**: Comprehensive metadata capture and management

### Data Storage & Organization
- **Storage Systems**: Implement scalable storage solutions
- **Organization Schemes**: Develop data organization frameworks
- **Version Control**: Data versioning and change tracking
- **Backup Systems**: Automated backup and recovery procedures
- **Access Control**: Role-based data access management

### Data Processing & Preprocessing
- **Cleaning Tools**: Data cleaning and preprocessing utilities
- **Transformation**: Data format conversion and standardization
- **Integration**: Data integration from multiple sources
- **Quality Enhancement**: Data quality improvement methods
- **Feature Engineering**: Feature extraction and engineering tools

### Data Validation & Quality
- **Validation Frameworks**: Comprehensive data validation systems
- **Quality Metrics**: Data quality assessment tools
- **Anomaly Detection**: Outlier and anomaly detection methods
- **Integrity Checks**: Data integrity verification procedures
- **Compliance**: Research compliance and standards adherence

### Data Security & Ethics
- **Security Protocols**: Data security and encryption
- **Privacy Protection**: Personal data protection measures
- **Access Control**: Fine-grained access control systems
- **Audit Trails**: Complete audit and logging systems
- **Ethical Compliance**: Research ethics and compliance tools

## Development Workflows

### Data Management Development Process
1. **Requirements Analysis**: Analyze data management requirements
2. **Standards Research**: Research data management standards and best practices
3. **System Design**: Design data management architecture and workflows
4. **Implementation**: Implement data management tools and systems
5. **Validation**: Validate against data management standards
6. **Testing**: Comprehensive testing including edge cases
7. **Documentation**: Create comprehensive data management documentation
8. **Training**: Develop user training and support materials
9. **Deployment**: Deploy with monitoring and maintenance plans
10. **Review**: Regular review and improvement cycles

### Data Collection Implementation
1. **Source Analysis**: Analyze data sources and requirements
2. **Protocol Design**: Design data collection protocols
3. **Tool Development**: Develop collection tools and interfaces
4. **Integration**: Integrate with existing data systems
5. **Testing**: Test collection systems thoroughly
6. **Validation**: Validate collected data quality
7. **Documentation**: Document collection procedures
8. **Training**: Train users on collection methods

### Data Storage Architecture
1. **Requirements Gathering**: Gather storage requirements
2. **Architecture Design**: Design storage architecture
3. **Technology Selection**: Select appropriate storage technologies
4. **Implementation**: Implement storage systems
5. **Performance Optimization**: Optimize for performance and scalability
6. **Security Implementation**: Implement security measures
7. **Backup Planning**: Design backup and recovery systems
8. **Monitoring**: Implement monitoring and alerting

## Quality Standards

### Data Quality Standards
- **Accuracy**: Ensure data accuracy and correctness
- **Completeness**: Maintain data completeness
- **Consistency**: Ensure data consistency across sources
- **Timeliness**: Maintain data currency and relevance
- **Validity**: Ensure data validity and format compliance

### Security Standards
- **Confidentiality**: Protect sensitive and confidential data
- **Integrity**: Maintain data integrity and prevent corruption
- **Availability**: Ensure data availability when needed
- **Authentication**: Implement proper authentication mechanisms
- **Authorization**: Fine-grained access control

### Ethical Standards
- **Privacy Protection**: Protect personal and sensitive data
- **Informed Consent**: Support informed consent processes
- **Data Sharing**: Enable ethical data sharing practices
- **Compliance**: Ensure regulatory compliance
- **Transparency**: Maintain transparent data practices

## Implementation Patterns

### Data Collection Framework
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging

@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    type: str  # api, database, file, sensor, etc.
    connection_info: Dict[str, Any]
    schema: Dict[str, Any]
    update_frequency: str  # real-time, hourly, daily, etc.
    metadata: Dict[str, Any] = None

@dataclass
class CollectionConfig:
    """Data collection configuration"""
    sources: List[DataSource]
    collection_interval: int  # seconds
    quality_checks: List[str]
    error_handling: str  # retry, skip, fail
    storage_config: Dict[str, Any]
    notification_settings: Dict[str, Any] = None

class BaseDataCollector(ABC):
    """Base class for data collection"""

    def __init__(self, config: CollectionConfig):
        """Initialize data collector"""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collection_stats = {
            'total_collected': 0,
            'errors': 0,
            'last_collection': None
        }

    @abstractmethod
    def connect_source(self, source: DataSource) -> bool:
        """Connect to data source"""
        pass

    @abstractmethod
    def collect_data(self, source: DataSource) -> Dict[str, Any]:
        """Collect data from source"""
        pass

    @abstractmethod
    def validate_data(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate collected data"""
        pass

    def run_collection_cycle(self) -> Dict[str, Any]:
        """Run complete data collection cycle"""
        cycle_results = {
            'timestamp': datetime.now(),
            'sources_processed': 0,
            'data_collected': 0,
            'errors': [],
            'validation_issues': []
        }

        for source in self.config.sources:
            try:
                # Connect to source
                if not self.connect_source(source):
                    error = f"Failed to connect to source: {source.name}"
                    self.logger.error(error)
                    cycle_results['errors'].append(error)
                    continue

                # Collect data
                raw_data = self.collect_data(source)
                self.collection_stats['total_collected'] += len(raw_data)

                # Validate data
                validation_issues = self.validate_data(raw_data, source.schema)
                if validation_issues:
                    cycle_results['validation_issues'].extend(validation_issues)
                    if self.config.error_handling == 'fail':
                        raise ValueError(f"Validation failed: {validation_issues}")

                # Store data
                self.store_data(raw_data, source)
                cycle_results['sources_processed'] += 1
                cycle_results['data_collected'] += len(raw_data)

                self.logger.info(f"Successfully collected {len(raw_data)} records from {source.name}")

            except Exception as e:
                error = f"Error collecting from {source.name}: {str(e)}"
                self.logger.error(error)
                cycle_results['errors'].append(error)
                self.collection_stats['errors'] += 1

                if self.config.error_handling == 'fail':
                    raise

        cycle_results['duration'] = (datetime.now() - cycle_results['timestamp']).total_seconds()
        self.collection_stats['last_collection'] = datetime.now()

        return cycle_results

    def store_data(self, data: Dict[str, Any], source: DataSource) -> None:
        """Store collected data"""
        # Implementation depends on storage backend
        pass

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return self.collection_stats.copy()

class APIDataCollector(BaseDataCollector):
    """API-based data collector implementation"""

    def connect_source(self, source: DataSource) -> bool:
        """Connect to API source"""
        try:
            import requests

            # Test connection
            response = requests.get(
                source.connection_info['base_url'] + '/health',
                timeout=10
            )
            return response.status_code == 200

        except Exception as e:
            self.logger.error(f"API connection failed: {str(e)}")
            return False

    def collect_data(self, source: DataSource) -> Dict[str, Any]:
        """Collect data from API"""
        import requests

        endpoint = source.connection_info.get('endpoint', '/data')
        params = source.connection_info.get('params', {})

        response = requests.get(
            source.connection_info['base_url'] + endpoint,
            params=params,
            headers=source.connection_info.get('headers', {}),
            timeout=30
        )

        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code}")

        return response.json()

    def validate_data(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate API data against schema"""
        issues = []

        # Check required fields
        for field, requirements in schema.items():
            if requirements.get('required', False):
                if field not in data:
                    issues.append(f"Required field '{field}' missing")

        # Type checking
        for field, value in data.items():
            if field in schema:
                expected_type = schema[field].get('type')
                if expected_type and not isinstance(value, expected_type):
                    issues.append(f"Field '{field}' type mismatch: expected {expected_type}, got {type(value)}")

        return issues

class DataCollectionManager:
    """Manager for multiple data collection processes"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data collection manager"""
        self.config = config
        self.collectors: Dict[str, BaseDataCollector] = {}
        self.collection_schedules: Dict[str, Any] = {}
        self.is_running = False

    def register_collector(self, name: str, collector: BaseDataCollector) -> None:
        """Register data collector"""
        self.collectors[name] = collector
        self.logger.info(f"Registered collector: {name}")

    def start_collection(self, collector_name: str) -> bool:
        """Start data collection for specified collector"""
        if collector_name not in self.collectors:
            raise ValueError(f"Unknown collector: {collector_name}")

        collector = self.collectors[collector_name]

        try:
            results = collector.run_collection_cycle()
            self.logger.info(f"Collection completed for {collector_name}: {results}")
            return True

        except Exception as e:
            self.logger.error(f"Collection failed for {collector_name}: {str(e)}")
            return False

    def start_scheduled_collection(self) -> None:
        """Start scheduled collection for all collectors"""
        import time
        import threading

        self.is_running = True

        def collection_worker():
            while self.is_running:
                for name, collector in self.collectors.items():
                    try:
                        self.start_collection(name)
                    except Exception as e:
                        self.logger.error(f"Scheduled collection failed for {name}: {str(e)}")

                # Wait for next collection cycle
                time.sleep(self.config.get('collection_interval', 3600))

        thread = threading.Thread(target=collection_worker, daemon=True)
        thread.start()
        self.logger.info("Started scheduled data collection")

    def stop_collection(self) -> None:
        """Stop scheduled collection"""
        self.is_running = False
        self.logger.info("Stopped data collection")

    def get_collection_status(self) -> Dict[str, Any]:
        """Get collection status for all collectors"""
        status = {}
        for name, collector in self.collectors.items():
            status[name] = collector.get_collection_stats()
        return status
```

### Data Storage Framework
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import sqlite3
import pandas as pd
from datetime import datetime

@dataclass
class StorageConfig:
    """Storage configuration"""
    type: str  # file, database, cloud, etc.
    location: str
    format: str  # json, csv, parquet, hdf5, etc.
    compression: Optional[str] = None
    encryption: Optional[Dict[str, Any]] = None
    backup_config: Optional[Dict[str, Any]] = None

class BaseDataStorage(ABC):
    """Base class for data storage"""

    def __init__(self, config: StorageConfig):
        """Initialize data storage"""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def store(self, data: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Store data with metadata"""
        pass

    @abstractmethod
    def retrieve(self, data_id: str) -> Dict[str, Any]:
        """Retrieve data by ID"""
        pass

    @abstractmethod
    def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query stored data"""
        pass

    @abstractmethod
    def delete(self, data_id: str) -> bool:
        """Delete data by ID"""
        pass

    @abstractmethod
    def list_data(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List stored data with optional filters"""
        pass

class FileStorage(BaseDataStorage):
    """File-based data storage"""

    def __init__(self, config: StorageConfig):
        """Initialize file storage"""
        super().__init__(config)
        self.storage_path = Path(config.location)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def store(self, data: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Store data to file"""
        import uuid

        data_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Create data package
        data_package = {
            'id': data_id,
            'timestamp': timestamp,
            'data': data,
            'metadata': metadata or {}
        }

        # Determine file format
        if self.config.format == 'json':
            file_path = self.storage_path / f"{data_id}.json"
            with open(file_path, 'w') as f:
                json.dump(data_package, f, indent=2)

        elif self.config.format == 'csv':
            file_path = self.storage_path / f"{data_id}.csv"
            df = pd.DataFrame([data])
            df.to_csv(file_path, index=False)

        elif self.config.format == 'parquet':
            file_path = self.storage_path / f"{data_id}.parquet"
            df = pd.DataFrame([data])
            df.to_parquet(file_path)

        self.logger.info(f"Stored data {data_id} to {file_path}")
        return data_id

    def retrieve(self, data_id: str) -> Dict[str, Any]:
        """Retrieve data from file"""
        file_path = self.storage_path / f"{data_id}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Data {data_id} not found")

        with open(file_path, 'r') as f:
            data_package = json.load(f)

        return data_package

    def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query files based on parameters"""
        results = []

        # Simple implementation - scan all files
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data_package = json.load(f)

                # Check if matches query parameters
                if self._matches_query(data_package, query_params):
                    results.append(data_package)

            except Exception as e:
                self.logger.warning(f"Error reading {file_path}: {str(e)}")

        return results

    def _matches_query(self, data_package: Dict[str, Any], query_params: Dict[str, Any]) -> bool:
        """Check if data package matches query parameters"""
        for key, value in query_params.items():
            if key == 'metadata':
                for meta_key, meta_value in value.items():
                    if data_package.get('metadata', {}).get(meta_key) != meta_value:
                        return False
            elif key == 'date_range':
                data_date = datetime.fromisoformat(data_package['timestamp'])
                start_date, end_date = value
                if not (start_date <= data_date <= end_date):
                    return False
            else:
                if data_package.get(key) != value:
                    return False

        return True

    def delete(self, data_id: str) -> bool:
        """Delete data file"""
        file_path = self.storage_path / f"{data_id}.json"

        if file_path.exists():
            file_path.unlink()
            self.logger.info(f"Deleted data {data_id}")
            return True

        return False

    def list_data(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List all data files"""
        data_list = []

        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data_package = json.load(f)
                data_list.append(data_package)
            except Exception as e:
                self.logger.warning(f"Error reading {file_path}: {str(e)}")

        return data_list

class DatabaseStorage(BaseDataStorage):
    """Database-based data storage"""

    def __init__(self, config: StorageConfig):
        """Initialize database storage"""
        super().__init__(config)
        self.db_path = config.location
        self.init_database()

    def init_database(self) -> None:
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_records (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_versions (
                    id TEXT,
                    version INTEGER,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    PRIMARY KEY (id, version)
                )
            ''')

            conn.commit()

    def store(self, data: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Store data in database"""
        import uuid

        data_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO data_records (id, timestamp, data, metadata) VALUES (?, ?, ?, ?)',
                (data_id, timestamp, json.dumps(data), json.dumps(metadata or {}))
            )

            # Store initial version
            conn.execute(
                'INSERT INTO data_versions (id, version, data, timestamp) VALUES (?, ?, ?, ?)',
                (data_id, 1, json.dumps(data), timestamp)
            )

            conn.commit()

        self.logger.info(f"Stored data {data_id} in database")
        return data_id

    def retrieve(self, data_id: str) -> Dict[str, Any]:
        """Retrieve data from database"""
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

    def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT id, data, metadata FROM data_records')

            results = []
            for row in cursor.fetchall():
                data_id, data, metadata = row
                data_package = {
                    'id': data_id,
                    'data': json.loads(data),
                    'metadata': json.loads(metadata) if metadata else {}
                }

                if self._matches_query(data_package, query_params):
                    results.append(data_package)

            return results

    def _matches_query(self, data_package: Dict[str, Any], query_params: Dict[str, Any]) -> bool:
        """Check if database record matches query"""
        # Implementation similar to file storage
        for key, value in query_params.items():
            if key == 'metadata':
                for meta_key, meta_value in value.items():
                    if data_package.get('metadata', {}).get(meta_key) != meta_value:
                        return False
            else:
                if data_package.get(key) != value:
                    return False

        return True

    def delete(self, data_id: str) -> bool:
        """Delete data from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('DELETE FROM data_records WHERE id = ?', (data_id,))
            deleted = cursor.rowcount > 0

            if deleted:
                conn.execute('DELETE FROM data_versions WHERE id = ?', (data_id,))
                conn.commit()
                self.logger.info(f"Deleted data {data_id} from database")

            return deleted

    def list_data(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List all data in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT id, data, metadata FROM data_records')

            results = []
            for row in cursor.fetchall():
                data_id, data, metadata = row
                results.append({
                    'id': data_id,
                    'data': json.loads(data),
                    'metadata': json.loads(metadata) if metadata else {}
                })

            return results
```

## Testing Guidelines

### Data Management Testing
- **Collection Testing**: Test data collection from various sources
- **Storage Testing**: Validate storage and retrieval operations
- **Validation Testing**: Test data validation and quality checks
- **Security Testing**: Validate security and access control
- **Performance Testing**: Test with large datasets and high throughput

### Quality Assurance
- **Data Integrity**: Ensure data integrity throughout lifecycle
- **Backup Testing**: Test backup and recovery procedures
- **Security Validation**: Validate security measures
- **Compliance Testing**: Ensure regulatory compliance
- **Performance Validation**: Validate performance requirements

## Performance Considerations

### Data Collection Performance
- **Throughput Optimization**: Optimize data collection throughput
- **Connection Pooling**: Efficient connection management
- **Batch Processing**: Batch data collection operations
- **Error Recovery**: Robust error handling and recovery

### Storage Performance
- **Query Optimization**: Optimize data queries
- **Indexing**: Implement appropriate indexing strategies
- **Compression**: Use data compression where beneficial
- **Caching**: Implement caching for frequently accessed data

## Maintenance and Evolution

### Data Management Updates
- **Schema Evolution**: Manage data schema changes
- **Migration Tools**: Develop data migration utilities
- **Performance Monitoring**: Monitor system performance
- **Capacity Planning**: Plan for data growth

### Security Updates
- **Security Patching**: Keep security measures current
- **Access Review**: Regular access control reviews
- **Audit Analysis**: Analyze audit logs for issues
- **Compliance Updates**: Update for regulatory changes

## Common Challenges and Solutions

### Challenge: Data Quality
**Solution**: Implement comprehensive validation at collection time and quality monitoring throughout the lifecycle.

### Challenge: Data Security
**Solution**: Use encryption, access controls, and regular security audits.

### Challenge: Scalability
**Solution**: Design for horizontal scaling and implement efficient storage strategies.

### Challenge: Integration
**Solution**: Use standard interfaces and comprehensive error handling.

## Getting Started as an Agent

### Development Setup
1. **Study Data Architecture**: Understand data management architecture
2. **Learn Standards**: Study data management standards and best practices
3. **Practice Implementation**: Practice implementing data management solutions
4. **Understand Security**: Learn data security and privacy requirements

### Contribution Process
1. **Identify Data Needs**: Find gaps in current data management capabilities
2. **Research Standards**: Study relevant data management standards
3. **Design Solutions**: Create detailed system designs
4. **Implement and Test**: Follow security and quality implementation standards
5. **Validate Thoroughly**: Ensure data integrity and security
6. **Document Completely**: Provide comprehensive documentation
7. **Security Review**: Submit for security and compliance review

### Learning Resources
- **Data Management**: Study data management methodologies
- **Security Standards**: Learn data security best practices
- **Privacy Regulations**: Understand privacy and compliance requirements
- **Database Design**: Master database and storage design
- **API Development**: Learn API development for data collection

## Related Documentation

- **[Data README](./README.md)**: Data management module overview
- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Research AGENTS.md](../AGENTS.md)**: Research tools module guidelines
- **[Security Tools](../../platform/security/)**: Security and compliance tools
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution processes

---

*"Active Inference for, with, by Generative AI"* - Advancing research through comprehensive data management, robust security practices, and ethical data stewardship.
