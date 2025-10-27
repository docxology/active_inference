# Research Data Collection - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Research Data Collection module of the Active Inference Knowledge Environment. It outlines data collection methodologies, implementation patterns, and best practices for robust, automated data acquisition throughout the research lifecycle.

## Data Collection Module Overview

The Research Data Collection module provides a comprehensive framework for acquiring research data from multiple sources including APIs, files, sensors, and real-time streams. It ensures data quality, security, and reproducibility while supporting both batch and real-time collection patterns.

## Core Responsibilities

### Automated Data Acquisition
- **API Integration**: Implement RESTful API data collection with authentication
- **File Monitoring**: Automated file system monitoring and batch import
- **Real-time Streaming**: Live data stream processing and event handling
- **Sensor Networks**: IoT sensor and device data acquisition
- **Web Collection**: Ethical web data collection and harvesting

### Data Quality Assurance
- **Real-time Validation**: Immediate data validation at collection point
- **Quality Gates**: Automated quality checkpoints and filtering
- **Error Handling**: Robust error recovery and retry mechanisms
- **Metadata Capture**: Comprehensive metadata and provenance tracking
- **Anomaly Detection**: Real-time anomaly and outlier detection

### Performance Optimization
- **High-throughput Collection**: Optimized for large-scale data acquisition
- **Parallel Processing**: Multi-source concurrent data collection
- **Resource Management**: Efficient CPU, memory, and network utilization
- **Load Balancing**: Dynamic resource allocation based on data volume
- **Caching Strategies**: Intelligent caching for improved performance

## Development Workflows

### Data Collection Development Process
1. **Requirements Analysis**: Analyze data collection requirements and constraints
2. **Source Evaluation**: Assess data source availability and quality
3. **Protocol Design**: Design collection protocols and data formats
4. **Implementation**: Implement collection interfaces and error handling
5. **Testing**: Test collection systems with real and simulated data
6. **Validation**: Validate data quality and completeness
7. **Optimization**: Optimize for performance and reliability
8. **Documentation**: Document collection procedures and APIs
9. **Deployment**: Deploy with monitoring and maintenance plans
10. **Review**: Regular review and improvement cycles

### API Collection Implementation
1. **API Analysis**: Analyze target API specifications and requirements
2. **Authentication Setup**: Implement secure authentication mechanisms
3. **Rate Limit Handling**: Implement rate limiting and backoff strategies
4. **Error Recovery**: Develop robust error handling and retry logic
5. **Data Transformation**: Transform API responses to internal formats
6. **Validation Integration**: Integrate with data validation pipeline
7. **Performance Testing**: Test with various data volumes and conditions
8. **Documentation**: Document API integration and usage patterns

### Real-time Stream Implementation
1. **Protocol Selection**: Choose appropriate streaming protocols (WebSocket, MQTT, etc.)
2. **Connection Management**: Implement robust connection handling
3. **Message Processing**: Develop efficient message parsing and processing
4. **Quality Filtering**: Implement real-time quality assessment
5. **Buffer Management**: Design efficient buffering and flushing strategies
6. **Monitoring**: Implement comprehensive monitoring and alerting
7. **Recovery**: Develop reconnection and recovery mechanisms
8. **Scaling**: Plan for horizontal scaling and load distribution

## Quality Standards

### Collection Quality Standards
- **Success Rate**: >99% successful data collection operations
- **Data Completeness**: >98% of expected data fields collected
- **Timeliness**: <5% of data older than collection interval
- **Accuracy**: <1% data corruption or transformation errors
- **Metadata Coverage**: 100% of collections include comprehensive metadata

### Performance Standards
- **Throughput**: Maintain target collection rates under various loads
- **Latency**: Minimize collection-to-storage latency
- **Resource Efficiency**: Optimize CPU, memory, and network usage
- **Scalability**: Support increasing data volumes and sources
- **Reliability**: Maintain operation during network and source issues

### Security Standards
- **Authentication**: Secure authentication for all data sources
- **Encryption**: Encrypt data in transit and at rest where required
- **Access Control**: Implement appropriate access controls
- **Audit Logging**: Comprehensive audit trails for all collection activities
- **Privacy Protection**: Respect data privacy and consent requirements

## Implementation Patterns

### Data Collection Framework
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio

@dataclass
class CollectionSource:
    """Configuration for data collection source"""
    source_id: str
    source_type: str  # api, file, stream, sensor, web
    connection_config: Dict[str, Any]
    collection_config: Dict[str, Any]
    validation_config: Dict[str, Any]
    security_config: Dict[str, Any]
    metadata: Dict[str, Any] = None

@dataclass
class CollectionResult:
    """Result of data collection operation"""
    source_id: str
    timestamp: datetime
    records_collected: int
    bytes_collected: int
    quality_score: float
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    duration_seconds: float

class BaseDataCollector(ABC):
    """Base class for all data collection implementations"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data collector with configuration"""
        self.config = config
        self.source_id = config.get('source_id', 'unknown')
        self.logger = logging.getLogger(f"collection.{self.__class__.__name__}")
        self.collection_stats = {
            'total_collected': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'last_collection': None,
            'average_quality': 0.0
        }

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source"""
        pass

    @abstractmethod
    async def collect_data(self) -> Dict[str, Any]:
        """Collect data from source"""
        pass

    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate collected data"""
        pass

    async def run_collection_cycle(self) -> CollectionResult:
        """Run complete data collection cycle"""
        start_time = datetime.now()
        cycle_result = CollectionResult(
            source_id=self.source_id,
            timestamp=start_time,
            records_collected=0,
            bytes_collected=0,
            quality_score=0.0,
            errors=[],
            warnings=[],
            metadata={},
            duration_seconds=0.0
        )

        try:
            self.logger.info(f"Starting collection cycle for {self.source_id}")

            # Establish connection
            connection_success = await self.connect()
            if not connection_success:
                error_msg = f"Failed to connect to source: {self.source_id}"
                self.logger.error(error_msg)
                cycle_result.errors.append(error_msg)
                cycle_result.duration_seconds = (datetime.now() - start_time).total_seconds()
                return cycle_result

            # Collect data
            raw_data = await self.collect_data()
            if not raw_data:
                warning_msg = f"No data collected from source: {self.source_id}"
                self.logger.warning(warning_msg)
                cycle_result.warnings.append(warning_msg)
                await self.disconnect()
                cycle_result.duration_seconds = (datetime.now() - start_time).total_seconds()
                return cycle_result

            # Validate data
            validation_result = self.validate_data(raw_data)
            cycle_result.records_collected = validation_result.get('record_count', 0)
            cycle_result.bytes_collected = validation_result.get('byte_count', 0)
            cycle_result.quality_score = validation_result.get('quality_score', 0.0)
            cycle_result.errors.extend(validation_result.get('errors', []))
            cycle_result.warnings.extend(validation_result.get('warnings', []))
            cycle_result.metadata = validation_result.get('metadata', {})

            # Update statistics
            self.collection_stats['total_collected'] += cycle_result.records_collected
            self.collection_stats['total_errors'] += len(cycle_result.errors)
            self.collection_stats['total_warnings'] += len(cycle_result.warnings)
            self.collection_stats['last_collection'] = datetime.now()

            # Update average quality
            if cycle_result.records_collected > 0:
                total_quality = self.collection_stats['average_quality'] * (self.collection_stats['total_collected'] - cycle_result.records_collected)
                total_quality += cycle_result.quality_score * cycle_result.records_collected
                self.collection_stats['average_quality'] = total_quality / self.collection_stats['total_collected']

            # Close connection
            await self.disconnect()

            self.logger.info(f"Collection cycle completed for {self.source_id}: "
                           f"{cycle_result.records_collected} records, "
                           f"quality: {cycle_result.quality_score:.3f}")

        except Exception as e:
            error_msg = f"Collection cycle failed for {self.source_id}: {str(e)}"
            self.logger.error(error_msg)
            cycle_result.errors.append(error_msg)
            cycle_result.duration_seconds = (datetime.now() - start_time).total_seconds()

            # Ensure connection is closed
            try:
                await self.disconnect()
            except Exception:
                pass  # Connection might already be closed

        return cycle_result

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return self.collection_stats.copy()

class APICollector(BaseDataCollector):
    """API-based data collection implementation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize API collector"""
        super().__init__(config)
        self.base_url = config.get('base_url')
        self.headers = config.get('headers', {})
        self.auth_config = config.get('auth', {})
        self.rate_limits = config.get('rate_limits', {})
        self.retry_config = config.get('retry', {})
        self.session = None

    async def connect(self) -> bool:
        """Connect to API endpoint"""
        try:
            import aiohttp

            # Set up authentication
            if self.auth_config.get('type') == 'bearer_token':
                self.headers['Authorization'] = f"Bearer {self.auth_config['token']}"
            elif self.auth_config.get('type') == 'api_key':
                self.headers[self.auth_config['header_name']] = self.auth_config['key']

            # Create session with timeout and rate limiting
            timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 30))
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )

            # Test connection
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200

        except Exception as e:
            self.logger.error(f"API connection failed: {str(e)}")
            return False

    async def disconnect(self) -> None:
        """Close API connection"""
        if self.session:
            await self.session.close()
            self.session = None

    async def collect_data(self) -> Dict[str, Any]:
        """Collect data from API"""
        if not self.session:
            raise RuntimeError("Not connected to API")

        endpoint = self.config.get('endpoint', '/data')
        params = self.config.get('params', {})

        async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
            if response.status != 200:
                raise ValueError(f"API request failed: {response.status}")

            return await response.json()

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate API response data"""
        validation_result = {
            'record_count': 0,
            'byte_count': 0,
            'quality_score': 0.0,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }

        try:
            # Count records
            if isinstance(data, list):
                validation_result['record_count'] = len(data)
            elif isinstance(data, dict):
                validation_result['record_count'] = 1
                validation_result['metadata'] = data.get('metadata', {})
            else:
                validation_result['errors'].append("Invalid data format: expected dict or list")
                return validation_result

            # Calculate data size
            validation_result['byte_count'] = len(str(data).encode('utf-8'))

            # Validate data structure
            schema = self.config.get('schema', {})
            if schema:
                schema_errors = self._validate_schema(data, schema)
                validation_result['errors'].extend(schema_errors)

            # Calculate quality score
            if validation_result['record_count'] > 0:
                validation_result['quality_score'] = max(0.0, 1.0 - (len(validation_result['errors']) / validation_result['record_count']))

        except Exception as e:
            validation_result['errors'].append(f"Validation failed: {str(e)}")

        return validation_result

    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate data against schema"""
        errors = []

        # Check required fields
        for field, requirements in schema.items():
            if requirements.get('required', False):
                if field not in data:
                    errors.append(f"Required field '{field}' missing")

        # Type checking
        for field, value in data.items():
            if field in schema:
                expected_type = schema[field].get('type')
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Field '{field}' type mismatch: expected {expected_type}, got {type(value)}")

        return errors

class CollectionManager:
    """Manager for multiple data collection processes"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize collection manager"""
        self.config = config
        self.collectors: Dict[str, BaseDataCollector] = {}
        self.schedules: Dict[str, Any] = {}
        self.is_running = False
        self.logger = logging.getLogger("collection.manager")

    def register_collector(self, source_id: str, collector: BaseDataCollector) -> None:
        """Register data collector"""
        self.collectors[source_id] = collector
        self.logger.info(f"Registered collector: {source_id}")

    async def run_collection_cycle(self, source_id: Optional[str] = None) -> Dict[str, CollectionResult]:
        """Run collection cycle for specified or all collectors"""
        results = {}

        if source_id:
            if source_id not in self.collectors:
                raise ValueError(f"Unknown collector: {source_id}")
            collectors_to_run = {source_id: self.collectors[source_id]}
        else:
            collectors_to_run = self.collectors

        for collector_id, collector in collectors_to_run.items():
            try:
                self.logger.info(f"Starting collection for {collector_id}")
                result = await collector.run_collection_cycle()
                results[collector_id] = result
                self.logger.info(f"Collection completed for {collector_id}: "
                               f"{result.records_collected} records, "
                               f"quality: {result.quality_score:.3f}")

            except Exception as e:
                self.logger.error(f"Collection failed for {collector_id}: {str(e)}")
                results[collector_id] = CollectionResult(
                    source_id=collector_id,
                    timestamp=datetime.now(),
                    records_collected=0,
                    bytes_collected=0,
                    quality_score=0.0,
                    errors=[str(e)],
                    warnings=[],
                    metadata={},
                    duration_seconds=0.0
                )

        return results

    async def start_scheduled_collection(self) -> None:
        """Start scheduled collection for all registered collectors"""
        self.is_running = True
        self.logger.info("Starting scheduled data collection")

        while self.is_running:
            try:
                # Run collection cycle
                results = await self.run_collection_cycle()

                # Check for critical errors
                critical_errors = []
                for collector_id, result in results.items():
                    if len(result.errors) > 0:
                        critical_errors.append(f"{collector_id}: {len(result.errors)} errors")

                if critical_errors:
                    self.logger.warning(f"Collection issues detected: {critical_errors}")

                # Wait for next cycle
                interval = self.config.get('collection_interval', 3600)  # 1 hour default
                self.logger.info(f"Waiting {interval} seconds for next collection cycle")
                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"Scheduled collection error: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def stop_collection(self) -> None:
        """Stop scheduled collection"""
        self.is_running = False
        self.logger.info("Stopped scheduled data collection")

    def get_collection_status(self) -> Dict[str, Any]:
        """Get status for all collectors"""
        status = {
            'total_collectors': len(self.collectors),
            'running': self.is_running,
            'collectors': {}
        }

        for collector_id, collector in self.collectors.items():
            status['collectors'][collector_id] = collector.get_collection_stats()

        return status
```

### Error Handling Pattern
```python
class RobustCollectionManager:
    """Robust collection manager with comprehensive error handling"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize robust collection manager"""
        self.config = config
        self.collectors: Dict[str, BaseDataCollector] = {}
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, bool] = {}
        self.logger = logging.getLogger("robust_collection")

    async def collect_with_recovery(self, source_id: str) -> Optional[CollectionResult]:
        """Collect data with comprehensive error recovery"""
        if source_id not in self.collectors:
            raise ValueError(f"Unknown collector: {source_id}")

        collector = self.collectors[source_id]

        # Check circuit breaker
        if self.circuit_breakers.get(source_id, False):
            self.logger.warning(f"Circuit breaker open for {source_id}")
            return None

        # Exponential backoff for repeated failures
        backoff_delay = self._calculate_backoff_delay(source_id)

        try:
            # Run collection with timeout
            result = await asyncio.wait_for(
                collector.run_collection_cycle(),
                timeout=self.config.get('collection_timeout', 300)  # 5 minutes
            )

            # Reset error count on success
            self.error_counts[source_id] = 0

            # Close circuit breaker on success
            if self.circuit_breakers.get(source_id, False):
                self.circuit_breakers[source_id] = False
                self.logger.info(f"Circuit breaker closed for {source_id}")

            return result

        except asyncio.TimeoutError:
            self.logger.error(f"Collection timeout for {source_id}")
            await self._handle_timeout_error(source_id)
            return None

        except Exception as e:
            self.logger.error(f"Collection error for {source_id}: {str(e)}")
            await self._handle_collection_error(source_id, str(e))
            return None

    async def _handle_timeout_error(self, source_id: str) -> None:
        """Handle collection timeout"""
        self.error_counts[source_id] = self.error_counts.get(source_id, 0) + 1

        # Open circuit breaker if too many timeouts
        if self.error_counts[source_id] >= 3:
            self.circuit_breakers[source_id] = True
            self.logger.error(f"Circuit breaker opened for {source_id} due to repeated timeouts")

        # Attempt recovery
        await self._attempt_recovery(source_id)

    async def _handle_collection_error(self, source_id: str, error: str) -> None:
        """Handle collection error"""
        self.error_counts[source_id] = self.error_counts.get(source_id, 0) + 1

        # Open circuit breaker if too many errors
        if self.error_counts[source_id] >= 5:
            self.circuit_breakers[source_id] = True
            self.logger.error(f"Circuit breaker opened for {source_id} due to repeated errors")

        # Log error details
        self.logger.error(f"Collection error details for {source_id}: {error}")

        # Attempt recovery
        await self._attempt_recovery(source_id)

    async def _attempt_recovery(self, source_id: str) -> None:
        """Attempt to recover failed collector"""
        try:
            collector = self.collectors[source_id]

            # Reset connection
            await collector.disconnect()
            await asyncio.sleep(1)  # Brief pause

            # Test connection
            connection_success = await collector.connect()
            if connection_success:
                self.logger.info(f"Successfully recovered connection for {source_id}")
                # Reset circuit breaker
                self.circuit_breakers[source_id] = False
            else:
                self.logger.warning(f"Recovery failed for {source_id}")

        except Exception as e:
            self.logger.error(f"Recovery attempt failed for {source_id}: {str(e)}")

    def _calculate_backoff_delay(self, source_id: str) -> float:
        """Calculate exponential backoff delay"""
        error_count = self.error_counts.get(source_id, 0)
        base_delay = self.config.get('backoff_base_delay', 1.0)
        max_delay = self.config.get('backoff_max_delay', 300.0)

        delay = min(base_delay * (2 ** error_count), max_delay)
        return delay
```

## Testing Guidelines

### Collection Testing
- **Unit Tests**: Test individual collection methods and error handling
- **Integration Tests**: Test collection with real and mock data sources
- **Performance Tests**: Test collection throughput and resource usage
- **Reliability Tests**: Test error recovery and retry mechanisms
- **Security Tests**: Validate authentication and data protection

### Quality Assurance
- **Data Validation**: Ensure collected data meets quality standards
- **Metadata Completeness**: Verify comprehensive metadata capture
- **Error Recovery**: Test recovery from various failure scenarios
- **Performance Monitoring**: Monitor collection performance metrics
- **Security Validation**: Validate security measures and access controls

## Performance Considerations

### Collection Performance
- **Throughput Optimization**: Maximize data collection rates
- **Connection Pooling**: Efficient connection management for APIs
- **Batch Processing**: Process data in optimal batch sizes
- **Memory Management**: Efficient memory usage for large datasets
- **Network Optimization**: Minimize network overhead and latency

### Scalability
- **Horizontal Scaling**: Support multiple collection instances
- **Load Balancing**: Distribute collection load across resources
- **Resource Monitoring**: Monitor and adjust resource allocation
- **Auto-scaling**: Automatically scale based on data volume
- **Performance Profiling**: Identify and optimize bottlenecks

## Maintenance and Evolution

### Collection System Updates
- **Source Monitoring**: Monitor data source availability and changes
- **Schema Evolution**: Handle changes in data source schemas
- **Performance Tuning**: Continuous performance optimization
- **Security Updates**: Update security measures and protocols

### Quality Improvements
- **Feedback Integration**: Incorporate user feedback for improvements
- **Quality Trend Analysis**: Analyze quality trends over time
- **Automated Optimization**: Automatically adjust collection parameters
- **Best Practice Updates**: Update based on industry best practices

## Common Challenges and Solutions

### Challenge: API Rate Limits
**Solution**: Implement intelligent rate limiting with exponential backoff, request queuing, and load distribution across multiple API keys.

### Challenge: Data Source Reliability
**Solution**: Implement circuit breakers, comprehensive error handling, fallback sources, and health monitoring with automatic recovery.

### Challenge: Data Quality Variability
**Solution**: Implement real-time quality assessment, adaptive collection strategies, and quality-based filtering and routing.

### Challenge: Security and Privacy
**Solution**: Use secure authentication, encrypt data in transit, implement access controls, and ensure compliance with privacy regulations.

## Getting Started as an Agent

### Development Setup
1. **Study Collection Architecture**: Understand data collection system design
2. **Learn Data Standards**: Study data formats and quality standards
3. **Practice Implementation**: Practice implementing collection interfaces
4. **Understand Security**: Learn security and privacy requirements

### Contribution Process
1. **Identify Collection Needs**: Find gaps in current collection capabilities
2. **Research Data Sources**: Study target data sources and APIs
3. **Design Collection Strategy**: Create detailed collection system designs
4. **Implement and Test**: Follow security and quality implementation standards
5. **Validate Thoroughly**: Ensure data quality and security compliance
6. **Document Completely**: Provide comprehensive documentation
7. **Security Review**: Submit for security and compliance review

### Learning Resources
- **API Design**: RESTful API design and integration patterns
- **Data Streaming**: Real-time data processing and event handling
- **Quality Assurance**: Data quality assessment and improvement
- **Security Standards**: Data security and privacy protection
- **Performance Optimization**: High-throughput data processing techniques

## Related Documentation

- **[Collection README](./README.md)**: Data collection module overview
- **[Data Management AGENTS.md](../AGENTS.md)**: Data management development guidelines
- **[Main AGENTS.md](../../../AGENTS.md)**: Project-wide agent guidelines
- **[Research AGENTS.md](../../AGENTS.md)**: Research tools module guidelines
- **[Contributing Guide](../../../CONTRIBUTING.md)**: Contribution processes

---

*"Active Inference for, with, by Generative AI"* - Advancing research through comprehensive data collection, real-time processing, and quality-assured data acquisition.
