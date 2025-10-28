# Integrations

This directory contains APIs, connectors, and integration tools for connecting Active Inference systems with external platforms, services, and data sources. These integrations enable seamless interoperability between Active Inference applications and the broader scientific and technical ecosystem.

## Overview

The Integrations module provides robust, well-tested connectors for external systems including scientific computing platforms, data repositories, visualization tools, cloud services, and research infrastructure. Each integration follows established patterns and provides comprehensive documentation for easy adoption.

## Directory Structure

```
integrations/
├── apis/                  # REST API connectors and clients
├── data_sources/          # Data source connectors (databases, files, streams)
├── platforms/            # Platform integrations (Jupyter, MATLAB, etc.)
├── services/             # Cloud and web service integrations
├── protocols/            # Communication protocol implementations
└── middleware/           # Integration middleware and adapters
```

## Core Components

### 🌐 API Integrations
- **REST APIs**: Connectors for RESTful web services and APIs
- **GraphQL Clients**: GraphQL API integration tools
- **WebSocket Support**: Real-time communication protocol implementations
- **Authentication**: OAuth, API key, and token-based authentication

### 💾 Data Source Connectors
- **Database Integration**: SQL and NoSQL database connectors
- **File System Access**: Local and remote file system integration
- **Data Streaming**: Real-time data stream processing
- **Format Conversion**: Data format conversion and standardization

### 🔬 Scientific Platforms
- **Jupyter Integration**: Jupyter notebook and lab integration
- **MATLAB Connectivity**: MATLAB toolbox integration
- **R Integration**: R statistical environment connectivity
- **Python Libraries**: Integration with scientific Python ecosystem

### ☁️ Cloud Services
- **Storage Services**: Cloud storage integration (S3, Azure, GCP)
- **Compute Resources**: Cloud computing resource management
- **Container Services**: Docker and Kubernetes integration
- **Monitoring**: Cloud monitoring and logging services

## Getting Started

### For Developers
1. **Choose Integration Type**: Select appropriate integration pattern for your needs
2. **Review Examples**: Study existing integration implementations
3. **Configure Connection**: Set up authentication and configuration
4. **Test Integration**: Verify integration functionality and performance
5. **Deploy**: Deploy integration in production environment

### For System Integrators
1. **Assess Requirements**: Analyze integration requirements and constraints
2. **Select Components**: Choose appropriate integration components
3. **Plan Architecture**: Design integration architecture and data flow
4. **Implement**: Develop custom integrations following established patterns
5. **Validate**: Test integration thoroughly before deployment

## Usage Examples

### REST API Integration
```python
from active_inference.applications.integrations import RESTAPIClient

class NeuroscienceDataAPI(RESTAPIClient):
    """Integration with neuroscience data repositories"""

    def __init__(self, api_config):
        super().__init__(api_config)
        self.base_url = api_config['base_url']
        self.api_key = api_config['api_key']
        self.setup_authentication()

    def setup_authentication(self):
        """Configure API authentication"""
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'Active-Inference-Knowledge-Environment'
        }

    def get_neural_data(self, subject_id: str, session: str) -> dict:
        """Retrieve neural data for specific subject and session"""
        endpoint = f'/subjects/{subject_id}/sessions/{session}/data'
        return self.get(endpoint)

    def upload_model_results(self, results: dict) -> str:
        """Upload Active Inference model results"""
        endpoint = '/model_results'
        response = self.post(endpoint, data=results)
        return response.get('result_id')
```

### Database Integration
```python
from active_inference.applications.integrations import DatabaseConnector

class ExperimentDatabase(DatabaseConnector):
    """Database integration for experiment management"""

    def __init__(self, db_config):
        super().__init__(db_config)
        self.setup_tables()

    def setup_tables(self):
        """Create database tables for experiment data"""
        self.create_experiments_table()
        self.create_results_table()
        self.create_metadata_table()

    def create_experiments_table(self):
        """Create experiments table"""
        query = """
        CREATE TABLE IF NOT EXISTS experiments (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            type VARCHAR(100),
            parameters JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.execute_query(query)

    def store_experiment(self, experiment_data: dict) -> int:
        """Store experiment data in database"""
        query = """
        INSERT INTO experiments (name, type, parameters)
        VALUES (%s, %s, %s)
        RETURNING id
        """
        return self.execute_query(query, (
            experiment_data['name'],
            experiment_data['type'],
            experiment_data['parameters']
        ))[0]['id']
```

### Real-time Data Streaming
```python
from active_inference.applications.integrations import StreamingClient

class NeuralStreamProcessor(StreamingClient):
    """Real-time neural data stream processing"""

    def __init__(self, stream_config):
        super().__init__(stream_config)
        self.data_buffer = []
        self.processing_queue = []
        self.setup_stream()

    def setup_stream(self):
        """Configure streaming connection"""
        self.connect_stream()
        self.subscribe_to_topics(['neural_data', 'behavioral_events'])

    def process_stream_data(self, message: dict) -> dict:
        """Process incoming stream data"""
        data_type = message.get('type')

        if data_type == 'neural_recording':
            return self.process_neural_data(message)
        elif data_type == 'behavioral_event':
            return self.process_behavioral_data(message)
        else:
            return self.handle_unknown_data(message)

    def process_neural_data(self, message: dict) -> dict:
        """Process neural recording data"""
        processed_data = {
            'timestamp': message['timestamp'],
            'channels': message['data'],
            'sampling_rate': message['sampling_rate'],
            'processed_features': self.extract_features(message['data'])
        }
        return processed_data
```

## Integration Patterns

### Authentication Patterns
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests

class BaseAuthenticator(ABC):
    """Base class for API authentication methods"""

    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, str]:
        """Perform authentication and return headers"""
        pass

    @abstractmethod
    def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh authentication token"""
        pass

class OAuth2Authenticator(BaseAuthenticator):
    """OAuth2 authentication implementation"""

    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, str]:
        """Authenticate using OAuth2"""
        token_response = requests.post(
            credentials['token_url'],
            data={
                'grant_type': 'client_credentials',
                'client_id': credentials['client_id'],
                'client_secret': credentials['client_secret']
            }
        )

        if token_response.status_code == 200:
            token_data = token_response.json()
            return {
                'Authorization': f'Bearer {token_data["access_token"]}',
                'Content-Type': 'application/json'
            }
        else:
            raise AuthenticationError("OAuth2 authentication failed")

    def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh OAuth2 token"""
        # Implementation for token refresh
        pass
```

### Error Handling Patterns
```python
from typing import Dict, Any, Optional
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

class IntegrationError(Exception):
    """Base exception for integration errors"""
    pass

class ConnectionError(IntegrationError):
    """Connection-related errors"""
    pass

class AuthenticationError(IntegrationError):
    """Authentication-related errors"""
    pass

class RateLimitError(IntegrationError):
    """Rate limiting errors"""
    pass

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, RateLimitError) as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")

                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff

            raise last_exception
        return wrapper
    return decorator

class ResilientIntegration:
    """Base class for resilient external integrations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self.retry_config = config.get('retry', {'max_retries': 3, 'delay': 1.0})

    @retry_on_failure()
    def execute_with_retry(self, operation: callable, *args, **kwargs):
        """Execute operation with retry logic"""
        return operation(*args, **kwargs)

    def handle_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Handle integration errors gracefully"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': time.time(),
            'recoverable': self.is_recoverable_error(error)
        }

        logger.error(f"Integration error in {context}: {error}")
        return error_info

    def is_recoverable_error(self, error: Exception) -> bool:
        """Determine if error is recoverable"""
        return isinstance(error, (ConnectionError, RateLimitError))
```

## Contributing

We welcome contributions to the integrations module! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

### Contribution Types
- **New Integrations**: Create connectors for new external systems
- **Enhanced Protocols**: Improve existing integration protocols
- **Authentication Methods**: Add new authentication mechanisms
- **Error Handling**: Improve error handling and resilience
- **Performance Optimization**: Optimize integration performance

### Quality Standards
- **Comprehensive Testing**: Include unit and integration tests
- **Error Handling**: Implement robust error handling and recovery
- **Documentation**: Provide clear API documentation and examples
- **Performance**: Ensure acceptable performance characteristics
- **Security**: Follow security best practices for external connections

## Learning Resources

- **Integration Patterns**: Study established integration patterns and examples
- **API Design**: Learn about RESTful API design and implementation
- **Authentication**: Understand various authentication mechanisms
- **Error Handling**: Study robust error handling patterns
- **Performance**: Learn about integration performance optimization

## Related Documentation

- **[Applications README](../README.md)**: Applications module overview
- **[Main README](../../README.md)**: Project overview and getting started
- **[Platform Documentation](../../platform/)**: Platform integration tools
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution guidelines
- **[Research Tools](../../research/)**: Research data integration

## Integration Checklist

Before submitting a new integration, ensure:

### ✅ **Technical Requirements**
- [ ] Comprehensive test suite with >90% coverage
- [ ] Complete API documentation with examples
- [ ] Error handling and recovery mechanisms
- [ ] Performance benchmarks and optimization
- [ ] Security considerations and validation

### ✅ **Documentation Requirements**
- [ ] Detailed README with setup instructions
- [ ] Usage examples and code samples
- [ ] Configuration documentation
- [ ] Troubleshooting guide
- [ ] Performance characteristics

### ✅ **Integration Standards**
- [ ] Follows established integration patterns
- [ ] Implements proper authentication
- [ ] Includes rate limiting and retry logic
- [ ] Provides health monitoring
- [ ] Supports graceful degradation

---

*"Active Inference for, with, by Generative AI"* - Building seamless integrations through robust connectors and comprehensive interoperability solutions.




