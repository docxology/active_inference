# Integrations - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Integrations module of the Active Inference Knowledge Environment. It outlines integration patterns, development workflows, and best practices for creating robust external system connectors.

## Integrations Module Overview

The Integrations module provides robust, well-tested connectors for external systems including scientific computing platforms, data repositories, visualization tools, cloud services, and research infrastructure. This module enables seamless interoperability between Active Inference applications and the broader scientific and technical ecosystem.

## Core Responsibilities

### Integration Development
- **API Design**: Create clean, intuitive APIs for external system integration
- **Protocol Implementation**: Implement communication protocols and standards
- **Authentication Systems**: Develop secure authentication mechanisms
- **Error Handling**: Implement comprehensive error handling and recovery
- **Performance Optimization**: Ensure efficient integration performance

### Quality Assurance
- **Testing**: Comprehensive testing including unit, integration, and performance tests
- **Security**: Ensure secure handling of credentials and sensitive data
- **Reliability**: Implement robust error handling and recovery mechanisms
- **Documentation**: Maintain clear documentation and usage examples
- **Validation**: Validate integration functionality and performance

### Maintenance and Support
- **API Evolution**: Keep integrations current with external API changes
- **Compatibility**: Ensure backward compatibility with existing systems
- **Monitoring**: Implement monitoring and alerting for integration health
- **Community Support**: Provide support for integration users and contributors

## Development Workflows

### Integration Creation Process
1. **Requirements Analysis**: Understand integration requirements and constraints
2. **API Research**: Research external system APIs and documentation
3. **Architecture Design**: Design integration architecture and data flow
4. **Implementation**: Implement integration following established patterns
5. **Testing**: Develop comprehensive test suite
6. **Documentation**: Create detailed documentation and examples
7. **Review**: Submit for peer review and validation
8. **Deployment**: Deploy and monitor integration in production

### Authentication Implementation Process
1. **Security Analysis**: Analyze authentication requirements and security implications
2. **Method Selection**: Choose appropriate authentication method
3. **Implementation**: Implement secure authentication mechanism
4. **Testing**: Test authentication under various scenarios
5. **Security Review**: Conduct security review and validation
6. **Documentation**: Document authentication setup and usage

### Error Handling Implementation Process
1. **Error Analysis**: Identify potential failure modes and error conditions
2. **Recovery Strategy**: Design error recovery and retry mechanisms
3. **Implementation**: Implement robust error handling
4. **Testing**: Test error scenarios and recovery mechanisms
5. **Monitoring**: Implement error monitoring and alerting
6. **Documentation**: Document error handling and troubleshooting

## Quality Standards

### Integration Quality
- **Reliability**: Integrations should handle errors gracefully and recover automatically
- **Performance**: Maintain acceptable performance characteristics
- **Security**: Implement secure authentication and data handling
- **Usability**: Provide clear, intuitive APIs and documentation
- **Maintainability**: Follow patterns that support long-term maintenance

### Code Quality
- **Test Coverage**: Maintain >95% test coverage for integration code
- **Error Handling**: Comprehensive error handling and recovery
- **Documentation**: Complete docstrings and usage examples
- **Type Safety**: Use complete type annotations
- **Standards Compliance**: Follow project coding standards

### Security Quality
- **Authentication**: Secure credential management and authentication
- **Data Protection**: Protect sensitive data in transit and at rest
- **Access Control**: Implement appropriate access controls
- **Audit Logging**: Maintain audit logs for security events
- **Compliance**: Follow security best practices and standards

## Implementation Patterns

### Base Integration Pattern
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for external integrations"""
    name: str
    base_url: str
    timeout: int = 30
    retry_config: Dict[str, Union[int, float]] = None
    auth_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.retry_config is None:
            self.retry_config = {'max_retries': 3, 'delay': 1.0}

class BaseIntegration(ABC):
    """Base class for external system integrations"""

    def __init__(self, config: IntegrationConfig):
        """Initialize integration with configuration"""
        self.config = config
        self.session = None
        self.authenticated = False
        self.connection_status = 'disconnected'
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure integration logging"""
        self.logger = logging.getLogger(f"{__name__}.{self.config.name}")
        self.logger.info(f"Integration {self.config.name} initialized")

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with external system"""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to external system"""
        pass

    def connect(self) -> bool:
        """Establish connection to external system"""
        try:
            self.logger.info(f"Connecting to {self.config.name}")
            self.session = self.create_session()

            if self.authenticate():
                self.authenticated = True
                self.connection_status = 'connected'
                self.logger.info(f"Successfully connected to {self.config.name}")
                return True
            else:
                self.logger.error(f"Authentication failed for {self.config.name}")
                return False

        except Exception as e:
            self.logger.error(f"Connection failed for {self.config.name}: {e}")
            self.connection_status = 'error'
            return False

    def disconnect(self) -> None:
        """Close connection to external system"""
        if self.session:
            self.session.close()
            self.session = None
        self.authenticated = False
        self.connection_status = 'disconnected'
        self.logger.info(f"Disconnected from {self.config.name}")

    @abstractmethod
    def create_session(self) -> Any:
        """Create session for external communication"""
        pass

    def get_health_status(self) -> Dict[str, Any]:
        """Get integration health status"""
        return {
            'integration': self.config.name,
            'status': self.connection_status,
            'authenticated': self.authenticated,
            'timestamp': time.time(),
            'config': {
                'base_url': self.config.base_url,
                'timeout': self.config.timeout
            }
        }

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
```

### Retry and Resilience Pattern
```python
from typing import Dict, Any, Callable, TypeVar, Generic
import time
import random
from functools import wraps

T = TypeVar('T')

class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(self,
                 max_retries: int = 3,
                 delay: float = 1.0,
                 backoff: float = 2.0,
                 jitter: bool = True,
                 exceptions: tuple = (Exception,)):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.jitter = jitter
        self.exceptions = exceptions

def retry_on_failure(retry_config: RetryConfig = None):
    """Decorator for implementing retry logic"""
    if retry_config is None:
        retry_config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_config.exceptions as e:
                    last_exception = e

                    if attempt == retry_config.max_retries:
                        raise e

                    # Calculate delay with jitter
                    delay = retry_config.delay * (retry_config.backoff ** attempt)
                    if retry_config.jitter:
                        delay *= (0.5 + random.random())

                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator

class ResilientIntegration(BaseIntegration):
    """Integration with built-in resilience and retry logic"""

    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.retry_config = RetryConfig(
            max_retries=config.retry_config.get('max_retries', 3),
            delay=config.retry_config.get('delay', 1.0),
            backoff=config.retry_config.get('backoff', 2.0),
            jitter=config.retry_config.get('jitter', True)
        )

    @retry_on_failure()
    def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        if not self.session or not self.authenticated:
            raise ConnectionError("Integration not connected or authenticated")

        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        response = self.session.request(
            method=method,
            url=url,
            timeout=self.config.timeout,
            **kwargs
        )

        response.raise_for_status()
        return response.json()

    def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """GET request with retry logic"""
        return self.make_request('GET', endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """POST request with retry logic"""
        return self.make_request('POST', endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """PUT request with retry logic"""
        return self.make_request('PUT', endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """DELETE request with retry logic"""
        return self.make_request('DELETE', endpoint, **kwargs)
```

### Authentication Pattern
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests
import time

class AuthenticationManager(ABC):
    """Base class for authentication management"""

    def __init__(self, auth_config: Dict[str, Any]):
        self.auth_config = auth_config
        self.token: Optional[str] = None
        self.token_expiry: Optional[float] = None
        self.refresh_token: Optional[str] = None

    @abstractmethod
    def authenticate(self) -> Dict[str, str]:
        """Perform authentication and return headers"""
        pass

    @abstractmethod
    def is_token_valid(self) -> bool:
        """Check if current token is valid"""
        pass

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if not self.is_token_valid():
            self.refresh_authentication()

        if self.token:
            return {'Authorization': f'Bearer {self.token}'}
        return {}

    def refresh_authentication(self) -> None:
        """Refresh authentication if needed"""
        if hasattr(self, 'refresh_token') and self.refresh_token:
            self.perform_token_refresh()
        else:
            self.authenticate()

    @abstractmethod
    def perform_token_refresh(self) -> None:
        """Perform token refresh"""
        pass

class OAuth2Manager(AuthenticationManager):
    """OAuth2 authentication manager"""

    def authenticate(self) -> Dict[str, str]:
        """Authenticate using OAuth2"""
        if self.auth_config.get('grant_type') == 'client_credentials':
            return self._authenticate_client_credentials()
        elif self.auth_config.get('grant_type') == 'authorization_code':
            return self._authenticate_authorization_code()
        else:
            raise ValueError("Unsupported OAuth2 grant type")

    def _authenticate_client_credentials(self) -> Dict[str, str]:
        """Client credentials OAuth2 flow"""
        response = requests.post(
            self.auth_config['token_url'],
            data={
                'grant_type': 'client_credentials',
                'client_id': self.auth_config['client_id'],
                'client_secret': self.auth_config['client_secret']
            },
            timeout=30
        )

        if response.status_code == 200:
            token_data = response.json()
            self.token = token_data['access_token']
            self.token_expiry = time.time() + token_data.get('expires_in', 3600)
            self.refresh_token = token_data.get('refresh_token')

            return {'Authorization': f'Bearer {self.token}'}
        else:
            raise AuthenticationError(f"OAuth2 authentication failed: {response.text}")

    def is_token_valid(self) -> bool:
        """Check if OAuth2 token is valid"""
        if not self.token or not self.token_expiry:
            return False

        # Check if token expires in next 5 minutes
        return time.time() < (self.token_expiry - 300)

    def perform_token_refresh(self) -> None:
        """Refresh OAuth2 token"""
        if not self.refresh_token:
            raise AuthenticationError("No refresh token available")

        response = requests.post(
            self.auth_config['token_url'],
            data={
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
                'client_id': self.auth_config['client_id'],
                'client_secret': self.auth_config['client_secret']
            },
            timeout=30
        )

        if response.status_code == 200:
            token_data = response.json()
            self.token = token_data['access_token']
            self.token_expiry = time.time() + token_data.get('expires_in', 3600)
            self.refresh_token = token_data.get('refresh_token', self.refresh_token)
        else:
            raise AuthenticationError(f"Token refresh failed: {response.text}")
```

## Testing Guidelines

### Integration Testing
- **Connection Testing**: Test connection establishment and teardown
- **Authentication Testing**: Validate authentication mechanisms
- **API Testing**: Test all API endpoints and methods
- **Error Testing**: Test error handling and recovery
- **Performance Testing**: Validate performance characteristics

### Security Testing
- **Authentication Testing**: Test authentication security
- **Data Protection**: Verify data protection mechanisms
- **Access Control**: Test access control implementations
- **Audit Logging**: Verify audit logging functionality
- **Compliance Testing**: Test compliance with security standards

### Reliability Testing
- **Retry Testing**: Test retry mechanisms under failure conditions
- **Recovery Testing**: Test recovery from various error states
- **Stress Testing**: Test under high load conditions
- **Longevity Testing**: Test long-running integration stability

## Performance Considerations

### Connection Management
- **Connection Pooling**: Use connection pooling for efficiency
- **Keep-Alive**: Implement keep-alive connections where appropriate
- **Timeouts**: Configure appropriate timeout values
- **Resource Cleanup**: Ensure proper cleanup of connections

### Data Transfer
- **Batch Processing**: Use batch processing for efficiency
- **Compression**: Implement data compression for large transfers
- **Streaming**: Use streaming for large data sets
- **Caching**: Implement caching for frequently accessed data

### Error Handling Performance
- **Fast Failure**: Fail fast for non-recoverable errors
- **Efficient Retry**: Implement efficient retry mechanisms
- **Resource Management**: Manage resources during error recovery
- **Monitoring Overhead**: Minimize performance impact of monitoring

## Maintenance and Evolution

### API Evolution
- **Version Management**: Handle API versioning and evolution
- **Backward Compatibility**: Maintain backward compatibility
- **Migration Support**: Support migration between API versions
- **Deprecation**: Handle deprecated API features

### Security Updates
- **Authentication Updates**: Update authentication mechanisms as needed
- **Security Patches**: Apply security patches promptly
- **Compliance Updates**: Update for changing compliance requirements
- **Audit Reviews**: Regular security audit and review

## Common Challenges and Solutions

### Challenge: API Rate Limiting
**Solution**: Implement intelligent rate limiting with exponential backoff and request queuing.

### Challenge: Authentication Complexity
**Solution**: Create flexible authentication framework supporting multiple methods and easy configuration.

### Challenge: Data Format Variations
**Solution**: Implement robust data transformation and validation layers.

### Challenge: Network Reliability
**Solution**: Implement comprehensive retry logic and graceful degradation.

## Getting Started as an Agent

### Development Setup
1. **Study Integration Patterns**: Review existing integration implementations
2. **Understand Authentication**: Learn about different authentication methods
3. **Practice Error Handling**: Implement robust error handling patterns
4. **Test Thoroughly**: Develop comprehensive test suites

### Contribution Process
1. **Identify Integration Needs**: Find systems that need integration
2. **Research APIs**: Study external system APIs and documentation
3. **Design Integration**: Create detailed integration design
4. **Implement and Test**: Follow TDD with comprehensive testing
5. **Document Completely**: Provide detailed documentation
6. **Security Review**: Ensure security best practices

### Learning Resources
- **API Design**: Study RESTful API design principles
- **Authentication Standards**: Learn OAuth2, API keys, and other auth methods
- **Network Protocols**: Understand HTTP, WebSocket, and other protocols
- **Security Best Practices**: Study integration security patterns
- **Testing Strategies**: Learn comprehensive testing approaches

## Related Documentation

- **[Integrations README](./README.md)**: Integrations module overview
- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Applications AGENTS.md](../AGENTS.md)**: Applications module guidelines
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution processes
- **[Platform Documentation](../../platform/)**: Platform integration tools

---

*"Active Inference for, with, by Generative AI"* - Building seamless integrations through robust connectors and comprehensive interoperability solutions.

