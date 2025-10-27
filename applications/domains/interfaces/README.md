# Domain Interfaces

**Interface definitions and integration protocols for domain-specific Active Inference applications.**

## 📖 Overview

**Standardized interfaces and integration protocols that enable seamless connectivity between Active Inference applications and external systems across different domains.**

This directory contains interface definitions, protocols, and integration patterns that enable Active Inference applications to connect with external systems, APIs, and domain-specific tools while maintaining consistency and reliability.

### 🎯 Mission & Role

This interfaces collection contributes to application interoperability by:

- **Standardization**: Consistent interface patterns across domains
- **Integration**: Seamless connectivity with external systems
- **Compatibility**: Standardized protocols for system interaction
- **Maintainability**: Well-defined interfaces for easy maintenance

## 🏗️ Architecture

### Interface Categories

```
applications/domains/interfaces/
├── api_interfaces/              # API integration interfaces
├── data_interfaces/             # Data exchange interfaces
├── protocol_interfaces/         # Communication protocol interfaces
├── service_interfaces/          # Service integration interfaces
└── domain_specific/            # Domain-specific interface definitions
```

### Integration Points

**Domain interfaces integrate with platform components:**

- **Application Framework**: Provides interface implementations for applications
- **Platform Services**: Leverages interfaces for external connectivity
- **Integration Tools**: Uses interfaces for system connectivity
- **Domain Applications**: Implements domain-specific interface protocols

### Interface Standards

#### Interface Design Principles
- **Consistency**: Uniform interface patterns across all domains
- **Extensibility**: Easy to extend interfaces for new requirements
- **Backward Compatibility**: Maintain compatibility with existing implementations
- **Documentation**: Comprehensive interface documentation and examples

#### Interface Types
- **API Interfaces**: REST, GraphQL, and custom API integrations
- **Data Interfaces**: File, database, and stream data exchange
- **Protocol Interfaces**: Communication and messaging protocols
- **Service Interfaces**: External service and platform integrations

## 🚀 Usage

### Basic Interface Usage

```python
# Import interface components
from applications.domains.interfaces.api_interfaces import RESTAPIInterface
from applications.domains.interfaces.data_interfaces import DataExchangeInterface
from applications.domains.interfaces.protocol_interfaces import MessageProtocolInterface

# Initialize interfaces with configuration
config = {
    "interface_type": "api",
    "domain": "neuroscience",
    "protocol": "rest",
    "authentication": "bearer_token"
}

# Create interface instance
api_interface = RESTAPIInterface(config)

# Configure interface
api_interface.configure_authentication("your_bearer_token")
api_interface.set_base_url("https://api.example.com")

# Use interface
response = api_interface.get("/neural/data")
print(f"Neural data retrieved: {response}")
```

### Advanced Interface Integration

```python
# Multi-interface integration
from applications.domains.interfaces import InterfaceManager

# Create interface manager
interface_manager = InterfaceManager()

# Register multiple interfaces
interface_manager.register_interface("neural_api", RESTAPIInterface(neural_config))
interface_manager.register_interface("data_stream", DataExchangeInterface(stream_config))
interface_manager.register_interface("message_bus", MessageProtocolInterface(bus_config))

# Configure cross-interface communication
interface_manager.configure_routing({
    "neural_api": ["data_stream", "message_bus"],
    "data_stream": ["message_bus"],
    "message_bus": ["neural_api"]
})

# Execute integrated workflow
workflow_result = interface_manager.execute_workflow({
    "name": "neural_data_processing",
    "steps": [
        {"interface": "neural_api", "action": "fetch_data"},
        {"interface": "data_stream", "action": "process_stream"},
        {"interface": "message_bus", "action": "broadcast_results"}
    ]
})
```

### Domain-Specific Interface Usage

```python
# Neuroscience domain interface
from applications.domains.interfaces.domain_specific.neuroscience import NeuralDataInterface

# Initialize neuroscience interface
neural_interface = NeuralDataInterface({
    "data_format": "edf",  # European Data Format
    "sampling_rate": 1000,
    "channels": ["EEG", "EMG", "EOG"],
    "real_time_processing": True
})

# Connect to neural data source
neural_interface.connect_data_source("neural_recorder_device")

# Process neural signals
neural_signals = neural_interface.acquire_signals(duration=30)  # 30 seconds
processed_signals = neural_interface.preprocess_signals(neural_signals)

# Apply Active Inference processing
ai_results = neural_interface.apply_active_inference(processed_signals)
print(f"Neural processing completed: {ai_results}")
```

## 🔧 Interface Categories

### API Interfaces

#### REST API Interface
```python
from applications.domains.interfaces.api_interfaces.rest_interface import RESTAPIInterface

class RESTAPIInterface:
    """REST API interface for external system integration"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize REST API interface"""
        self.config = config
        self.session = self.create_session()
        self.endpoints = self.load_endpoint_definitions()

    def configure_authentication(self, auth_config: Dict[str, Any]) -> None:
        """Configure API authentication"""
        auth_type = auth_config.get("type", "none")

        if auth_type == "bearer_token":
            self.session.headers.update({
                "Authorization": f"Bearer {auth_config['token']}"
            })
        elif auth_type == "api_key":
            self.session.headers.update({
                "X-API-Key": auth_config['api_key']
            })
        elif auth_type == "basic_auth":
            self.session.auth = (
                auth_config['username'],
                auth_config['password']
            )

    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make GET request to API endpoint"""
        url = self.build_url(endpoint)

        try:
            response = self.session.get(url, params=params, timeout=self.config.get("timeout", 30))
            response.raise_for_status()

            return {
                "status": "success",
                "data": response.json(),
                "headers": dict(response.headers),
                "status_code": response.status_code
            }

        except requests.RequestException as e:
            return {
                "status": "error",
                "error": str(e),
                "url": url
            }

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request to API endpoint"""
        url = self.build_url(endpoint)
        headers = {"Content-Type": "application/json"}

        try:
            response = self.session.post(
                url,
                json=data,
                headers=headers,
                timeout=self.config.get("timeout", 30)
            )
            response.raise_for_status()

            return {
                "status": "success",
                "data": response.json(),
                "status_code": response.status_code
            }

        except requests.RequestException as e:
            return {
                "status": "error",
                "error": str(e),
                "url": url
            }

    def build_url(self, endpoint: str) -> str:
        """Build complete URL from base URL and endpoint"""
        base_url = self.config.get("base_url", "")
        return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
```

#### GraphQL API Interface
```python
from applications.domains.interfaces.api_interfaces.graphql_interface import GraphQLInterface

class GraphQLInterface:
    """GraphQL API interface for flexible data querying"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize GraphQL interface"""
        self.config = config
        self.client = self.create_graphql_client()

    def execute_query(self, query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute GraphQL query"""
        try:
            # Execute query with variables
            result = self.client.execute(query, variables=variables)

            return {
                "status": "success",
                "data": result,
                "query": query,
                "variables": variables
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }

    def execute_mutation(self, mutation: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute GraphQL mutation"""
        try:
            result = self.client.execute(mutation, variables=variables)

            return {
                "status": "success",
                "data": result,
                "mutation": mutation
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "mutation": mutation
            }

    def create_query_builder(self) -> GraphQLQueryBuilder:
        """Create GraphQL query builder for complex queries"""
        return GraphQLQueryBuilder(self.config)
```

### Data Interfaces

#### Data Exchange Interface
```python
from applications.domains.interfaces.data_interfaces.data_exchange import DataExchangeInterface

class DataExchangeInterface:
    """Interface for data exchange between systems"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data exchange interface"""
        self.config = config
        self.data_formatters = self.load_data_formatters()

    def import_data(self, source_path: str, target_format: str = None) -> Dict[str, Any]:
        """Import data from external source"""
        try:
            # Detect source format
            source_format = self.detect_data_format(source_path)

            # Read source data
            raw_data = self.read_source_data(source_path, source_format)

            # Convert to target format if needed
            if target_format and target_format != source_format:
                converted_data = self.convert_data_format(raw_data, source_format, target_format)
            else:
                converted_data = raw_data

            # Validate converted data
            validation_result = self.validate_converted_data(converted_data, target_format)

            if validation_result["valid"]:
                return {
                    "status": "success",
                    "data": converted_data,
                    "source_format": source_format,
                    "target_format": target_format or source_format
                }
            else:
                return {
                    "status": "validation_error",
                    "error": validation_result["errors"],
                    "data": converted_data
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def export_data(self, data: Any, target_path: str, target_format: str = None) -> Dict[str, Any]:
        """Export data to external target"""
        try:
            # Determine target format
            export_format = target_format or self.detect_target_format(target_path)

            # Convert data to target format
            formatted_data = self.format_data_for_export(data, export_format)

            # Write to target
            write_result = self.write_target_data(formatted_data, target_path, export_format)

            return {
                "status": "success",
                "target_path": target_path,
                "format": export_format,
                "data_size": write_result["size"]
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
```

### Protocol Interfaces

#### Message Protocol Interface
```python
from applications.domains.interfaces.protocol_interfaces.message_protocol import MessageProtocolInterface

class MessageProtocolInterface:
    """Interface for message-based communication protocols"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize message protocol interface"""
        self.config = config
        self.protocol = self.initialize_protocol()

    def send_message(self, destination: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message to destination"""
        try:
            # Format message for protocol
            formatted_message = self.format_message_for_protocol(message)

            # Send via protocol
            send_result = self.protocol.send(destination, formatted_message)

            # Track message delivery
            delivery_id = self.track_message_delivery(send_result)

            return {
                "status": "sent",
                "delivery_id": delivery_id,
                "destination": destination,
                "message_size": len(str(message))
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "destination": destination
            }

    def receive_message(self, source: str = None) -> Dict[str, Any]:
        """Receive message from source"""
        try:
            # Receive via protocol
            raw_message = self.protocol.receive(source=source)

            # Parse message from protocol format
            parsed_message = self.parse_message_from_protocol(raw_message)

            # Acknowledge receipt
            self.acknowledge_message_receipt(raw_message)

            return {
                "status": "received",
                "message": parsed_message,
                "source": raw_message.get("source"),
                "timestamp": raw_message.get("timestamp")
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def subscribe_to_topic(self, topic: str, callback: Callable) -> str:
        """Subscribe to message topic"""
        subscription_id = self.generate_subscription_id()

        # Register subscription
        self.protocol.subscribe(topic, callback, subscription_id)

        return subscription_id

    def unsubscribe_from_topic(self, subscription_id: str) -> bool:
        """Unsubscribe from message topic"""
        return self.protocol.unsubscribe(subscription_id)
```

## 🧪 Testing

### Interface Testing

```python
# Interface testing framework
def test_api_interface():
    """Test API interface functionality"""
    config = {
        "base_url": "https://test-api.example.com",
        "authentication": {"type": "bearer_token", "token": "test_token"},
        "timeout": 10
    }

    interface = RESTAPIInterface(config)

    # Test GET request
    response = interface.get("/test/endpoint")
    assert response["status"] == "success"
    assert "data" in response

    # Test POST request
    post_data = {"key": "value"}
    post_response = interface.post("/test/endpoint", post_data)
    assert post_response["status"] == "success"

def test_data_interface():
    """Test data exchange interface"""
    config = {
        "supported_formats": ["json", "csv", "xml"],
        "validation": "strict"
    }

    interface = DataExchangeInterface(config)

    # Test data import
    import_result = interface.import_data("test_data.json")
    assert import_result["status"] == "success"
    assert import_result["data"] is not None

    # Test data export
    test_data = {"test": "data"}
    export_result = interface.export_data(test_data, "output.json")
    assert export_result["status"] == "success"

def test_protocol_interface():
    """Test message protocol interface"""
    config = {
        "protocol_type": "mqtt",
        "broker_url": "test-broker.example.com",
        "client_id": "test_client"
    }

    interface = MessageProtocolInterface(config)

    # Test message sending
    test_message = {"action": "test", "data": "test_data"}
    send_result = interface.send_message("test_destination", test_message)
    assert send_result["status"] == "sent"

    # Test message receiving
    receive_result = interface.receive_message()
    assert receive_result["status"] in ["received", "error"]
```

## 🔄 Development Workflow

### Interface Development Process

1. **Interface Requirements Analysis**:
   ```bash
   # Analyze integration requirements
   ai-interfaces analyze --domain neuroscience --requirements integration.yaml

   # Study existing interface patterns
   ai-interfaces patterns --extract --category api_interfaces
   ```

2. **Interface Design and Implementation**:
   ```bash
   # Design interface architecture
   ai-interfaces design --template api_interface --domain neuroscience

   # Implement interface following TDD
   ai-interfaces implement --test-first --template interface_implementation
   ```

3. **Interface Integration**:
   ```bash
   # Integrate with domain applications
   ai-interfaces integrate --interface neural_api --domain neuroscience

   # Validate integration
   ai-interfaces validate --integration --comprehensive
   ```

4. **Interface Documentation**:
   ```bash
   # Generate interface documentation
   ai-interfaces docs --generate --interface neural_api

   # Update interface registry
   ai-interfaces registry --update
   ```

### Interface Quality Assurance

```python
# Interface quality validation
def validate_interface_quality(interface: DomainInterface) -> Dict[str, Any]:
    """Validate interface quality and functionality"""

    quality_metrics = {
        "functionality": validate_interface_functionality(interface),
        "integration": validate_interface_integration(interface),
        "performance": validate_interface_performance(interface),
        "documentation": validate_interface_documentation(interface),
        "testing": validate_interface_testing(interface)
    }

    # Overall quality assessment
    overall_score = calculate_overall_interface_quality(quality_metrics)

    return {
        "metrics": quality_metrics,
        "overall_score": overall_score,
        "certified": overall_score >= INTERFACE_QUALITY_THRESHOLD,
        "recommendations": generate_interface_improvements(quality_metrics)
    }
```

## 🤝 Contributing

### Interface Development Guidelines

When contributing domain interfaces:

1. **Domain Expertise**: Deep understanding of target domain requirements
2. **Integration Focus**: Design interfaces for seamless system integration
3. **Standards Compliance**: Follow established interface standards
4. **Documentation**: Provide comprehensive interface documentation
5. **Testing**: Include comprehensive interface testing

### Interface Review Process

1. **Functionality Review**: Validate interface functionality and features
2. **Integration Review**: Verify integration with domain applications
3. **Standards Review**: Ensure compliance with interface standards
4. **Quality Review**: Validate interface quality and reliability
5. **Documentation Review**: Validate interface documentation completeness

## 📚 Resources

### Interface Documentation
- **[API Interfaces](api_interfaces/README.md)**: API integration interfaces
- **[Data Interfaces](data_interfaces/README.md)**: Data exchange interfaces
- **[Protocol Interfaces](protocol_interfaces/README.md)**: Communication protocols

### Integration References
- **[Integration Patterns](../../../applications/integrations/README.md)**: Integration best practices
- **[API Standards](../../../docs/api/README.md)**: API design standards
- **[Domain Interfaces](../../../knowledge/applications/domains/README.md)**: Domain-specific interfaces

## 📄 License

This domain interfaces collection is part of the Active Inference Knowledge Environment and follows the same [MIT License](../../../LICENSE).

---

**Domain Interfaces Version**: 1.0.0 | **Last Updated**: October 2024 | **Development Status**: Active Development

*"Active Inference for, with, by Generative AI"* - Enabling seamless integration through standardized interfaces and comprehensive connectivity.
