# Utilities - Agent Development Guide

**Guidelines for AI agents working with helper functions and development tools.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with utility systems:**

### Primary Responsibilities
- **Utility Development**: Create reusable helper functions and tools
- **Data Processing**: Implement common data transformation utilities
- **Configuration Management**: Build centralized configuration systems
- **Error Handling**: Standardize error handling across platform
- **Performance Monitoring**: Implement performance measurement tools
- **Validation Systems**: Create input validation and data integrity checks

### Development Focus Areas
1. **Core Utilities**: Essential helper functions for platform
2. **Configuration Systems**: Centralized configuration management
3. **Error Management**: Platform-wide error handling
4. **Performance Tools**: Monitoring and optimization utilities
5. **Validation Systems**: Input validation and data integrity

## 🏗️ Architecture & Integration

### Utility Architecture

**Understanding the utility system structure:**

```
Platform Utilities
├── Data Utilities (processing, transformation)
├── Configuration Management (loading, validation)
├── Error Handling (standardized, reporting)
├── Performance Monitoring (metrics, profiling)
├── Validation Systems (input, data integrity)
└── Integration Utilities (service coordination)
```

### Integration Points

**Key integration points for utilities:**

#### Platform Integration
- **All Components**: Utilities used across entire platform
- **Configuration System**: Centralized configuration management
- **Logging System**: Platform-wide logging utilities
- **Error Reporting**: Standardized error handling
- **Performance Monitoring**: Platform-wide performance tracking

#### External Systems
- **File Systems**: Configuration file management
- **Databases**: Connection and query utilities
- **Cache Systems**: Caching and optimization utilities
- **Monitoring Systems**: Performance and health monitoring

### Utility Categories

```python
# Utility organization pattern
utilities/
├── data/                    # Data processing and transformation
├── config/                  # Configuration management
├── errors/                  # Error handling and reporting
├── performance/             # Performance monitoring
├── validation/              # Input validation and integrity
└── integration/             # Service integration utilities
```

## 💻 Development Patterns

### Required Implementation Patterns

**All utility development must follow these patterns:**

#### 1. Configuration Management Pattern
```python
class ConfigurationManager:
    """Centralized configuration management"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.config_cache = {}
        self.validation_rules = self.load_validation_rules()

    def load_config(self, config_path: str, environment: str = None) -> Dict[str, Any]:
        """Load configuration with validation"""
        # Load configuration file
        raw_config = self.load_config_file(config_path)

        # Apply environment overrides
        if environment:
            env_overrides = self.load_environment_config(environment)
            raw_config = self.merge_configs(raw_config, env_overrides)

        # Validate configuration
        validation_result = self.validate_config(raw_config)
        if not validation_result["valid"]:
            raise ConfigurationError(f"Invalid configuration: {validation_result['errors']}")

        # Cache configuration
        self.cache_config(config_path, raw_config)

        return raw_config

    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        # Parse key path
        keys = key_path.split(".")

        # Navigate configuration
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def update_config(self, key_path: str, value: Any) -> bool:
        """Update configuration value"""
        # Validate update
        if not self.validate_config_update(key_path, value):
            return False

        # Apply update
        self.apply_config_update(key_path, value)

        # Clear relevant caches
        self.clear_affected_caches(key_path)

        return True
```

#### 2. Error Handling Pattern
```python
class ErrorHandler:
    """Standardized error handling"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_log = []
        self.error_reporter = ErrorReporter(config)

    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Handle error with context"""
        # Create error information
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "timestamp": datetime.utcnow(),
            "traceback": traceback.format_exc() if self.config.get("include_traceback") else None
        }

        # Classify error severity
        error_info["severity"] = self.classify_error_severity(error, context)

        # Log error
        self.log_error(error_info)

        # Report if critical
        if error_info["severity"] == "critical":
            self.report_critical_error(error_info)

        return error_info

    def create_user_friendly_message(self, error_info: ErrorInfo) -> str:
        """Create user-friendly error message"""
        # Map technical errors to user messages
        user_messages = {
            "ValidationError": "Invalid input provided. Please check your data.",
            "NetworkError": "Connection issue. Please check your network and try again.",
            "PermissionError": "You don't have permission to perform this action.",
            "ResourceError": "System resource issue. Please try again later."
        }

        error_type = error_info["error_type"]
        base_message = user_messages.get(error_type, "An unexpected error occurred.")

        # Add context-specific information
        if error_info["context"].get("field"):
            base_message += f" Field: {error_info['context']['field']}"

        return base_message

    def setup_global_error_handling(self) -> None:
        """Set up global error handling"""
        # Configure logging
        self.setup_error_logging()

        # Set up exception handlers
        sys.excepthook = self.global_exception_handler

        # Configure warnings
        warnings.filterwarnings("error", category=DeprecationWarning)
```

#### 3. Data Processing Pattern
```python
class DataProcessor:
    """Data processing and transformation utilities"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processors = self.initialize_processors()

    def process_knowledge_content(self, content: Dict) -> ProcessedContent:
        """Process knowledge content for platform"""
        # Normalize format
        normalized = self.normalize_content_format(content)

        # Extract metadata
        metadata = self.extract_content_metadata(normalized)

        # Validate structure
        validation = self.validate_content_structure(normalized)

        # Enhance content
        enhanced = self.enhance_content_features(normalized)

        return ProcessedContent(
            content=enhanced,
            metadata=metadata,
            validation=validation
        )

    def transform_data_format(self, data: Any, from_format: str, to_format: str) -> Any:
        """Transform data between formats"""
        # Validate formats
        self.validate_data_formats(from_format, to_format)

        # Apply transformation
        if from_format == "json" and to_format == "structured":
            return self.json_to_structured(data)
        elif from_format == "structured" and to_format == "json":
            return self.structured_to_json(data)
        else:
            return self.generic_transform(data, from_format, to_format)

    def validate_data_integrity(self, data: Dict, schema: Dict) -> ValidationResult:
        """Validate data integrity against schema"""
        # Schema validation
        schema_validation = self.validate_schema_compliance(data, schema)

        # Data type validation
        type_validation = self.validate_data_types(data, schema)

        # Business rule validation
        business_validation = self.validate_business_rules(data)

        # Cross-reference validation
        reference_validation = self.validate_cross_references(data)

        return {
            "schema_valid": schema_validation["valid"],
            "types_valid": type_validation["valid"],
            "business_valid": business_validation["valid"],
            "references_valid": reference_validation["valid"],
            "errors": [
                schema_validation["errors"],
                type_validation["errors"],
                business_validation["errors"],
                reference_validation["errors"]
            ]
        }
```

## 🧪 Testing Standards

### Test Categories (MANDATORY)

#### 1. Configuration Testing
```python
class TestConfigurationManagement:
    """Test configuration utilities"""

    def test_config_loading(self):
        """Test configuration loading"""
        # Test valid configuration
        config = self.config_manager.load_config("valid_config.yaml")
        assert config is not None
        assert "database_url" in config

        # Test invalid configuration
        with pytest.raises(ConfigurationError):
            self.config_manager.load_config("invalid_config.yaml")

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid configuration
        valid_config = create_valid_config()
        result = self.config_manager.validate_config(valid_config)
        assert result["valid"] == True

        # Invalid configuration
        invalid_config = create_invalid_config()
        result = self.config_manager.validate_config(invalid_config)
        assert result["valid"] == False
        assert len(result["errors"]) > 0
```

#### 2. Error Handling Testing
```python
class TestErrorHandling:
    """Test error handling utilities"""

    def test_error_classification(self):
        """Test error classification"""
        # Test different error types
        validation_error = ValidationError("Invalid input")
        network_error = NetworkError("Connection failed")
        unexpected_error = Exception("Unexpected error")

        # Test classification
        assert self.error_handler.classify_severity(validation_error) == "medium"
        assert self.error_handler.classify_severity(network_error) == "high"
        assert self.error_handler.classify_severity(unexpected_error) == "critical"

    def test_user_friendly_messages(self):
        """Test user-friendly error messages"""
        error_info = {
            "error_type": "ValidationError",
            "error_message": "Field 'email' is required",
            "context": {"field": "email"}
        }

        message = self.error_handler.create_user_friendly_message(error_info)
        assert "email" in message.lower()
        assert "required" in message.lower()
```

#### 3. Data Processing Testing
```python
class TestDataProcessing:
    """Test data processing utilities"""

    def test_content_normalization(self):
        """Test content normalization"""
        # Create test content
        raw_content = create_raw_content()

        # Normalize content
        normalized = self.data_processor.normalize_content(raw_content)

        # Validate normalization
        assert normalized["format_version"] == "1.0"
        assert "metadata" in normalized
        assert "validation" in normalized

    def test_data_transformation(self):
        """Test data format transformation"""
        # Test JSON to structured
        json_data = {"id": "test", "content": "test content"}
        structured = self.data_processor.transform_data_format(
            json_data, "json", "structured"
        )
        assert structured["id"] == "test"

        # Test structured to JSON
        back_to_json = self.data_processor.transform_data_format(
            structured, "structured", "json"
        )
        assert back_to_json["id"] == "test"
```

## 📖 Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. API Documentation
**All utility functions must be documented:**

```python
def load_configuration(config_path: str, environment: str = None, validate: bool = True) -> Dict[str, Any]:
    """
    Load and validate configuration from file.

    This function loads configuration from the specified path, applies
    environment-specific overrides, and validates against schema.

    Args:
        config_path: Path to configuration file (YAML, JSON, or INI)
        environment: Environment name for overrides (optional)
        validate: Whether to validate configuration (default: True)

    Returns:
        Dictionary containing validated configuration

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValidationError: If configuration validation fails
        ConfigurationError: If configuration loading fails

    Examples:
        >>> config = load_configuration("config/production.yaml", "production")
        >>> print(config["database_url"])
        "postgresql://..."
    """
    pass
```

#### 2. Configuration Schema Documentation
**Configuration schemas must be documented:**

```python
# Platform configuration schema
platform_config_schema = {
    "database": {
        "type": "object",
        "required": ["url", "pool_size"],
        "properties": {
            "url": {"type": "string", "format": "uri"},
            "pool_size": {"type": "integer", "minimum": 1, "maximum": 100},
            "timeout": {"type": "number", "default": 30.0}
        }
    },
    "cache": {
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean", "default": True},
            "backend": {"type": "string", "enum": ["redis", "memory"]},
            "ttl": {"type": "integer", "default": 3600}
        }
    }
}
```

## 🚀 Performance Optimization

### Performance Requirements

**Utility system must meet these performance standards:**

- **Configuration Load Time**: <100ms for typical configurations
- **Data Processing**: <10ms for typical transformations
- **Error Handling**: <1ms overhead for error processing
- **Validation**: <5ms for typical data validation

### Optimization Techniques

#### 1. Configuration Caching
```python
class ConfigCache:
    """Cache configuration for performance"""

    def __init__(self, config: Dict[str, Any]):
        self.cache = {}
        self.cache_timestamps = {}
        self.config = config

    def get_cached_config(self, config_path: str) -> Dict[str, Any]:
        """Get cached configuration"""
        # Check cache validity
        if self.is_cache_valid(config_path):
            return self.cache[config_path]

        # Load and cache configuration
        config = self.load_config(config_path)
        self.cache[config_path] = config
        self.cache_timestamps[config_path] = datetime.now()

        return config

    def is_cache_valid(self, config_path: str) -> bool:
        """Check if cached configuration is valid"""
        if config_path not in self.cache_timestamps:
            return False

        # Check file modification time
        file_mtime = os.path.getmtime(config_path)
        cache_time = self.cache_timestamps[config_path]

        return cache_time > file_mtime
```

#### 2. Data Processing Optimization
```python
class DataProcessingOptimizer:
    """Optimize data processing operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processing_cache = {}

    def optimize_processing_pipeline(self, data: Any) -> OptimizedPipeline:
        """Create optimized processing pipeline"""
        # Analyze data characteristics
        data_analysis = self.analyze_data_characteristics(data)

        # Select optimal processors
        processors = self.select_optimal_processors(data_analysis)

        # Create processing pipeline
        pipeline = self.create_optimized_pipeline(processors)

        # Pre-compile transformations
        self.precompile_transformations(pipeline)

        return pipeline

    def batch_process_data(self, data_batch: List[Any]) -> List[ProcessedData]:
        """Process data in optimized batches"""
        # Determine optimal batch size
        batch_size = self.calculate_optimal_batch_size(data_batch)

        # Process in batches
        results = []
        for i in range(0, len(data_batch), batch_size):
            batch = data_batch[i:i + batch_size]
            batch_result = self.process_batch_optimized(batch)
            results.extend(batch_result)

        return results
```

## 🔒 Security Standards

### Security Requirements (MANDATORY)

#### 1. Configuration Security
```python
class ConfigSecurity:
    """Secure configuration management"""

    def validate_config_security(self, config: Dict[str, Any]) -> SecurityReport:
        """Validate configuration security"""
        # Check for secrets in config
        secrets_check = self.check_for_secrets(config)

        # Validate secure defaults
        defaults_check = self.validate_secure_defaults(config)

        # Check access permissions
        permissions_check = self.check_config_permissions(config)

        return {
            "secrets_detected": secrets_check,
            "secure_defaults": defaults_check,
            "permissions_valid": permissions_check
        }

    def sanitize_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration for security"""
        # Remove sensitive information
        sanitized = self.remove_sensitive_data(config)

        # Encrypt secrets
        encrypted = self.encrypt_secrets(sanitized)

        # Validate sanitization
        validation = self.validate_sanitization(encrypted)

        return encrypted if validation["valid"] else {}
```

#### 2. Data Validation Security
```python
class ValidationSecurity:
    """Secure data validation"""

    def validate_input_security(self, input_data: Dict[str, Any]) -> SecurityReport:
        """Validate input for security threats"""
        # SQL injection prevention
        sql_check = self.prevent_sql_injection(input_data)

        # XSS prevention
        xss_check = self.prevent_xss(input_data)

        # Command injection prevention
        cmd_check = self.prevent_command_injection(input_data)

        return {
            "sql_injection_safe": sql_check["safe"],
            "xss_safe": xss_check["safe"],
            "command_injection_safe": cmd_check["safe"]
        }
```

## 🐛 Debugging & Troubleshooting

### Debug Configuration

```python
# Enable utility debugging
debug_config = {
    "debug_mode": True,
    "log_level": "DEBUG",
    "config_debug": True,
    "error_debug": True,
    "performance_debug": True
}
```

### Common Debugging Patterns

#### 1. Configuration Debugging
```python
class ConfigDebugger:
    """Debug configuration issues"""

    def debug_config_loading(self, config_path: str) -> DebugReport:
        """Debug configuration loading"""
        # Check file existence
        file_check = self.check_file_exists(config_path)

        # Check file permissions
        permission_check = self.check_file_permissions(config_path)

        # Check syntax
        syntax_check = self.check_config_syntax(config_path)

        # Check validation
        validation_check = self.check_config_validation(config_path)

        return {
            "file_exists": file_check,
            "permissions": permission_check,
            "syntax_valid": syntax_check,
            "validation": validation_check
        }
```

## 🔄 Development Workflow

### Agent Development Process

1. **Task Assessment**
   - Identify utility requirements
   - Analyze platform needs
   - Consider reusability requirements

2. **Architecture Planning**
   - Design utility interfaces
   - Plan configuration management
   - Consider error handling needs

3. **Test-Driven Development**
   - Write utility tests first
   - Test configuration scenarios
   - Validate error handling

4. **Implementation**
   - Implement utility functions
   - Add configuration management
   - Implement error handling

5. **Quality Assurance**
   - Test utility performance
   - Validate security compliance
   - Cross-platform testing

6. **Integration**
   - Test with platform components
   - Validate configuration integration
   - Performance optimization

### Code Review Checklist

**Before submitting utility code for review:**

- [ ] **Functionality Tests**: Comprehensive utility function tests
- [ ] **Configuration Tests**: Configuration loading and validation tests
- [ ] **Error Tests**: Error handling and recovery tests
- [ ] **Performance Tests**: Utility performance validation
- [ ] **Security Tests**: Input validation and security tests
- [ ] **Documentation**: Complete API and configuration documentation

## 📚 Learning Resources

### Utility Development Resources

- **[Python Utilities](https://example.com/python-utils)**: Utility function patterns
- **[Configuration Management](https://example.com/config-mgmt)**: Configuration best practices
- **[Error Handling](https://example.com/error-handling)**: Error management patterns
- **[Performance Monitoring](https://example.com/perf-monitoring)**: Performance tools

### Platform Integration

- **[Platform Architecture](../../platform/README.md)**: Platform structure
- **[Service Patterns](../../../src/active_inference/README.md)**: Integration patterns
- **[Quality Standards](../../../.cursorrules)**: Development standards

## 🎯 Success Metrics

### Quality Metrics

- **Utility Coverage**: All platform components have utilities
- **Performance**: Efficient utility operations
- **Security**: Secure configuration and data handling
- **Maintainability**: Clean, reusable utility code

### Development Metrics

- **Code Reuse**: High utility function reuse across platform
- **Configuration Management**: Centralized, validated configuration
- **Error Handling**: Consistent error management
- **Performance**: Optimized utility operations

---

**Component**: Utilities | **Version**: 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Essential utilities for robust platform operation.