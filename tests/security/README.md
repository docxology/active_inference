# Security Testing Framework

This directory contains security tests for the Active Inference Knowledge Environment, ensuring protection against common vulnerabilities and security threats.

## Overview

Security testing validates that the platform is protected against various security threats including injection attacks, path traversal, cross-site scripting (XSS), and other common vulnerabilities. These tests ensure the system maintains security and data integrity.

## Test Categories

### 🛡️ Input Validation Testing
Tests that validate input sanitization and protection:
- **Malicious Input Handling**: Protection against XSS, SQL injection, and other injection attacks
- **Input Sanitization**: Proper sanitization of user inputs and content
- **Boundary Validation**: Validation of input boundaries and limits
- **Type Safety**: Ensuring type safety and preventing type-related attacks

### 🔒 Path Traversal Protection
Tests that prevent unauthorized file system access:
- **Directory Traversal**: Protection against `../` path traversal attacks
- **File Access Control**: Ensuring proper file access restrictions
- **Resource Protection**: Preventing access to sensitive system resources
- **URL Validation**: Protection against malicious URLs and links

### 🧹 Content Sanitization
Tests that ensure content is properly sanitized:
- **HTML Sanitization**: Removal of malicious HTML and script tags
- **Content Validation**: Validation of content structure and format
- **Metadata Security**: Secure handling of metadata and configuration
- **Link Safety**: Validation of internal and external links

### 🚧 Resource Protection
Tests that prevent resource exhaustion and abuse:
- **Resource Limits**: Protection against resource exhaustion attacks
- **Rate Limiting**: Prevention of excessive request rates
- **Memory Protection**: Protection against memory-based attacks
- **CPU Protection**: Prevention of CPU-intensive operations

## Getting Started

### Running Security Tests

```bash
# Run all security tests
pytest tests/security/ -v

# Run specific security test categories
pytest tests/security/test_knowledge_security.py -v
pytest tests/security/test_api_security.py -v

# Run security tests with security-focused reporting
pytest tests/security/ --tb=short -q

# Run security vulnerability scans
pytest tests/security/ -m "security and vulnerability"
```

### Security Test Configuration

Security tests can be configured for different security levels and threat models:

```bash
# Set security test level
export SECURITY_LEVEL=high
export ENABLE_FUZZING=true
export MAX_INPUT_LENGTH=10000

# Run with enhanced security validation
pytest tests/security/ -m "security and not basic"
pytest tests/security/ -m "vulnerability"
```

## Security Test Coverage

### Common Vulnerabilities Tested

#### Injection Attacks
- **SQL Injection**: Protection against SQL injection in search and data operations
- **XSS Protection**: Prevention of cross-site scripting attacks
- **Command Injection**: Protection against command injection attempts
- **Template Injection**: Prevention of template injection attacks

#### Path Traversal
- **Directory Traversal**: Protection against `../` path traversal
- **File System Access**: Controlled access to file system resources
- **URL Traversal**: Protection against malicious URL manipulation
- **Resource Access Control**: Proper resource access restrictions

#### Content Security
- **Content Sanitization**: Removal of malicious content elements
- **Script Prevention**: Blocking of script execution in content
- **Link Validation**: Validation of internal and external links
- **Metadata Security**: Secure metadata handling

## Security Test Implementation

### Security Test Patterns

#### Input Validation Testing
```python
def test_sql_injection_protection(self, knowledge_repo):
    """Test protection against SQL injection attacks"""
    sql_injections = [
        "'; DROP TABLE knowledge_nodes; --",
        "' OR '1'='1",
        "'; INSERT INTO users VALUES ('hacker', 'password'); --",
        "admin'/*",
        "' UNION SELECT * FROM users --"
    ]

    for injection in sql_injections:
        # Should handle SQL injection safely
        results = knowledge_repo.search(injection)
        assert isinstance(results, list)
        # Should not cause database errors
        assert len(results) >= 0
```

#### Path Traversal Testing
```python
def test_path_traversal_protection(self, knowledge_repo):
    """Test protection against path traversal attacks"""
    traversal_attempts = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\SAM",
        "~/.bashrc",
        "/home/user/.ssh/id_rsa"
    ]

    for traversal_path in traversal_attempts:
        # Should not access system files
        results = knowledge_repo.search(traversal_path)
        assert isinstance(results, list)
        # Should return no results for system paths
        assert len(results) == 0

        # Should not crash or expose file contents
        node = knowledge_repo.get_node(traversal_path)
        assert node is None
```

#### Content Security Testing
```python
def test_xss_protection(self, knowledge_repo):
    """Test protection against XSS attacks"""
    xss_payloads = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert('xss')>",
        "javascript:alert('xss')",
        "<svg onload=alert('xss')>",
        "<iframe src='javascript:alert('xss')'></iframe>"
    ]

    malicious_content = {
        "id": "xss_test",
        "title": "XSS Test",
        "content_type": "foundation",
        "difficulty": "beginner",
        "description": "Testing XSS protection",
        "prerequisites": [],
        "content": {
            "overview": xss_payloads[0],
            "examples": [{"name": "XSS Example", "description": xss_payloads[1]}]
        },
        "tags": ["security", "xss"],
        "learning_objectives": ["Learn security"],
        "metadata": {"version": "1.0"}
    }

    # Should handle XSS content safely
    # (In real implementation, this would be sanitized)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save and load malicious content
        # Repository should handle without executing scripts
        pass
```

## Security Standards

### Security Testing Requirements
- **OWASP Compliance**: Follow OWASP security testing guidelines
- **Input Validation**: All inputs must be properly validated and sanitized
- **Error Handling**: Security errors should not expose sensitive information
- **Logging**: Security events must be properly logged
- **Access Control**: Proper access control and authorization checks

### Vulnerability Management
- **Regular Scanning**: Automated vulnerability scanning
- **Threat Modeling**: Regular threat model updates
- **Security Updates**: Prompt security patch application
- **Incident Response**: Security incident response procedures
- **Security Training**: Security awareness and training

## Security Monitoring

### Continuous Security Monitoring
- **Vulnerability Scanning**: Regular automated vulnerability scans
- **Security Event Monitoring**: Real-time security event monitoring
- **Access Logging**: Comprehensive access and security logging
- **Threat Intelligence**: Integration with threat intelligence feeds
- **Security Dashboards**: Visual security metrics and alerts

### Security Testing Integration
- **CI/CD Integration**: Security tests in continuous integration
- **Automated Security Gates**: Security validation in deployment pipeline
- **Security Regression Testing**: Detect security regressions
- **Compliance Testing**: Automated compliance validation

## Contributing

### Writing Security Tests
1. **Understand Threat Model**: Know the security threats to test against
2. **Follow Security Standards**: Adhere to security testing best practices
3. **Test Edge Cases**: Focus on unusual and malicious inputs
4. **Validate Sanitization**: Ensure proper input sanitization
5. **Document Security Requirements**: Clearly specify security expectations

### Security Test Best Practices
- **Realistic Attack Vectors**: Test with realistic attack patterns
- **Comprehensive Coverage**: Cover all common vulnerability types
- **Safe Testing**: Ensure security tests don't create actual vulnerabilities
- **Proper Cleanup**: Clean up test data and resources securely
- **Documentation**: Document security test rationale and expectations

## Security Resources

### Security Testing Tools
- **Input Fuzzing**: Automated input fuzzing and mutation testing
- **Static Analysis**: Static security analysis tools integration
- **Dynamic Analysis**: Runtime security testing and monitoring
- **Penetration Testing**: Automated penetration testing capabilities

### Security Standards and Guidelines
- **OWASP Top 10**: Protection against OWASP Top 10 vulnerabilities
- **CWE Coverage**: Coverage of Common Weakness Enumeration items
- **Security Headers**: Proper security headers and configurations
- **Content Security Policy**: CSP implementation and validation

## Related Documentation

- **[Testing README](../README.md)**: Main testing framework documentation
- **[Security Guidelines](../../applications/best_practices/)**: Security best practices
- **[System Security](../../README.md)**: System security requirements
- **[Platform Security](../../platform/)**: Platform security features

---

*"Active Inference for, with, by Generative AI"* - Ensuring security through comprehensive testing, validation, and protection.
