# Utilities - Source Code Implementation

This directory contains the source code implementation of the Active Inference utilities system, providing helper functions, data processing tools, configuration management, and mathematical utilities for development and scientific computing.

## Overview

The utilities module provides comprehensive helper functions and tools for Active Inference development, including data processing, configuration management, file operations, mathematical utilities, and general development assistance functions.

## Module Structure

```
src/active_inference/tools/utilities/
├── __init__.py         # Utilities module exports
├── utilities.py        # Core utilities and helper functions
└── [implementations]   # Specialized utility implementations
    ├── data_processing/    # Data processing and analysis utilities
    ├── configuration/      # Configuration management utilities
    ├── file_management/    # File and I/O utilities
    ├── mathematical/       # Mathematical utilities for Active Inference
    └── helper_functions/   # General development helper functions
```

## Core Components

### 🔧 Core Utilities (`utilities.py`)
**Comprehensive utility functions and tools**
- Data processing and normalization utilities
- Configuration management and validation
- File management and safe I/O operations
- Mathematical utilities specialized for Active Inference
- General helper functions for development workflows

**Key Methods to Implement:**
```python
def normalize_data(self, data: List[float], method: str = "minmax") -> List[float]:
    """Normalize data using various normalization methods with validation"""

def smooth_data(self, data: List[float], window_size: int = 5) -> List[float]:
    """Apply smoothing algorithms to time series data with error handling"""

def compute_statistics(self, data: List[float]) -> Dict[str, float]:
    """Compute comprehensive statistical measures with edge case handling"""

def load_config(self, config_path: Path, config_type: str = "auto") -> Dict[str, Any]:
    """Load configuration with validation, error handling, and format detection"""

def save_config(self, config: Dict[str, Any], config_path: Path, config_type: str = "json") -> bool:
    """Save configuration with backup, validation, and atomic operations"""

def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configuration files with proper precedence and validation"""

def softmax(self, x: List[float], temperature: float = 1.0) -> List[float]:
    """Compute softmax with temperature scaling and numerical stability"""

def compute_entropy(self, probabilities: List[float]) -> float:
    """Compute Shannon entropy for probability distributions with validation"""

def kl_divergence(self, p: List[float], q: List[float]) -> float:
    """Compute KL divergence between probability distributions with numerical stability"""

def safe_write_file(self, file_path: Path, content: str) -> bool:
    """Safely write content to file with atomic operations and backup"""

def load_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file with comprehensive error handling and validation"""

def save_json_file(self, data: Dict[str, Any], file_path: Path) -> bool:
    """Save data as JSON file with atomic operations and validation"""
```

## Implementation Architecture

### Utility System Architecture
The utilities system implements a comprehensive framework with:
- **Data Processing**: Comprehensive data processing and analysis tools
- **Configuration Management**: Centralized configuration with validation
- **File Management**: Safe file operations with backup and recovery
- **Mathematical Tools**: Specialized mathematical utilities for Active Inference
- **Development Helpers**: Common development and debugging utilities

### Specialized Utility Modules
Each utility category provides specialized functionality:
- **Data Processing**: Normalization, smoothing, statistical analysis
- **Configuration**: Loading, merging, validation, and persistence
- **File Management**: Safe I/O operations with backup and atomicity
- **Mathematical**: Information theory and probabilistic utilities
- **Helpers**: General development assistance functions

## Development Guidelines

### Utility Development Standards
- **Reusability**: Design utilities for maximum reusability across projects
- **Reliability**: Implement comprehensive error handling and validation
- **Performance**: Optimize utilities for efficiency and speed
- **Documentation**: Provide comprehensive documentation and examples
- **Testing**: Create extensive test suites for utility reliability

### Quality Standards
- **Code Quality**: Follow established utility patterns and best practices
- **Error Handling**: Comprehensive error handling with informative messages
- **Performance**: Optimize utilities for development workflow efficiency
- **Documentation**: Complete documentation with usage examples
- **Testing**: Comprehensive testing including edge cases and validation

## Usage Examples

### Data Processing Utilities
```python
from active_inference.tools.utilities import Utilities

# Initialize utilities
utils = Utilities(config)

# Data processing
data = [1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 1.0, 7.0]

# Normalize data
normalized = utils.data_processing.normalize_data(data, "zscore")
print(f"Normalized data: {normalized}")

# Smooth data
smoothed = utils.data_processing.smooth_data(data, window_size=3)
print(f"Smoothed data: {smoothed}")

# Compute statistics
statistics = utils.data_processing.compute_statistics(data)
print(f"Data statistics: {statistics}")

# Mathematical utilities
probabilities = [0.1, 0.2, 0.3, 0.4]
entropy = utils.mathematics.compute_entropy(probabilities)
print(f"Entropy: {entropy}")

kl_div = utils.mathematics.kl_divergence([0.25, 0.25, 0.25, 0.25], probabilities)
print(f"KL divergence: {kl_div}")
```

### Configuration Management
```python
from active_inference.tools.utilities import Utilities

# Initialize utilities
utils = Utilities(config)

# Load configuration
base_config = utils.configuration.load_config(Path("config/base.json"))
override_config = utils.configuration.load_config(Path("config/override.yaml"))

# Merge configurations
merged_config = utils.configuration.merge_configs(base_config, override_config)
print(f"Merged config keys: {list(merged_config.keys())}")

# Save configuration
success = utils.configuration.save_config(merged_config, Path("config/merged.json"))
print(f"Config saved: {success}")
```

### File Management
```python
from active_inference.tools.utilities import Utilities

# Initialize utilities
utils = Utilities(config)

# Safe file operations
content = "This is test content for file operations"
file_path = Path("test_output.txt")

# Safe write with backup
success = utils.file_manager.safe_write_file(file_path, content)
print(f"File written safely: {success}")

# Load JSON with validation
data = utils.file_manager.load_json_file(Path("data.json"))
if data:
    print(f"Loaded data: {len(data)} items")

# Save JSON atomically
test_data = {"test": "data", "numbers": [1, 2, 3]}
success = utils.file_manager.save_json_file(test_data, Path("test.json"))
print(f"JSON saved: {success}")
```

## Testing Framework

### Utility Testing Requirements
- **Function Testing**: Test individual utility functions in isolation
- **Data Testing**: Test data processing with various data types and edge cases
- **Configuration Testing**: Test configuration loading, merging, and validation
- **File Testing**: Test file operations with various scenarios and error conditions
- **Mathematical Testing**: Test mathematical utilities with known values

### Test Structure
```python
class TestUtilitiesFramework(unittest.TestCase):
    """Test utilities framework functionality and accuracy"""

    def setUp(self):
        """Set up test environment"""
        self.utilities = Utilities(test_config)

    def test_data_processing_utilities(self):
        """Test data processing utility functions"""
        # Test data normalization
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        minmax_normalized = self.utilities.data_processing.normalize_data(data, "minmax")
        self.assertAlmostEqual(min(minmax_normalized), 0.0, places=6)
        self.assertAlmostEqual(max(minmax_normalized), 1.0, places=6)

        zscore_normalized = self.utilities.data_processing.normalize_data(data, "zscore")
        self.assertAlmostEqual(np.mean(zscore_normalized), 0.0, places=6)
        self.assertAlmostEqual(np.std(zscore_normalized), 1.0, places=6)

        # Test data smoothing
        smoothed = self.utilities.data_processing.smooth_data(data, window_size=3)
        self.assertEqual(len(smoothed), len(data))

        # Test statistics computation
        stats = self.utilities.data_processing.compute_statistics(data)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertEqual(stats["count"], len(data))

    def test_mathematical_utilities(self):
        """Test mathematical utility functions"""
        # Test entropy computation
        uniform_dist = [0.25, 0.25, 0.25, 0.25]
        entropy_uniform = self.utilities.mathematics.compute_entropy(uniform_dist)
        self.assertAlmostEqual(entropy_uniform, 2.0, places=6)

        skewed_dist = [0.9, 0.05, 0.03, 0.02]
        entropy_skewed = self.utilities.mathematics.compute_entropy(skewed_dist)
        self.assertLess(entropy_skewed, entropy_uniform)

        # Test KL divergence
        p = [0.2, 0.3, 0.3, 0.2]
        q = [0.25, 0.25, 0.25, 0.25]

        kl_div = self.utilities.mathematics.kl_divergence(p, q)
        self.assertGreaterEqual(kl_div, 0)

        # Test softmax
        logits = [1.0, 2.0, 0.5]
        softmax_probs = self.utilities.mathematics.softmax(logits)
        self.assertAlmostEqual(sum(softmax_probs), 1.0, places=6)

        # Test with temperature
        softmax_temp = self.utilities.mathematics.softmax(logits, temperature=0.5)
        self.assertAlmostEqual(sum(softmax_temp), 1.0, places=6)

    def test_configuration_management(self):
        """Test configuration management utilities"""
        # Test configuration loading
        config_path = Path("test_config.json")
        test_config = {"database": {"host": "localhost", "port": 5432}, "debug": True}

        success = self.utilities.file_manager.save_json_file(test_config, config_path)
        self.assertTrue(success)

        loaded_config = self.utilities.configuration.load_config(config_path)
        self.assertEqual(loaded_config["database"]["host"], "localhost")
        self.assertEqual(loaded_config["database"]["port"], 5432)
        self.assertTrue(loaded_config["debug"])

        # Test configuration merging
        base_config = {"database": {"host": "localhost"}, "debug": False}
        override_config = {"database": {"port": 3306}, "cache": {"enabled": True}}

        merged = self.utilities.configuration.merge_configs(base_config, override_config)
        self.assertEqual(merged["database"]["host"], "localhost")
        self.assertEqual(merged["database"]["port"], 3306)
        self.assertTrue(merged["cache"]["enabled"])
        self.assertTrue(merged["debug"])  # Should preserve original value

    def test_file_management_utilities(self):
        """Test file management utility functions"""
        # Test safe file writing
        content = "This is test content for safe file operations"
        file_path = Path("test_safe_write.txt")

        success = self.utilities.file_manager.safe_write_file(file_path, content)
        self.assertTrue(success)
        self.assertTrue(file_path.exists())

        # Verify content
        with open(file_path, 'r') as f:
            saved_content = f.read()
        self.assertEqual(saved_content, content)

        # Test JSON file operations
        json_data = {"test": "data", "numbers": [1, 2, 3], "nested": {"key": "value"}}

        json_file = Path("test_data.json")
        json_success = self.utilities.file_manager.save_json_file(json_data, json_file)
        self.assertTrue(json_success)

        loaded_json = self.utilities.file_manager.load_json_file(json_file)
        self.assertEqual(loaded_json, json_data)

        # Cleanup
        file_path.unlink()
        json_file.unlink()
```

## Performance Considerations

### Utility Performance
- **Function Speed**: Optimize utility functions for fast execution
- **Memory Efficiency**: Efficient memory usage for large data processing
- **Resource Management**: Proper resource cleanup and management
- **Caching**: Intelligent caching for expensive operations

### Data Processing Performance
- **Algorithm Efficiency**: Use efficient algorithms for data processing
- **Vectorization**: Leverage NumPy vectorization for performance
- **Parallel Processing**: Utilize parallel processing where beneficial
- **Streaming**: Support streaming for large dataset processing

## Utility Integration

### Platform Integration
- **Service Integration**: Seamless integration with platform services
- **Component Communication**: Efficient communication between utilities
- **Data Flow**: Smooth data flow between utility functions
- **Configuration Management**: Unified configuration across utilities

### Development Environment Integration
- **IDE Integration**: Integration with popular development environments
- **Testing Integration**: Integration with testing frameworks and tools
- **Debugging Integration**: Support for debugging and development workflows
- **Performance Integration**: Performance monitoring and optimization

## Contributing Guidelines

When contributing to the utilities module:

1. **Utility Design**: Design utilities following established patterns and best practices
2. **Performance**: Optimize utilities for efficiency and speed
3. **Testing**: Include comprehensive testing for utility reliability
4. **Documentation**: Provide comprehensive documentation and examples
5. **Integration**: Ensure integration with platform utility systems
6. **Validation**: Validate utilities against real usage scenarios

## Related Documentation

- **[Main Tools README](../README.md)**: Tools module overview
- **[Utilities AGENTS.md](AGENTS.md)**: Agent development guidelines for this module
- **[Utilities Documentation](utilities.py)**: Core utilities implementation details
- **[Data Processing Documentation](utilities.py)**: Data processing utilities details
- **[Configuration Documentation](utilities.py)**: Configuration management details

---

*"Active Inference for, with, by Generative AI"* - Building utility systems through collaborative intelligence and comprehensive development assistance.