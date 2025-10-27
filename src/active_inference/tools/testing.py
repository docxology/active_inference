"""
Tools - Testing Framework

Comprehensive testing framework for Active Inference components and applications.
Provides unit testing, integration testing, performance testing, and quality
assurance tools for ensuring reliable and robust implementations.
"""

import logging
import unittest
import time
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a test execution"""
    test_name: str
    status: str  # passed, failed, error, skipped
    execution_time: float
    error_message: Optional[str] = None
    traceback: Optional[str] = None


class TestRunner:
    """Runs and manages test execution"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_results: List[TestResult] = []

        logger.info("TestRunner initialized")

    def run_test(self, test_function: Callable, test_name: str) -> TestResult:
        """Run a single test function"""
        logger.debug(f"Running test: {test_name}")

        start_time = time.time()
        result = TestResult(
            test_name=test_name,
            status="pending",
            execution_time=0.0
        )

        try:
            # Run the test
            test_function()
            result.status = "passed"

        except AssertionError as e:
            result.status = "failed"
            result.error_message = str(e)

        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            result.traceback = traceback.format_exc()

        finally:
            result.execution_time = time.time() - start_time

        self.test_results.append(result)
        logger.debug(f"Test {test_name} completed: {result.status} ({result.execution_time".3f"}s)")

        return result

    def run_test_class(self, test_class: type) -> List[TestResult]:
        """Run all tests in a test class"""
        logger.info(f"Running test class: {test_class.__name__}")

        results = []

        # Get all test methods
        test_methods = [
            method for method in dir(test_class)
            if method.startswith('test_') and callable(getattr(test_class, method))
        ]

        # Create instance
        test_instance = test_class()

        # Run setup if available
        if hasattr(test_instance, 'setUp'):
            test_instance.setUp()

        try:
            # Run each test method
            for method_name in test_methods:
                test_function = getattr(test_instance, method_name)
                result = self.run_test(test_function, f"{test_class.__name__}.{method_name}")
                results.append(result)

        finally:
            # Run teardown if available
            if hasattr(test_instance, 'tearDown'):
                test_instance.tearDown()

        logger.info(f"Test class {test_class.__name__} completed: {len([r for r in results if r.status == 'passed'])}/{len(results)} passed")

        return results

    def run_test_suite(self, test_modules: List[str]) -> Dict[str, Any]:
        """Run a complete test suite"""
        logger.info(f"Running test suite with {len(test_modules)} modules")

        all_results = []
        suite_start_time = time.time()

        for module_name in test_modules:
            try:
                # Import test module
                module = __import__(module_name, fromlist=[''])

                # Find test classes
                test_classes = [
                    obj for name, obj in module.__dict__.items()
                    if (isinstance(obj, type) and
                        issubclass(obj, unittest.TestCase) and
                        obj != unittest.TestCase)
                ]

                # Run each test class
                for test_class in test_classes:
                    results = self.run_test_class(test_class)
                    all_results.extend(results)

            except Exception as e:
                logger.error(f"Failed to run test module {module_name}: {e}")
                all_results.append(TestResult(
                    test_name=f"module_{module_name}",
                    status="error",
                    execution_time=0.0,
                    error_message=str(e)
                ))

        suite_time = time.time() - suite_start_time

        # Generate summary
        passed = len([r for r in all_results if r.status == "passed"])
        failed = len([r for r in all_results if r.status == "failed"])
        errors = len([r for r in all_results if r.status == "error"])

        summary = {
            "total_tests": len(all_results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "success_rate": passed / len(all_results) if all_results else 0,
            "total_time": suite_time,
            "results": all_results
        }

        logger.info(f"Test suite completed: {summary}")
        return summary

    def generate_report(self, output_path: Path) -> bool:
        """Generate test report"""
        if not self.test_results:
            logger.warning("No test results to report")
            return False

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.test_results),
            "results_by_status": {},
            "results": [
                {
                    "test_name": result.test_name,
                    "status": result.status,
                    "execution_time": result.execution_time,
                    "error_message": result.error_message
                }
                for result in self.test_results
            ]
        }

        # Count by status
        for result in self.test_results:
            status = result.status
            if status not in report["results_by_status"]:
                report["results_by_status"][status] = 0
            report["results_by_status"][status] += 1

        # Save report
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Test report saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save test report: {e}")
            return False


class QualityAssurance:
    """Quality assurance and validation tools"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules: Dict[str, Callable] = {}

        self._initialize_validation_rules()
        logger.info("QualityAssurance initialized")

    def _initialize_validation_rules(self) -> None:
        """Initialize validation rules"""
        self.validation_rules = {
            "probability_distribution": self._validate_probability_distribution,
            "numerical_stability": self._validate_numerical_stability,
            "parameter_ranges": self._validate_parameter_ranges,
            "data_integrity": self._validate_data_integrity
        }

    def _validate_probability_distribution(self, data: List[float]) -> Dict[str, Any]:
        """Validate that data forms a valid probability distribution"""
        issues = []

        if not data:
            issues.append("Empty distribution")
            return {"valid": False, "issues": issues}

        # Check for negative values
        negative_indices = [i for i, x in enumerate(data) if x < 0]
        if negative_indices:
            issues.append(f"Negative values at indices: {negative_indices}")

        # Check normalization
        total = sum(data)
        if abs(total - 1.0) > 1e-6:
            issues.append(f"Distribution does not sum to 1.0 (sum = {total})")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "sum": total,
            "min_value": min(data),
            "max_value": max(data)
        }

    def _validate_numerical_stability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate numerical stability of computations"""
        issues = []

        # Check for NaN values
        for key, value in data.items():
            if isinstance(value, (int, float)) and np.isnan(value):
                issues.append(f"NaN value in {key}")
            elif isinstance(value, (list, np.ndarray)):
                if np.any(np.isnan(value)):
                    issues.append(f"NaN values in {key}")

        # Check for infinite values
        for key, value in data.items():
            if isinstance(value, (int, float)) and np.isinf(value):
                issues.append(f"Infinite value in {key}")
            elif isinstance(value, (list, np.ndarray)):
                if np.any(np.isinf(value)):
                    issues.append(f"Infinite values in {key}")

        # Check for extremely large values
        for key, value in data.items():
            if isinstance(value, (int, float)) and abs(value) > 1e10:
                issues.append(f"Extremely large value in {key}: {value}")
            elif isinstance(value, (list, np.ndarray)):
                if np.any(np.abs(value) > 1e10):
                    issues.append(f"Extremely large values in {key}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "numerically_stable": len(issues) == 0
        }

    def _validate_parameter_ranges(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameter ranges"""
        issues = []

        # Common parameter validations
        for param_name, value in parameters.items():
            if param_name.endswith("_rate") or param_name.endswith("_learning_rate"):
                if not (0 < value <= 1):
                    issues.append(f"Learning rate {param_name} should be in (0, 1], got {value}")

            if param_name.endswith("_temperature"):
                if value <= 0:
                    issues.append(f"Temperature {param_name} should be positive, got {value}")

            if param_name.endswith("_precision") or param_name.endswith("_inverse_temperature"):
                if value <= 0:
                    issues.append(f"Precision {param_name} should be positive, got {value}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "parameters_checked": len(parameters)
        }

    def _validate_data_integrity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data integrity"""
        issues = []

        # Check for missing required fields
        required_fields = ["timestamp", "data"]
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")

        # Check data consistency
        if "data" in data:
            data_field = data["data"]

            if isinstance(data_field, dict):
                # Check for consistent key types
                key_types = {type(k).__name__ for k in data_field.keys()}
                if len(key_types) > 1:
                    issues.append(f"Inconsistent key types in data: {key_types}")

            elif isinstance(data_field, list):
                # Check for consistent item types
                if data_field:
                    item_types = {type(item).__name__ for item in data_field}
                    if len(item_types) > 1:
                        issues.append(f"Inconsistent item types in data list: {item_types}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "data_integrity_ok": len(issues) == 0
        }

    def validate_component(self, component_name: str, component_data: Dict[str, Any],
                          validation_types: List[str] = None) -> Dict[str, Any]:
        """Validate a component using specified validation rules"""
        if validation_types is None:
            validation_types = list(self.validation_rules.keys())

        logger.info(f"Validating component {component_name} with {len(validation_types)} validation types")

        validation_results = {}

        for validation_type in validation_types:
            if validation_type in self.validation_rules:
                try:
                    result = self.validation_rules[validation_type](component_data)
                    validation_results[validation_type] = result
                except Exception as e:
                    logger.error(f"Validation {validation_type} failed: {e}")
                    validation_results[validation_type] = {
                        "valid": False,
                        "error": str(e)
                    }

        # Overall validation
        overall_valid = all(
            result.get("valid", False)
            for result in validation_results.values()
        )

        return {
            "component": component_name,
            "overall_valid": overall_valid,
            "validation_results": validation_results,
            "validation_types": validation_types,
            "timestamp": datetime.now().isoformat()
        }


class TestingFramework:
    """Main testing framework coordinating all testing activities"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_runner = TestRunner(config.get("runner", {}))
        self.quality_assurance = QualityAssurance(config.get("qa", {}))

        logger.info("TestingFramework initialized")

    def run_tests(self, test_modules: List[str]) -> Dict[str, Any]:
        """Run complete test suite"""
        return self.test_runner.run_test_suite(test_modules)

    def validate_model(self, model_name: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an Active Inference model"""
        return self.quality_assurance.validate_component(
            model_name,
            model_data,
            ["numerical_stability", "parameter_ranges", "probability_distribution"]
        )

    def run_performance_test(self, test_function: Callable, test_name: str,
                            iterations: int = 100) -> Dict[str, Any]:
        """Run performance test"""
        logger.info(f"Running performance test: {test_name} ({iterations} iterations)")

        execution_times = []

        for i in range(iterations):
            start_time = time.time()
            try:
                test_function()
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            except Exception as e:
                logger.error(f"Performance test iteration {i} failed: {e}")
                execution_times.append(float('inf'))  # Mark as failed

        # Compute statistics
        valid_times = [t for t in execution_times if not np.isinf(t)]

        if valid_times:
            performance_stats = {
                "test_name": test_name,
                "iterations": iterations,
                "successful_iterations": len(valid_times),
                "failed_iterations": iterations - len(valid_times),
                "min_time": min(valid_times),
                "max_time": max(valid_times),
                "mean_time": sum(valid_times) / len(valid_times),
                "median_time": sorted(valid_times)[len(valid_times) // 2],
                "total_time": sum(valid_times),
                "throughput": len(valid_times) / sum(valid_times) if sum(valid_times) > 0 else 0
            }
        else:
            performance_stats = {
                "test_name": test_name,
                "iterations": iterations,
                "successful_iterations": 0,
                "failed_iterations": iterations,
                "error": "All iterations failed"
            }

        logger.info(f"Performance test completed: {performance_stats}")
        return performance_stats

    def generate_test_report(self, output_path: Path) -> bool:
        """Generate comprehensive test report"""
        return self.test_runner.generate_report(output_path)
