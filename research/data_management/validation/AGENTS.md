# Research Data Validation - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Research Data Validation module of the Active Inference Knowledge Environment. It outlines data validation methodologies, implementation patterns, and best practices for ensuring data quality, integrity, and compliance throughout the research lifecycle.

## Data Validation Module Overview

The Research Data Validation module provides a comprehensive framework for validating research data quality, integrity, and compliance. It supports multiple validation types, provides detailed quality assessment, and enables continuous quality improvement through automated validation pipelines and quality monitoring.

## Core Responsibilities

### Data Quality Validation
- **Schema Validation**: Comprehensive schema and format validation
- **Statistical Validation**: Statistical property and distribution validation
- **Quality Assessment**: Multi-dimensional quality scoring and assessment
- **Integrity Checks**: Data integrity and consistency verification
- **Compliance Validation**: Regulatory and ethical compliance checking

### Quality Assurance & Monitoring
- **Real-time Validation**: Continuous data validation and quality monitoring
- **Quality Gates**: Automated quality checkpoints and filtering
- **Anomaly Detection**: Statistical and pattern-based anomaly detection
- **Quality Reporting**: Comprehensive quality assessment and improvement reports
- **Continuous Improvement**: Automated quality improvement and optimization

### Validation Pipeline Management
- **Validation Orchestration**: Multi-stage validation pipeline management
- **Quality Workflow**: Automated quality assurance workflows
- **Validation Integration**: Integration with data processing and analysis pipelines
- **Quality Standards**: Establishment and maintenance of quality standards
- **Compliance Monitoring**: Continuous compliance monitoring and reporting

## Development Workflows

### Data Validation Development Process
1. **Requirements Analysis**: Analyze validation requirements and quality standards
2. **Validation Design**: Design validation methods and quality assessment frameworks
3. **Implementation**: Implement validation algorithms and quality assessment tools
4. **Testing**: Test validation methods with diverse datasets and edge cases
5. **Quality Calibration**: Calibrate validation methods and quality thresholds
6. **Integration**: Integrate validation into data processing pipelines
7. **Documentation**: Document validation methods and quality standards
8. **Deployment**: Deploy with monitoring and maintenance plans
9. **Review**: Regular review and improvement of validation methods

### Schema Validation Implementation
1. **Schema Analysis**: Analyze data schemas and validation requirements
2. **Rule Design**: Design validation rules and constraints
3. **Implementation**: Implement schema validation algorithms
4. **Testing**: Test with various data formats and edge cases
5. **Performance Testing**: Validate performance with large datasets
6. **Documentation**: Document schema requirements and validation rules

### Statistical Validation Implementation
1. **Statistical Analysis**: Analyze statistical requirements and validation needs
2. **Method Selection**: Select appropriate statistical validation methods
3. **Implementation**: Implement statistical validation algorithms
4. **Calibration**: Calibrate statistical tests and thresholds
5. **Testing**: Test with diverse statistical scenarios
6. **Documentation**: Document statistical validation methods and interpretations

## Quality Standards

### Validation Quality Standards
- **Accuracy**: >99% validation accuracy and correctness
- **Coverage**: 100% validation of defined quality dimensions
- **Consistency**: <1% variation in validation results across runs
- **Timeliness**: <100ms validation time for typical datasets
- **Reliability**: 99.9% validation system availability

### Quality Assessment Standards
- **Multi-dimensional**: Assessment across multiple quality dimensions
- **Quantitative**: Quantitative quality scoring and metrics
- **Actionable**: Clear recommendations for quality improvement
- **Comparative**: Comparison against established quality benchmarks
- **Trend Analysis**: Analysis of quality trends over time

### Compliance Standards
- **Regulatory Compliance**: 100% adherence to applicable regulations
- **Ethical Standards**: Complete research ethics compliance
- **Data Protection**: Comprehensive data protection and privacy
- **Audit Compliance**: Complete audit trails and compliance reporting
- **Standards Adherence**: Adherence to established quality standards

## Implementation Patterns

### Data Validation Framework
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging

@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    validation_types: List[str]
    quality_thresholds: Dict[str, float]
    strict_mode: bool = False
    auto_fix: bool = False
    generate_report: bool = True

@dataclass
class ValidationResult:
    """Result of validation operation"""
    validator_name: str
    timestamp: datetime
    data_id: str
    validation_passed: bool
    quality_score: float
    issues_found: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]
    processing_time: float

class BaseValidator(ABC):
    """Base class for data validation"""

    def __init__(self, config: ValidationConfig):
        """Initialize validator with configuration"""
        self.config = config
        self.validator_name = self.__class__.__name__.replace('Validator', '').lower()
        self.logger = logging.getLogger(f"validation.{self.validator_name}")
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'average_quality': 0.0,
            'last_validation': None
        }

    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input data"""
        pass

    @abstractmethod
    def assess_quality(self, data: Dict[str, Any]) -> float:
        """Assess data quality"""
        pass

    def update_stats(self, result: ValidationResult) -> None:
        """Update validation statistics"""
        self.validation_stats['total_validations'] += 1
        if result.validation_passed:
            self.validation_stats['passed_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1

        self.validation_stats['last_validation'] = datetime.now()

        # Update average quality
        current_total = self.validation_stats['average_quality'] * (self.validation_stats['total_validations'] - 1)
        self.validation_stats['average_quality'] = (current_total + result.quality_score) / self.validation_stats['total_validations']

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self.validation_stats.copy()

class SchemaValidator(BaseValidator):
    """Schema-based data validation"""

    def __init__(self, config: ValidationConfig, schema: Dict[str, Any]):
        """Initialize schema validator"""
        super().__init__(config)
        self.schema = schema
        self.schema_cache = {}

    def validate_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against schema"""
        start_time = datetime.now()
        data_id = data.get('id', 'unknown')

        try:
            # Schema validation
            schema_issues = self._validate_schema_structure(data)

            # Type validation
            type_issues = self._validate_data_types(data)

            # Format validation
            format_issues = self._validate_data_formats(data)

            # Cross-field validation
            cross_issues = self._validate_cross_field_constraints(data)

            # Combine all issues
            all_issues = schema_issues + type_issues + format_issues + cross_issues

            # Calculate quality score
            quality_score = self.assess_quality(data)

            # Determine if validation passed
            validation_passed = len(all_issues) == 0 or (not self.config.strict_mode and quality_score >= 0.8)

            # Generate suggestions
            suggestions = self._generate_suggestions(data, all_issues)

            result = ValidationResult(
                validator_name=self.validator_name,
                timestamp=datetime.now(),
                data_id=data_id,
                validation_passed=validation_passed,
                quality_score=quality_score,
                issues_found=all_issues,
                suggestions=suggestions,
                metadata={
                    'schema_version': self.schema.get('version', '1.0'),
                    'validation_type': 'schema',
                    'strict_mode': self.config.strict_mode
                },
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            self.update_stats(result)
            return result

        except Exception as e:
            self.logger.error(f"Schema validation failed for {data_id}: {str(e)}")
            result = ValidationResult(
                validator_name=self.validator_name,
                timestamp=datetime.now(),
                data_id=data_id,
                validation_passed=False,
                quality_score=0.0,
                issues_found=[f"Validation error: {str(e)}"],
                suggestions=["Check data format and schema compatibility"],
                metadata={'error': str(e)},
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            return result

    def _validate_schema_structure(self, data: Dict[str, Any]) -> List[str]:
        """Validate data structure against schema"""
        issues = []

        # Check required fields
        required_fields = self.schema.get('required', [])
        for field in required_fields:
            if field not in data:
                issues.append(f"Required field '{field}' is missing")

        # Check field types
        properties = self.schema.get('properties', {})
        for field, value in data.items():
            if field in properties:
                field_schema = properties[field]
                expected_type = field_schema.get('type')

                if expected_type:
                    if not isinstance(value, self._get_python_type(expected_type)):
                        issues.append(f"Field '{field}' has incorrect type: expected {expected_type}, got {type(value).__name__}")

        return issues

    def _get_python_type(self, schema_type: str):
        """Convert schema type to Python type"""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        return type_map.get(schema_type, object)

    def _validate_data_types(self, data: Dict[str, Any]) -> List[str]:
        """Validate data types"""
        issues = []

        # Type-specific validation
        for field, value in data.items():
            if isinstance(value, str):
                # String validation
                if len(value.strip()) == 0:
                    issues.append(f"Field '{field}' is empty string")
            elif isinstance(value, (int, float)):
                # Numeric validation
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    issues.append(f"Field '{field}' has invalid numeric value: {value}")

        return issues

    def _validate_data_formats(self, data: Dict[str, Any]) -> List[str]:
        """Validate data formats"""
        issues = []

        # Format validation based on schema
        properties = self.schema.get('properties', {})
        for field, value in data.items():
            if field in properties:
                field_schema = properties[field]

                # Date format validation
                if field_schema.get('format') == 'date-time':
                    try:
                        datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except ValueError:
                        issues.append(f"Field '{field}' has invalid date-time format: {value}")

                # Pattern validation
                pattern = field_schema.get('pattern')
                if pattern and isinstance(value, str):
                    import re
                    if not re.match(pattern, value):
                        issues.append(f"Field '{field}' does not match required pattern: {pattern}")

        return issues

    def _validate_cross_field_constraints(self, data: Dict[str, Any]) -> List[str]:
        """Validate cross-field constraints"""
        issues = []

        # Example: validate that end_time > start_time
        if 'start_time' in data and 'end_time' in data:
            try:
                start = datetime.fromisoformat(data['start_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(data['end_time'].replace('Z', '+00:00'))
                if end <= start:
                    issues.append("End time must be after start time")
            except ValueError:
                pass  # Let format validation handle this

        return issues

    def assess_quality(self, data: Dict[str, Any]) -> float:
        """Assess data quality based on schema compliance"""
        issues = self._validate_schema_structure(data)
        max_possible_issues = len(self.schema.get('required', [])) + len(self.schema.get('properties', []))

        if max_possible_issues == 0:
            return 1.0

        return max(0.0, 1.0 - (len(issues) / max_possible_issues))

    def _generate_suggestions(self, data: Dict[str, Any], issues: List[str]) -> List[str]:
        """Generate suggestions for data improvement"""
        suggestions = []

        if any('missing' in issue.lower() for issue in issues):
            suggestions.append("Add missing required fields")

        if any('type' in issue.lower() for issue in issues):
            suggestions.append("Check and correct data types")

        if any('format' in issue.lower() for issue in issues):
            suggestions.append("Validate date and other format requirements")

        if len(suggestions) == 0:
            suggestions.append("Data structure looks good")

        return suggestions

class StatisticalValidator(BaseValidator):
    """Statistical validation of data"""

    def __init__(self, config: ValidationConfig):
        """Initialize statistical validator"""
        super().__init__(config)
        self.statistical_tests = config.validation_types
        self.distribution_tests = ['normality', 'homogeneity']
        self.outlier_detection = True

    def validate_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data using statistical methods"""
        start_time = datetime.now()
        data_id = data.get('id', 'unknown')

        try:
            # Convert data to appropriate format
            numeric_data = self._extract_numeric_data(data)

            # Statistical validation
            statistical_issues = self._validate_statistical_properties(numeric_data)

            # Distribution validation
            distribution_issues = self._validate_distributions(numeric_data)

            # Outlier detection
            outlier_issues = self._detect_outliers(numeric_data)

            # Combine all issues
            all_issues = statistical_issues + distribution_issues + outlier_issues

            # Calculate quality score
            quality_score = self.assess_quality(data)

            # Determine if validation passed
            validation_passed = len(all_issues) == 0 or quality_score >= self.config.quality_thresholds.get('statistical', 0.8)

            # Generate suggestions
            suggestions = self._generate_statistical_suggestions(data, all_issues)

            result = ValidationResult(
                validator_name=self.validator_name,
                timestamp=datetime.now(),
                data_id=data_id,
                validation_passed=validation_passed,
                quality_score=quality_score,
                issues_found=all_issues,
                suggestions=suggestions,
                metadata={
                    'tests_performed': self.statistical_tests,
                    'data_points': len(numeric_data),
                    'variables': len(numeric_data.columns) if hasattr(numeric_data, 'columns') else 0
                },
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            self.update_stats(result)
            return result

        except Exception as e:
            self.logger.error(f"Statistical validation failed for {data_id}: {str(e)}")
            result = ValidationResult(
                validator_name=self.validator_name,
                timestamp=datetime.now(),
                data_id=data_id,
                validation_passed=False,
                quality_score=0.0,
                issues_found=[f"Statistical validation error: {str(e)}"],
                suggestions=["Check data format and statistical test compatibility"],
                metadata={'error': str(e)},
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            return result

    def _extract_numeric_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Extract numeric data for statistical analysis"""
        # This would extract numeric columns from the data
        # For now, return a placeholder
        return pd.DataFrame()

    def _validate_statistical_properties(self, data: pd.DataFrame) -> List[str]:
        """Validate statistical properties"""
        issues = []

        # Check for sufficient data
        if len(data) < 10:
            issues.append("Insufficient data for reliable statistical analysis")

        # Check for missing values
        missing_rate = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_rate > 0.1:
            issues.append(f"High missing data rate: {missing_rate:.1%}")

        # Check for constant columns
        for column in data.columns:
            if data[column].std() == 0:
                issues.append(f"Column '{column}' has zero variance (constant)")

        return issues

    def _validate_distributions(self, data: pd.DataFrame) -> List[str]:
        """Validate data distributions"""
        issues = []

        for column in data.select_dtypes(include=[np.number]).columns:
            # Simple normality test
            try:
                from scipy import stats
                _, p_value = stats.shapiro(data[column].dropna())

                if p_value < 0.05:
                    issues.append(f"Column '{column}' may not be normally distributed (p={p_value:.3f})")

            except Exception:
                pass  # Skip if test fails

        return issues

    def _detect_outliers(self, data: pd.DataFrame) -> List[str]:
        """Detect outliers in data"""
        issues = []

        for column in data.select_dtypes(include=[np.number]).columns:
            # Simple outlier detection using IQR
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            outlier_count = len(outliers)

            if outlier_count > 0:
                issues.append(f"Column '{column}' has {outlier_count} potential outliers")

        return issues

    def assess_quality(self, data: Dict[str, Any]) -> float:
        """Assess statistical quality of data"""
        # This would implement comprehensive statistical quality assessment
        return 0.85  # Placeholder

    def _generate_statistical_suggestions(self, data: Dict[str, Any], issues: List[str]) -> List[str]:
        """Generate statistical improvement suggestions"""
        suggestions = []

        if any('outlier' in issue.lower() for issue in issues):
            suggestions.append("Consider outlier removal or robust statistical methods")

        if any('missing' in issue.lower() for issue in issues):
            suggestions.append("Address missing data through imputation or data collection")

        if any('distribution' in issue.lower() for issue in issues):
            suggestions.append("Consider data transformation or non-parametric methods")

        if len(suggestions) == 0:
            suggestions.append("Statistical properties look acceptable")

        return suggestions

class ValidationPipeline:
    """Pipeline for comprehensive data validation"""

    def __init__(self, config: ValidationConfig):
        """Initialize validation pipeline"""
        self.config = config
        self.validators: List[BaseValidator] = []
        self._initialize_validators()

    def _initialize_validators(self) -> None:
        """Initialize validation pipeline validators"""
        # This would initialize different types of validators
        # For now, add placeholder validators
        self.validators = [
            SchemaValidator(self.config, {}),
            StatisticalValidator(self.config)
        ]

    async def validate_data(self, data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Run complete validation pipeline"""
        results = {}

        for validator in self.validators:
            try:
                result = validator.validate_data(data)
                results[validator.validator_name] = result

                # Early exit if strict mode and validation failed
                if self.config.strict_mode and not result.validation_passed:
                    self.logger.warning(f"Validation failed in strict mode at {validator.validator_name}")
                    break

            except Exception as e:
                self.logger.error(f"Validator {validator.validator_name} failed: {str(e)}")
                results[validator.validator_name] = ValidationResult(
                    validator_name=validator.validator_name,
                    timestamp=datetime.now(),
                    data_id=data.get('id', 'unknown'),
                    validation_passed=False,
                    quality_score=0.0,
                    issues_found=[f"Validator error: {str(e)}"],
                    suggestions=["Check validator configuration and data compatibility"],
                    metadata={'error': str(e)},
                    processing_time=0.0
                )

        return results

    def generate_validation_report(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_passed': all(result.validation_passed for result in results.values()),
            'overall_quality': sum(result.quality_score for result in results.values()) / len(results),
            'validation_results': {},
            'summary': {}
        }

        for validator_name, result in results.items():
            report['validation_results'][validator_name] = {
                'passed': result.validation_passed,
                'quality_score': result.quality_score,
                'issues_count': len(result.issues_found),
                'suggestions_count': len(result.suggestions),
                'processing_time': result.processing_time
            }

        # Generate summary
        total_issues = sum(len(result.issues_found) for result in results.values())
        total_suggestions = sum(len(result.suggestions) for result in results.values())

        report['summary'] = {
            'total_validators': len(results),
            'passed_validators': sum(1 for result in results.values() if result.validation_passed),
            'total_issues': total_issues,
            'total_suggestions': total_suggestions,
            'average_quality': report['overall_quality']
        }

        return report
```

### Quality Assessment Pattern
```python
class QualityAssessor:
    """Comprehensive quality assessment framework"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize quality assessor"""
        self.config = config
        self.dimensions = config.get('dimensions', ['completeness', 'accuracy', 'consistency'])
        self.weights = config.get('weights', {dim: 1.0/len(self.dimensions) for dim in self.dimensions})

    def assess_comprehensive_quality(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Assess data quality across multiple dimensions"""
        quality_scores = {}

        for dimension in self.dimensions:
            score = self._assess_dimension(data, dimension)
            quality_scores[dimension] = score

        # Calculate weighted overall score
        overall_score = sum(score * self.weights.get(dimension, 1.0) for dimension, score in quality_scores.items())

        return {
            **quality_scores,
            'overall': overall_score
        }

    def _assess_dimension(self, data: Dict[str, Any], dimension: str) -> float:
        """Assess specific quality dimension"""
        if dimension == 'completeness':
            return self._assess_completeness(data)
        elif dimension == 'accuracy':
            return self._assess_accuracy(data)
        elif dimension == 'consistency':
            return self._assess_consistency(data)
        else:
            return 0.5  # Default score

    def _assess_completeness(self, data: Dict[str, Any]) -> float:
        """Assess data completeness"""
        total_fields = len(data)
        if total_fields == 0:
            return 0.0

        filled_fields = sum(1 for value in data.values() if value is not None and value != '')
        return filled_fields / total_fields

    def _assess_accuracy(self, data: Dict[str, Any]) -> float:
        """Assess data accuracy"""
        # This would implement accuracy assessment
        # For now, return a placeholder
        return 0.9

    def _assess_consistency(self, data: Dict[str, Any]) -> float:
        """Assess data consistency"""
        # This would implement consistency assessment
        # For now, return a placeholder
        return 0.85
```

## Testing Guidelines

### Validation Testing
- **Unit Tests**: Test individual validation methods and algorithms
- **Integration Tests**: Test validation pipelines and workflows
- **Quality Tests**: Validate quality assessment accuracy
- **Performance Tests**: Test validation speed and memory usage
- **Edge Case Tests**: Test with problematic data and edge cases

### Quality Assurance
- **Quality Validation**: Ensure validation improves data quality
- **Consistency Testing**: Verify consistent validation results
- **Performance Validation**: Validate validation performance
- **Documentation Testing**: Test validation documentation accuracy
- **Integration Testing**: Test validation pipeline integration

## Performance Considerations

### Validation Performance
- **Speed Optimization**: Optimize validation algorithms for speed
- **Memory Efficiency**: Minimize memory usage during validation
- **Parallel Processing**: Utilize parallel processing for large datasets
- **Caching**: Cache validation results for repeated validations
- **Streaming**: Support streaming validation for large datasets

### Scalability
- **Data Volume Scaling**: Maintain performance with increasing data sizes
- **Validation Rule Scaling**: Scale validation rules and complexity
- **Quality Assessment Scaling**: Scale quality assessment with data complexity
- **Reporting Scaling**: Scale reporting with validation volume
- **Monitoring Scaling**: Scale monitoring with validation frequency

## Maintenance and Evolution

### Validation System Updates
- **Method Updates**: Update validation methods based on research
- **Quality Improvements**: Improve validation quality and accuracy
- **Performance Optimization**: Optimize validation performance
- **Documentation Updates**: Keep validation documentation current

### Quality Standards Evolution
- **Standards Updates**: Update quality standards based on research
- **Threshold Adjustment**: Adjust quality thresholds based on experience
- **Method Enhancement**: Enhance validation methods and algorithms
- **Integration Improvements**: Improve validation pipeline integration

## Common Challenges and Solutions

### Challenge: Validation Accuracy
**Solution**: Implement comprehensive validation testing, calibration against known standards, and continuous validation accuracy monitoring.

### Challenge: Performance with Large Datasets
**Solution**: Use streaming validation, parallel processing, and efficient algorithms with appropriate sampling for large datasets.

### Challenge: Validation Rule Complexity
**Solution**: Implement hierarchical validation rules, configurable validation levels, and validation rule optimization.

### Challenge: Quality Assessment Subjectivity
**Solution**: Use quantitative quality metrics, establish clear quality standards, and implement automated quality scoring.

## Getting Started as an Agent

### Development Setup
1. **Study Validation Architecture**: Understand validation system design and patterns
2. **Learn Quality Standards**: Study data quality standards and best practices
3. **Practice Implementation**: Practice implementing validation algorithms
4. **Understand Quality Assessment**: Learn quality assessment and scoring

### Contribution Process
1. **Identify Validation Needs**: Find gaps in current validation capabilities
2. **Research Validation Methods**: Study relevant validation algorithms and methods
3. **Design Validation Solutions**: Create detailed validation system designs
4. **Implement and Test**: Follow quality and performance implementation standards
5. **Validate Thoroughly**: Ensure validation accuracy and performance
6. **Document Completely**: Provide comprehensive validation documentation
7. **Quality Review**: Submit for quality and accuracy review

### Learning Resources
- **Data Validation**: Study data validation methodologies and standards
- **Quality Assurance**: Learn quality assessment and improvement techniques
- **Statistical Validation**: Study statistical validation and testing
- **Schema Design**: Learn schema design and validation patterns
- **Performance Optimization**: Study validation performance optimization

## Related Documentation

- **[Validation README](./README.md)**: Data validation module overview
- **[Data Management AGENTS.md](../AGENTS.md)**: Data management development guidelines
- **[Main AGENTS.md](../../../AGENTS.md)**: Project-wide agent guidelines
- **[Research AGENTS.md](../../AGENTS.md)**: Research tools module guidelines
- **[Contributing Guide](../../../CONTRIBUTING.md)**: Contribution processes

---

*"Active Inference for, with, by Generative AI"* - Advancing research through comprehensive data validation, quality assurance, and integrity verification.
