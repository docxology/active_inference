# Integration Testing - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Integration Testing module of the Active Inference Knowledge Environment. It outlines integration testing methodologies, implementation patterns, and best practices for validating component interactions and system-wide functionality.

## Integration Testing Module Overview

Integration testing validates the interactions between different components of the Active Inference Knowledge Environment, ensuring that data flows correctly between modules, APIs function properly, and the system as a whole behaves as expected when components are combined.

## Core Responsibilities

### Component Integration Testing
- **Module Interaction Validation**: Test interaction between major system modules
- **Data Flow Validation**: Verify data flows correctly through component boundaries
- **API Integration Testing**: Validate REST API functionality and data exchange
- **Service Integration Testing**: Test service-to-service communication
- **External System Integration**: Validate integration with external systems

### System Integration Testing
- **End-to-End Testing**: Complete system functionality validation
- **Cross-Module Testing**: Test interactions across multiple modules
- **Data Pipeline Testing**: End-to-end data processing pipeline validation
- **Workflow Testing**: Test complete user and system workflows
- **Performance Integration Testing**: System performance under integrated load

### Integration Quality Assurance
- **Regression Testing**: Ensure changes don't break existing integrations
- **Compatibility Testing**: Validate component compatibility and versions
- **Scalability Testing**: Test system scaling with component integration
- **Reliability Testing**: Test system reliability under integrated conditions
- **Security Integration Testing**: Validate security across component boundaries

## Development Workflows

### Integration Testing Development Process
1. **Integration Analysis**: Analyze component interactions and integration points
2. **Test Scenario Design**: Design realistic integration test scenarios
3. **Test Environment Setup**: Configure integrated test environments
4. **Test Implementation**: Implement integration test cases and fixtures
5. **Test Data Preparation**: Prepare realistic test data for integration scenarios
6. **Test Execution**: Run integration tests with proper monitoring
7. **Result Analysis**: Analyze integration test results and identify issues
8. **Issue Resolution**: Resolve integration issues and retest
9. **Documentation**: Document integration requirements and test procedures
10. **Maintenance**: Maintain integration tests as components evolve

### Component Integration Implementation
1. **Interface Analysis**: Analyze component interfaces and interaction points
2. **Integration Design**: Design integration test scenarios and data flow
3. **Mock Setup**: Set up appropriate mocks for external dependencies
4. **Test Implementation**: Implement integration test cases
5. **Data Flow Testing**: Test data flow through component boundaries
6. **Error Testing**: Test error handling across component boundaries
7. **Performance Testing**: Test integration performance characteristics

### System Integration Implementation
1. **System Analysis**: Analyze complete system architecture and workflows
2. **Workflow Design**: Design end-to-end test workflows and scenarios
3. **Environment Setup**: Set up complete system test environments
4. **Integration Testing**: Test complete system integration scenarios
5. **Performance Validation**: Validate system performance under load
6. **Scalability Testing**: Test system scaling with integration complexity

## Quality Standards

### Integration Testing Standards
- **Realistic Scenarios**: Test with realistic data volumes and usage patterns
- **Component Isolation**: Test component interactions without external dependencies
- **Data Integrity**: Verify data integrity through integration points
- **Error Propagation**: Test error handling across component boundaries
- **Performance Validation**: Validate performance under integrated load

### Test Quality Standards
- **Test Reliability**: <1% flaky test rate in integration tests
- **Test Coverage**: >80% integration point coverage
- **Test Performance**: Integration tests complete within acceptable time limits
- **Test Maintainability**: Tests are easy to understand and maintain
- **Test Documentation**: All integration tests are properly documented

### System Quality Standards
- **System Stability**: System maintains stability under integrated testing
- **Data Consistency**: Data remains consistent across component boundaries
- **Error Recovery**: System recovers gracefully from integration failures
- **Performance**: System meets performance requirements under integrated load
- **Scalability**: System scales appropriately with integration complexity

## Implementation Patterns

### Integration Test Framework
```python
import pytest
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock
import logging

class BaseIntegrationTest:
    """Base class for integration testing"""

    def __init__(self):
        """Initialize integration test"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_components = {}
        self.test_data = {}
        self.mocks = {}

    def setup_integration_environment(self) -> None:
        """Set up integrated test environment"""
        # Initialize all components under test
        self.test_components = self._initialize_test_components()

        # Set up test data
        self.test_data = self._prepare_test_data()

        # Set up mocks for external dependencies
        self.mocks = self._setup_mocks()

    def teardown_integration_environment(self) -> None:
        """Clean up integration test environment"""
        # Clean up test components
        for component_name, component in self.test_components.items():
            try:
                if hasattr(component, 'cleanup'):
                    component.cleanup()
                elif hasattr(component, 'close'):
                    component.close()
            except Exception as e:
                self.logger.warning(f"Error cleaning up component {component_name}: {str(e)}")

        # Clean up test data
        self.test_data.clear()

        # Clean up mocks
        self.mocks.clear()

    @abstractmethod
    def _initialize_test_components(self) -> Dict[str, Any]:
        """Initialize components for integration testing"""
        pass

    @abstractmethod
    def _prepare_test_data(self) -> Dict[str, Any]:
        """Prepare test data for integration scenarios"""
        pass

    @abstractmethod
    def _setup_mocks(self) -> Dict[str, Any]:
        """Set up mocks for external dependencies"""
        pass

    def run_integration_test(self, test_scenario: str) -> Dict[str, Any]:
        """Run integration test scenario"""
        try:
            # Execute test scenario
            result = self._execute_test_scenario(test_scenario)

            # Validate integration result
            validation = self._validate_integration_result(result)

            return {
                'scenario': test_scenario,
                'success': validation['passed'],
                'result': result,
                'validation': validation,
                'execution_time': result.get('execution_time', 0)
            }

        except Exception as e:
            self.logger.error(f"Integration test failed: {str(e)}")
            return {
                'scenario': test_scenario,
                'success': False,
                'error': str(e),
                'result': None,
                'validation': {'passed': False, 'issues': [str(e)]}
            }

    @abstractmethod
    def _execute_test_scenario(self, scenario: str) -> Dict[str, Any]:
        """Execute specific test scenario"""
        pass

    @abstractmethod
    def _validate_integration_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate integration test result"""
        pass

class KnowledgeResearchIntegrationTest(BaseIntegrationTest):
    """Integration tests for knowledge and research components"""

    def _initialize_test_components(self) -> Dict[str, Any]:
        """Initialize knowledge and research components"""
        from active_inference.knowledge import KnowledgeRepository
        from active_inference.research import ExperimentManager

        return {
            'knowledge_repo': KnowledgeRepository(test_config=True),
            'experiment_manager': ExperimentManager(test_config=True)
        }

    def _prepare_test_data(self) -> Dict[str, Any]:
        """Prepare test data for knowledge-research integration"""
        return {
            'knowledge_content': {
                'title': 'Test Active Inference Concept',
                'content': 'Test content for integration testing',
                'tags': ['test', 'integration', 'active_inference']
            },
            'experiment_config': {
                'name': 'integration_test_experiment',
                'type': 'active_inference_study',
                'parameters': {'iterations': 100}
            }
        }

    def _setup_mocks(self) -> Dict[str, Any]:
        """Set up mocks for external dependencies"""
        # Mock external APIs or services if needed
        return {
            'external_api': Mock(),
            'database': Mock()
        }

    def _execute_test_scenario(self, scenario: str) -> Dict[str, Any]:
        """Execute knowledge-research integration scenario"""
        start_time = asyncio.get_event_loop().time()

        if scenario == 'knowledge_to_experiment':
            # Test: Create knowledge content and use in experiment
            components = self.test_components
            test_data = self.test_data

            # Create knowledge content
            knowledge_id = components['knowledge_repo'].create_node(
                test_data['knowledge_content']
            )

            # Create experiment referencing knowledge
            experiment_config = test_data['experiment_config'].copy()
            experiment_config['knowledge_reference'] = knowledge_id

            experiment_id = components['experiment_manager'].create_experiment(
                experiment_config
            )

            # Run experiment
            experiment_results = components['experiment_manager'].run_experiment(
                experiment_id
            )

            return {
                'knowledge_id': knowledge_id,
                'experiment_id': experiment_id,
                'experiment_results': experiment_results,
                'execution_time': asyncio.get_event_loop().time() - start_time
            }

        elif scenario == 'experiment_to_knowledge':
            # Test: Generate knowledge from experiment results
            components = self.test_components

            # Run experiment
            experiment_results = components['experiment_manager'].run_experiment(
                'test_experiment_id'
            )

            # Generate knowledge content from results
            knowledge_content = {
                'title': 'Experiment Results Summary',
                'content': f'Experiment completed with results: {experiment_results}',
                'tags': ['experiment', 'results', 'generated']
            }

            knowledge_id = components['knowledge_repo'].create_node(knowledge_content)

            return {
                'knowledge_id': knowledge_id,
                'experiment_results': experiment_results,
                'execution_time': asyncio.get_event_loop().time() - start_time
            }

        else:
            raise ValueError(f"Unknown test scenario: {scenario}")

    def _validate_integration_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate knowledge-research integration result"""
        issues = []

        # Validate knowledge creation
        if 'knowledge_id' not in result:
            issues.append("Knowledge ID not found in result")

        # Validate experiment execution
        if 'experiment_results' not in result:
            issues.append("Experiment results not found in result")

        # Validate data integrity
        if result.get('experiment_results', {}).get('status') != 'completed':
            issues.append("Experiment did not complete successfully")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'validation_details': {
                'knowledge_created': 'knowledge_id' in result,
                'experiment_completed': result.get('experiment_results', {}).get('status') == 'completed'
            }
        }

class PlatformKnowledgeIntegrationTest(BaseIntegrationTest):
    """Integration tests for platform and knowledge components"""

    def _initialize_test_components(self) -> Dict[str, Any]:
        """Initialize platform and knowledge components"""
        from active_inference.platform import PlatformServices
        from active_inference.knowledge import KnowledgeRepository

        return {
            'platform': PlatformServices(test_config=True),
            'knowledge': KnowledgeRepository(test_config=True)
        }

    def _prepare_test_data(self) -> Dict[str, Any]:
        """Prepare test data for platform-knowledge integration"""
        return {
            'user_data': {
                'user_id': 'test_user_001',
                'preferences': {'learning_style': 'visual', 'difficulty': 'intermediate'}
            },
            'knowledge_query': {
                'query': 'active inference fundamentals',
                'filters': {'difficulty': 'beginner'},
                'limit': 10
            },
            'platform_request': {
                'service': 'knowledge_search',
                'parameters': {'personalized': True}
            }
        }

    def _setup_mocks(self) -> Dict[str, Any]:
        """Set up mocks for external dependencies"""
        return {
            'authentication_service': Mock(),
            'user_service': Mock(),
            'notification_service': Mock()
        }

    def _execute_test_scenario(self, scenario: str) -> Dict[str, Any]:
        """Execute platform-knowledge integration scenario"""
        start_time = asyncio.get_event_loop().time()

        if scenario == 'personalized_knowledge_search':
            # Test: Personalized knowledge search through platform
            components = self.test_components
            test_data = self.test_data

            # Authenticate user
            user_session = components['platform'].authenticate_user(
                test_data['user_data']['user_id']
            )

            # Search knowledge with personalization
            search_results = components['platform'].search_knowledge(
                test_data['knowledge_query'],
                user_session
            )

            # Personalize results based on user preferences
            personalized_results = components['platform'].personalize_results(
                search_results,
                user_session
            )

            return {
                'user_session': user_session,
                'search_results': search_results,
                'personalized_results': personalized_results,
                'execution_time': asyncio.get_event_loop().time() - start_time
            }

        elif scenario == 'knowledge_collaboration':
            # Test: Collaborative knowledge creation through platform
            components = self.test_components

            # Create collaborative knowledge session
            session = components['platform'].create_knowledge_session(
                participants=['user1', 'user2', 'user3'],
                topic='active_inference_collaboration'
            )

            # Contribute knowledge content
            contributions = components['knowledge'].create_collaborative_content(
                session,
                {'content': 'Collaborative Active Inference content'}
            )

            return {
                'session': session,
                'contributions': contributions,
                'execution_time': asyncio.get_event_loop().time() - start_time
            }

        else:
            raise ValueError(f"Unknown test scenario: {scenario}")

    def _validate_integration_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate platform-knowledge integration result"""
        issues = []

        # Validate platform integration
        if 'user_session' not in result and 'session' not in result:
            issues.append("Platform integration not found in result")

        # Validate knowledge integration
        if 'search_results' not in result and 'contributions' not in result:
            issues.append("Knowledge integration not found in result")

        # Validate personalization
        if 'personalized_results' in result:
            if not result['personalized_results'].get('personalized', False):
                issues.append("Results were not properly personalized")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'validation_details': {
                'platform_integrated': 'user_session' in result or 'session' in result,
                'knowledge_integrated': 'search_results' in result or 'contributions' in result,
                'personalized': result.get('personalized_results', {}).get('personalized', False)
            }
        }

class DataFlowIntegrationTest(BaseIntegrationTest):
    """Integration tests for data flow through system"""

    def _initialize_test_components(self) -> Dict[str, Any]:
        """Initialize data flow components"""
        from active_inference.research.data_management import DataManager
        from active_inference.knowledge import KnowledgeRepository
        from active_inference.research.analysis import AnalysisManager

        return {
            'data_manager': DataManager(test_config=True),
            'knowledge_repo': KnowledgeRepository(test_config=True),
            'analysis_manager': AnalysisManager(test_config=True)
        }

    def _prepare_test_data(self) -> Dict[str, Any]:
        """Prepare test data for data flow integration"""
        return {
            'raw_data': {
                'experiment': 'active_inference_test',
                'trials': 100,
                'subjects': 10,
                'conditions': ['control', 'experimental']
            },
            'processing_config': {
                'cleaning': {'remove_outliers': True, 'impute_missing': True},
                'transformation': {'normalize': True, 'encode_categorical': True},
                'analysis': {'statistical_tests': True, 'visualization': True}
            }
        }

    def _setup_mocks(self) -> Dict[str, Any]:
        """Set up mocks for external dependencies"""
        return {
            'external_database': Mock(),
            'file_storage': Mock(),
            'api_services': Mock()
        }

    def _execute_test_scenario(self, scenario: str) -> Dict[str, Any]:
        """Execute data flow integration scenario"""
        start_time = asyncio.get_event_loop().time()

        if scenario == 'complete_data_pipeline':
            # Test: Complete data processing pipeline
            components = self.test_components
            test_data = self.test_data

            # Store raw data
            data_id = components['data_manager'].store_data(
                test_data['raw_data'],
                metadata={'source': 'test_generation'}
            )

            # Process data through pipeline
            processed_data = components['data_manager'].process_data(
                data_id,
                test_data['processing_config']
            )

            # Analyze processed data
            analysis_results = components['analysis_manager'].analyze_data(
                processed_data,
                test_data['processing_config']['analysis']
            )

            # Generate knowledge from analysis
            knowledge_content = {
                'title': 'Analysis Results Summary',
                'content': f'Analysis completed: {analysis_results}',
                'tags': ['analysis', 'results', 'generated']
            }

            knowledge_id = components['knowledge_repo'].create_node(knowledge_content)

            return {
                'data_id': data_id,
                'processed_data': processed_data,
                'analysis_results': analysis_results,
                'knowledge_id': knowledge_id,
                'execution_time': asyncio.get_event_loop().time() - start_time
            }

        elif scenario == 'knowledge_driven_analysis':
            # Test: Knowledge-driven data analysis
            components = self.test_components

            # Retrieve relevant knowledge
            knowledge_results = components['knowledge_repo'].search_nodes(
                query='statistical analysis methods',
                limit=5
            )

            # Apply knowledge to data analysis
            analysis_config = {
                'methods': [node['content'] for node in knowledge_results],
                'parameters': {'significance_level': 0.05}
            }

            analysis_results = components['analysis_manager'].analyze_data(
                self.test_data['raw_data'],
                analysis_config
            )

            return {
                'knowledge_results': knowledge_results,
                'analysis_results': analysis_results,
                'execution_time': asyncio.get_event_loop().time() - start_time
            }

        else:
            raise ValueError(f"Unknown test scenario: {scenario}")

    def _validate_integration_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data flow integration result"""
        issues = []

        # Validate data processing
        if 'processed_data' not in result:
            issues.append("Data processing result not found")

        # Validate analysis
        if 'analysis_results' not in result:
            issues.append("Analysis results not found")

        # Validate knowledge generation
        if 'knowledge_id' not in result:
            issues.append("Knowledge generation result not found")

        # Validate data integrity
        if result.get('analysis_results', {}).get('data_integrity', False) != True:
            issues.append("Data integrity not maintained through pipeline")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'validation_details': {
                'data_processed': 'processed_data' in result,
                'data_analyzed': 'analysis_results' in result,
                'knowledge_generated': 'knowledge_id' in result,
                'integrity_maintained': result.get('analysis_results', {}).get('data_integrity', False)
            }
        }
```

### Integration Test Organization
```python
class IntegrationTestSuite:
    """Suite for organizing and executing integration tests"""

    def __init__(self):
        """Initialize integration test suite"""
        self.test_categories = {
            'component_integration': [],
            'data_flow_integration': [],
            'api_integration': [],
            'platform_integration': [],
            'system_integration': []
        }
        self.test_results = {}

    def register_integration_test(self, test_class: type, category: str) -> None:
        """Register integration test class"""
        if category in self.test_categories:
            self.test_categories[category].append(test_class)
        else:
            self.logger.warning(f"Unknown test category: {category}")

    def run_integration_suite(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Run integration test suite"""
        results = {}

        categories_to_run = [category] if category else self.test_categories.keys()

        for test_category in categories_to_run:
            if test_category not in self.test_categories:
                continue

            category_results = []
            test_classes = self.test_categories[test_category]

            for test_class in test_classes:
                try:
                    # Create and run test instance
                    test_instance = test_class()
                    test_instance.setup_integration_environment()

                    # Run all test scenarios
                    for scenario in test_instance.get_test_scenarios():
                        result = test_instance.run_integration_test(scenario)
                        category_results.append(result)

                    test_instance.teardown_integration_environment()

                except Exception as e:
                    self.logger.error(f"Integration test {test_class.__name__} failed: {str(e)}")
                    category_results.append({
                        'test_class': test_class.__name__,
                        'success': False,
                        'error': str(e)
                    })

            results[test_category] = {
                'total_tests': len(category_results),
                'passed_tests': sum(1 for r in category_results if r.get('success', False)),
                'failed_tests': sum(1 for r in category_results if not r.get('success', False)),
                'results': category_results
            }

        return results

    def generate_integration_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive integration test report"""
        report = ["# Integration Test Report", ""]

        total_tests = sum(category['total_tests'] for category in results.values())
        total_passed = sum(category['passed_tests'] for category in results.values())
        total_failed = sum(category['failed_tests'] for category in results.values())

        report.append(f"## Summary")
        report.append(f"- Total Integration Tests: {total_tests}")
        report.append(f"- Passed: {total_passed}")
        report.append(f"- Failed: {total_failed}")
        report.append(f"- Success Rate: {total_passed/total_tests*100:.1f}%" if total_tests > 0 else "- Success Rate: N/A")
        report.append("")

        for category, category_result in results.items():
            report.append(f"## {category.replace('_', ' ').title()}")
            report.append(f"- Tests: {category_result['total_tests']}")
            report.append(f"- Passed: {category_result['passed_tests']}")
            report.append(f"- Failed: {category_result['failed_tests']}")

            if category_result['failed_tests'] > 0:
                report.append("### Failed Tests")
                for result in category_result['results']:
                    if not result.get('success', False):
                        report.append(f"- {result.get('scenario', result.get('test_class', 'Unknown'))}: {result.get('error', 'Unknown error')}")
                report.append("")

        return "\n".join(report)
```

## Testing Guidelines

### Integration Test Guidelines
- **Realistic Scenarios**: Test with realistic data volumes and usage patterns
- **Component Isolation**: Test component interactions without external dependencies
- **Data Integrity**: Verify data integrity through integration points
- **Error Propagation**: Test error handling across component boundaries
- **Performance Validation**: Validate performance under integrated load

### Test Data Management
- **Realistic Test Data**: Use realistic data for integration testing
- **Data Cleanup**: Proper cleanup of test data between test runs
- **Data Isolation**: Isolate test data from production data
- **Data Versioning**: Manage test data versions and updates

### Test Environment Management
- **Environment Setup**: Proper setup of integrated test environments
- **Component Initialization**: Correct initialization of all test components
- **Dependency Management**: Proper management of component dependencies
- **Resource Cleanup**: Complete cleanup of test resources

## Performance Considerations

### Integration Performance Testing
- **Load Testing**: Test system under various integrated load conditions
- **Stress Testing**: Test system beyond normal integrated operating parameters
- **Concurrency Testing**: Test concurrent access and data flow
- **Scalability Testing**: Test system scaling with component integration
- **Resource Monitoring**: Monitor resource usage during integration tests

### Performance Benchmarks
- **Integration Latency**: <100ms for typical integration operations
- **Data Flow Throughput**: Maintain target throughput through integration points
- **Component Communication**: Efficient inter-component communication
- **Memory Usage**: Efficient memory usage with multiple components
- **Error Recovery**: Fast error recovery in integrated scenarios

## Maintenance and Evolution

### Integration Test Maintenance
- **Regular Updates**: Keep integration tests current with component changes
- **Test Refactoring**: Refactor tests as component interfaces evolve
- **Environment Updates**: Update test environments as components change
- **Documentation**: Keep integration test documentation current
- **Performance Monitoring**: Monitor integration test performance

### Integration Test Evolution
- **New Integration Points**: Add tests for new component interactions
- **Test Framework Updates**: Update integration testing frameworks
- **Best Practice Adoption**: Adopt new integration testing best practices
- **Automation Improvements**: Improve integration test automation

## Common Challenges and Solutions

### Challenge: Component Interface Changes
**Solution**: Implement comprehensive interface testing, use abstraction layers, and maintain backward compatibility testing.

### Challenge: Test Data Complexity
**Solution**: Use realistic but manageable test data, implement test data factories, and ensure proper test data cleanup.

### Challenge: Test Environment Setup
**Solution**: Automate test environment setup, use containerization, and implement environment validation.

### Challenge: Flaky Integration Tests
**Solution**: Identify sources of non-determinism, implement proper test isolation, and use retry mechanisms for inherently flaky operations.

## Getting Started as an Agent

### Development Setup
1. **Study Integration Architecture**: Understand component integration patterns and interfaces
2. **Learn Integration Testing**: Study integration testing methodologies and approaches
3. **Practice Test Writing**: Practice writing integration test cases
4. **Understand Data Flow**: Learn system data flow and integration points

### Contribution Process
1. **Identify Integration Points**: Find component interaction points needing testing
2. **Study Component Interfaces**: Understand components requiring integration testing
3. **Design Test Scenarios**: Create realistic integration test scenarios
4. **Implement Integration Tests**: Write comprehensive integration test cases
5. **Validate Test Quality**: Ensure integration tests are reliable and maintainable
6. **Document Tests**: Provide comprehensive integration test documentation
7. **Code Review**: Submit integration tests for peer review and validation

### Learning Resources
- **Integration Testing**: Study integration testing principles and practices
- **Component Architecture**: Learn component design and interface patterns
- **Data Flow Analysis**: Study system data flow and integration points
- **Test Design**: Learn integration test design and implementation
- **Quality Assurance**: Learn integration quality assurance principles

## Related Documentation

- **[Integration README](./README.md)**: Integration testing module overview
- **[Testing AGENTS.md](../AGENTS.md)**: Testing framework development guidelines
- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution processes

---

*"Active Inference for, with, by Generative AI"* - Ensuring system reliability through comprehensive integration testing and component interaction validation.
