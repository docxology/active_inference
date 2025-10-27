# Comprehensive Integration Testing and Validation Prompt

**"Active Inference for, with, by Generative AI"**

## 🎯 Mission: Develop Comprehensive Integration Testing Frameworks

You are tasked with developing comprehensive integration testing frameworks for the Active Inference Knowledge Environment that validate system-wide interactions, data flows, and component integrations to ensure reliable, cohesive platform operation.

## 📋 Integration Testing Requirements

### Core Integration Standards (MANDATORY)
1. **End-to-End Testing**: Complete user journey validation from request to response
2. **Data Flow Validation**: Ensure data integrity across component boundaries
3. **API Contract Testing**: Validate all API contracts and interfaces
4. **Cross-Component Testing**: Test interactions between all platform components
5. **Performance Under Load**: Integration testing with realistic load patterns
6. **Failure Scenario Testing**: Test system behavior under failure conditions

### Integration Testing Architecture
```
tests/integration/
├── api_integration/            # API contract and interface testing
│   ├── test_knowledge_api.py   # Knowledge base API integration
│   ├── test_search_api.py      # Search service API integration
│   ├── test_visualization_api.py # Visualization API integration
│   └── test_collaboration_api.py # Collaboration API integration
├── data_flow/                  # Data flow and pipeline testing
│   ├── test_knowledge_pipeline.py   # Knowledge processing pipeline
│   ├── test_search_indexing.py      # Search indexing pipeline
│   ├── test_user_session_flow.py    # User session data flow
│   └── test_content_synchronization.py # Content sync across services
├── component_interaction/      # Cross-component interaction testing
│   ├── test_knowledge_search_integration.py
│   ├── test_visualization_knowledge_integration.py
│   ├── test_collaboration_platform_integration.py
│   └── test_research_platform_integration.py
├── performance_integration/    # Performance under integration load
│   ├── test_concurrent_user_load.py
│   ├── test_data_processing_load.py
│   ├── test_search_query_load.py
│   └── test_visualization_rendering_load.py
├── failure_scenarios/          # Failure and recovery testing
│   ├── test_service_failure_recovery.py
│   ├── test_network_partition_handling.py
│   ├── test_database_connection_failure.py
│   └── test_external_service_degradation.py
└── end_to_end/                 # Complete user journey testing
    ├── test_user_registration_flow.py
    ├── test_knowledge_exploration_flow.py
    ├── test_collaborative_editing_flow.py
    └── test_research_workflow.py
```

## 🏗️ Integration Testing Framework

### Phase 1: API Integration Testing

#### 1.1 API Contract Testing Framework
```python
import pytest
import requests
from typing import Dict, List, Any, Optional, Callable
import json
import logging
from dataclasses import dataclass

@dataclass
class APIContract:
    """API contract specification"""
    endpoint: str
    method: str
    request_schema: Dict[str, Any]
    response_schema: Dict[str, Any]
    expected_status_codes: List[int]
    authentication_required: bool = False
    rate_limit: Optional[int] = None

@dataclass
class APIResponse:
    """API response wrapper"""
    status_code: int
    headers: Dict[str, str]
    data: Any
    response_time: float

class APIIntegrationTester:
    """Comprehensive API integration testing framework"""

    def __init__(self, base_url: str, config: Dict[str, Any]):
        """Initialize API integration tester"""
        self.base_url = base_url.rstrip('/')
        self.config = config
        self.logger = logging.getLogger('APIIntegrationTester')
        self.session = requests.Session()

        # Setup authentication if configured
        if 'auth_token' in config:
            self.session.headers.update({
                'Authorization': f'Bearer {config["auth_token"]}'
            })

        # Load API contracts
        self.api_contracts = self.load_api_contracts()

    def load_api_contracts(self) -> Dict[str, APIContract]:
        """Load API contract specifications"""
        # In practice, this would load from contract files
        return {
            'get_knowledge_node': APIContract(
                endpoint='/api/knowledge/{node_id}',
                method='GET',
                request_schema={},
                response_schema={
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'string'},
                        'title': {'type': 'string'},
                        'content_type': {'type': 'string'},
                        'difficulty': {'type': 'string'},
                        'content': {'type': 'object'}
                    },
                    'required': ['id', 'title', 'content_type']
                },
                expected_status_codes=[200, 404],
                authentication_required=False
            ),
            'search_knowledge': APIContract(
                endpoint='/api/search',
                method='POST',
                request_schema={
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string'},
                        'filters': {'type': 'object'}
                    },
                    'required': ['query']
                },
                response_schema={
                    'type': 'object',
                    'properties': {
                        'results': {'type': 'array'},
                        'total': {'type': 'integer'},
                        'took': {'type': 'number'}
                    }
                },
                expected_status_codes=[200],
                authentication_required=False
            ),
            'create_visualization': APIContract(
                endpoint='/api/visualization',
                method='POST',
                request_schema={
                    'type': 'object',
                    'properties': {
                        'type': {'type': 'string'},
                        'data': {'type': 'object'},
                        'config': {'type': 'object'}
                    },
                    'required': ['type', 'data']
                },
                response_schema={
                    'type': 'object',
                    'properties': {
                        'visualization_id': {'type': 'string'},
                        'url': {'type': 'string'}
                    }
                },
                expected_status_codes=[201],
                authentication_required=True
            )
        }

    def test_api_contract(self, contract_name: str) -> Dict[str, Any]:
        """Test a specific API contract"""
        if contract_name not in self.api_contracts:
            raise ValueError(f"Unknown API contract: {contract_name}")

        contract = self.api_contracts[contract_name]
        test_results = {
            'contract_name': contract_name,
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }

        # Test valid request
        valid_result = self.test_valid_request(contract)
        test_results['details'].append(valid_result)
        if valid_result['passed']:
            test_results['tests_passed'] += 1
        else:
            test_results['tests_failed'] += 1

        # Test invalid requests
        invalid_results = self.test_invalid_requests(contract)
        for result in invalid_results:
            test_results['details'].append(result)
            if result['passed']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1

        # Test rate limiting if specified
        if contract.rate_limit:
            rate_limit_result = self.test_rate_limiting(contract)
            test_results['details'].append(rate_limit_result)
            if rate_limit_result['passed']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1

        return test_results

    def test_valid_request(self, contract: APIContract) -> Dict[str, Any]:
        """Test valid request against contract"""
        try:
            # Generate valid test data
            test_data = self.generate_valid_test_data(contract)

            # Make request
            response = self.make_api_request(contract, test_data)

            # Validate response
            validation_result = self.validate_response(response, contract)

            return {
                'test_type': 'valid_request',
                'passed': validation_result['valid'],
                'response_time': response.response_time,
                'status_code': response.status_code,
                'validation_errors': validation_result.get('errors', [])
            }

        except Exception as e:
            return {
                'test_type': 'valid_request',
                'passed': False,
                'error': str(e)
            }

    def test_invalid_requests(self, contract: APIContract) -> List[Dict[str, Any]]:
        """Test invalid requests to ensure proper error handling"""
        invalid_tests = []

        # Test missing required fields
        if contract.request_schema.get('required'):
            missing_field_test = self.test_missing_required_field(contract)
            invalid_tests.append(missing_field_test)

        # Test invalid data types
        invalid_type_test = self.test_invalid_data_types(contract)
        invalid_tests.append(invalid_type_test)

        # Test malformed JSON
        if contract.method in ['POST', 'PUT', 'PATCH']:
            malformed_json_test = self.test_malformed_json(contract)
            invalid_tests.append(malformed_json_test)

        return invalid_tests

    def test_missing_required_field(self, contract: APIContract) -> Dict[str, Any]:
        """Test request with missing required field"""
        try:
            # Generate test data with missing required field
            required_fields = contract.request_schema.get('required', [])
            if not required_fields:
                return {'test_type': 'missing_required_field', 'passed': True, 'skipped': True}

            test_data = self.generate_valid_test_data(contract)
            missing_field = required_fields[0]
            del test_data[missing_field]

            response = self.make_api_request(contract, test_data)

            # Should return 400 Bad Request
            return {
                'test_type': 'missing_required_field',
                'passed': response.status_code == 400,
                'status_code': response.status_code
            }

        except Exception as e:
            return {
                'test_type': 'missing_required_field',
                'passed': False,
                'error': str(e)
            }

    def test_invalid_data_types(self, contract: APIContract) -> Dict[str, Any]:
        """Test request with invalid data types"""
        try:
            test_data = self.generate_valid_test_data(contract)

            # Change a field to wrong type (if possible)
            if test_data:
                first_key = list(test_data.keys())[0]
                original_value = test_data[first_key]

                # Try to make it the wrong type
                if isinstance(original_value, str):
                    test_data[first_key] = 123  # String to int
                elif isinstance(original_value, int):
                    test_data[first_key] = "not_a_number"  # Int to string
                elif isinstance(original_value, list):
                    test_data[first_key] = "not_a_list"  # List to string

                response = self.make_api_request(contract, test_data)

                # Should return 400 Bad Request
                return {
                    'test_type': 'invalid_data_types',
                    'passed': response.status_code == 400,
                    'status_code': response.status_code
                }

        except Exception as e:
            return {
                'test_type': 'invalid_data_types',
                'passed': False,
                'error': str(e)
            }

    def test_malformed_json(self, contract: APIContract) -> Dict[str, Any]:
        """Test request with malformed JSON"""
        try:
            # Send malformed JSON
            url = self.build_url(contract.endpoint)
            response = self.session.post(
                url,
                data="{invalid json",
                headers={'Content-Type': 'application/json'}
            )

            return {
                'test_type': 'malformed_json',
                'passed': response.status_code == 400,
                'status_code': response.status_code
            }

        except Exception as e:
            return {
                'test_type': 'malformed_json',
                'passed': False,
                'error': str(e)
            }

    def test_rate_limiting(self, contract: APIContract) -> Dict[str, Any]:
        """Test rate limiting behavior"""
        try:
            rate_limit = contract.rate_limit
            test_data = self.generate_valid_test_data(contract)

            # Make requests up to rate limit
            responses = []
            for i in range(rate_limit + 5):  # Go over limit
                response = self.make_api_request(contract, test_data)
                responses.append(response.status_code)

                if response.status_code == 429:  # Too Many Requests
                    break

            # Check if rate limiting kicked in
            rate_limited = any(code == 429 for code in responses)

            return {
                'test_type': 'rate_limiting',
                'passed': rate_limited,
                'requests_made': len(responses),
                'rate_limit_hit': rate_limited
            }

        except Exception as e:
            return {
                'test_type': 'rate_limiting',
                'passed': False,
                'error': str(e)
            }

    def generate_valid_test_data(self, contract: APIContract) -> Dict[str, Any]:
        """Generate valid test data for contract"""
        # Simple test data generation based on schema
        test_data = {}

        if contract.endpoint == '/api/knowledge/{node_id}':
            # For knowledge API, use a test node ID
            pass  # GET request, no body needed
        elif contract.endpoint == '/api/search':
            test_data = {
                'query': 'test search query',
                'filters': {'content_type': 'foundation'}
            }
        elif contract.endpoint == '/api/visualization':
            test_data = {
                'type': 'concept_map',
                'data': {'nodes': [], 'edges': []},
                'config': {'interactive': True}
            }

        return test_data

    def make_api_request(self, contract: APIContract, data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make API request and return response wrapper"""
        import time

        url = self.build_url(contract.endpoint)

        start_time = time.time()

        try:
            if contract.method == 'GET':
                response = self.session.get(url)
            elif contract.method == 'POST':
                response = self.session.post(url, json=data)
            elif contract.method == 'PUT':
                response = self.session.put(url, json=data)
            elif contract.method == 'DELETE':
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {contract.method}")

            response_time = time.time() - start_time

            return APIResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                data=response.json() if response.content else None,
                response_time=response_time
            )

        except requests.RequestException as e:
            response_time = time.time() - start_time
            return APIResponse(
                status_code=0,  # Network error
                headers={},
                data=None,
                response_time=response_time
            )

    def build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint template"""
        # Replace path parameters with test values
        if '{node_id}' in endpoint:
            endpoint = endpoint.replace('{node_id}', 'test_node_123')

        return f"{self.base_url}{endpoint}"

    def validate_response(self, response: APIResponse, contract: APIContract) -> Dict[str, Any]:
        """Validate response against contract"""
        errors = []

        # Check status code
        if response.status_code not in contract.expected_status_codes:
            errors.append(f"Unexpected status code: {response.status_code}, expected: {contract.expected_status_codes}")

        # Validate response schema (basic validation)
        if response.data and contract.response_schema:
            schema_errors = self.validate_against_schema(response.data, contract.response_schema)
            errors.extend(schema_errors)

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def validate_against_schema(self, data: Any, schema: Dict[str, Any]) -> List[str]:
        """Basic schema validation"""
        errors = []

        if not isinstance(data, dict):
            return errors  # Skip complex validation for now

        # Check required properties
        required = schema.get('properties', {}).keys()
        for prop in required:
            if prop not in data:
                errors.append(f"Missing required property: {prop}")

        # Basic type checking
        properties = schema.get('properties', {})
        for prop, prop_schema in properties.items():
            if prop in data:
                expected_type = prop_schema.get('type')
                actual_value = data[prop]

                if expected_type == 'string' and not isinstance(actual_value, str):
                    errors.append(f"Property {prop} should be string, got {type(actual_value)}")
                elif expected_type == 'integer' and not isinstance(actual_value, int):
                    errors.append(f"Property {prop} should be integer, got {type(actual_value)}")
                elif expected_type == 'array' and not isinstance(actual_value, list):
                    errors.append(f"Property {prop} should be array, got {type(actual_value)}")

        return errors

    def run_full_api_test_suite(self) -> Dict[str, Any]:
        """Run complete API integration test suite"""
        results = {
            'total_contracts': len(self.api_contracts),
            'passed_contracts': 0,
            'failed_contracts': 0,
            'contract_results': {}
        }

        for contract_name in self.api_contracts.keys():
            contract_result = self.test_api_contract(contract_name)
            results['contract_results'][contract_name] = contract_result

            if contract_result['tests_failed'] == 0:
                results['passed_contracts'] += 1
            else:
                results['failed_contracts'] += 1

        results['overall_success'] = results['failed_contracts'] == 0
        return results
```

#### 1.2 API Integration Test Examples
```python
import pytest
from tests.integration.api_integration.api_tester import APIIntegrationTester

class TestKnowledgeAPIIntegration:
    """Integration tests for Knowledge API"""

    @pytest.fixture
    def api_tester(self):
        """Create API integration tester"""
        config = {
            'base_url': 'http://localhost:8000',
            'auth_token': 'test_token_123'
        }
        return APIIntegrationTester(config['base_url'], config)

    def test_knowledge_api_contract_compliance(self, api_tester):
        """Test that Knowledge API complies with contracts"""
        results = api_tester.test_api_contract('get_knowledge_node')

        assert results['tests_passed'] > 0, "Some tests should pass"
        assert results['tests_failed'] == 0, f"Contract tests failed: {results['details']}"

    def test_search_api_integration(self, api_tester):
        """Test search API integration"""
        results = api_tester.test_api_contract('search_knowledge')

        assert results['tests_passed'] > 0
        assert results['tests_failed'] == 0

    def test_cross_api_data_consistency(self, api_tester):
        """Test data consistency across APIs"""
        # Create a knowledge node via API
        create_response = api_tester.make_api_request(
            api_tester.api_contracts['create_knowledge_node'],
            {
                'title': 'Integration Test Node',
                'content_type': 'foundation',
                'difficulty': 'beginner',
                'content': {'overview': 'Test content'}
            }
        )

        assert create_response.status_code == 201
        node_id = create_response.data['id']

        # Retrieve via search API
        search_response = api_tester.make_api_request(
            api_tester.api_contracts['search_knowledge'],
            {'query': 'Integration Test Node'}
        )

        assert search_response.status_code == 200
        assert len(search_response.data['results']) > 0

        # Verify data consistency
        found_node = search_response.data['results'][0]
        assert found_node['id'] == node_id
        assert found_node['title'] == 'Integration Test Node'

    def test_api_error_handling_integration(self, api_tester):
        """Test error handling across API endpoints"""
        # Test non-existent node
        response = api_tester.make_api_request(
            api_tester.api_contracts['get_knowledge_node']
        )

        # Should return 404 for non-existent node
        assert response.status_code == 404

        # Verify error response format
        assert 'error' in response.data
        assert response.data['error']['code'] == 'NODE_NOT_FOUND'

    def test_api_rate_limiting_integration(self, api_tester):
        """Test rate limiting across API endpoints"""
        # Make multiple requests quickly
        responses = []
        for i in range(15):  # More than typical rate limit
            response = api_tester.make_api_request(
                api_tester.api_contracts['get_knowledge_node']
            )
            responses.append(response.status_code)

        # Should eventually hit rate limit
        rate_limited = any(code == 429 for code in responses)
        assert rate_limited, "Rate limiting should activate under load"

class TestVisualizationAPIIntegration:
    """Integration tests for Visualization API"""

    @pytest.fixture
    def api_tester(self):
        """Create API integration tester with auth"""
        config = {
            'base_url': 'http://localhost:8000',
            'auth_token': 'test_token_123'
        }
        return APIIntegrationTester(config['base_url'], config)

    def test_visualization_creation_workflow(self, api_tester):
        """Test complete visualization creation workflow"""
        # Create visualization
        create_response = api_tester.make_api_request(
            api_tester.api_contracts['create_visualization'],
            {
                'type': 'concept_map',
                'data': {
                    'nodes': [
                        {'id': 'node1', 'title': 'Test Node 1'},
                        {'id': 'node2', 'title': 'Test Node 2'}
                    ],
                    'edges': [
                        {'source': 'node1', 'target': 'node2', 'type': 'related'}
                    ]
                },
                'config': {
                    'interactive': True,
                    'layout': 'force_directed'
                }
            }
        )

        assert create_response.status_code == 201
        viz_id = create_response.data['visualization_id']
        viz_url = create_response.data['url']

        # Verify visualization is accessible
        get_response = api_tester.make_api_request(
            api_tester.api_contracts['get_visualization'],
            None,  # GET request
            endpoint_override=f'/api/visualization/{viz_id}'
        )

        assert get_response.status_code == 200
        assert get_response.data['id'] == viz_id

    def test_visualization_knowledge_integration(self, api_tester):
        """Test visualization integration with knowledge base"""
        # First create knowledge node
        knowledge_response = api_tester.make_api_request(
            api_tester.api_contracts['create_knowledge_node'],
            {
                'title': 'Visualization Test Concept',
                'content_type': 'foundation',
                'difficulty': 'intermediate',
                'content': {'overview': 'Test for visualization integration'}
            }
        )

        assert knowledge_response.status_code == 201
        node_id = knowledge_response.data['id']

        # Create visualization based on knowledge
        viz_response = api_tester.make_api_request(
            api_tester.api_contracts['create_visualization_from_knowledge'],
            {
                'knowledge_node_id': node_id,
                'visualization_type': 'mind_map',
                'include_related': True
            }
        )

        assert viz_response.status_code == 201

        # Verify visualization contains knowledge data
        viz_data = viz_response.data
        assert node_id in [node['id'] for node in viz_data['data']['nodes']]
```

### Phase 2: Data Flow Integration Testing

#### 2.1 Data Pipeline Testing Framework
```python
import pytest
from typing import Dict, List, Any, Optional, Callable
import asyncio
import time
import logging

class DataFlow:
    """Represents a data flow through the system"""

    def __init__(self, flow_id: str, description: str, steps: List[Dict[str, Any]]):
        """Initialize data flow"""
        self.flow_id = flow_id
        self.description = description
        self.steps = steps
        self.current_step = 0
        self.flow_data = {}
        self.errors = []

    def get_current_step(self) -> Optional[Dict[str, Any]]:
        """Get current step in flow"""
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def advance_step(self, step_output: Any = None) -> bool:
        """Advance to next step"""
        if step_output is not None:
            self.flow_data[f"step_{self.current_step}_output"] = step_output

        self.current_step += 1
        return self.current_step < len(self.steps)

    def record_error(self, error: str, step_index: Optional[int] = None) -> None:
        """Record error in flow"""
        step_idx = step_index if step_index is not None else self.current_step
        self.errors.append({
            'step': step_idx,
            'error': error,
            'timestamp': time.time()
        })

class DataFlowTester:
    """Data flow integration testing framework"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data flow tester"""
        self.config = config
        self.logger = logging.getLogger('DataFlowTester')
        self.data_flows = self.define_data_flows()

    def define_data_flows(self) -> Dict[str, DataFlow]:
        """Define data flows for testing"""
        return {
            'user_knowledge_exploration': DataFlow(
                'user_knowledge_exploration',
                'Complete user journey from search to knowledge consumption',
                [
                    {
                        'name': 'user_search',
                        'component': 'search_service',
                        'action': 'search',
                        'input_data': {'query': 'active inference basics'},
                        'expected_output': {'results_count': lambda x: len(x.get('results', [])) > 0}
                    },
                    {
                        'name': 'knowledge_retrieval',
                        'component': 'knowledge_service',
                        'action': 'get_node',
                        'input_data': {'use_search_result': True},
                        'expected_output': {'node_data': lambda x: x.get('content') is not None}
                    },
                    {
                        'name': 'visualization_creation',
                        'component': 'visualization_service',
                        'action': 'create_concept_map',
                        'input_data': {'use_knowledge_node': True},
                        'expected_output': {'visualization_url': lambda x: x.get('url') is not None}
                    },
                    {
                        'name': 'user_interaction_tracking',
                        'component': 'analytics_service',
                        'action': 'track_interaction',
                        'input_data': {'interaction_type': 'knowledge_exploration'},
                        'expected_output': {'tracking_id': lambda x: x.get('id') is not None}
                    }
                ]
            ),
            'content_creation_workflow': DataFlow(
                'content_creation_workflow',
                'Content creation and publication workflow',
                [
                    {
                        'name': 'content_draft',
                        'component': 'content_service',
                        'action': 'create_draft',
                        'input_data': {
                            'title': 'Test Content',
                            'content_type': 'foundation',
                            'content': {'overview': 'Draft content'}
                        },
                        'expected_output': {'draft_id': lambda x: x.get('id') is not None}
                    },
                    {
                        'name': 'content_validation',
                        'component': 'validation_service',
                        'action': 'validate_content',
                        'input_data': {'use_draft_id': True},
                        'expected_output': {'validation_passed': lambda x: x.get('passed', False)}
                    },
                    {
                        'name': 'content_indexing',
                        'component': 'search_service',
                        'action': 'index_content',
                        'input_data': {'use_draft_id': True},
                        'expected_output': {'indexing_success': lambda x: x.get('success', False)}
                    },
                    {
                        'name': 'content_publication',
                        'component': 'content_service',
                        'action': 'publish_content',
                        'input_data': {'use_draft_id': True},
                        'expected_output': {'publication_success': lambda x: x.get('published', False)}
                    }
                ]
            )
        }

    async def test_data_flow(self, flow_name: str) -> Dict[str, Any]:
        """Test a complete data flow"""
        if flow_name not in self.data_flows:
            raise ValueError(f"Unknown data flow: {flow_name}")

        flow = self.data_flows[flow_name]
        test_results = {
            'flow_name': flow_name,
            'steps_completed': 0,
            'steps_failed': 0,
            'total_steps': len(flow.steps),
            'execution_time': 0,
            'errors': [],
            'step_results': []
        }

        start_time = time.time()

        try:
            while True:
                current_step = flow.get_current_step()
                if not current_step:
                    break  # Flow completed

                step_result = await self.execute_flow_step(flow, current_step)
                test_results['step_results'].append(step_result)

                if step_result['success']:
                    test_results['steps_completed'] += 1
                    flow.advance_step(step_result.get('output'))
                else:
                    test_results['steps_failed'] += 1
                    flow.record_error(step_result.get('error', 'Unknown error'))
                    break  # Stop on first failure

            test_results['flow_completed'] = test_results['steps_failed'] == 0
            test_results['execution_time'] = time.time() - start_time

        except Exception as e:
            test_results['flow_completed'] = False
            test_results['execution_time'] = time.time() - start_time
            test_results['errors'].append(str(e))
            self.logger.error(f"Data flow test failed: {e}")

        return test_results

    async def execute_flow_step(self, flow: DataFlow, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in the data flow"""
        step_name = step['name']
        component = step['component']
        action = step['action']

        self.logger.info(f"Executing step: {step_name} ({component}.{action})")

        try:
            # Prepare input data
            input_data = self.prepare_step_input(flow, step)

            # Execute step (mock implementation)
            output = await self.mock_component_call(component, action, input_data)

            # Validate output
            validation_result = self.validate_step_output(output, step.get('expected_output', {}))

            return {
                'step_name': step_name,
                'component': component,
                'action': action,
                'success': validation_result['valid'],
                'output': output,
                'validation_errors': validation_result.get('errors', []),
                'execution_time': time.time()
            }

        except Exception as e:
            self.logger.error(f"Step execution failed: {step_name} - {e}")
            return {
                'step_name': step_name,
                'component': component,
                'action': action,
                'success': False,
                'error': str(e),
                'execution_time': time.time()
            }

    def prepare_step_input(self, flow: DataFlow, step: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input data for step execution"""
        input_data = step.get('input_data', {}).copy()

        # Replace placeholders with flow data
        if input_data.get('use_search_result'):
            # Use output from previous search step
            search_output = flow.flow_data.get('step_0_output', {})
            if 'results' in search_output and search_output['results']:
                input_data['node_id'] = search_output['results'][0]['id']
            del input_data['use_search_result']

        if input_data.get('use_knowledge_node'):
            # Use output from knowledge retrieval step
            knowledge_output = flow.flow_data.get('step_1_output', {})
            input_data['knowledge_data'] = knowledge_output
            del input_data['use_knowledge_node']

        return input_data

    async def mock_component_call(self, component: str, action: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock component call for testing (replace with real calls in production)"""
        # Simulate network delay
        await asyncio.sleep(0.1)

        # Mock responses based on component and action
        if component == 'search_service' and action == 'search':
            return {
                'results': [
                    {'id': 'test_node_1', 'title': 'Test Result 1'},
                    {'id': 'test_node_2', 'title': 'Test Result 2'}
                ],
                'total': 2,
                'took': 0.05
            }
        elif component == 'knowledge_service' and action == 'get_node':
            return {
                'id': input_data.get('node_id', 'test_node'),
                'title': 'Test Knowledge Node',
                'content_type': 'foundation',
                'content': {'overview': 'Test content'}
            }
        elif component == 'visualization_service' and action == 'create_concept_map':
            return {
                'visualization_id': 'test_viz_123',
                'url': 'http://localhost:8000/viz/test_viz_123'
            }
        elif component == 'analytics_service' and action == 'track_interaction':
            return {
                'tracking_id': 'track_123',
                'recorded': True
            }

        # Default mock response
        return {'success': True, 'mock': True}

    def validate_step_output(self, output: Any, expected_output: Dict[str, Callable]) -> Dict[str, Any]:
        """Validate step output against expectations"""
        errors = []

        if not expected_output:
            return {'valid': True}

        for field_name, validator in expected_output.items():
            if field_name not in output:
                errors.append(f"Missing expected field: {field_name}")
                continue

            field_value = output[field_name]
            try:
                if not validator(field_value):
                    errors.append(f"Field {field_name} failed validation")
            except Exception as e:
                errors.append(f"Validation error for {field_name}: {e}")

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    async def test_data_flow_under_load(self, flow_name: str, concurrent_users: int) -> Dict[str, Any]:
        """Test data flow under concurrent load"""
        load_test_results = {
            'flow_name': flow_name,
            'concurrent_users': concurrent_users,
            'individual_results': [],
            'aggregate_metrics': {}
        }

        # Execute flow concurrently
        tasks = []
        for i in range(concurrent_users):
            task = asyncio.create_task(self.test_data_flow(flow_name))
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_runs = 0
        total_execution_time = 0
        all_errors = []

        for result in results:
            if isinstance(result, Exception):
                all_errors.append(str(result))
            else:
                load_test_results['individual_results'].append(result)
                if result.get('flow_completed', False):
                    successful_runs += 1
                    total_execution_time += result.get('execution_time', 0)

        # Calculate aggregate metrics
        success_rate = successful_runs / concurrent_users if concurrent_users > 0 else 0
        avg_execution_time = total_execution_time / successful_runs if successful_runs > 0 else 0

        load_test_results['aggregate_metrics'] = {
            'success_rate': success_rate,
            'successful_runs': successful_runs,
            'failed_runs': len(all_errors),
            'average_execution_time': avg_execution_time,
            'errors': all_errors
        }

        return load_test_results

class TestDataFlowIntegration:
    """Integration tests for data flows"""

    @pytest.fixture
    def flow_tester(self):
        """Create data flow tester"""
        config = {'mock_components': True}
        return DataFlowTester(config)

    @pytest.mark.asyncio
    async def test_user_knowledge_exploration_flow(self, flow_tester):
        """Test complete user knowledge exploration flow"""
        results = await flow_tester.test_data_flow('user_knowledge_exploration')

        assert results['flow_completed'] == True
        assert results['steps_completed'] == results['total_steps']
        assert results['steps_failed'] == 0
        assert len(results['errors']) == 0

    @pytest.mark.asyncio
    async def test_content_creation_workflow(self, flow_tester):
        """Test content creation and publication workflow"""
        results = await flow_tester.test_data_flow('content_creation_workflow')

        assert results['flow_completed'] == True
        assert results['steps_completed'] == 4  # All steps in workflow
        assert results['execution_time'] > 0

    @pytest.mark.asyncio
    async def test_data_flow_under_load(self, flow_tester):
        """Test data flow performance under concurrent load"""
        results = await flow_tester.test_data_flow_under_load(
            'user_knowledge_exploration',
            concurrent_users=5
        )

        assert results['concurrent_users'] == 5
        assert results['aggregate_metrics']['success_rate'] >= 0.8  # At least 80% success rate
        assert results['aggregate_metrics']['average_execution_time'] > 0

    @pytest.mark.asyncio
    async def test_flow_error_handling(self, flow_tester):
        """Test error handling in data flows"""
        # Test with a flow that has a failing step
        flow_tester.data_flows['error_test_flow'] = DataFlow(
            'error_test_flow',
            'Test flow with intentional failure',
            [
                {
                    'name': 'working_step',
                    'component': 'test_component',
                    'action': 'working_action',
                    'expected_output': {'success': lambda x: True}
                },
                {
                    'name': 'failing_step',
                    'component': 'test_component',
                    'action': 'failing_action',
                    'expected_output': {'success': lambda x: False}  # Will fail
                }
            ]
        )

        results = await flow_tester.test_data_flow('error_test_flow')

        assert results['flow_completed'] == False
        assert results['steps_failed'] == 1
        assert len(results['errors']) > 0

    @pytest.mark.asyncio
    async def test_flow_data_persistence(self, flow_tester):
        """Test data persistence across flow steps"""
        # Create a custom flow that passes data between steps
        flow_tester.data_flows['data_persistence_test'] = DataFlow(
            'data_persistence_test',
            'Test data flow between steps',
            [
                {
                    'name': 'create_data',
                    'component': 'data_service',
                    'action': 'create',
                    'input_data': {'test_value': 'initial_data'},
                    'expected_output': {'data_id': lambda x: x is not None}
                },
                {
                    'name': 'process_data',
                    'component': 'data_service',
                    'action': 'process',
                    'input_data': {'use_previous_data': True},
                    'expected_output': {'processed': lambda x: x == True}
                }
            ]
        )

        results = await flow_tester.test_data_flow('data_persistence_test')

        assert results['flow_completed'] == True
        # Verify data was passed correctly between steps
        step_0_output = results['step_results'][0]['output']
        step_1_input = flow_tester.data_flows['data_persistence_test'].flow_data.get('step_0_output')

        assert step_1_input is not None
        assert step_1_input == step_0_output
```

### Phase 3: Performance Integration Testing

#### 3.1 Load Testing Framework
```python
import pytest
import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional
import logging

class LoadTestScenario:
    """Represents a load testing scenario"""

    def __init__(self, name: str, description: str, config: Dict[str, Any]):
        """Initialize load test scenario"""
        self.name = name
        self.description = description
        self.config = config
        self.results: List[Dict[str, Any]] = []

    def get_concurrent_users(self) -> int:
        """Get number of concurrent users for this scenario"""
        return self.config.get('concurrent_users', 10)

    def get_duration_seconds(self) -> int:
        """Get test duration in seconds"""
        return self.config.get('duration_seconds', 60)

    def get_ramp_up_seconds(self) -> int:
        """Get ramp-up time in seconds"""
        return self.config.get('ramp_up_seconds', 10)

    async def execute_scenario(self) -> Dict[str, Any]:
        """Execute the load testing scenario"""
        concurrent_users = self.get_concurrent_users()
        duration = self.get_duration_seconds()
        ramp_up = self.get_ramp_up_seconds()

        start_time = time.time()
        end_time = start_time + duration

        # Create user tasks
        user_tasks = []
        for user_id in range(concurrent_users):
            # Stagger user start times during ramp-up
            delay = (user_id / concurrent_users) * ramp_up if ramp_up > 0 else 0
            task = asyncio.create_task(self.simulate_user_session(user_id, delay, end_time))
            user_tasks.append(task)

        # Wait for all users to complete
        user_results = await asyncio.gather(*user_tasks)

        # Aggregate results
        scenario_results = self.aggregate_scenario_results(user_results, duration)

        return scenario_results

    async def simulate_user_session(self, user_id: int, start_delay: float, end_time: float) -> Dict[str, Any]:
        """Simulate a single user session"""
        await asyncio.sleep(start_delay)

        user_results = {
            'user_id': user_id,
            'actions_completed': 0,
            'errors_encountered': 0,
            'response_times': [],
            'start_time': time.time(),
            'end_time': None
        }

        while time.time() < end_time:
            try:
                # Execute user action
                action_start = time.time()
                await self.execute_user_action(user_id)
                action_end = time.time()

                response_time = action_end - action_start
                user_results['response_times'].append(response_time)
                user_results['actions_completed'] += 1

                # Wait between actions (simulate think time)
                await asyncio.sleep(self.config.get('think_time_seconds', 1.0))

            except Exception as e:
                user_results['errors_encountered'] += 1
                await asyncio.sleep(0.1)  # Brief pause on error

        user_results['end_time'] = time.time()
        user_results['session_duration'] = user_results['end_time'] - user_results['start_time']

        return user_results

    async def execute_user_action(self, user_id: int) -> None:
        """Execute a single user action (to be implemented by subclasses)"""
        # Default implementation - override in subclasses
        await asyncio.sleep(0.1)  # Simulate action

    def aggregate_scenario_results(self, user_results: List[Dict[str, Any]], duration: int) -> Dict[str, Any]:
        """Aggregate results from all user sessions"""
        total_actions = sum(user['actions_completed'] for user in user_results)
        total_errors = sum(user['errors_encountered'] for user in user_results)

        # Collect all response times
        all_response_times = []
        for user in user_results:
            all_response_times.extend(user['response_times'])

        # Calculate statistics
        results = {
            'scenario_name': self.name,
            'concurrent_users': len(user_results),
            'duration_seconds': duration,
            'total_actions': total_actions,
            'total_errors': total_errors,
            'actions_per_second': total_actions / duration if duration > 0 else 0,
            'error_rate': total_errors / total_actions if total_actions > 0 else 0,
            'response_time_stats': {}
        }

        if all_response_times:
            results['response_time_stats'] = {
                'min': min(all_response_times),
                'max': max(all_response_times),
                'mean': statistics.mean(all_response_times),
                'median': statistics.median(all_response_times),
                'p95': statistics.quantiles(all_response_times, n=20)[18],  # 95th percentile
                'p99': statistics.quantiles(all_response_times, n=100)[98] if len(all_response_times) >= 100 else max(all_response_times)
            }

        # Individual user stats
        results['user_stats'] = []
        for user in user_results:
            user_stat = {
                'user_id': user['user_id'],
                'actions_completed': user['actions_completed'],
                'errors_encountered': user['errors_encountered'],
                'session_duration': user['session_duration'],
                'actions_per_second': user['actions_completed'] / user['session_duration'] if user['session_duration'] > 0 else 0
            }
            results['user_stats'].append(user_stat)

        return results

class KnowledgeExplorationLoadTest(LoadTestScenario):
    """Load test for knowledge exploration functionality"""

    async def execute_user_action(self, user_id: int) -> None:
        """Simulate user exploring knowledge base"""
        # Simulate search
        search_query = f"active inference concepts user_{user_id}"
        # In real implementation, make actual API call
        await asyncio.sleep(0.2)  # Simulate search time

        # Simulate reading results and clicking on a concept
        await asyncio.sleep(0.5)  # Simulate reading time

        # Simulate requesting detailed view
        # In real implementation, make API call for knowledge node
        await asyncio.sleep(0.3)  # Simulate content loading

class VisualizationLoadTest(LoadTestScenario):
    """Load test for visualization functionality"""

    async def execute_user_action(self, user_id: int) -> None:
        """Simulate user creating and interacting with visualizations"""
        # Simulate creating a concept map
        await asyncio.sleep(0.8)  # Simulate visualization creation

        # Simulate interacting with visualization (zoom, pan, click)
        for _ in range(5):
            await asyncio.sleep(0.1)  # Simulate interaction

class SearchLoadTest(LoadTestScenario):
    """Load test for search functionality"""

    async def execute_user_action(self, user_id: int) -> None:
        """Simulate user performing searches"""
        # Simulate different types of searches
        search_types = ['basic', 'advanced', 'autocomplete']

        for search_type in search_types:
            if search_type == 'basic':
                # Basic search
                await asyncio.sleep(0.15)
            elif search_type == 'advanced':
                # Advanced search with filters
                await asyncio.sleep(0.25)
            elif search_type == 'autocomplete':
                # Autocomplete suggestions
                await asyncio.sleep(0.05)

class LoadTestingFramework:
    """Comprehensive load testing framework"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize load testing framework"""
        self.config = config
        self.logger = logging.getLogger('LoadTestingFramework')
        self.test_scenarios = self.define_test_scenarios()

    def define_test_scenarios(self) -> Dict[str, LoadTestScenario]:
        """Define load testing scenarios"""
        return {
            'light_load': KnowledgeExplorationLoadTest(
                'light_load',
                'Light load test with 10 concurrent users',
                {
                    'concurrent_users': 10,
                    'duration_seconds': 60,
                    'ramp_up_seconds': 5,
                    'think_time_seconds': 2.0
                }
            ),
            'medium_load': KnowledgeExplorationLoadTest(
                'medium_load',
                'Medium load test with 50 concurrent users',
                {
                    'concurrent_users': 50,
                    'duration_seconds': 120,
                    'ramp_up_seconds': 15,
                    'think_time_seconds': 1.5
                }
            ),
            'heavy_load': KnowledgeExplorationLoadTest(
                'heavy_load',
                'Heavy load test with 100 concurrent users',
                {
                    'concurrent_users': 100,
                    'duration_seconds': 180,
                    'ramp_up_seconds': 30,
                    'think_time_seconds': 1.0
                }
            ),
            'visualization_load': VisualizationLoadTest(
                'visualization_load',
                'Visualization load test with 20 concurrent users',
                {
                    'concurrent_users': 20,
                    'duration_seconds': 90,
                    'ramp_up_seconds': 10,
                    'think_time_seconds': 3.0
                }
            ),
            'search_load': SearchLoadTest(
                'search_load',
                'Search load test with 30 concurrent users',
                {
                    'concurrent_users': 30,
                    'duration_seconds': 120,
                    'ramp_up_seconds': 10,
                    'think_time_seconds': 0.5
                }
            )
        }

    async def run_load_test(self, scenario_name: str) -> Dict[str, Any]:
        """Run a specific load test scenario"""
        if scenario_name not in self.test_scenarios:
            raise ValueError(f"Unknown load test scenario: {scenario_name}")

        scenario = self.test_scenarios[scenario_name]
        self.logger.info(f"Starting load test: {scenario_name}")

        start_time = time.time()
        results = await scenario.execute_scenario()
        end_time = time.time()

        # Add test metadata
        results['test_metadata'] = {
            'scenario_name': scenario_name,
            'start_time': start_time,
            'end_time': end_time,
            'total_duration': end_time - start_time,
            'framework_version': '1.0.0'
        }

        # Validate results against thresholds
        results['threshold_validation'] = self.validate_performance_thresholds(results)

        self.logger.info(f"Completed load test: {scenario_name}")
        return results

    def validate_performance_thresholds(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate results against performance thresholds"""
        validation = {
            'passed': True,
            'failed_thresholds': [],
            'warnings': []
        }

        # Define thresholds based on scenario
        scenario_name = results.get('scenario_name', '')
        thresholds = self.get_performance_thresholds(scenario_name)

        # Check response time thresholds
        response_stats = results.get('response_time_stats', {})
        if response_stats:
            p95_time = response_stats.get('p95', 0)
            if p95_time > thresholds['max_p95_response_time']:
                validation['passed'] = False
                validation['failed_thresholds'].append({
                    'metric': 'p95_response_time',
                    'actual': p95_time,
                    'threshold': thresholds['max_p95_response_time']
                })

            mean_time = response_stats.get('mean', 0)
            if mean_time > thresholds['max_mean_response_time']:
                validation['warnings'].append({
                    'metric': 'mean_response_time',
                    'actual': mean_time,
                    'threshold': thresholds['max_mean_response_time']
                })

        # Check error rate threshold
        error_rate = results.get('error_rate', 0)
        if error_rate > thresholds['max_error_rate']:
            validation['passed'] = False
            validation['failed_thresholds'].append({
                'metric': 'error_rate',
                'actual': error_rate,
                'threshold': thresholds['max_error_rate']
            })

        # Check throughput threshold
        actions_per_second = results.get('actions_per_second', 0)
        if actions_per_second < thresholds['min_actions_per_second']:
            validation['passed'] = False
            validation['failed_thresholds'].append({
                'metric': 'actions_per_second',
                'actual': actions_per_second,
                'threshold': thresholds['min_actions_per_second']
            })

        return validation

    def get_performance_thresholds(self, scenario_name: str) -> Dict[str, float]:
        """Get performance thresholds for a scenario"""
        # Default thresholds
        default_thresholds = {
            'max_p95_response_time': 2.0,  # 2 seconds
            'max_mean_response_time': 1.0,  # 1 second
            'max_error_rate': 0.05,  # 5%
            'min_actions_per_second': 1.0  # 1 action per second
        }

        # Scenario-specific thresholds
        scenario_thresholds = {
            'light_load': {
                'max_p95_response_time': 1.0,
                'max_mean_response_time': 0.5,
                'max_error_rate': 0.01,
                'min_actions_per_second': 2.0
            },
            'medium_load': {
                'max_p95_response_time': 2.0,
                'max_mean_response_time': 1.0,
                'max_error_rate': 0.03,
                'min_actions_per_second': 1.5
            },
            'heavy_load': {
                'max_p95_response_time': 5.0,
                'max_mean_response_time': 2.0,
                'max_error_rate': 0.08,
                'min_actions_per_second': 1.0
            },
            'visualization_load': {
                'max_p95_response_time': 3.0,
                'max_mean_response_time': 1.5,
                'max_error_rate': 0.05,
                'min_actions_per_second': 0.5
            },
            'search_load': {
                'max_p95_response_time': 0.5,
                'max_mean_response_time': 0.2,
                'max_error_rate': 0.02,
                'min_actions_per_second': 5.0
            }
        }

        return scenario_thresholds.get(scenario_name, default_thresholds)

    async def run_performance_regression_test(self, baseline_results: Dict[str, Any],
                                           current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance regression test comparing current results to baseline"""
        regression_results = {
            'regression_detected': False,
            'significant_changes': [],
            'baseline_comparison': {}
        }

        # Compare key metrics
        metrics_to_compare = [
            'response_time_stats.mean',
            'response_time_stats.p95',
            'error_rate',
            'actions_per_second'
        ]

        for metric_path in metrics_to_compare:
            baseline_value = self.get_nested_value(baseline_results, metric_path)
            current_value = self.get_nested_value(current_results, metric_path)

            if baseline_value is not None and current_value is not None:
                change_percent = ((current_value - baseline_value) / baseline_value) * 100

                # Check for significant regression (more than 20% degradation)
                if change_percent > 20:
                    regression_results['regression_detected'] = True
                    regression_results['significant_changes'].append({
                        'metric': metric_path,
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_percent': change_percent,
                        'severity': 'regression'
                    })
                elif change_percent < -20:
                    # Significant improvement
                    regression_results['significant_changes'].append({
                        'metric': metric_path,
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_percent': change_percent,
                        'severity': 'improvement'
                    })

                regression_results['baseline_comparison'][metric_path] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'change_percent': change_percent
                }

        return regression_results

    def get_nested_value(self, data: Dict[str, Any], path: str):
        """Get nested value from dictionary using dot notation"""
        keys = path.split('.')
        current = data

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return None

    def generate_load_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive load test report"""
        scenario_name = results.get('scenario_name', 'Unknown')
        threshold_validation = results.get('threshold_validation', {})

        report = f"""
# Load Test Report: {scenario_name}

## Test Summary
- **Scenario**: {scenario_name}
- **Concurrent Users**: {results.get('concurrent_users', 0)}
- **Duration**: {results.get('duration_seconds', 0)} seconds
- **Total Actions**: {results.get('total_actions', 0)}
- **Actions/Second**: {results.get('actions_per_second', 0):.2f}
- **Error Rate**: {results.get('error_rate', 0)*100:.2f}%

## Performance Metrics

### Response Times
"""

        response_stats = results.get('response_time_stats', {})
        if response_stats:
            report += f"""- **Min**: {response_stats.get('min', 0):.3f}s
- **Max**: {response_stats.get('max', 0):.3f}s
- **Mean**: {response_stats.get('mean', 0):.3f}s
- **Median**: {response_stats.get('median', 0):.3f}s
- **95th Percentile**: {response_stats.get('p95', 0):.3f}s
- **99th Percentile**: {response_stats.get('p99', 0):.3f}s

"""

        # Threshold validation
        if threshold_validation.get('passed', True):
            report += "## ✅ Threshold Validation: PASSED\n\n"
        else:
            report += "## ❌ Threshold Validation: FAILED\n\n"
            for failure in threshold_validation.get('failed_thresholds', []):
                report += f"- **{failure['metric']}**: {failure['actual']:.3f} (threshold: {failure['threshold']:.3f})\n"

        # Warnings
        warnings = threshold_validation.get('warnings', [])
        if warnings:
            report += "\n## ⚠️ Performance Warnings\n\n"
            for warning in warnings:
                report += f"- **{warning['metric']}**: {warning['actual']:.3f} (threshold: {warning['threshold']:.3f})\n"

        # User statistics
        user_stats = results.get('user_stats', [])
        if user_stats:
            report += "\n## User Statistics\n\n"
            report += "| User ID | Actions | Errors | Actions/Second |\n"
            report += "|---------|---------|--------|----------------|\n"

            for user in user_stats[:10]:  # Show first 10 users
                report += f"| {user['user_id']} | {user['actions_completed']} | {user['errors_encountered']} | {user['actions_per_second']:.2f} |\n"

            if len(user_stats) > 10:
                report += f"| ... | ... | ... | ... |\n"

        return report

class TestLoadTestingIntegration:
    """Integration tests for load testing framework"""

    @pytest.fixture
    def load_tester(self):
        """Create load testing framework"""
        config = {'test_mode': True}
        return LoadTestingFramework(config)

    @pytest.mark.asyncio
    async def test_light_load_scenario(self, load_tester):
        """Test light load scenario execution"""
        results = await load_tester.run_load_test('light_load')

        assert results['concurrent_users'] == 10
        assert results['duration_seconds'] == 60
        assert results['total_actions'] > 0
        assert results['error_rate'] < 0.1  # Less than 10% error rate

        # Check threshold validation
        threshold_validation = results.get('threshold_validation', {})
        assert 'passed' in threshold_validation

    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, load_tester):
        """Test performance regression detection"""
        # Run baseline test
        baseline_results = await load_tester.run_load_test('light_load')

        # Simulate degraded performance in current results
        current_results = baseline_results.copy()
        current_results['response_time_stats']['mean'] *= 1.5  # 50% slower

        # Run regression test
        regression_results = await load_tester.run_performance_regression_test(
            baseline_results, current_results
        )

        assert regression_results['regression_detected'] == True
        assert len(regression_results['significant_changes']) > 0

        # Check that response time regression was detected
        changes = regression_results['significant_changes']
        response_time_change = next(
            (c for c in changes if 'response_time' in c['metric']), None
        )
        assert response_time_change is not None
        assert response_time_change['severity'] == 'regression'

    @pytest.mark.asyncio
    async def test_multiple_concurrent_scenarios(self, load_tester):
        """Test running multiple load scenarios concurrently"""
        scenarios = ['light_load', 'search_load', 'visualization_load']

        # Run all scenarios concurrently
        tasks = [load_tester.run_load_test(scenario) for scenario in scenarios]
        results_list = await asyncio.gather(*tasks)

        # Verify all scenarios completed
        assert len(results_list) == len(scenarios)

        for i, results in enumerate(results_list):
            scenario_name = scenarios[i]
            assert results['scenario_name'] == scenario_name
            assert results['total_actions'] > 0

    def test_load_test_report_generation(self, load_tester):
        """Test load test report generation"""
        # Create mock results
        mock_results = {
            'scenario_name': 'test_scenario',
            'concurrent_users': 5,
            'duration_seconds': 30,
            'total_actions': 150,
            'actions_per_second': 5.0,
            'error_rate': 0.02,
            'response_time_stats': {
                'min': 0.1,
                'max': 2.0,
                'mean': 0.5,
                'median': 0.4,
                'p95': 1.2,
                'p99': 1.8
            },
            'threshold_validation': {
                'passed': True,
                'failed_thresholds': [],
                'warnings': []
            },
            'user_stats': [
                {'user_id': 0, 'actions_completed': 30, 'errors_encountered': 1, 'actions_per_second': 1.0},
                {'user_id': 1, 'actions_completed': 32, 'errors_encountered': 0, 'actions_per_second': 1.07}
            ]
        }

        # Generate report
        report = load_tester.generate_load_test_report(mock_results)

        # Verify report content
        assert 'test_scenario' in report
        assert '5.0' in report  # actions per second
        assert '2.00%' in report  # error rate
        assert 'PASSED' in report  # threshold validation
        assert 'User Statistics' in report

    @pytest.mark.asyncio
    async def test_scalability_boundary_testing(self, load_tester):
        """Test system behavior at scalability boundaries"""
        # Test with increasing load levels
        load_levels = [10, 25, 50, 100]

        scalability_results = []

        for load in load_levels:
            # Create custom scenario for this load level
            scenario_name = f'scalability_test_{load}'
            load_tester.test_scenarios[scenario_name] = KnowledgeExplorationLoadTest(
                scenario_name,
                f'Scalability test with {load} users',
                {
                    'concurrent_users': load,
                    'duration_seconds': 30,  # Shorter duration for scalability testing
                    'ramp_up_seconds': min(10, load // 2),
                    'think_time_seconds': 1.0
                }
            )

            # Run test
            results = await load_tester.run_load_test(scenario_name)
            scalability_results.append({
                'load_level': load,
                'results': results
            })

        # Analyze scalability
        for i in range(1, len(scalability_results)):
            prev_load = scalability_results[i-1]['load_level']
            curr_load = scalability_results[i]['load_level']
            prev_actions = scalability_results[i-1]['results']['actions_per_second']
            curr_actions = scalability_results[i]['results']['actions_per_second']

            # Check for performance degradation
            load_increase = curr_load / prev_load
            performance_ratio = curr_actions / prev_actions

            # Performance shouldn't degrade more than load increases
            # (allowing for some overhead)
            assert performance_ratio >= (1.0 / load_increase) * 0.7, \
                f"Performance degraded too much at load {curr_load}: {performance_ratio:.2f}"
```

---

**"Active Inference for, with, by Generative AI"** - Building comprehensive integration testing frameworks that validate system-wide interactions, data flows, and component integrations for reliable, cohesive platform operation.
