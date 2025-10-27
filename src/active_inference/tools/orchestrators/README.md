# Orchestrators

**Thin orchestration components for workflow management in the Active Inference Knowledge Environment.**

## Overview

The orchestrators module provides lightweight orchestration components that coordinate workflows across different platform services, enabling complex multi-step processes and automated task management.

### Core Features

- **Workflow Orchestration**: Coordinate complex multi-step processes
- **Service Integration**: Seamless integration with platform services
- **Event-Driven Processing**: Reactive processing based on events
- **State Management**: Persistent workflow state and progress tracking
- **Error Recovery**: Robust error handling and recovery mechanisms
- **Monitoring**: Real-time workflow monitoring and analytics

## Architecture

### Orchestration Components

```
┌─────────────────┐
│   Workflow      │ ← Define and manage workflows
│   Manager       │
├─────────────────┤
│   Task          │ ← Execute individual tasks
│   Executor      │
├─────────────────┤
│   State         │ ← Track workflow state and progress
│   Manager       │
├─────────────────┤
│ Event            │ ← Event-driven processing and triggers
│   Processor     │
└─────────────────┘
```

### Workflow Types

- **Sequential Workflows**: Linear task execution
- **Parallel Workflows**: Concurrent task execution
- **Conditional Workflows**: Branching based on conditions
- **Event-Driven Workflows**: Reactive processing
- **Batch Workflows**: Process multiple items

## Usage

### Basic Setup

```python
from active_inference.tools.orchestrators import WorkflowManager

# Initialize orchestrator
config = {
    "storage_backend": "redis",  # or "memory", "database"
    "max_concurrent_workflows": 100,
    "enable_monitoring": True
}

orchestrator = WorkflowManager(config)
```

### Creating Workflows

```python
# Define a simple sequential workflow
workflow = orchestrator.create_workflow("knowledge_processing")

# Add tasks to workflow
workflow.add_task("validate_content", validate_content_task)
workflow.add_task("index_content", index_content_task)
workflow.add_task("generate_summary", generate_summary_task)
workflow.add_task("notify_completion", notify_completion_task)

# Set dependencies
workflow.set_dependencies({
    "index_content": ["validate_content"],
    "generate_summary": ["index_content"],
    "notify_completion": ["generate_summary"]
})
```

### Executing Workflows

```python
# Execute workflow with input data
result = await orchestrator.execute_workflow(
    workflow_id="knowledge_processing",
    input_data={
        "content_path": "knowledge/foundations/active_inference.json",
        "user_id": "user123",
        "priority": "high"
    }
)

# Monitor execution
status = orchestrator.get_workflow_status(workflow_id)
print(f"Status: {status['state']}")
print(f"Progress: {status['progress']}%")
```

### Event-Driven Orchestration

```python
# Create event-driven workflow
event_workflow = orchestrator.create_event_workflow("content_update")

# Define event handlers
@event_workflow.on("content_created")
async def handle_content_created(event_data):
    """Handle new content creation"""
    return await process_new_content(event_data)

@event_workflow.on("content_updated")
async def handle_content_updated(event_data):
    """Handle content updates"""
    return await update_indexes(event_data)

# Start event processing
await orchestrator.start_event_processing()
```

## Configuration

### Storage Configuration

```python
storage_config = {
    "backend": "redis",
    "connection": {
        "host": "localhost",
        "port": 6379,
        "db": 0
    },
    "serialization": "json",
    "compression": True
}
```

### Execution Configuration

```python
execution_config = {
    "max_concurrent_workflows": 50,
    "max_concurrent_tasks": 100,
    "task_timeout": 300,
    "retry_attempts": 3,
    "retry_delay": 5
}
```

### Monitoring Configuration

```python
monitoring_config = {
    "enabled": True,
    "metrics": [
        "workflow_duration",
        "task_success_rate",
        "error_rate",
        "throughput"
    ],
    "alerting": {
        "error_threshold": 0.1,
        "duration_threshold": 600
    }
}
```

## API Reference

### WorkflowManager

Main interface for workflow orchestration.

#### Core Methods

- `create_workflow(name: str, workflow_type: str = "sequential") -> Workflow`: Create workflow
- `execute_workflow(workflow_id: str, input_data: Dict) -> WorkflowResult`: Execute workflow
- `get_workflow_status(workflow_id: str) -> Dict`: Get workflow status
- `pause_workflow(workflow_id: str) -> bool`: Pause workflow execution
- `resume_workflow(workflow_id: str) -> bool`: Resume workflow execution
- `cancel_workflow(workflow_id: str) -> bool`: Cancel workflow

### TaskExecutor

Executes individual workflow tasks.

#### Methods

- `execute_task(task: Task, input_data: Dict) -> TaskResult`: Execute single task
- `execute_parallel(tasks: List[Task]) -> List[TaskResult]`: Execute tasks in parallel
- `retry_task(task: Task, max_attempts: int) -> TaskResult`: Retry failed task
- `get_task_status(task_id: str) -> Dict`: Get task execution status

### StateManager

Manages workflow state and persistence.

#### Methods

- `save_workflow_state(workflow_id: str, state: Dict) -> bool`: Save workflow state
- `load_workflow_state(workflow_id: str) -> Dict`: Load workflow state
- `update_workflow_progress(workflow_id: str, progress: float) -> bool`: Update progress
- `get_workflow_history(workflow_id: str) -> List[State]`: Get state history

## Advanced Features

### Conditional Workflows

```python
# Create conditional workflow
conditional_workflow = orchestrator.create_conditional_workflow("content_validation")

# Define conditions
@conditional_workflow.condition("needs_review")
def check_review_needed(task_result):
    """Check if content needs review"""
    return task_result["validation_score"] < 0.8

@conditional_workflow.condition("is_complex")
def check_complexity(task_result):
    """Check content complexity"""
    return task_result["complexity_score"] > 0.7

# Define conditional tasks
conditional_workflow.add_conditional_task(
    "human_review",
    condition="needs_review",
    task=human_review_task
)

conditional_workflow.add_conditional_task(
    "expert_review",
    condition="is_complex",
    task=expert_review_task
)
```

### Parallel Processing

```python
# Create parallel workflow
parallel_workflow = orchestrator.create_parallel_workflow("batch_processing")

# Add parallel tasks
parallel_workflow.add_parallel_tasks([
    {"name": "process_item_1", "task": process_item_task, "data": item1},
    {"name": "process_item_2", "task": process_item_task, "data": item2},
    {"name": "process_item_3", "task": process_item_task, "data": item3}
])

# Execute with concurrency control
result = await orchestrator.execute_parallel_workflow(
    workflow_id="batch_processing",
    max_concurrency=5,
    timeout=3600
)
```

### Event-Driven Processing

```python
# Create event-driven orchestrator
event_orchestrator = EventDrivenOrchestrator(config)

# Define event handlers
@event_orchestrator.on("content.created")
async def handle_content_created(event):
    """Handle new content creation"""
    workflow = orchestrator.create_workflow("content_processing")
    # ... workflow setup

@event_orchestrator.on("user.requested_analysis")
async def handle_analysis_request(event):
    """Handle analysis requests"""
    workflow = orchestrator.create_workflow("analysis_pipeline")
    # ... workflow setup

# Start event processing
await event_orchestrator.start()
```

## Performance

### Optimization

```python
# Enable performance optimizations
orchestrator.enable_optimizations()

# Configure resource management
orchestrator.set_resource_limits(
    max_memory="2GB",
    max_cpu="4",
    max_concurrent=50
)

# Enable caching
orchestrator.enable_task_caching()
```

### Metrics

```python
# Monitor performance
metrics = orchestrator.get_performance_metrics()

print(f"Workflow throughput: {metrics['workflows_per_hour']}")
print(f"Average task duration: {metrics['avg_task_duration']}s")
print(f"Error rate: {metrics['error_rate']}%")
print(f"Resource utilization: {metrics['resource_utilization']}%")
```

## Testing

### Running Tests

```bash
# Run orchestrator tests
make test-orchestrators

# Or run specific tests
pytest src/active_inference/tools/orchestrators/tests/ -v

# Load testing
pytest src/active_inference/tools/orchestrators/tests/test_load.py -v

# Integration tests
pytest src/active_inference/tools/orchestrators/tests/test_integration.py -v
```

### Test Coverage

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Cross-service workflow execution
- **Performance Tests**: Concurrent workflow processing
- **Reliability Tests**: Error handling and recovery

## Monitoring

### Health Checks

```python
# Orchestrator health check
health = orchestrator.health_check()

print(f"Active workflows: {health['active_workflows']}")
print(f"Queued tasks: {health['queued_tasks']}")
print(f"Worker status: {health['worker_status']}")
print(f"Resource usage: {health['resource_usage']}%")
```

### Workflow Analytics

```bash
# Start workflow analytics
make orchestrator-analytics

# View real-time metrics
curl http://localhost:8080/orchestrator/analytics
```

## Integration

### Service Integration

```python
# Integrate with platform services
orchestrator.integrate_with_knowledge_service(knowledge_service)
orchestrator.integrate_with_search_service(search_service)
orchestrator.integrate_with_visualization_service(visualization_service)

# Configure service dependencies
orchestrator.set_service_dependencies({
    "knowledge_processing": ["knowledge_service", "search_service"],
    "content_generation": ["knowledge_service", "visualization_service"]
})
```

### Platform Integration

```python
# Platform-wide orchestration
platform_orchestrator = PlatformOrchestrator(config)

# Coordinate all platform services
await platform_orchestrator.coordinate_services(
    services=["knowledge", "research", "visualization"],
    workflow="platform_sync"
)
```

## Contributing

See [AGENTS.md](AGENTS.md) for development guidelines and [.cursorrules](../../../.cursorrules) for comprehensive development standards.

### Development Process

1. **Environment Setup**:
   ```bash
   cd src/active_inference/tools/orchestrators
   make setup
   ```

2. **Testing**:
   ```bash
   make test
   make test-integration
   ```

3. **Documentation**:
   - Update README.md for new features
   - Update AGENTS.md for development patterns
   - Add comprehensive examples

## Security

### Workflow Security

```python
# Enable workflow security
security_config = {
    "access_control": True,
    "audit_logging": True,
    "data_encryption": True,
    "sandbox_execution": True
}
```

### Access Control

- Role-based workflow execution
- User permission validation
- Secure task execution environment
- Audit trails for all operations

## Troubleshooting

### Common Issues

#### Workflow Failures
```bash
# Check workflow logs
orchestrator.get_workflow_logs(workflow_id)

# Debug workflow execution
orchestrator.debug_workflow(workflow_id)

# Retry failed tasks
orchestrator.retry_failed_tasks(workflow_id)
```

#### Performance Issues
```bash
# Monitor resource usage
orchestrator.monitor_resources()

# Optimize workflow configuration
orchestrator.optimize_configuration()

# Scale orchestration resources
orchestrator.scale_resources()
```

#### State Issues
```bash
# Check workflow state consistency
orchestrator.validate_state_consistency()

# Repair corrupted state
orchestrator.repair_state()

# Clear stale workflows
orchestrator.cleanup_stale_workflows()
```

---

**Component Version**: 1.0.0 | **Development Status**: Active Development

*"Active Inference for, with, by Generative AI"* - Orchestrated intelligence through coordinated workflows.