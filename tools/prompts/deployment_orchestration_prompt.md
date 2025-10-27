# Deployment Orchestration and Scaling Prompt

**"Active Inference for, with, by Generative AI"**

## 🎯 Mission: Automated Deployment and Scaling Orchestration

You are tasked with developing comprehensive deployment orchestration and scaling systems for the Active Inference Knowledge Environment. This involves creating automated deployment pipelines, container orchestration, scaling strategies, and infrastructure management that ensure reliable, scalable platform operations.

## 📋 Deployment Orchestration Requirements

### Core Deployment Standards (MANDATORY)
1. **Infrastructure as Code**: All infrastructure defined as code with version control
2. **Automated Pipelines**: CI/CD pipelines for automated testing and deployment
3. **Container Orchestration**: Kubernetes-based deployment with service mesh
4. **Blue-Green Deployments**: Zero-downtime deployment strategies
5. **Rollback Capabilities**: Automated rollback mechanisms for failed deployments
6. **Monitoring Integration**: Comprehensive monitoring and alerting for deployed services

### Deployment Architecture Components
```
deployment/
├── orchestrator/
│   ├── deployment_manager.py     # Main deployment orchestration
│   ├── pipeline_engine.py        # CI/CD pipeline management
│   ├── rollback_manager.py       # Deployment rollback system
│   └── health_checker.py         # Deployment health validation
├── scaling/
│   ├── auto_scaler.py           # Automatic scaling management
│   ├── load_balancer.py         # Load balancing configuration
│   ├── resource_manager.py      # Resource allocation and management
│   └── performance_monitor.py   # Performance-based scaling decisions
├── infrastructure/
│   ├── terraform/               # Infrastructure as Code definitions
│   ├── kubernetes/              # Kubernetes manifests
│   ├── docker/                  # Container definitions
│   └── monitoring/              # Infrastructure monitoring
└── backup/
    ├── snapshot_manager.py      # System snapshot creation
    ├── backup_scheduler.py      # Automated backup scheduling
    ├── recovery_manager.py      # System recovery orchestration
    └── disaster_recovery.py     # Disaster recovery procedures
```

## 🏗️ Deployment Orchestration Engine

### Phase 1: CI/CD Pipeline Management

#### 1.1 Pipeline Engine Architecture
```python
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import asyncio
import logging
from datetime import datetime
import json

class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLING_BACK = "rolling_back"

class DeploymentStage(Enum):
    """Deployment pipeline stages"""
    BUILD = "build"
    TEST = "test"
    DEPLOY_STAGING = "deploy_staging"
    INTEGRATION_TEST = "integration_test"
    DEPLOY_PRODUCTION = "deploy_production"
    POST_DEPLOYMENT_CHECKS = "post_deployment_checks"
    CLEANUP = "cleanup"

class PipelineStep:
    """Individual pipeline step definition"""

    def __init__(self, name: str, stage: DeploymentStage, command: str,
                 timeout: int = 300, dependencies: Optional[List[str]] = None):
        """Initialize pipeline step"""
        self.name = name
        self.stage = stage
        self.command = command
        self.timeout = timeout
        self.dependencies = dependencies or []
        self.status = PipelineStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.output: str = ""
        self.error: str = ""
        self.exit_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary"""
        return {
            'name': self.name,
            'stage': self.stage.value,
            'command': self.command,
            'timeout': self.timeout,
            'dependencies': self.dependencies,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'exit_code': self.exit_code,
            'has_error': bool(self.error)
        }

class DeploymentPipeline:
    """Complete deployment pipeline definition"""

    def __init__(self, pipeline_id: str, name: str, config: Dict[str, Any]):
        """Initialize deployment pipeline"""
        self.pipeline_id = pipeline_id
        self.name = name
        self.config = config
        self.steps: Dict[str, PipelineStep] = {}
        self.status = PipelineStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}

    def add_step(self, step: PipelineStep) -> None:
        """Add step to pipeline"""
        self.steps[step.name] = step

    def get_executable_steps(self) -> List[PipelineStep]:
        """Get steps that can be executed (dependencies satisfied)"""
        executable = []

        for step in self.steps.values():
            if step.status == PipelineStatus.PENDING:
                # Check if all dependencies are completed successfully
                dependencies_satisfied = all(
                    self.steps[dep].status == PipelineStatus.SUCCESS
                    for dep in step.dependencies
                )

                if dependencies_satisfied:
                    executable.append(step)

        return executable

    def update_step_status(self, step_name: str, status: PipelineStatus,
                          exit_code: Optional[int] = None, output: str = "",
                          error: str = "") -> None:
        """Update step execution status"""
        if step_name in self.steps:
            step = self.steps[step_name]
            step.status = status
            step.exit_code = exit_code
            step.output = output
            step.error = error

            if status in [PipelineStatus.RUNNING, PipelineStatus.ROLLING_BACK]:
                step.start_time = step.start_time or datetime.now()
            elif status in [PipelineStatus.SUCCESS, PipelineStatus.FAILED, PipelineStatus.CANCELLED]:
                step.end_time = datetime.now()

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        completed_steps = sum(1 for s in self.steps.values()
                            if s.status in [PipelineStatus.SUCCESS, PipelineStatus.FAILED])
        total_steps = len(self.steps)

        # Determine overall pipeline status
        if all(s.status == PipelineStatus.SUCCESS for s in self.steps.values()):
            overall_status = PipelineStatus.SUCCESS
        elif any(s.status == PipelineStatus.FAILED for s in self.steps.values()):
            overall_status = PipelineStatus.FAILED
        elif any(s.status in [PipelineStatus.RUNNING, PipelineStatus.ROLLING_BACK]
                for s in self.steps.values()):
            overall_status = PipelineStatus.RUNNING
        else:
            overall_status = PipelineStatus.PENDING

        return {
            'pipeline_id': self.pipeline_id,
            'name': self.name,
            'status': overall_status.value,
            'progress': f"{completed_steps}/{total_steps}",
            'progress_percent': (completed_steps / total_steps) * 100 if total_steps > 0 else 0,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'steps': {name: step.to_dict() for name, step in self.steps.items()},
            'metadata': self.metadata
        }

class PipelineEngine:
    """CI/CD pipeline execution engine"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline engine"""
        self.config = config
        self.logger = logging.getLogger('PipelineEngine')
        self.pipelines: Dict[str, DeploymentPipeline] = {}
        self.active_pipelines: Dict[str, asyncio.Task] = {}
        self.pipeline_hooks: Dict[str, List[Callable]] = {}

    async def create_pipeline(self, name: str, config: Dict[str, Any]) -> str:
        """Create new deployment pipeline"""
        pipeline_id = f"pipeline_{datetime.now().timestamp()}_{name.replace(' ', '_')}"

        pipeline = DeploymentPipeline(pipeline_id, name, config)
        self.pipelines[pipeline_id] = pipeline

        # Add standard pipeline steps based on config
        await self.build_pipeline_steps(pipeline, config)

        self.logger.info(f"Created deployment pipeline: {pipeline_id}")
        return pipeline_id

    async def build_pipeline_steps(self, pipeline: DeploymentPipeline,
                                 config: Dict[str, Any]) -> None:
        """Build pipeline steps based on configuration"""

        # Build stage
        build_step = PipelineStep(
            "build_application",
            DeploymentStage.BUILD,
            self.get_build_command(config),
            timeout=config.get('build_timeout', 600)
        )
        pipeline.add_step(build_step)

        # Test stage
        test_step = PipelineStep(
            "run_tests",
            DeploymentStage.TEST,
            self.get_test_command(config),
            timeout=config.get('test_timeout', 300),
            dependencies=["build_application"]
        )
        pipeline.add_step(test_step)

        # Staging deployment
        staging_step = PipelineStep(
            "deploy_staging",
            DeploymentStage.DEPLOY_STAGING,
            self.get_staging_deploy_command(config),
            timeout=config.get('deploy_timeout', 300),
            dependencies=["run_tests"]
        )
        pipeline.add_step(staging_step)

        # Integration tests
        integration_step = PipelineStep(
            "integration_tests",
            DeploymentStage.INTEGRATION_TEST,
            self.get_integration_test_command(config),
            timeout=config.get('integration_timeout', 300),
            dependencies=["deploy_staging"]
        )
        pipeline.add_step(integration_step)

        # Production deployment
        production_step = PipelineStep(
            "deploy_production",
            DeploymentStage.DEPLOY_PRODUCTION,
            self.get_production_deploy_command(config),
            timeout=config.get('deploy_timeout', 600),
            dependencies=["integration_tests"]
        )
        pipeline.add_step(production_step)

        # Post-deployment checks
        health_step = PipelineStep(
            "health_checks",
            DeploymentStage.POST_DEPLOYMENT_CHECKS,
            self.get_health_check_command(config),
            timeout=config.get('health_check_timeout', 120),
            dependencies=["deploy_production"]
        )
        pipeline.add_step(health_step)

    def get_build_command(self, config: Dict[str, Any]) -> str:
        """Get build command based on technology stack"""
        build_type = config.get('build_type', 'python')

        if build_type == 'python':
            return "python setup.py build && python -m pip install -e ."
        elif build_type == 'docker':
            return f"docker build -t {config.get('image_name', 'app')} ."
        elif build_type == 'node':
            return "npm run build"
        else:
            return config.get('custom_build_command', 'echo "Custom build command"')

    def get_test_command(self, config: Dict[str, Any]) -> str:
        """Get test command"""
        test_type = config.get('test_type', 'pytest')
        coverage_threshold = config.get('coverage_threshold', 95)

        if test_type == 'pytest':
            return f"pytest --cov --cov-report=xml --cov-report=term --cov-fail-under={coverage_threshold}"
        elif test_type == 'jest':
            return "npm test -- --coverage --coverageReporters=json-summary"
        else:
            return config.get('custom_test_command', 'echo "Running tests"')

    def get_staging_deploy_command(self, config: Dict[str, Any]) -> str:
        """Get staging deployment command"""
        deploy_type = config.get('deploy_type', 'kubernetes')

        if deploy_type == 'kubernetes':
            return f"kubectl apply -f k8s/staging/ --namespace={config.get('staging_namespace', 'staging')}"
        elif deploy_type == 'docker':
            return f"docker-compose -f docker-compose.staging.yml up -d"
        else:
            return config.get('custom_staging_deploy', 'echo "Staging deployment"')

    def get_integration_test_command(self, config: Dict[str, Any]) -> str:
        """Get integration test command"""
        return config.get('integration_test_command',
                         'pytest tests/integration/ --maxfail=5')

    def get_production_deploy_command(self, config: Dict[str, Any]) -> str:
        """Get production deployment command"""
        deploy_type = config.get('deploy_type', 'kubernetes')

        if deploy_type == 'kubernetes':
            return f"kubectl apply -f k8s/production/ --namespace={config.get('production_namespace', 'production')}"
        elif deploy_type == 'docker':
            return f"docker-compose -f docker-compose.production.yml up -d"
        else:
            return config.get('custom_production_deploy', 'echo "Production deployment"')

    def get_health_check_command(self, config: Dict[str, Any]) -> str:
        """Get health check command"""
        health_check_url = config.get('health_check_url', 'http://localhost:8000/health')
        return f"curl -f {health_check_url} || exit 1"

    async def execute_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """Execute deployment pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        pipeline = self.pipelines[pipeline_id]
        pipeline.started_at = datetime.now()
        pipeline.status = PipelineStatus.RUNNING

        self.logger.info(f"Starting pipeline execution: {pipeline_id}")

        try:
            # Execute pipeline asynchronously
            task = asyncio.create_task(self._execute_pipeline_async(pipeline))
            self.active_pipelines[pipeline_id] = task

            result = await task

            pipeline.completed_at = datetime.now()
            pipeline.status = PipelineStatus.SUCCESS

            return result

        except Exception as e:
            pipeline.status = PipelineStatus.FAILED
            pipeline.completed_at = datetime.now()
            self.logger.error(f"Pipeline execution failed: {e}")
            raise

        finally:
            if pipeline_id in self.active_pipelines:
                del self.active_pipelines[pipeline_id]

    async def _execute_pipeline_async(self, pipeline: DeploymentPipeline) -> Dict[str, Any]:
        """Execute pipeline steps asynchronously"""
        execution_log = []

        while True:
            # Get executable steps
            executable_steps = pipeline.get_executable_steps()

            if not executable_steps:
                # Check if all steps are completed
                all_completed = all(
                    step.status in [PipelineStatus.SUCCESS, PipelineStatus.FAILED, PipelineStatus.CANCELLED]
                    for step in pipeline.steps.values()
                )

                if all_completed:
                    break

                # Wait for steps to complete
                await asyncio.sleep(1)
                continue

            # Execute steps concurrently if no dependencies conflict
            execution_tasks = []

            for step in executable_steps:
                task = asyncio.create_task(self.execute_step(pipeline, step))
                execution_tasks.append(task)

            # Wait for all executable steps to complete
            await asyncio.gather(*execution_tasks)

        return {
            'pipeline_id': pipeline.pipeline_id,
            'status': pipeline.status.value,
            'execution_log': execution_log,
            'completed_at': datetime.now().isoformat()
        }

    async def execute_step(self, pipeline: DeploymentPipeline, step: PipelineStep) -> None:
        """Execute individual pipeline step"""
        self.logger.info(f"Executing step: {step.name}")

        pipeline.update_step_status(step.name, PipelineStatus.RUNNING)

        try:
            # Execute step command
            process = await asyncio.create_subprocess_shell(
                step.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=step.timeout
                )

                exit_code = process.returncode
                output = stdout.decode()
                error = stderr.decode()

                if exit_code == 0:
                    pipeline.update_step_status(
                        step.name, PipelineStatus.SUCCESS,
                        exit_code, output, error
                    )
                    self.logger.info(f"Step {step.name} completed successfully")
                else:
                    pipeline.update_step_status(
                        step.name, PipelineStatus.FAILED,
                        exit_code, output, error
                    )
                    self.logger.error(f"Step {step.name} failed with exit code {exit_code}")

            except asyncio.TimeoutError:
                process.kill()
                pipeline.update_step_status(
                    step.name, PipelineStatus.FAILED,
                    -1, "", f"Step timed out after {step.timeout} seconds"
                )
                self.logger.error(f"Step {step.name} timed out")

        except Exception as e:
            pipeline.update_step_status(
                step.name, PipelineStatus.FAILED,
                -1, "", str(e)
            )
            self.logger.error(f"Step {step.name} execution error: {e}")

    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel running pipeline"""
        if pipeline_id in self.active_pipelines:
            task = self.active_pipelines[pipeline_id]
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Update pipeline status
            if pipeline_id in self.pipelines:
                pipeline = self.pipelines[pipeline_id]
                pipeline.status = PipelineStatus.CANCELLED
                pipeline.completed_at = datetime.now()

            del self.active_pipelines[pipeline_id]
            self.logger.info(f"Cancelled pipeline: {pipeline_id}")
            return True

        return False

    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline execution status"""
        if pipeline_id in self.pipelines:
            pipeline = self.pipelines[pipeline_id]
            return pipeline.get_pipeline_status()
        return None

    def list_pipelines(self, status_filter: Optional[PipelineStatus] = None) -> List[Dict[str, Any]]:
        """List all pipelines with optional status filter"""
        pipelines = []

        for pipeline in self.pipelines.values():
            pipeline_status = pipeline.get_pipeline_status()

            if status_filter is None or pipeline.status == status_filter:
                pipelines.append(pipeline_status)

        return pipelines

    def register_pipeline_hook(self, event: str, callback: Callable) -> None:
        """Register pipeline event hook"""
        if event not in self.pipeline_hooks:
            self.pipeline_hooks[event] = []

        self.pipeline_hooks[event].append(callback)

    async def trigger_pipeline_event(self, event: str, pipeline_id: str, data: Dict[str, Any]) -> None:
        """Trigger pipeline event hooks"""
        if event in self.pipeline_hooks:
            for callback in self.pipeline_hooks[event]:
                try:
                    await callback(pipeline_id, data)
                except Exception as e:
                    self.logger.error(f"Pipeline hook error: {e}")
```

### Phase 2: Container Orchestration Management

#### 2.1 Kubernetes Deployment Manager
```python
import kubernetes.client as k8s_client
from kubernetes.client.rest import ApiException
from typing import Dict, List, Any, Optional
import yaml
import logging
from datetime import datetime

class KubernetesDeploymentManager:
    """Kubernetes deployment management for the platform"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Kubernetes deployment manager"""
        self.config = config
        self.logger = logging.getLogger('K8sDeploymentManager')

        # Initialize Kubernetes clients
        configuration = k8s_client.Configuration()
        configuration.host = config.get('kubernetes_host', 'https://localhost:6443')

        if 'kubernetes_token' in config:
            configuration.api_key = {'authorization': f"Bearer {config['kubernetes_token']}"}
        elif 'kubernetes_cert' in config:
            # Certificate-based authentication
            configuration.cert_file = config.get('kubernetes_cert')
            configuration.key_file = config.get('kubernetes_key')
            if 'kubernetes_ca' in config:
                configuration.ssl_ca_cert = config['kubernetes_ca']

        self.apps_v1 = k8s_client.AppsV1Api(k8s_client.ApiClient(configuration))
        self.core_v1 = k8s_client.CoreV1Api(k8s_client.ApiClient(configuration))
        self.networking_v1 = k8s_client.NetworkingV1Api(k8s_client.ApiClient(configuration))

        self.namespace = config.get('namespace', 'active-inference')

    def create_namespace_if_not_exists(self) -> None:
        """Create namespace if it doesn't exist"""
        try:
            # Check if namespace exists
            self.core_v1.read_namespace(self.namespace)
            self.logger.info(f"Namespace {self.namespace} already exists")
        except ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace = k8s_client.V1Namespace(
                    metadata=k8s_client.V1ObjectMeta(name=self.namespace)
                )
                self.core_v1.create_namespace(namespace)
                self.logger.info(f"Created namespace: {self.namespace}")
            else:
                raise

    def deploy_application(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy application to Kubernetes"""
        app_name = deployment_config['name']
        image = deployment_config['image']
        replicas = deployment_config.get('replicas', 1)
        port = deployment_config.get('port', 8000)

        self.logger.info(f"Deploying application: {app_name}")

        # Create deployment
        deployment = self.create_deployment_manifest(app_name, image, replicas, port)
        self.apps_v1.create_namespaced_deployment(self.namespace, deployment)

        # Create service
        service = self.create_service_manifest(app_name, port)
        self.core_v1.create_namespaced_service(self.namespace, service)

        # Create ingress if configured
        if deployment_config.get('ingress', False):
            ingress = self.create_ingress_manifest(app_name, deployment_config)
            self.networking_v1.create_namespaced_ingress(self.namespace, ingress)

        # Wait for deployment to be ready
        self.wait_for_deployment_ready(app_name, timeout=300)

        return {
            'deployment_name': app_name,
            'namespace': self.namespace,
            'replicas': replicas,
            'status': 'deployed',
            'timestamp': datetime.now().isoformat()
        }

    def create_deployment_manifest(self, name: str, image: str, replicas: int, port: int) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest"""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': name,
                'labels': {
                    'app': name,
                    'component': 'active-inference'
                }
            },
            'spec': {
                'replicas': replicas,
                'selector': {
                    'matchLabels': {
                        'app': name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': name,
                            'component': 'active-inference'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': name,
                            'image': image,
                            'ports': [{
                                'containerPort': port
                            }],
                            'env': self.get_environment_variables(),
                            'resources': self.get_resource_limits(),
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': port
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': port
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }

    def create_service_manifest(self, name: str, port: int) -> Dict[str, Any]:
        """Create Kubernetes service manifest"""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{name}-service",
                'labels': {
                    'app': name,
                    'component': 'active-inference'
                }
            },
            'spec': {
                'selector': {
                    'app': name
                },
                'ports': [{
                    'name': 'http',
                    'port': port,
                    'targetPort': port,
                    'protocol': 'TCP'
                }],
                'type': 'ClusterIP'
            }
        }

    def create_ingress_manifest(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Kubernetes ingress manifest"""
        domain = config.get('domain', f"{name}.example.com")

        return {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': f"{name}-ingress",
                'annotations': {
                    'kubernetes.io/ingress.class': 'nginx',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
                }
            },
            'spec': {
                'tls': [{
                    'hosts': [domain],
                    'secretName': f"{name}-tls"
                }],
                'rules': [{
                    'host': domain,
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': f"{name}-service",
                                    'port': {
                                        'number': config.get('port', 8000)
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }

    def get_environment_variables(self) -> List[Dict[str, str]]:
        """Get environment variables for containers"""
        return [
            {'name': 'ENVIRONMENT', 'value': 'production'},
            {'name': 'LOG_LEVEL', 'value': 'INFO'},
            {'name': 'NAMESPACE', 'value': self.namespace}
        ]

    def get_resource_limits(self) -> Dict[str, Dict[str, str]]:
        """Get resource limits for containers"""
        return {
            'requests': {
                'cpu': '100m',
                'memory': '128Mi'
            },
            'limits': {
                'cpu': '500m',
                'memory': '512Mi'
            }
        }

    def wait_for_deployment_ready(self, deployment_name: str, timeout: int = 300) -> None:
        """Wait for deployment to be ready"""
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(
                    deployment_name, self.namespace
                )

                ready_replicas = deployment.status.ready_replicas or 0
                desired_replicas = deployment.spec.replicas

                if ready_replicas == desired_replicas:
                    self.logger.info(f"Deployment {deployment_name} is ready")
                    return

                self.logger.debug(f"Waiting for deployment {deployment_name}: "
                                f"{ready_replicas}/{desired_replicas} ready")

            except ApiException as e:
                self.logger.warning(f"Error checking deployment status: {e}")

            time.sleep(5)

        raise TimeoutError(f"Deployment {deployment_name} did not become ready within {timeout} seconds")

    def scale_deployment(self, deployment_name: str, replicas: int) -> Dict[str, Any]:
        """Scale deployment to specified number of replicas"""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                deployment_name, self.namespace
            )

            # Update replicas
            deployment.spec.replicas = replicas

            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                deployment_name, self.namespace, deployment
            )

            # Wait for scaling to complete
            self.wait_for_deployment_ready(deployment_name)

            return {
                'deployment': deployment_name,
                'new_replicas': replicas,
                'status': 'scaled',
                'timestamp': datetime.now().isoformat()
            }

        except ApiException as e:
            self.logger.error(f"Failed to scale deployment {deployment_name}: {e}")
            raise

    def rollback_deployment(self, deployment_name: str, to_revision: Optional[int] = None) -> Dict[str, Any]:
        """Rollback deployment to previous revision"""
        try:
            # Create rollback spec
            rollback_spec = k8s_client.V1beta1RollbackConfig(
                revision=to_revision
            ) if to_revision else None

            # Perform rollback
            rollback = self.apps_v1.create_namespaced_deployment_rollback(
                deployment_name,
                self.namespace,
                {
                    'name': deployment_name,
                    'rollbackTo': rollback_spec
                }
            )

            # Wait for rollback to complete
            self.wait_for_deployment_ready(deployment_name)

            return {
                'deployment': deployment_name,
                'rollback_to_revision': to_revision,
                'status': 'rolled_back',
                'timestamp': datetime.now().isoformat()
            }

        except ApiException as e:
            self.logger.error(f"Failed to rollback deployment {deployment_name}: {e}")
            raise

    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get deployment status"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                deployment_name, self.namespace
            )

            return {
                'name': deployment_name,
                'namespace': self.namespace,
                'replicas': deployment.spec.replicas,
                'ready_replicas': deployment.status.ready_replicas,
                'available_replicas': deployment.status.available_replicas,
                'unavailable_replicas': deployment.status.unavailable_replicas,
                'conditions': [
                    {
                        'type': condition.type,
                        'status': condition.status,
                        'reason': condition.reason,
                        'message': condition.message
                    }
                    for condition in deployment.status.conditions or []
                ]
            }

        except ApiException as e:
            self.logger.error(f"Failed to get deployment status: {e}")
            raise

    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments in namespace"""
        try:
            deployments = self.apps_v1.list_namespaced_deployment(self.namespace)

            deployment_list = []
            for deployment in deployments.items:
                deployment_list.append({
                    'name': deployment.metadata.name,
                    'replicas': deployment.spec.replicas,
                    'ready_replicas': deployment.status.ready_replicas,
                    'created': deployment.metadata.creation_timestamp.isoformat()
                })

            return deployment_list

        except ApiException as e:
            self.logger.error(f"Failed to list deployments: {e}")
            return []

    def delete_deployment(self, deployment_name: str, delete_services: bool = True) -> None:
        """Delete deployment and optionally associated services"""
        try:
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                deployment_name, self.namespace
            )

            # Delete service
            if delete_services:
                try:
                    self.core_v1.delete_namespaced_service(
                        f"{deployment_name}-service", self.namespace
                    )
                except ApiException:
                    pass  # Service might not exist

            # Delete ingress
            try:
                self.networking_v1.delete_namespaced_ingress(
                    f"{deployment_name}-ingress", self.namespace
                )
            except ApiException:
                pass  # Ingress might not exist

            self.logger.info(f"Deleted deployment: {deployment_name}")

        except ApiException as e:
            self.logger.error(f"Failed to delete deployment {deployment_name}: {e}")
            raise
```

### Phase 3: Auto-scaling Management

#### 3.1 Intelligent Auto-scaler
```python
from typing import Dict, List, Any, Optional, Callable
import asyncio
import logging
from datetime import datetime, timedelta
import statistics

class ScalingMetric:
    """Represents a scaling metric"""

    def __init__(self, name: str, value: float, timestamp: datetime,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize scaling metric"""
        self.name = name
        self.value = value
        self.timestamp = timestamp
        self.metadata = metadata or {}

class ScalingDecision:
    """Represents a scaling decision"""

    def __init__(self, service_name: str, action: str, target_replicas: int,
                 reason: str, confidence: float):
        """Initialize scaling decision"""
        self.service_name = service_name
        self.action = action  # 'scale_up', 'scale_down', 'no_action'
        self.target_replicas = target_replicas
        self.reason = reason
        self.confidence = confidence
        self.timestamp = datetime.now()

class AutoScaler:
    """Intelligent auto-scaling system for platform services"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize auto-scaler"""
        self.config = config
        self.logger = logging.getLogger('AutoScaler')

        # Scaling configuration
        self.min_replicas = config.get('min_replicas', 1)
        self.max_replicas = config.get('max_replicas', 10)
        self.scale_up_threshold = config.get('scale_up_threshold', 70.0)  # CPU %
        self.scale_down_threshold = config.get('scale_down_threshold', 30.0)  # CPU %
        self.cooldown_period = config.get('cooldown_period', 300)  # seconds

        # Metrics and decision tracking
        self.metrics_history: List[ScalingMetric] = []
        self.decisions_history: List[ScalingDecision] = []
        self.last_scaling_actions: Dict[str, datetime] = {}

        # Scaling policies
        self.scaling_policies: Dict[str, Callable] = {
            'cpu_based': self.cpu_based_scaling_policy,
            'request_based': self.request_based_scaling_policy,
            'latency_based': self.latency_based_scaling_policy,
            'custom': self.custom_scaling_policy
        }

        # Kubernetes client for scaling
        self.k8s_client = KubernetesDeploymentManager(config)

    async def start_auto_scaling(self) -> None:
        """Start auto-scaling monitoring loop"""
        self.logger.info("Starting auto-scaling service")

        while True:
            try:
                await self.scaling_cycle()
                await asyncio.sleep(self.config.get('check_interval', 60))  # Check every 60 seconds

            except Exception as e:
                self.logger.error(f"Auto-scaling cycle error: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    async def scaling_cycle(self) -> None:
        """Execute one complete scaling cycle"""
        # Collect metrics from all services
        services = await self.get_monitored_services()
        metrics = await self.collect_service_metrics(services)

        # Make scaling decisions
        decisions = []
        for service_name, service_metrics in metrics.items():
            decision = await self.make_scaling_decision(service_name, service_metrics)
            if decision:
                decisions.append(decision)

        # Execute scaling decisions
        for decision in decisions:
            await self.execute_scaling_decision(decision)

        # Cleanup old metrics and decisions
        self.cleanup_old_data()

    async def get_monitored_services(self) -> List[str]:
        """Get list of services being monitored for scaling"""
        # In practice, this would query Kubernetes for deployments
        # with auto-scaling annotations
        return [
            'knowledge-graph-service',
            'search-service',
            'api-gateway',
            'visualization-service'
        ]

    async def collect_service_metrics(self, services: List[str]) -> Dict[str, Dict[str, Any]]:
        """Collect current metrics for all services"""
        metrics = {}

        for service in services:
            try:
                service_metrics = await self.get_service_metrics(service)
                metrics[service] = service_metrics

                # Store metrics for analysis
                for metric_name, value in service_metrics.items():
                    if isinstance(value, (int, float)):
                        metric = ScalingMetric(
                            name=f"{service}.{metric_name}",
                            value=float(value),
                            timestamp=datetime.now(),
                            metadata={'service': service}
                        )
                        self.metrics_history.append(metric)

            except Exception as e:
                self.logger.error(f"Failed to collect metrics for {service}: {e}")

        return metrics

    async def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get current metrics for a specific service"""
        # In practice, this would query monitoring systems like Prometheus
        # For now, simulate metrics

        # Simulate CPU usage (40-90%)
        cpu_usage = 40 + (50 * (hash(service_name + str(datetime.now().hour)) % 100) / 100)

        # Simulate memory usage (30-85%)
        memory_usage = 30 + (55 * (hash(service_name + str(datetime.now().minute)) % 100) / 100)

        # Simulate request rate (10-200 req/min)
        request_rate = 10 + (190 * (hash(service_name) % 100) / 100)

        # Simulate response time (50-500ms)
        response_time = 50 + (450 * (hash(service_name + "response") % 100) / 100)

        return {
            'cpu_percent': cpu_usage,
            'memory_percent': memory_usage,
            'request_rate': request_rate,
            'response_time_ms': response_time,
            'active_connections': 10 + (hash(service_name) % 90),
            'error_rate': (hash(service_name) % 100) / 1000.0  # 0-0.1
        }

    async def make_scaling_decision(self, service_name: str, metrics: Dict[str, Any]) -> Optional[ScalingDecision]:
        """Make scaling decision for a service"""

        # Check cooldown period
        if self.is_in_cooldown_period(service_name):
            return None

        current_replicas = await self.get_current_replicas(service_name)

        # Apply scaling policies
        for policy_name, policy_func in self.scaling_policies.items():
            if self.should_apply_policy(policy_name, service_name):
                decision = await policy_func(service_name, metrics, current_replicas)
                if decision:
                    return decision

        return None

    def is_in_cooldown_period(self, service_name: str) -> bool:
        """Check if service is in cooldown period after scaling"""
        last_action = self.last_scaling_actions.get(service_name)
        if last_action:
            time_since_action = (datetime.now() - last_action).total_seconds()
            return time_since_action < self.cooldown_period
        return False

    async def get_current_replicas(self, service_name: str) -> int:
        """Get current number of replicas for service"""
        try:
            status = self.k8s_client.get_deployment_status(service_name)
            return status.get('ready_replicas', 1)
        except Exception:
            return 1  # Default to 1 if unable to determine

    async def cpu_based_scaling_policy(self, service_name: str, metrics: Dict[str, Any],
                                     current_replicas: int) -> Optional[ScalingDecision]:
        """CPU-based scaling policy"""
        cpu_percent = metrics.get('cpu_percent', 0)

        if cpu_percent > self.scale_up_threshold and current_replicas < self.max_replicas:
            target_replicas = min(current_replicas + 1, self.max_replicas)
            return ScalingDecision(
                service_name=service_name,
                action='scale_up',
                target_replicas=target_replicas,
                reason=f"CPU usage {cpu_percent:.1f}% exceeds threshold {self.scale_up_threshold}%",
                confidence=min(0.9, cpu_percent / 100.0)
            )

        elif cpu_percent < self.scale_down_threshold and current_replicas > self.min_replicas:
            target_replicas = max(current_replicas - 1, self.min_replicas)
            return ScalingDecision(
                service_name=service_name,
                action='scale_down',
                target_replicas=target_replicas,
                reason=f"CPU usage {cpu_percent:.1f}% below threshold {self.scale_down_threshold}%",
                confidence=min(0.8, (50 - cpu_percent) / 50.0)
            )

        return None

    async def request_based_scaling_policy(self, service_name: str, metrics: Dict[str, Any],
                                         current_replicas: int) -> Optional[ScalingDecision]:
        """Request rate based scaling policy"""
        request_rate = metrics.get('request_rate', 0)
        scale_up_threshold = self.config.get('request_scale_up_threshold', 100)  # req/min
        scale_down_threshold = self.config.get('request_scale_down_threshold', 20)  # req/min

        if request_rate > scale_up_threshold and current_replicas < self.max_replicas:
            target_replicas = min(current_replicas + 1, self.max_replicas)
            return ScalingDecision(
                service_name=service_name,
                action='scale_up',
                target_replicas=target_replicas,
                reason=f"Request rate {request_rate:.1f}/min exceeds threshold {scale_up_threshold}",
                confidence=min(0.85, request_rate / (scale_up_threshold * 2))
            )

        elif request_rate < scale_down_threshold and current_replicas > self.min_replicas:
            target_replicas = max(current_replicas - 1, self.min_replicas)
            return ScalingDecision(
                service_name=service_name,
                action='scale_down',
                target_replicas=target_replicas,
                reason=f"Request rate {request_rate:.1f}/min below threshold {scale_down_threshold}",
                confidence=min(0.75, (scale_down_threshold - request_rate) / scale_down_threshold)
            )

        return None

    async def latency_based_scaling_policy(self, service_name: str, metrics: Dict[str, Any],
                                         current_replicas: int) -> Optional[ScalingDecision]:
        """Response time based scaling policy"""
        response_time = metrics.get('response_time_ms', 0)
        latency_threshold = self.config.get('latency_scale_up_threshold', 300)  # ms

        if response_time > latency_threshold and current_replicas < self.max_replicas:
            target_replicas = min(current_replicas + 1, self.max_replicas)
            return ScalingDecision(
                service_name=service_name,
                action='scale_up',
                target_replicas=target_replicas,
                reason=f"Response time {response_time:.1f}ms exceeds threshold {latency_threshold}ms",
                confidence=min(0.8, response_time / (latency_threshold * 2))
            )

        return None

    async def custom_scaling_policy(self, service_name: str, metrics: Dict[str, Any],
                                  current_replicas: int) -> Optional[ScalingDecision]:
        """Custom scaling policy based on multiple metrics"""
        # Implement more sophisticated scaling logic
        # Consider multiple metrics together for better decisions

        cpu_weight = 0.4
        memory_weight = 0.3
        request_weight = 0.3

        cpu_score = metrics.get('cpu_percent', 0) / 100.0
        memory_score = metrics.get('memory_percent', 0) / 100.0
        request_score = min(metrics.get('request_rate', 0) / 200.0, 1.0)  # Normalize to 200 req/min max

        combined_score = (
            cpu_score * cpu_weight +
            memory_score * memory_weight +
            request_score * request_weight
        )

        scale_up_threshold = 0.7
        scale_down_threshold = 0.3

        if combined_score > scale_up_threshold and current_replicas < self.max_replicas:
            target_replicas = min(current_replicas + 1, self.max_replicas)
            return ScalingDecision(
                service_name=service_name,
                action='scale_up',
                target_replicas=target_replicas,
                reason=f"Combined load score {combined_score:.2f} exceeds threshold {scale_up_threshold}",
                confidence=min(0.9, combined_score)
            )

        elif combined_score < scale_down_threshold and current_replicas > self.min_replicas:
            target_replicas = max(current_replicas - 1, self.min_replicas)
            return ScalingDecision(
                service_name=service_name,
                action='scale_down',
                target_replicas=target_replicas,
                reason=f"Combined load score {combined_score:.2f} below threshold {scale_down_threshold}",
                confidence=min(0.7, 1.0 - combined_score)
            )

        return None

    def should_apply_policy(self, policy_name: str, service_name: str) -> bool:
        """Determine if a scaling policy should be applied"""
        # Allow multiple policies to be active
        # In practice, you might have service-specific policy configurations
        return True

    async def execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute a scaling decision"""
        try:
            self.logger.info(f"Executing scaling decision: {decision.service_name} "
                           f"{decision.action} to {decision.target_replicas} replicas "
                           f"(reason: {decision.reason})")

            # Execute scaling via Kubernetes
            result = self.k8s_client.scale_deployment(
                decision.service_name,
                decision.target_replicas
            )

            # Record scaling action
            self.last_scaling_actions[decision.service_name] = datetime.now()
            self.decisions_history.append(decision)

            self.logger.info(f"Scaling completed: {result}")

        except Exception as e:
            self.logger.error(f"Failed to execute scaling decision: {e}")

    def cleanup_old_data(self) -> None:
        """Clean up old metrics and decisions"""
        cutoff_time = datetime.now() - timedelta(hours=24)

        # Clean old metrics
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]

        # Clean old decisions (keep more history for decisions)
        decision_cutoff = datetime.now() - timedelta(days=7)
        self.decisions_history = [
            d for d in self.decisions_history
            if d.timestamp > decision_cutoff
        ]

    def get_scaling_history(self, service_name: Optional[str] = None,
                          hours: int = 24) -> List[ScalingDecision]:
        """Get scaling decision history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        decisions = [
            d for d in self.decisions_history
            if d.timestamp > cutoff_time
        ]

        if service_name:
            decisions = [d for d in decisions if d.service_name == service_name]

        return decisions

    def get_scaling_statistics(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get scaling statistics"""
        decisions = self.get_scaling_history(service_name, hours=168)  # Last 7 days

        if not decisions:
            return {'message': 'No scaling decisions in the last week'}

        scale_up_count = sum(1 for d in decisions if d.action == 'scale_up')
        scale_down_count = sum(1 for d in decisions if d.action == 'scale_down')

        avg_confidence = statistics.mean(d.confidence for d in decisions)

        return {
            'total_decisions': len(decisions),
            'scale_up_decisions': scale_up_count,
            'scale_down_decisions': scale_down_count,
            'average_confidence': avg_confidence,
            'time_period': 'last_7_days'
        }

    def predict_optimal_replicas(self, service_name: str, forecast_hours: int = 24) -> Dict[str, Any]:
        """Predict optimal number of replicas based on historical data"""
        # Simple prediction based on historical scaling patterns
        recent_decisions = self.get_scaling_history(service_name, hours=forecast_hours)

        if not recent_decisions:
            return {
                'recommended_replicas': self.min_replicas,
                'confidence': 0.5,
                'reason': 'No historical data available'
            }

        # Analyze scaling patterns
        scale_up_trend = sum(1 for d in recent_decisions[-10:] if d.action == 'scale_up')
        scale_down_trend = sum(1 for d in recent_decisions[-10:] if d.action == 'scale_down')

        current_replicas = self.get_current_replicas(service_name) if asyncio.iscoroutinefunction(self.get_current_replicas) else 1

        if scale_up_trend > scale_down_trend:
            recommended = min(current_replicas + 1, self.max_replicas)
            confidence = min(0.8, scale_up_trend / 10)
            reason = f"Recent upscaling trend ({scale_up_trend} vs {scale_down_trend} down)"
        elif scale_down_trend > scale_up_trend:
            recommended = max(current_replicas - 1, self.min_replicas)
            confidence = min(0.8, scale_down_trend / 10)
            reason = f"Recent downscaling trend ({scale_down_trend} vs {scale_up_trend} up)"
        else:
            recommended = current_replicas
            confidence = 0.6
            reason = "Stable scaling pattern"

        return {
            'recommended_replicas': recommended,
            'confidence': confidence,
            'reason': reason,
            'forecast_period': f'{forecast_hours}_hours'
        }
```

---

**"Active Inference for, with, by Generative AI"** - Building robust deployment and scaling infrastructure that ensures reliable, high-performance platform operations with intelligent automation and comprehensive monitoring.
