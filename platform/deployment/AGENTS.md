# Platform Deployment - Agent Development Guide

**Guidelines for AI agents working with platform deployment and infrastructure.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with deployment systems:**

### Primary Responsibilities
- **Infrastructure Management**: Deploy and manage platform infrastructure
- **Container Orchestration**: Kubernetes and Docker deployment
- **Environment Management**: Development, staging, production environments
- **Scaling Systems**: Auto-scaling and load balancing
- **Monitoring**: Health monitoring and alerting
- **Backup & Recovery**: Data backup and disaster recovery

### Development Focus Areas
1. **Containerization**: Docker and container management
2. **Orchestration**: Kubernetes deployment and management
3. **CI/CD Pipelines**: Continuous integration and deployment
4. **Monitoring**: Health checks and performance monitoring
5. **Security**: Secure deployment and access control

## 🏗️ Architecture & Integration

### Deployment Architecture

**Understanding the deployment system structure:**

```
Infrastructure Layer
├── Container Runtime (Docker)
├── Orchestration (Kubernetes)
├── Load Balancing (Nginx, HAProxy)
├── Storage (Persistent volumes)
└── Networking (Service mesh, ingress)
```

### Integration Points

**Key integration points for deployment:**

#### Platform Services
- **Web Services**: Application deployment and scaling
- **Database**: Data persistence and backup
- **Cache**: Redis and caching layer
- **Search**: Elasticsearch deployment
- **Monitoring**: Metrics collection and alerting

#### Infrastructure
- **Cloud Providers**: AWS, GCP, Azure integration
- **Container Registry**: Docker image management
- **CDN**: Content delivery network
- **DNS**: Domain name system
- **Security**: Firewalls and access control

### Deployment Pipeline

```bash
# Development workflow
code_changes → tests → build → deploy_dev → validate

# Production workflow
release → tests → build → deploy_staging → validate → deploy_production
```

## 💻 Development Patterns

### Required Implementation Patterns

**All deployment development must follow these patterns:**

#### 1. Container Management Pattern
```python
class ContainerManager:
    """Manage Docker containers and images"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docker_client = DockerClient(config)
        self.image_registry = ImageRegistry(config)

    async def build_image(self, service_name: str, tag: str = "latest") -> str:
        """Build Docker image for service"""
        # Validate Dockerfile
        await self.validate_dockerfile(service_name)

        # Build image
        image_id = await self.docker_client.build_image(
            service_name=service_name,
            dockerfile_path=f"deployment/{service_name}/Dockerfile",
            tag=tag
        )

        # Push to registry
        await self.image_registry.push_image(image_id, tag)

        return image_id

    async def deploy_service(self, service_name: str, image_tag: str) -> DeploymentResult:
        """Deploy service with specified image"""
        # Validate deployment configuration
        config = await self.load_deployment_config(service_name)

        # Create deployment
        deployment = await self.create_kubernetes_deployment(
            service_name, image_tag, config
        )

        # Wait for readiness
        await self.wait_for_deployment_ready(deployment)

        return {"success": True, "deployment_id": deployment.id}
```

#### 2. Environment Management Pattern
```python
class EnvironmentManager:
    """Manage deployment environments"""

    async def create_environment(self, env_name: str, config: Dict) -> Environment:
        """Create new deployment environment"""
        # Validate environment configuration
        await self.validate_environment_config(config)

        # Create Kubernetes namespace
        namespace = await self.k8s_client.create_namespace(env_name)

        # Deploy environment services
        await self.deploy_environment_services(namespace, config)

        # Configure monitoring
        await self.setup_environment_monitoring(namespace)

        return Environment(namespace, config)

    async def scale_environment(self, env_name: str, scaling_config: Dict) -> bool:
        """Scale environment resources"""
        # Validate scaling configuration
        await self.validate_scaling_config(scaling_config)

        # Scale services
        for service, scale_config in scaling_config.items():
            await self.scale_service(env_name, service, scale_config)

        # Update load balancers
        await self.update_load_balancers(env_name, scaling_config)

        return True
```

#### 3. Monitoring Pattern
```python
class DeploymentMonitor:
    """Monitor deployment health and performance"""

    async def setup_monitoring(self, deployment: Deployment) -> None:
        """Set up monitoring for deployment"""
        # Health checks
        await self.setup_health_checks(deployment)

        # Metrics collection
        await self.setup_metrics_collection(deployment)

        # Alerting rules
        await self.setup_alerting_rules(deployment)

        # Log aggregation
        await self.setup_log_aggregation(deployment)

    async def monitor_deployment(self, deployment_id: str) -> MonitoringData:
        """Monitor deployment performance and health"""
        # Get deployment status
        status = await self.get_deployment_status(deployment_id)

        # Check health endpoints
        health = await self.check_health_endpoints(deployment_id)

        # Collect metrics
        metrics = await self.collect_metrics(deployment_id)

        # Analyze performance
        analysis = await self.analyze_performance(metrics)

        return {
            "status": status,
            "health": health,
            "metrics": metrics,
            "analysis": analysis
        }
```

## 🧪 Testing Standards

### Test Categories (MANDATORY)

#### 1. Deployment Testing
```python
class TestDeployment:
    """Test deployment functionality"""

    async def test_container_build(self):
        """Test Docker image building"""
        # Build test image
        image_id = await self.container_manager.build_image("test_service")

        # Validate image
        validation = await self.validate_image(image_id)
        assert validation["valid"] == True

        # Test image functionality
        test_result = await self.test_image_functionality(image_id)
        assert test_result["success"] == True

    async def test_kubernetes_deployment(self):
        """Test Kubernetes deployment"""
        # Create test deployment
        deployment_config = create_test_deployment_config()
        deployment = await self.k8s_manager.create_deployment(deployment_config)

        # Wait for readiness
        await self.wait_for_deployment_ready(deployment)

        # Validate deployment
        status = await self.get_deployment_status(deployment.id)
        assert status["ready"] == True
        assert status["replicas"] == deployment_config["replicas"]
```

#### 2. Scaling Testing
```python
class TestScaling:
    """Test scaling functionality"""

    async def test_horizontal_scaling(self):
        """Test horizontal scaling"""
        # Initial deployment
        deployment = await self.create_deployment(replicas=2)

        # Scale up
        await self.scale_deployment(deployment.id, replicas=5)
        status = await self.get_deployment_status(deployment.id)
        assert status["replicas"] == 5

        # Scale down
        await self.scale_deployment(deployment.id, replicas=2)
        status = await self.get_deployment_status(deployment.id)
        assert status["replicas"] == 2

    async def test_auto_scaling(self):
        """Test auto-scaling functionality"""
        # Configure auto-scaling
        scaling_config = {
            "min_replicas": 2,
            "max_replicas": 10,
            "target_cpu": 70
        }

        await self.configure_auto_scaling("test_service", scaling_config)

        # Generate load
        await self.generate_cpu_load("test_service", 80)

        # Wait for scaling
        await self.wait_for_scaling_event()

        # Validate scaling
        status = await self.get_deployment_status("test_service")
        assert scaling_config["min_replicas"] <= status["replicas"] <= scaling_config["max_replicas"]
```

#### 3. Monitoring Testing
```python
class TestMonitoring:
    """Test monitoring functionality"""

    async def test_health_checks(self):
        """Test health check functionality"""
        # Set up health checks
        await self.setup_health_checks("test_deployment")

        # Test health endpoints
        health_status = await self.check_health_endpoints("test_deployment")

        assert health_status["overall"] == "healthy"
        assert all(check["status"] == "passing" for check in health_status["checks"])

    async def test_metrics_collection(self):
        """Test metrics collection"""
        # Generate test metrics
        await self.generate_test_metrics()

        # Collect metrics
        metrics = await self.collect_deployment_metrics("test_deployment")

        # Validate metrics
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "response_time" in metrics
        assert all(isinstance(value, (int, float)) for value in metrics.values())
```

## 📖 Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. Deployment Documentation
**All deployment configurations must be documented:**

```yaml
# deployment/production.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: platform-web
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: platform-web
  template:
    metadata:
      labels:
        app: platform-web
    spec:
      containers:
      - name: platform
        image: active-inference/platform:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: platform-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

#### 2. Environment Documentation
**Environment configurations must be documented:**

```python
# Environment configuration schema
environment_config_schema = {
    "development": {
        "description": "Development environment for testing",
        "replicas": 1,
        "resources": {"memory": "512Mi", "cpu": "250m"},
        "debug": True,
        "monitoring": False
    },
    "staging": {
        "description": "Staging environment for validation",
        "replicas": 2,
        "resources": {"memory": "1Gi", "cpu": "500m"},
        "debug": False,
        "monitoring": True
    },
    "production": {
        "description": "Production environment",
        "replicas": 5,
        "resources": {"memory": "2Gi", "cpu": "1000m"},
        "debug": False,
        "monitoring": True,
        "backup": True
    }
}
```

## 🚀 Performance Optimization

### Performance Requirements

**Deployment system must meet these performance standards:**

- **Deployment Time**: <5 minutes for typical services
- **Scaling Response**: <2 minutes for scaling operations
- **Monitoring Latency**: <10 seconds for metric updates
- **Recovery Time**: <5 minutes for service recovery

### Optimization Techniques

#### 1. Image Optimization
```python
class ImageOptimizer:
    """Optimize Docker images"""

    def optimize_base_image(self, service_name: str) -> str:
        """Select optimal base image"""
        # Multi-stage builds
        if self.is_python_service(service_name):
            return "python:3.9-slim"
        elif self.is_node_service(service_name):
            return "node:18-alpine"
        else:
            return "ubuntu:20.04"

    def optimize_image_size(self, dockerfile: str) -> str:
        """Optimize image size"""
        # Layer optimization
        optimized_dockerfile = self.optimize_layers(dockerfile)

        # Package optimization
        optimized_dockerfile = self.optimize_packages(optimized_dockerfile)

        # Cleanup optimization
        optimized_dockerfile = self.add_cleanup_steps(optimized_dockerfile)

        return optimized_dockerfile
```

#### 2. Deployment Optimization
```python
class DeploymentOptimizer:
    """Optimize deployment performance"""

    async def optimize_deployment_strategy(self, service: str) -> DeploymentStrategy:
        """Select optimal deployment strategy"""
        if await self.is_stateless_service(service):
            return DeploymentStrategy.ROLLING_UPDATE
        elif await self.requires_downtime(service):
            return DeploymentStrategy.RECREATE
        else:
            return DeploymentStrategy.BLUE_GREEN

    async def optimize_resource_allocation(self, service: str) -> ResourceConfig:
        """Optimize resource allocation"""
        # Analyze historical usage
        usage_patterns = await self.analyze_usage_patterns(service)

        # Calculate optimal resources
        optimal_resources = self.calculate_optimal_resources(usage_patterns)

        # Set resource limits
        resource_config = {
            "requests": optimal_resources["baseline"],
            "limits": optimal_resources["peak"]
        }

        return resource_config
```

## 🔒 Security Standards

### Security Requirements (MANDATORY)

#### 1. Container Security
```python
class ContainerSecurity:
    """Secure container deployment"""

    def validate_image_security(self, image: DockerImage) -> SecurityReport:
        """Validate container image security"""
        # Vulnerability scanning
        vulnerabilities = self.scan_image_vulnerabilities(image)

        # Security compliance
        compliance = self.check_security_compliance(image)

        # Access control
        access_control = self.validate_image_access_control(image)

        return {
            "vulnerabilities": vulnerabilities,
            "compliance": compliance,
            "access_control": access_control
        }

    def secure_container_runtime(self, deployment: Deployment) -> None:
        """Secure container runtime environment"""
        # Runtime security
        self.enable_runtime_security(deployment)

        # Network security
        self.configure_network_policies(deployment)

        # Resource security
        self.set_security_limits(deployment)
```

#### 2. Infrastructure Security
```python
class InfrastructureSecurity:
    """Secure infrastructure deployment"""

    async def setup_network_security(self, environment: str) -> None:
        """Set up network security"""
        # Firewall rules
        await self.configure_firewall_rules(environment)

        # Security groups
        await self.configure_security_groups(environment)

        # VPN setup
        await self.setup_vpn_access(environment)

    async def configure_access_control(self, environment: str) -> None:
        """Configure access control"""
        # Role-based access
        await self.setup_rbac(environment)

        # Service accounts
        await self.create_service_accounts(environment)

        # API access control
        await self.configure_api_access(environment)
```

## 🐛 Debugging & Troubleshooting

### Debug Configuration

```python
# Enable deployment debugging
debug_config = {
    "debug_mode": True,
    "log_level": "DEBUG",
    "deployment_debug": True,
    "monitoring_debug": True,
    "scaling_debug": True
}
```

### Common Debugging Patterns

#### 1. Deployment Debugging
```python
class DeploymentDebugger:
    """Debug deployment issues"""

    async def debug_deployment_failure(self, deployment_id: str) -> DebugReport:
        """Debug failed deployment"""
        # Get deployment logs
        logs = await self.get_deployment_logs(deployment_id)

        # Check resource availability
        resources = await self.check_resource_availability()

        # Validate configuration
        config_validation = await self.validate_deployment_config(deployment_id)

        # Network connectivity
        network_check = await self.check_network_connectivity(deployment_id)

        return {
            "logs": logs,
            "resources": resources,
            "config": config_validation,
            "network": network_check
        }
```

## 🔄 Development Workflow

### Agent Development Process

1. **Task Assessment**
   - Understand infrastructure requirements
   - Analyze deployment constraints
   - Consider scalability needs

2. **Architecture Planning**
   - Design container architecture
   - Plan Kubernetes orchestration
   - Consider monitoring requirements

3. **Test-Driven Development**
   - Write infrastructure tests first
   - Test deployment automation
   - Validate scaling scenarios

4. **Implementation**
   - Implement container definitions
   - Add orchestration configuration
   - Implement monitoring

5. **Quality Assurance**
   - Test in multiple environments
   - Validate scaling performance
   - Ensure security compliance

6. **Integration**
   - Test with platform services
   - Validate production readiness
   - Performance optimization

### Code Review Checklist

**Before submitting deployment code for review:**

- [ ] **Container Tests**: Docker build and deployment tests
- [ ] **Orchestration Tests**: Kubernetes deployment validation
- [ ] **Scaling Tests**: Auto-scaling and load balancing tests
- [ ] **Monitoring Tests**: Health checks and metrics collection
- [ ] **Security Tests**: Container and infrastructure security
- [ ] **Performance Tests**: Deployment and scaling performance
- [ ] **Documentation**: Complete deployment and configuration docs

## 📚 Learning Resources

### Deployment Resources

- **[Kubernetes Best Practices](https://example.com/k8s)**: Container orchestration
- **[Docker Optimization](https://example.com/docker)**: Container optimization
- **[Infrastructure as Code](https://example.com/iac)**: Infrastructure automation
- **[Monitoring Systems](https://example.com/monitoring)**: System monitoring

### Platform Integration

- **[Platform Services](../../platform/README.md)**: Platform architecture
- **[Service Integration](../../../src/active_inference/platform/README.md)**: Service patterns
- **[Configuration Management](../../../tools/README.md)**: Configuration tools

## 🎯 Success Metrics

### Quality Metrics

- **Deployment Success Rate**: >99% successful deployments
- **Scaling Response Time**: <2 minutes for scaling operations
- **Monitoring Coverage**: 100% of critical services monitored
- **Security Compliance**: Zero critical vulnerabilities

### Development Metrics

- **Infrastructure Code Quality**: Clean, maintainable deployment code
- **Automation Coverage**: 100% automated deployment processes
- **Environment Consistency**: Identical staging and production environments
- **Recovery Capability**: <5 minute recovery from failures

---

**Component**: Platform Deployment | **Version**: 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Scalable infrastructure through automated deployment.
