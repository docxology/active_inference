"""
Platform - Deployment and Monitoring

Deployment management, service orchestration, and monitoring tools for the
Active Inference Knowledge Environment. Provides containerization, scaling,
health monitoring, and operational management capabilities.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health status of a service"""
    service_name: str
    status: ServiceStatus
    response_time: float = 0.0
    last_check: datetime = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now()


class MonitoringTools:
    """System monitoring and health checking"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_checks: Dict[str, ServiceHealth] = {}
        self.metrics: Dict[str, List[float]] = {}

        logger.info("MonitoringTools initialized")

    def register_service(self, service_name: str) -> None:
        """Register a service for monitoring"""
        self.health_checks[service_name] = ServiceHealth(
            service_name=service_name,
            status=ServiceStatus.UNKNOWN
        )

        self.metrics[service_name] = []
        logger.info(f"Registered service for monitoring: {service_name}")

    def update_service_health(self, service_name: str, status: ServiceStatus,
                            response_time: float = 0.0, error_message: Optional[str] = None) -> None:
        """Update service health status"""
        if service_name not in self.health_checks:
            self.register_service(service_name)

        self.health_checks[service_name].status = status
        self.health_checks[service_name].response_time = response_time
        self.health_checks[service_name].last_check = datetime.now()
        self.health_checks[service_name].error_message = error_message

        # Store metric
        self.metrics[service_name].append(response_time)
        if len(self.metrics[service_name]) > 100:  # Keep last 100 measurements
            self.metrics[service_name].pop(0)

        logger.debug(f"Updated health for {service_name}: {status.value}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        if not self.health_checks:
            return {"status": "no_services", "services": {}}

        healthy_count = sum(1 for h in self.health_checks.values() if h.status == ServiceStatus.HEALTHY)
        total_count = len(self.health_checks)

        overall_status = ServiceStatus.HEALTHY
        if healthy_count < total_count:
            overall_status = ServiceStatus.DEGRADED
        if healthy_count == 0:
            overall_status = ServiceStatus.UNHEALTHY

        return {
            "overall_status": overall_status.value,
            "healthy_services": healthy_count,
            "total_services": total_count,
            "health_percentage": (healthy_count / total_count) * 100 if total_count > 0 else 0,
            "services": {
                name: {
                    "status": health.status.value,
                    "response_time": health.response_time,
                    "last_check": health.last_check.isoformat(),
                    "error_message": health.error_message
                }
                for name, health in self.health_checks.items()
            },
            "timestamp": datetime.now().isoformat()
        }

    def get_performance_metrics(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        if service_name:
            if service_name not in self.metrics:
                return {"error": f"Service {service_name} not found"}

            metrics = self.metrics[service_name]
            if not metrics:
                return {"service": service_name, "metrics": {}}

            return {
                "service": service_name,
                "metrics": {
                    "average_response_time": sum(metrics) / len(metrics),
                    "min_response_time": min(metrics),
                    "max_response_time": max(metrics),
                    "samples": len(metrics)
                }
            }

        # All services
        all_metrics = {}
        for name in self.metrics:
            all_metrics[name] = self.get_performance_metrics(name)["metrics"]

        return {"all_services": all_metrics}


class ServiceOrchestrator:
    """Orchestrates platform services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services: Dict[str, Dict[str, Any]] = {}
        self.dependencies: Dict[str, List[str]] = {}

        logger.info("ServiceOrchestrator initialized")

    def register_service(self, service_name: str, service_config: Dict[str, Any],
                        dependencies: List[str] = None) -> None:
        """Register a service"""
        self.services[service_name] = {
            "config": service_config,
            "status": "stopped",
            "started_at": None,
            "dependencies": dependencies or []
        }

        self.dependencies[service_name] = dependencies or []

        logger.info(f"Registered service: {service_name}")

    def start_service(self, service_name: str) -> bool:
        """Start a service"""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not found")
            return False

        service = self.services[service_name]

        # Check dependencies
        for dep in service["dependencies"]:
            if self.services.get(dep, {}).get("status") != "running":
                logger.error(f"Cannot start {service_name}: dependency {dep} not running")
                return False

        # Placeholder for actual service startup
        logger.info(f"Starting service: {service_name}")

        service["status"] = "running"
        service["started_at"] = datetime.now().isoformat()

        return True

    def stop_service(self, service_name: str) -> bool:
        """Stop a service"""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not found")
            return False

        service = self.services[service_name]

        # Placeholder for actual service shutdown
        logger.info(f"Stopping service: {service_name}")

        service["status"] = "stopped"
        service["started_at"] = None

        return True

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        return {
            "services": {
                name: {
                    "status": service["status"],
                    "started_at": service["started_at"],
                    "dependencies": service["dependencies"]
                }
                for name, service in self.services.items()
            },
            "total_services": len(self.services),
            "running_services": len([s for s in self.services.values() if s["status"] == "running"]),
            "timestamp": datetime.now().isoformat()
        }


class DeploymentManager:
    """Main deployment and operations management"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring = MonitoringTools(config.get("monitoring", {}))
        self.orchestrator = ServiceOrchestrator(config.get("orchestrator", {}))

        # Register core platform services
        self._register_core_services()

        logger.info("DeploymentManager initialized")

    def _register_core_services(self) -> None:
        """Register core platform services"""
        core_services = {
            "knowledge_graph": {
                "port": 8001,
                "health_endpoint": "/health",
                "dependencies": []
            },
            "search_engine": {
                "port": 8002,
                "health_endpoint": "/health",
                "dependencies": ["knowledge_graph"]
            },
            "visualization_engine": {
                "port": 8003,
                "health_endpoint": "/health",
                "dependencies": []
            },
            "collaboration_service": {
                "port": 8004,
                "health_endpoint": "/health",
                "dependencies": ["knowledge_graph"]
            }
        }

        for service_name, service_config in core_services.items():
            self.monitoring.register_service(service_name)
            self.orchestrator.register_service(service_name, service_config)

    def deploy_platform(self, environment: str = "development") -> Dict[str, Any]:
        """Deploy the platform in specified environment"""
        logger.info(f"Deploying platform in {environment} environment")

        # Start services in dependency order
        deployment_results = {}

        for service_name in ["knowledge_graph", "search_engine", "visualization_engine", "collaboration_service"]:
            success = self.orchestrator.start_service(service_name)
            deployment_results[service_name] = success

            # Update monitoring
            status = ServiceStatus.HEALTHY if success else ServiceStatus.UNHEALTHY
            self.monitoring.update_service_health(service_name, status)

        success_count = sum(1 for success in deployment_results.values() if success)
        total_count = len(deployment_results)

        return {
            "deployment_id": f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "environment": environment,
            "success": success_count == total_count,
            "services_deployed": success_count,
            "total_services": total_count,
            "service_results": deployment_results,
            "timestamp": datetime.now().isoformat()
        }

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        return {
            "platform_status": self.monitoring.get_system_health(),
            "service_status": self.orchestrator.get_service_status(),
            "deployment_info": {
                "environment": self.config.get("environment", "unknown"),
                "version": self.config.get("version", "0.1.0"),
                "deployed_at": self.config.get("deployed_at", "unknown")
            }
        }

    def scale_service(self, service_name: str, replicas: int) -> bool:
        """Scale a service to specified number of replicas"""
        logger.info(f"Scaling service {service_name} to {replicas} replicas")

        # Placeholder for actual scaling logic
        if service_name in self.orchestrator.services:
            self.orchestrator.services[service_name]["replicas"] = replicas
            return True

        return False
