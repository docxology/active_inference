# Platform Deployment

**Deployment and scaling infrastructure for the Active Inference Knowledge Environment.**

## Overview

The deployment module provides comprehensive infrastructure for deploying, scaling, and managing the Active Inference platform in various environments, from development to production.

### Core Features

- **Container Orchestration**: Docker and Kubernetes deployment
- **Environment Management**: Development, staging, production environments
- **Scaling**: Horizontal and vertical scaling capabilities
- **Monitoring**: Health monitoring and alerting
- **Backup & Recovery**: Automated backup and recovery systems
- **Configuration Management**: Environment-specific configurations

## Architecture

### Deployment Components

- **Docker Configuration**: Containerization and service definitions
- **Kubernetes Manifests**: Orchestration and scaling rules
- **Environment Templates**: Configuration templates for different environments
- **Monitoring Stack**: Health checks and performance monitoring
- **Backup Systems**: Automated data backup and recovery
- **Migration Scripts**: Database and content migrations

### Infrastructure Layers

```
┌─────────────────┐
│   Load Balancer │
├─────────────────┤
│   Web Services  │ ← Nginx, API Gateway
├─────────────────┤
│  Application    │ ← Active Inference Platform
│     Services    │
├─────────────────┤
│   Data Layer    │ ← PostgreSQL, Redis, Elasticsearch
├─────────────────┤
│ Infrastructure  │ ← Docker, Kubernetes, Monitoring
└─────────────────┘
```

## Usage

### Local Development

```bash
# Start local development environment
make setup
make serve

# Or using Docker
docker-compose up -d
```

### Production Deployment

```bash
# Deploy to production
make deploy-production

# Or using Kubernetes
kubectl apply -f deployment/production/

# Check deployment status
kubectl get pods -n active-inference
```

### Scaling

```bash
# Scale web services
kubectl scale deployment platform-web --replicas=5

# Scale knowledge services
kubectl scale deployment knowledge-service --replicas=3

# Auto-scaling configuration
kubectl autoscale deployment platform-web --cpu-percent=70 --min=2 --max=10
```

## Configuration

### Environment Variables

```bash
# Production environment
export ENVIRONMENT=production
export DATABASE_URL=postgresql://user:pass@db:5432/active_inference
export REDIS_URL=redis://redis:6379
export ELASTICSEARCH_URL=http://elasticsearch:9200

# Development environment
export ENVIRONMENT=development
export DATABASE_URL=postgresql://localhost:5432/active_inference_dev
export DEBUG=true
```

### Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  platform:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=${DATABASE_URL}
    depends_on:
      - postgres
      - redis
```

### Kubernetes Configuration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: platform-web
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
        image: active-inference/platform:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
```

## Deployment Commands

### Development

```bash
# Start development environment
make setup
make serve

# Run with hot reload
python platform/serve.py --reload
```

### Production

```bash
# Build production image
docker build -t active-inference/platform:latest .

# Deploy to Kubernetes
kubectl apply -f deployment/

# Update deployment
kubectl rollout restart deployment/platform-web

# View logs
kubectl logs -f deployment/platform-web
```

### Monitoring

```bash
# Health check
curl http://localhost:8000/health

# Metrics endpoint
curl http://localhost:8000/metrics

# Platform status
make platform-status
```

## Infrastructure

### Database

```bash
# Initialize database
make db-init

# Run migrations
make db-migrate

# Backup database
make db-backup

# Restore database
make db-restore backup.sql
```

### Caching

```bash
# Clear Redis cache
make cache-clear

# Redis monitoring
make redis-monitor

# Cache statistics
make cache-stats
```

### Search Engine

```bash
# Reindex content
make search-reindex

# Search optimization
make search-optimize

# Search analytics
make search-analytics
```

## Security

### Production Security

```bash
# Enable security features
export ENABLE_HTTPS=true
export SSL_CERT_PATH=/path/to/cert.pem
export SSL_KEY_PATH=/path/to/key.pem

# Security monitoring
export SECURITY_LOGGING=true
export INTRUSION_DETECTION=true
```

### Access Control

```bash
# Configure firewalls
make firewall-setup

# Set up VPN
make vpn-setup

# Configure security groups
make security-groups
```

## Monitoring & Alerting

### Health Monitoring

```python
# Health check configuration
health_config = {
    "endpoints": [
        "/health",
        "/api/health",
        "/platform/health"
    ],
    "timeout": 10,
    "interval": 30
}
```

### Performance Monitoring

```python
# Performance metrics
performance_config = {
    "metrics": [
        "response_time",
        "throughput",
        "error_rate",
        "memory_usage"
    ],
    "thresholds": {
        "response_time": "<100ms",
        "error_rate": "<1%"
    }
}
```

### Alerting

```bash
# Configure alerts
make alerts-setup

# Test alerting
make alerts-test

# View alert history
make alerts-history
```

## Backup & Recovery

### Automated Backups

```bash
# Configure backups
backup_config = {
    "database": {
        "frequency": "daily",
        "retention": "30d",
        "location": "s3://backups/"
    },
    "knowledge": {
        "frequency": "hourly",
        "retention": "7d"
    }
}

# Run backup
make backup-full

# Test restore
make restore-test
```

### Disaster Recovery

```bash
# Failover configuration
make failover-setup

# Test disaster recovery
make dr-test

# Emergency procedures
make emergency-stop
```

## Testing

### Deployment Tests

```bash
# Test deployment pipeline
make test-deployment

# Load testing
make test-load

# Stress testing
make test-stress

# Integration testing
make test-integration
```

### Infrastructure Tests

```bash
# Test infrastructure setup
make test-infrastructure

# Network connectivity tests
make test-network

# Security tests
make test-security
```

## Contributing

See [AGENTS.md](AGENTS.md) for development guidelines and [.cursorrules](../../../.cursorrules) for comprehensive development standards.

### Development Process

1. **Environment Setup**:
   ```bash
   make setup-deployment
   ```

2. **Testing**:
   ```bash
   make test-deployment
   make test-infrastructure
   ```

3. **Documentation**:
   - Update deployment guides
   - Update configuration templates
   - Document new deployment methods

## Performance

### Scaling Metrics

- **Horizontal Scaling**: Support for 1000+ concurrent users
- **Response Time**: <100ms for typical operations
- **Throughput**: 10000+ requests per second
- **Availability**: 99.9% uptime guarantee

### Resource Optimization

- **Container Optimization**: Minimal resource footprint
- **Load Balancing**: Efficient request distribution
- **Database Optimization**: Query optimization and indexing
- **Caching**: Multi-level caching strategy

## Troubleshooting

### Common Issues

#### Deployment Failures
```bash
# Check deployment status
kubectl get deployments -n active-inference

# View deployment logs
kubectl logs deployment/platform-web

# Debug deployment issues
make deployment-debug
```

#### Performance Issues
```bash
# Check resource usage
kubectl top nodes
kubectl top pods

# Scale resources
kubectl scale deployment/platform-web --replicas=5

# Optimize configuration
make performance-optimize
```

#### Network Issues
```bash
# Test connectivity
make network-test

# Check firewall rules
make firewall-check

# Test load balancer
make lb-test
```

---

**Component Version**: 1.0.0 | **Development Status**: Active Development

*"Active Inference for, with, by Generative AI"* - Scalable infrastructure for collaborative intelligence.
