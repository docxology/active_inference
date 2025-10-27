# Platform Infrastructure - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Platform Infrastructure module of the Active Inference Knowledge Environment. It outlines platform architecture, service development, deployment strategies, and best practices for building scalable platform components.

## Platform Infrastructure Module Overview

The Platform Infrastructure module provides a comprehensive backend infrastructure for the Active Inference Knowledge Environment, including REST APIs, web services, knowledge graph management, intelligent search, collaboration features, and deployment tools. All components are designed to be scalable, maintainable, and extensible.

## Core Responsibilities

### Platform Architecture
- **Service Design**: Design scalable, maintainable platform services
- **API Development**: Create comprehensive REST and Python APIs
- **System Integration**: Integrate platform components seamlessly
- **Performance Optimization**: Optimize platform performance and scalability
- **Security Implementation**: Implement robust security measures

### Infrastructure Management
- **Deployment**: Manage platform deployment and scaling
- **Monitoring**: Implement comprehensive monitoring and alerting
- **Maintenance**: Maintain platform reliability and performance
- **Backup and Recovery**: Ensure data protection and recovery
- **Configuration Management**: Manage platform configuration

### Service Development
- **Knowledge Services**: Develop knowledge management services
- **Search Services**: Implement intelligent search capabilities
- **Collaboration Services**: Create collaboration and community features
- **Integration Services**: Build external system integration
- **Analytics Services**: Implement usage analytics and insights

## Development Workflows

### Platform Service Development Process
1. **Requirements Analysis**: Analyze platform service requirements
2. **Architecture Design**: Design service architecture and interfaces
3. **API Design**: Design comprehensive service APIs
4. **Implementation**: Implement services following best practices
5. **Integration**: Integrate with existing platform services
6. **Testing**: Comprehensive testing including integration tests
7. **Documentation**: Create comprehensive API documentation
8. **Review**: Submit for technical and security review
9. **Deployment**: Deploy service with monitoring
10. **Maintenance**: Monitor and maintain service performance

### API Development Process
1. **API Specification**: Define complete API specification
2. **Endpoint Design**: Design RESTful endpoints and parameters
3. **Implementation**: Implement API endpoints with validation
4. **Testing**: Test API functionality and edge cases
5. **Documentation**: Generate comprehensive API documentation
6. **Security Review**: Review API security implementation
7. **Performance Testing**: Validate API performance
8. **Integration Testing**: Test API integration

### Infrastructure Development Process
1. **Infrastructure Planning**: Plan platform infrastructure needs
2. **Component Design**: Design infrastructure components
3. **Implementation**: Implement infrastructure components
4. **Integration**: Integrate with platform services
5. **Testing**: Test infrastructure reliability and performance
6. **Security**: Implement security measures
7. **Deployment**: Deploy infrastructure components
8. **Monitoring**: Set up monitoring and alerting

## Quality Standards

### Platform Quality
- **Scalability**: Platform must scale to support growing usage
- **Reliability**: High availability and reliability required
- **Performance**: Acceptable response times and throughput
- **Security**: Robust security implementation
- **Maintainability**: Clean, maintainable code and architecture

### Service Quality
- **API Design**: Clean, intuitive API design
- **Error Handling**: Comprehensive error handling and recovery
- **Documentation**: Complete API documentation
- **Testing**: Extensive testing coverage
- **Monitoring**: Comprehensive service monitoring

### Infrastructure Quality
- **Deployment**: Reliable deployment processes
- **Monitoring**: Effective monitoring and alerting
- **Security**: Secure infrastructure configuration
- **Performance**: Optimized infrastructure performance
- **Backup**: Reliable backup and recovery systems

## Implementation Patterns

### Service Base Pattern
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from fastapi import FastAPI, HTTPException
import uvicorn

logger = logging.getLogger(__name__)

class BasePlatformService(ABC):
    """Base class for platform services"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize service with configuration"""
        self.config = config
        self.service_name = self.__class__.__name__.lower()
        self.app: Optional[FastAPI] = None
        self.setup_logging()
        self.setup_service()

    def setup_logging(self) -> None:
        """Configure service logging"""
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s - {self.service_name} - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"platform.{self.service_name}")

    @abstractmethod
    def setup_service(self) -> None:
        """Set up service-specific configuration"""
        pass

    @abstractmethod
    def create_endpoints(self) -> None:
        """Create service API endpoints"""
        pass

    def create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title=f"{self.service_name.title()} Service",
            description=f"Platform service for {self.service_name}",
            version="1.0.0"
        )

        # Add health endpoint
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": self.service_name}

        # Add service info endpoint
        @app.get("/info")
        async def service_info():
            return self.get_service_info()

        # Create service-specific endpoints
        self.create_endpoints()

        self.app = app
        return app

    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            'service_name': self.service_name,
            'version': '1.0.0',
            'config': self.config,
            'endpoints': [route.path for route in self.app.routes if hasattr(route, 'path')]
        }

    def start_service(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the service"""
        if not self.app:
            self.create_app()

        self.logger.info(f"Starting {self.service_name} service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

class KnowledgeService(BasePlatformService):
    """Knowledge management service"""

    def setup_service(self) -> None:
        """Set up knowledge service"""
        self.knowledge_repo = self.config.get('knowledge_repository')
        self.search_engine = self.config.get('search_engine')

    def create_endpoints(self) -> None:
        """Create knowledge service endpoints"""
        if not self.app:
            return

        @self.app.get("/knowledge/nodes/{node_id}")
        async def get_knowledge_node(node_id: str):
            """Get knowledge node by ID"""
            try:
                node = self.knowledge_repo.get_node(node_id)
                if not node:
                    raise HTTPException(status_code=404, detail="Node not found")
                return node
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/knowledge/search")
        async def search_knowledge(q: str, limit: int = 10):
            """Search knowledge repository"""
            try:
                results = self.search_engine.search(q, limit=limit)
                return {"results": results, "query": q}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/knowledge/paths/{path_id}")
        async def get_learning_path(path_id: str):
            """Get learning path by ID"""
            try:
                path = self.knowledge_repo.get_learning_path(path_id)
                if not path:
                    raise HTTPException(status_code=404, detail="Path not found")
                return path
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
```

### Knowledge Graph Pattern
```python
from typing import Dict, Any, List, Optional
import networkx as nx
from dataclasses import dataclass

@dataclass
class KnowledgeNode:
    """Knowledge graph node"""
    id: str
    type: str  # concept, theory, method, implementation, etc.
    title: str
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None

@dataclass
class KnowledgeRelationship:
    """Knowledge graph relationship"""
    source: str
    target: str
    type: str  # prerequisite, related_to, implements, etc.
    weight: float = 1.0
    metadata: Dict[str, Any] = None

class KnowledgeGraphManager:
    """Manager for knowledge graph operations"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize knowledge graph manager"""
        self.config = config
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.relationships: List[KnowledgeRelationship] = []
        self.setup_graph()

    def setup_graph(self) -> None:
        """Set up knowledge graph"""
        # Initialize graph with configuration
        self.graph.graph['name'] = 'Active Inference Knowledge Graph'
        self.graph.graph['version'] = '1.0.0'
        self.logger.info("Knowledge graph initialized")

    def add_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """Add node to knowledge graph"""
        node = KnowledgeNode(
            id=node_id,
            type=node_data.get('type', 'concept'),
            title=node_data.get('title', node_id),
            content=node_data.get('content', ''),
            metadata=node_data.get('metadata', {}),
            embeddings=node_data.get('embeddings')
        )

        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node_data)

        self.logger.info(f"Added node: {node_id}")

    def add_relationship(self, source: str, target: str, relationship_type: str,
                        weight: float = 1.0, metadata: Dict[str, Any] = None) -> None:
        """Add relationship between nodes"""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Nodes {source} or {target} not found")

        relationship = KnowledgeRelationship(
            source=source,
            target=target,
            type=relationship_type,
            weight=weight,
            metadata=metadata or {}
        )

        self.relationships.append(relationship)
        self.graph.add_edge(source, target, type=relationship_type, weight=weight, **metadata)

        self.logger.info(f"Added relationship: {source} --[{relationship_type}]--> {target}")

    def query(self, query: str, node_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Query knowledge graph"""
        # Simple query implementation
        results = []

        # Filter by node type if specified
        nodes_to_search = self.nodes.values()
        if node_type:
            nodes_to_search = [node for node in nodes_to_search if node.type == node_type]

        # Simple text matching
        for node in nodes_to_search:
            if query.lower() in node.title.lower() or query.lower() in node.content.lower():
                results.append({
                    'id': node.id,
                    'title': node.title,
                    'type': node.type,
                    'content': node.content[:200] + '...' if len(node.content) > 200 else node.content,
                    'metadata': node.metadata
                })

        return results[:limit]

    def get_related_nodes(self, node_id: str, relationship_type: str = None,
                         max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get nodes related to specified node"""
        if node_id not in self.nodes:
            return []

        related_nodes = []

        # Get direct neighbors
        neighbors = list(self.graph.neighbors(node_id))

        for neighbor in neighbors:
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            if not relationship_type or edge_data.get('type') == relationship_type:
                node = self.nodes[neighbor]
                related_nodes.append({
                    'id': node.id,
                    'title': node.title,
                    'type': node.type,
                    'relationship': edge_data.get('type'),
                    'weight': edge_data.get('weight', 1.0)
                })

        return related_nodes

    def get_learning_path(self, start_node: str, end_node: str) -> List[str]:
        """Find learning path between two nodes"""
        try:
            # Find shortest path in knowledge graph
            path = nx.shortest_path(self.graph, start_node, end_node, weight='weight')

            # Convert to learning path with prerequisites
            learning_path = []
            for i in range(len(path) - 1):
                current = path[i]
                next_node = path[i + 1]

                # Get edge information
                edge_data = self.graph.get_edge_data(current, next_node)

                learning_path.append({
                    'node_id': next_node,
                    'node': self.nodes[next_node],
                    'prerequisite': current,
                    'relationship': edge_data.get('type'),
                    'step': i + 1
                })

            return learning_path

        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            self.logger.error(f"Error finding learning path: {e}")
            return []

    def export_graph(self, format: str = 'json') -> str:
        """Export knowledge graph in specified format"""
        if format.lower() == 'json':
            export_data = {
                'nodes': [
                    {
                        'id': node.id,
                        'type': node.type,
                        'title': node.title,
                        'content': node.content,
                        'metadata': node.metadata
                    }
                    for node in self.nodes.values()
                ],
                'relationships': [
                    {
                        'source': rel.source,
                        'target': rel.target,
                        'type': rel.type,
                        'weight': rel.weight,
                        'metadata': rel.metadata
                    }
                    for rel in self.relationships
                ],
                'metadata': {
                    'total_nodes': len(self.nodes),
                    'total_relationships': len(self.relationships),
                    'export_format': format,
                    'timestamp': self.get_timestamp()
                }
            }

            import json
            return json.dumps(export_data, indent=2)

        elif format.lower() == 'graphml':
            import tempfile
            import os

            # Export to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.graphml', delete=False)
            nx.write_graphml(self.graph, temp_file.name)

            # Read and return content
            with open(temp_file.name, 'r') as f:
                content = f.read()

            # Clean up
            os.unlink(temp_file.name)
            return content

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            'total_nodes': len(self.nodes),
            'total_relationships': len(self.relationships),
            'node_types': list(set(node.type for node in self.nodes.values())),
            'relationship_types': list(set(rel.type for rel in self.relationships)),
            'connected_components': nx.number_connected_components(self.graph.to_undirected()),
            'average_degree': sum(dict(self.graph.degree()).values()) / len(self.nodes) if self.nodes else 0,
            'density': nx.density(self.graph)
        }
```

### Search Engine Pattern
```python
from typing import Dict, Any, List, Optional
import whoosh
from whoosh import index, fields
from whoosh.qparser import QueryParser
import os

class SearchEngine:
    """Intelligent search engine for knowledge environment"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize search engine"""
        self.config = config
        self.index_path = config.get('index_path', './search_index')
        self.content_sources = config.get('content_sources', [])
        self.index = None
        self.schema = self.create_schema()
        self.setup_index()

    def create_schema(self) -> fields.Schema:
        """Create search index schema"""
        return fields.Schema(
            id=fields.ID(stored=True),
            title=fields.TEXT(stored=True),
            content=fields.TEXT(stored=True),
            type=fields.KEYWORD(stored=True),
            tags=fields.KEYWORD(stored=True, commas=True),
            difficulty=fields.KEYWORD(stored=True),
            author=fields.TEXT(stored=True),
            url=fields.ID(stored=True),
            metadata=fields.TEXT(stored=True),
            last_modified=fields.DATETIME(stored=True)
        )

    def setup_index(self) -> None:
        """Set up search index"""
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)

        # Create or open index
        if index.exists_in(self.index_path):
            self.index = index.open_dir(self.index_path)
        else:
            self.index = index.create_in(self.index_path, self.schema)

    def index_content(self, content_path: str) -> int:
        """Index content from specified path"""
        indexed_count = 0

        # Implementation for content indexing
        # This would recursively find and index content files

        self.logger.info(f"Indexed {indexed_count} documents")
        return indexed_count

    def search(self, query: str, filters: Dict[str, Any] = None,
               limit: int = 10, ranking: str = 'relevance') -> List[Dict[str, Any]]:
        """Perform search with optional filters"""
        if not self.index:
            return []

        with self.index.searcher() as searcher:
            # Parse query
            parser = QueryParser("content", self.index.schema)
            parsed_query = parser.parse(query)

            # Apply filters if provided
            filter_query = self.build_filter_query(filters)

            # Combine queries
            if filter_query:
                from whoosh.query import And
                combined_query = And([parsed_query, filter_query])
            else:
                combined_query = parsed_query

            # Perform search
            results = searcher.search(combined_query, limit=limit)

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result['id'],
                    'title': result['title'],
                    'content': result['content'][:200] + '...' if len(result['content']) > 200 else result['content'],
                    'type': result['type'],
                    'tags': result.get('tags', '').split(',') if result.get('tags') else [],
                    'difficulty': result.get('difficulty'),
                    'author': result.get('author'),
                    'url': result.get('url'),
                    'score': result.score,
                    'metadata': result.get('metadata', {})
                })

            return formatted_results

    def build_filter_query(self, filters: Dict[str, Any]) -> Optional[Any]:
        """Build filter query from filter parameters"""
        if not filters:
            return None

        from whoosh.query import Term, And, Or

        filter_terms = []
        for field, value in filters.items():
            if field in self.schema.names:
                if isinstance(value, list):
                    # Multiple values for field
                    field_terms = [Term(field, v) for v in value]
                    filter_terms.append(Or(field_terms))
                else:
                    filter_terms.append(Term(field, value))

        if filter_terms:
            return And(filter_terms)

        return None

    def get_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions for partial query"""
        if not self.index:
            return []

        with self.index.searcher() as searcher:
            # Get suggestions from title and content fields
            suggestions = set()

            # Title suggestions
            title_parser = QueryParser("title", self.index.schema)
            title_suggestions = searcher.suggest(title_parser, partial_query, limit=limit//2)
            suggestions.update(title_suggestions)

            # Content suggestions
            content_parser = QueryParser("content", self.index.schema)
            content_suggestions = searcher.suggest(content_parser, partial_query, limit=limit//2)
            suggestions.update(content_suggestions)

            return list(suggestions)[:limit]

    def update_index(self, content_path: str) -> None:
        """Update search index with new or modified content"""
        # Implementation for index updates
        pass

    def optimize_index(self) -> None:
        """Optimize search index for better performance"""
        if self.index:
            with self.index.writer() as writer:
                writer.commit(optimize=True)
```

## Testing Guidelines

### Platform Testing
- **Service Testing**: Test individual platform services
- **Integration Testing**: Test service integration
- **API Testing**: Test platform APIs thoroughly
- **Performance Testing**: Test platform performance and scalability
- **Security Testing**: Test platform security measures

### Infrastructure Testing
- **Deployment Testing**: Test deployment processes
- **Monitoring Testing**: Test monitoring and alerting
- **Backup Testing**: Test backup and recovery procedures
- **Load Testing**: Test platform under various loads
- **Stress Testing**: Test platform limits and failure modes

## Performance Considerations

### Service Performance
- **Response Time**: Optimize API response times
- **Throughput**: Maximize request throughput
- **Resource Usage**: Optimize resource consumption
- **Caching**: Implement effective caching strategies
- **Load Balancing**: Distribute load effectively

### Search Performance
- **Indexing Speed**: Fast content indexing
- **Search Speed**: Quick search response times
- **Index Size**: Manage index storage efficiently
- **Update Performance**: Fast index updates
- **Scalability**: Scale with growing content

## Maintenance and Evolution

### Platform Updates
- **Service Updates**: Keep services current and secure
- **API Evolution**: Manage API changes and versions
- **Performance Optimization**: Continuous performance improvement
- **Security Updates**: Regular security updates and patches
- **Feature Addition**: Add new platform features

### Infrastructure Evolution
- **Scaling**: Scale infrastructure as needed
- **Technology Updates**: Update underlying technologies
- **Architecture Evolution**: Evolve platform architecture
- **Integration Updates**: Update external integrations
- **Monitoring Enhancement**: Improve monitoring and alerting

## Common Challenges and Solutions

### Challenge: Service Integration
**Solution**: Implement clear service interfaces and comprehensive integration testing.

### Challenge: Performance Scaling
**Solution**: Design for horizontal scaling and implement performance monitoring.

### Challenge: Data Consistency
**Solution**: Implement distributed data consistency mechanisms.

### Challenge: Security
**Solution**: Follow security best practices and regular security audits.

## Getting Started as an Agent

### Development Setup
1. **Study Platform Architecture**: Understand current platform structure
2. **Learn Service Patterns**: Study service development patterns
3. **Practice API Development**: Practice REST API development
4. **Understand Deployment**: Learn deployment and scaling concepts

### Contribution Process
1. **Identify Platform Needs**: Find gaps in platform capabilities
2. **Design Solutions**: Create detailed platform component designs
3. **Implement Services**: Follow service development best practices
4. **Test Thoroughly**: Ensure comprehensive testing
5. **Document APIs**: Provide complete API documentation
6. **Security Review**: Ensure security best practices
7. **Performance Testing**: Validate performance characteristics
8. **Community Review**: Submit for technical review

### Learning Resources
- **Platform Architecture**: Study platform and service architecture
- **API Design**: Learn REST API design principles
- **Service Development**: Master service-oriented architecture
- **Security**: Study platform security best practices
- **Performance**: Learn performance optimization techniques

## Related Documentation

- **[Platform README](./README.md)**: Platform infrastructure overview
- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution processes
- **[Knowledge Repository](../../knowledge/)**: Knowledge management
- **[Search Tools](../../platform/search/)**: Search implementation
- **[Deployment Tools](../../platform/deployment/)**: Deployment infrastructure

---

*"Active Inference for, with, by Generative AI"* - Building robust platform infrastructure through scalable services, comprehensive APIs, and collaborative features.
