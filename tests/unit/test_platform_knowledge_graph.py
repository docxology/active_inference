"""
Tests for Platform Knowledge Graph Module

Unit tests for the knowledge graph system, ensuring proper operation of
graph management, node and edge operations, and graph analytics.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch
from pathlib import Path

from active_inference.platform.knowledge_graph import KnowledgeGraphManager, KnowledgeNode, KnowledgeEdge, NodeType, RelationshipType


class TestKnowledgeNode:
    """Test cases for KnowledgeNode dataclass"""

    def test_knowledge_node_creation(self):
        """Test creating knowledge nodes"""
        node = KnowledgeNode(
            id="test_node",
            title="Test Node",
            node_type=NodeType.CONCEPT,
            content={"description": "Test concept"},
            properties={"domain": "test"}
        )

        assert node.id == "test_node"
        assert node.title == "Test Node"
        assert node.node_type == NodeType.CONCEPT
        assert node.content["description"] == "Test concept"
        assert node.properties["domain"] == "test"

    def test_knowledge_node_post_init(self):
        """Test KnowledgeNode post-initialization"""
        node = KnowledgeNode(
            id="test_node",
            title="Test Node",
            node_type=NodeType.CONCEPT,
            content={"description": "Test"}
        )

        # Properties should be initialized if None
        assert node.properties == {}


class TestKnowledgeEdge:
    """Test cases for KnowledgeEdge dataclass"""

    def test_knowledge_edge_creation(self):
        """Test creating knowledge edges"""
        edge = KnowledgeEdge(
            source="node1",
            target="node2",
            relationship_type=RelationshipType.PREREQUISITE,
            weight=0.8,
            properties={"strength": "strong"}
        )

        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.relationship_type == RelationshipType.PREREQUISITE
        assert edge.weight == 0.8
        assert edge.properties["strength"] == "strong"

    def test_knowledge_edge_post_init(self):
        """Test KnowledgeEdge post-initialization"""
        edge = KnowledgeEdge(
            source="node1",
            target="node2",
            relationship_type=RelationshipType.RELATED_TO
        )

        # Properties should be initialized if None
        assert edge.properties == {}


class TestKnowledgeGraph:
    """Test cases for KnowledgeGraph class"""

    @pytest.fixture
    def knowledge_graph(self):
        """Create KnowledgeGraphManager instance for testing"""
        return KnowledgeGraphManager({"storage_backend": "memory"})

    def test_knowledge_graph_initialization(self, knowledge_graph):
        """Test KnowledgeGraphManager initialization"""
        assert hasattr(knowledge_graph, 'nodes')
        assert hasattr(knowledge_graph, 'edges')
        assert hasattr(knowledge_graph, 'node_index')
        assert hasattr(knowledge_graph, 'storage_backend')

        assert isinstance(knowledge_graph.nodes, dict)
        assert isinstance(knowledge_graph.edges, dict)
        assert isinstance(knowledge_graph.node_index, dict)

    def test_add_node(self, knowledge_graph):
        """Test adding nodes to the graph"""
        node = KnowledgeNode(
            id="test_concept",
            title="Test Concept",
            node_type=NodeType.CONCEPT,
            content={"description": "A test concept"}
        )

        success = knowledge_graph.add_node(node)

        assert success is True
        assert "test_concept" in knowledge_graph.nodes
        assert knowledge_graph.nodes["test_concept"] == node

    def test_add_duplicate_node(self, knowledge_graph):
        """Test adding duplicate node (should update)"""
        node1 = KnowledgeNode(
            id="test_concept",
            title="Test Concept 1",
            node_type=NodeType.CONCEPT,
            content={"description": "First version"}
        )

        node2 = KnowledgeNode(
            id="test_concept",
            title="Test Concept 2",
            node_type=NodeType.CONCEPT,
            content={"description": "Updated version"}
        )

        # Add first node
        success1 = knowledge_graph.add_node(node1)
        assert success1 is True

        # Add duplicate (should update)
        success2 = knowledge_graph.add_node(node2)
        assert success2 is True

        # Check that node was updated
        stored_node = knowledge_graph.nodes["test_concept"]
        assert stored_node.title == "Test Concept 2"
        assert stored_node.content["description"] == "Updated version"

    def test_add_edge(self, knowledge_graph):
        """Test adding edges to the graph"""
        # Add nodes first
        node1 = KnowledgeNode(id="node1", title="Node 1", node_type=NodeType.CONCEPT)
        node2 = KnowledgeNode(id="node2", title="Node 2", node_type=NodeType.CONCEPT)

        knowledge_graph.add_node(node1)
        knowledge_graph.add_node(node2)

        # Add edge
        edge = KnowledgeEdge(
            source="node1",
            target="node2",
            relationship_type=RelationshipType.PREREQUISITE
        )

        success = knowledge_graph.add_edge(edge)

        assert success is True
        edge_id = "node1_prerequisite_node2"
        assert edge_id in knowledge_graph.edges

    def test_get_node(self, knowledge_graph):
        """Test getting nodes from the graph"""
        node = KnowledgeNode(id="test_node", title="Test Node", node_type=NodeType.CONCEPT)
        knowledge_graph.add_node(node)

        # Get existing node
        retrieved = knowledge_graph.get_node("test_node")
        assert retrieved == node

        # Get non-existent node
        nonexistent = knowledge_graph.get_node("nonexistent")
        assert nonexistent is None

    def test_get_related_nodes(self, knowledge_graph):
        """Test getting related nodes"""
        # Add nodes and edges
        node1 = KnowledgeNode(id="node1", title="Node 1", node_type=NodeType.CONCEPT)
        node2 = KnowledgeNode(id="node2", title="Node 2", node_type=NodeType.CONCEPT)
        node3 = KnowledgeNode(id="node3", title="Node 3", node_type=NodeType.CONCEPT)

        knowledge_graph.add_node(node1)
        knowledge_graph.add_node(node2)
        knowledge_graph.add_node(node3)

        # Add edges
        edge1 = KnowledgeEdge(source="node1", target="node2", relationship_type=RelationshipType.PREREQUISITE)
        edge2 = KnowledgeEdge(source="node2", target="node3", relationship_type=RelationshipType.EXTENDS)

        knowledge_graph.add_edge(edge1)
        knowledge_graph.add_edge(edge2)

        # Get related nodes for node1
        related = knowledge_graph.get_related_nodes("node1")

        assert len(related) == 1
        assert related[0]['node'].id == "node2"

        # Get related nodes for node2
        related = knowledge_graph.get_related_nodes("node2")

        assert len(related) == 2  # node1 (incoming) and node3 (outgoing)

    def test_find_shortest_path(self, knowledge_graph):
        """Test finding shortest path between nodes"""
        # Create a simple graph: node1 -> node2 -> node3
        nodes = [
            KnowledgeNode(id="node1", title="Node 1", node_type=NodeType.CONCEPT),
            KnowledgeNode(id="node2", title="Node 2", node_type=NodeType.CONCEPT),
            KnowledgeNode(id="node3", title="Node 3", node_type=NodeType.CONCEPT)
        ]

        for node in nodes:
            knowledge_graph.add_node(node)

        edges = [
            KnowledgeEdge(source="node1", target="node2", relationship_type=RelationshipType.PREREQUISITE),
            KnowledgeEdge(source="node2", target="node3", relationship_type=RelationshipType.EXTENDS)
        ]

        for edge in edges:
            knowledge_graph.add_edge(edge)

        # Find path from node1 to node3
        path = knowledge_graph.find_shortest_path("node1", "node3")

        assert path == ["node1", "node2", "node3"]

        # Test non-existent path
        path = knowledge_graph.find_shortest_path("node1", "nonexistent")
        assert path is None

    def test_get_graph_statistics(self, knowledge_graph):
        """Test getting graph statistics"""
        # Add some nodes and edges
        for i in range(5):
            node = KnowledgeNode(id=f"node{i}", title=f"Node {i}", node_type=NodeType.CONCEPT)
            knowledge_graph.add_node(node)

        # Add some edges
        for i in range(4):
            edge = KnowledgeEdge(source=f"node{i}", target=f"node{i+1}", relationship_type=RelationshipType.RELATED_TO)
            knowledge_graph.add_edge(edge)

        stats = knowledge_graph.get_statistics()

        assert isinstance(stats, dict)
        assert stats["total_nodes"] == 5
        assert stats["total_edges"] == 4
        assert "node_types" in stats
        assert "relationship_types" in stats

    # TODO: Implement test_validate_graph when validation functionality is added
    # def test_validate_graph(self, knowledge_graph):
    #     """Test graph validation"""
    #     # Add valid nodes and edges
    #     node1 = KnowledgeNode(id="valid_node1", title="Valid Node 1", node_type=NodeType.CONCEPT)
    #     node2 = KnowledgeNode(id="valid_node2", title="Valid Node 2", node_type=NodeType.CONCEPT)

    #     knowledge_graph.add_node(node1)
    #     knowledge_graph.add_node(node2)

    #     edge = KnowledgeEdge(source="valid_node1", target="valid_node2", relationship_type=RelationshipType.RELATED_TO)
    #     knowledge_graph.add_edge(edge)

    #     validation = knowledge_graph.validate_graph()

    #     assert isinstance(validation, dict)
    #     assert validation["valid"] is True
    #     assert len(validation["issues"]) == 0

    def test_export_graph(self, knowledge_graph):
        """Test graph export functionality"""
        # Add test data
        node = KnowledgeNode(id="export_test", title="Export Test", node_type=NodeType.CONCEPT)
        knowledge_graph.add_node(node)

        # Export as JSON
        export_json = knowledge_graph.export_graph("json")
        assert isinstance(export_json, str)

        # Parse and validate structure
        export_data = json.loads(export_json)
        assert "nodes" in export_data
        assert "edges" in export_data
        assert len(export_data["nodes"]) == 1

    # TODO: Implement test_import_graph when import functionality is added
    # def test_import_graph(self, knowledge_graph):
    #     """Test graph import functionality"""
    #     # Create export data
    #     export_data = {
    #         "nodes": [
    #             {
    #                 "id": "import_test",
    #                 "label": "Import Test Node",
    #                 "type": "concept",
    #                 "content": {"description": "Imported node"},
    #                 "properties": {"imported": True}
    #             }
    #         ],
    #         "edges": [],
    #         "metadata": {
    #             "total_nodes": 1,
    #             "total_edges": 0,
    #             "export_format": "json"
    #         }
    #     }

    #     success = knowledge_graph.import_graph(export_data)

    #     assert success is True
    #     assert "import_test" in knowledge_graph.nodes

    #     imported_node = knowledge_graph.nodes["import_test"]
    #     assert imported_node.label == "Import Test Node"
    #     assert imported_node.properties["imported"] is True

    # TODO: Implement test_get_subgraph when subgraph functionality is added
    # def test_get_subgraph(self, knowledge_graph):
    #     """Test subgraph extraction"""
    #     # Create test graph
    #     nodes = []
    #     for i in range(5):
    #         node = KnowledgeNode(id=f"node{i}", label=f"Node {i}", node_type="concept")
    #         knowledge_graph.add_node(node)
    #         nodes.append(node)

    #     # Add edges to create a connected graph
    #     edges = [
    #         KnowledgeEdge(source="node0", target="node1", relation_type="related"),
    #         KnowledgeEdge(source="node1", target="node2", relation_type="related"),
    #         KnowledgeEdge(source="node2", target="node3", relation_type="related"),
    #         KnowledgeEdge(source="node3", target="node4", relation_type="related")
    #     ]

    #     for edge in edges:
    #         knowledge_graph.add_edge(edge)

    #     # Extract subgraph containing node0, node1, node2
    #     subgraph = knowledge_graph.get_subgraph(["node0", "node1", "node2"], max_depth=1)

    #     assert len(subgraph.nodes) >= 3  # At least the specified nodes
    #     assert "node0" in subgraph.nodes
    #     assert "node1" in subgraph.nodes
    #     assert "node2" in subgraph.nodes

    # TODO: Implement test_compute_graph_metrics when graph metrics functionality is added
    # def test_compute_graph_metrics(self, knowledge_graph):
    #     """Test graph metrics computation"""
    #     # Create a simple graph
    #     for i in range(3):
    #         node = KnowledgeNode(id=f"node{i}", label=f"Node {i}", node_type="concept")
    #         knowledge_graph.add_node(node)

    #     knowledge_graph.add_edge(KnowledgeEdge(source="node0", target="node1", relation_type="related"))
    #     knowledge_graph.add_edge(KnowledgeEdge(source="node1", target="node2", relation_type="related"))

    #     metrics = knowledge_graph.compute_graph_metrics()

    #     assert isinstance(metrics, dict)
    #     assert "basic" in metrics
    #     assert "connectivity" in metrics
    #     assert "centrality" in metrics
    #     assert "clustering" in metrics

    #     # Check basic metrics
    #     basic = metrics["basic"]
    #     assert basic["total_nodes"] == 3
    #     assert basic["total_edges"] == 2

    # TODO: Implement test_find_similar_concepts when similarity functionality is added
    # def test_find_similar_concepts(self, knowledge_graph):
    #     """Test finding similar concepts"""
    #     # Add nodes with embeddings
    #     node1 = KnowledgeNode(id="concept1", label="Concept 1", node_type="concept")
    #     node2 = KnowledgeNode(id="concept2", label="Concept 2", node_type="concept")

    #     knowledge_graph.add_node(node1)
    #     knowledge_graph.add_node(node2)

    #     # Test finding similar concepts
    #     similar = knowledge_graph.find_similar_concepts("concept1", limit=5)

    #     assert isinstance(similar, list)
    #     assert len(similar) <= 5

    # TODO: Implement test_get_concept_hierarchy when hierarchy functionality is added
    # def test_get_concept_hierarchy(self, knowledge_graph):
    #     """Test concept hierarchy extraction"""
    #     node = KnowledgeNode(id="main_concept", label="Main Concept", node_type="concept")
    #     knowledge_graph.add_node(node)

    #     hierarchy = knowledge_graph.get_concept_hierarchy("main_concept")

    #     assert isinstance(hierarchy, dict)
    #     assert hierarchy["concept"] == "main_concept"
    #     assert "related_concepts" in hierarchy
    #     assert "sub_concepts" in hierarchy
    #     assert "super_concepts" in hierarchy
    #     assert "sibling_concepts" in hierarchy

    # TODO: Implement test_optimize_graph_structure when optimization functionality is added
    # def test_optimize_graph_structure(self, knowledge_graph):
    #     """Test graph structure optimization"""
    #     # Add some nodes and edges
    #     for i in range(3):
    #         node = KnowledgeNode(id=f"node{i}", title=f"Node {i}", node_type=NodeType.CONCEPT)
    #         knowledge_graph.add_node(node)

    #     knowledge_graph.add_edge(KnowledgeEdge(source="node0", target="node1", relationship_type=RelationshipType.RELATED_TO))

    #     optimization = knowledge_graph.optimize_graph_structure()

    #     assert isinstance(optimization, dict)
    #     assert optimization["success"] is True
    #     assert "optimizations_applied" in optimization
    #     assert "final_graph_size" in optimization

    # TODO: Implement test_backup_and_restore_graph when backup functionality is added
    # def test_backup_and_restore_graph(self, knowledge_graph, tmp_path):
    #     """Test graph backup and restore functionality"""
    #     # Add test data
    #     node = KnowledgeNode(id="backup_test", title="Backup Test", node_type=NodeType.CONCEPT)
    #     knowledge_graph.add_node(node)

    #     # Create backup
    #     backup_path = knowledge_graph.backup_graph()

    #     # Verify backup file exists
    #     assert Path(backup_path).exists()

    #     # Create new graph and restore
    #     new_graph = KnowledgeGraphManager({"storage_backend": "memory"})
    #     success = new_graph.restore_graph(backup_path)

    #     assert success is True
    #     assert "backup_test" in new_graph.nodes

    #     # Clean up
    #     Path(backup_path).unlink()

    # TODO: Implement test_compute_connectivity_metrics when metrics functionality is added
    # def test_compute_connectivity_metrics(self, knowledge_graph):
    #     """Test connectivity metrics computation"""
    #     # Create a connected graph
    #     for i in range(3):
    #         node = KnowledgeNode(id=f"node{i}", title=f"Node {i}", node_type=NodeType.CONCEPT)
    #         knowledge_graph.add_node(node)

    #     knowledge_graph.add_edge(KnowledgeEdge(source="node0", target="node1", relationship_type=RelationshipType.RELATED_TO))
    #     knowledge_graph.add_edge(KnowledgeEdge(source="node1", target="node2", relationship_type=RelationshipType.RELATED_TO))

    #     metrics = knowledge_graph.get_statistics()
    #     connectivity = metrics.get("connectivity", {})

    #     assert isinstance(connectivity, dict)
    #     assert "connected_components" in connectivity
    #     assert "is_connected" in connectivity

    #     # Should be connected (1 component)
    #     assert connectivity["connected_components"] == 1
    #     assert connectivity["is_connected"] is True

    # TODO: Implement test_compute_centrality_metrics when centrality functionality is added
    # def test_compute_centrality_metrics(self, knowledge_graph):
    #     """Test centrality metrics computation"""
    #     # Create graph with different degrees
    #     for i in range(4):
    #         node = KnowledgeNode(id=f"node{i}", title=f"Node {i}", node_type=NodeType.CONCEPT)
    #         knowledge_graph.add_node(node)

    #     # Node0 connected to node1 and node2
    #     knowledge_graph.add_edge(KnowledgeEdge(source="node0", target="node1", relationship_type=RelationshipType.RELATED_TO))
    #     knowledge_graph.add_edge(KnowledgeEdge(source="node0", target="node2", relationship_type=RelationshipType.RELATED_TO))

    #     # Node3 is isolated
    #     # Node1 connected to node0 only

    #     metrics = knowledge_graph.get_statistics()
    #     centrality = metrics.get("centrality", {})

    #     assert isinstance(centrality, dict)
    #     assert "degree_centrality" in centrality
    #     assert "max_degree" in centrality
    #     assert "most_central_nodes" in centrality

    #     # Node0 should have highest degree (2)
    #     assert centrality["degree_centrality"]["node0"] == 2
    #     assert centrality["max_degree"] == 2
    #     assert "node0" in centrality["most_central_nodes"]


if __name__ == "__main__":
    pytest.main([__file__])

