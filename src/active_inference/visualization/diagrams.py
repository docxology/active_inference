"""
Visualization Engine - Interactive Diagrams

Interactive diagram system for visualizing Active Inference concepts, models,
and processes. Provides dynamic, educational diagrams with real-time updates
and interactive exploration capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DiagramType(Enum):
    """Types of diagrams supported"""
    CONCEPT_MAP = "concept_map"
    FLOWCHART = "flowchart"
    STATE_DIAGRAM = "state_diagram"
    ARCHITECTURE = "architecture"
    TIMELINE = "timeline"
    HIERARCHY = "hierarchy"


@dataclass
class DiagramNode:
    """Represents a node in a diagram"""
    id: str
    label: str
    position: Tuple[float, float] = (0, 0)
    node_type: str = "default"
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class DiagramEdge:
    """Represents an edge/connection in a diagram"""
    source: str
    target: str
    label: str = ""
    edge_type: str = "directed"
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class InteractiveDiagram:
    """Interactive diagram with dynamic updates"""

    def __init__(self, diagram_type: DiagramType, title: str):
        self.diagram_type = diagram_type
        self.title = title
        self.nodes: Dict[str, DiagramNode] = {}
        self.edges: List[DiagramEdge] = []
        self.properties: Dict[str, Any] = {}
        self.interactive_elements: List[Dict[str, Any]] = []

        logger.info(f"Created interactive diagram: {title} ({diagram_type.value})")

    def add_node(self, node: DiagramNode) -> None:
        """Add a node to the diagram"""
        self.nodes[node.id] = node
        logger.debug(f"Added node {node.id} to diagram {self.title}")

    def add_edge(self, edge: DiagramEdge) -> None:
        """Add an edge to the diagram"""
        # Validate that source and target nodes exist
        if edge.source not in self.nodes:
            logger.warning(f"Source node {edge.source} not found in diagram")
            return
        if edge.target not in self.nodes:
            logger.warning(f"Target node {edge.target} not found in diagram")
            return

        self.edges.append(edge)
        logger.debug(f"Added edge {edge.source} -> {edge.target}")

    def update_node_position(self, node_id: str, position: Tuple[float, float]) -> bool:
        """Update the position of a node"""
        if node_id not in self.nodes:
            return False

        self.nodes[node_id].position = position
        logger.debug(f"Updated position of node {node_id}")
        return True

    def highlight_path(self, node_ids: List[str], color: str = "#ff0000") -> None:
        """Highlight a path through the diagram"""
        for node_id in node_ids:
            if node_id in self.nodes:
                self.nodes[node_id].properties["highlight"] = color

        logger.debug(f"Highlighted path: {node_ids}")

    def to_dict(self) -> Dict[str, Any]:
        """Export diagram to dictionary format"""
        return {
            "type": self.diagram_type.value,
            "title": self.title,
            "nodes": {node_id: {
                "id": node.id,
                "label": node.label,
                "position": node.position,
                "node_type": node.node_type,
                "properties": node.properties
            } for node, node in self.nodes.items()},
            "edges": [{
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
                "edge_type": edge.edge_type,
                "properties": edge.properties
            } for edge in self.edges],
            "properties": self.properties
        }


class ConceptDiagram:
    """Specialized diagrams for Active Inference concepts"""

    def __init__(self):
        self.diagrams: Dict[str, InteractiveDiagram] = {}

    def create_active_inference_overview(self) -> InteractiveDiagram:
        """Create overview diagram of Active Inference"""
        diagram = InteractiveDiagram(DiagramType.CONCEPT_MAP, "Active Inference Overview")

        # Add key concepts as nodes
        nodes = [
            DiagramNode("perception", "Perception", (100, 100), "process"),
            DiagramNode("action", "Action", (300, 100), "process"),
            DiagramNode("learning", "Learning", (200, 50), "process"),
            DiagramNode("generative_model", "Generative Model", (200, 150), "component"),
            DiagramNode("free_energy", "Free Energy", (200, 250), "principle"),
            DiagramNode("variational_inference", "Variational Inference", (100, 200), "method"),
            DiagramNode("active_learning", "Active Learning", (300, 200), "method")
        ]

        for node in nodes:
            diagram.add_node(node)

        # Add relationships
        edges = [
            DiagramEdge("perception", "generative_model", "updates"),
            DiagramEdge("generative_model", "action", "guides"),
            DiagramEdge("perception", "free_energy", "minimizes"),
            DiagramEdge("action", "free_energy", "minimizes"),
            DiagramEdge("learning", "generative_model", "improves"),
            DiagramEdge("variational_inference", "free_energy", "computes"),
            DiagramEdge("active_learning", "action", "drives")
        ]

        for edge in edges:
            diagram.add_edge(edge)

        return diagram

    def create_free_energy_principle_diagram(self) -> InteractiveDiagram:
        """Create diagram explaining the Free Energy Principle"""
        diagram = InteractiveDiagram(DiagramType.FLOWCHART, "Free Energy Principle")

        nodes = [
            DiagramNode("system", "System", (100, 100), "component"),
            DiagramNode("model", "Internal Model", (300, 100), "component"),
            DiagramNode("sensory_input", "Sensory Input", (50, 200), "input"),
            DiagramNode("prediction", "Prediction", (200, 200), "output"),
            DiagramNode("prediction_error", "Prediction Error", (350, 200), "signal"),
            DiagramNode("free_energy", "Free Energy", (200, 300), "measure"),
            DiagramNode("action", "Action", (350, 300), "response")
        ]

        for node in nodes:
            diagram.add_node(node)

        edges = [
            DiagramEdge("sensory_input", "system", "received by"),
            DiagramEdge("system", "model", "uses"),
            DiagramEdge("model", "prediction", "generates"),
            DiagramEdge("sensory_input", "prediction_error", "compared with"),
            DiagramEdge("prediction", "prediction_error", "compared with"),
            DiagramEdge("prediction_error", "free_energy", "quantifies"),
            DiagramEdge("free_energy", "action", "drives")
        ]

        for edge in edges:
            diagram.add_edge(edge)

        return diagram


class VisualizationEngine:
    """Main visualization engine coordinating all diagram types"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.concept_diagram = ConceptDiagram()
        self.diagrams: Dict[str, InteractiveDiagram] = {}

        logger.info("VisualizationEngine initialized")

    def create_diagram(self, diagram_type: DiagramType, title: str,
                      nodes: List[DiagramNode] = None,
                      edges: List[DiagramEdge] = None) -> InteractiveDiagram:
        """Create a new interactive diagram"""
        diagram = InteractiveDiagram(diagram_type, title)

        if nodes:
            for node in nodes:
                diagram.add_node(node)

        if edges:
            for edge in edges:
                diagram.add_edge(edge)

        self.diagrams[title] = diagram
        logger.info(f"Created diagram: {title}")

        return diagram

    def get_concept_diagram(self, concept: str) -> Optional[InteractiveDiagram]:
        """Get a pre-built concept diagram"""
        if concept == "active_inference":
            return self.concept_diagram.create_active_inference_overview()
        elif concept == "free_energy_principle":
            return self.concept_diagram.create_free_energy_principle_diagram()
        else:
            logger.warning(f"Concept diagram not found: {concept}")
            return None

    def export_diagram(self, diagram_name: str, format: str = "json") -> Optional[Dict[str, Any]]:
        """Export diagram in specified format"""
        if diagram_name not in self.diagrams:
            logger.error(f"Diagram not found: {diagram_name}")
            return None

        diagram = self.diagrams[diagram_name]

        if format == "json":
            return diagram.to_dict()
        else:
            logger.warning(f"Export format not supported: {format}")
            return None

