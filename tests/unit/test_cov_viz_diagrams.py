"""Behavioural tests for active_inference.visualization.diagrams."""

import logging

import pytest

from active_inference.visualization.diagrams import (
    ConceptDiagram,
    DiagramEdge,
    DiagramNode,
    DiagramType,
    InteractiveDiagram,
    VisualizationEngine,
)


def test_node_and_edge_defaults():
    node = DiagramNode("n", "N")
    assert node.position == (0, 0)
    assert node.node_type == "default"
    assert node.properties == {}
    assert DiagramNode("m", "M").properties is not node.properties

    edge = DiagramEdge("a", "b")
    assert (edge.label, edge.edge_type, edge.properties) == ("", "directed", {})

    kept = DiagramNode("k", "K", properties={"x": 1})
    assert kept.properties == {"x": 1}


class TestInteractiveDiagram:
    def _diagram(self):
        diagram = InteractiveDiagram(DiagramType.FLOWCHART, "Flow")
        diagram.add_node(DiagramNode("a", "A", (1, 2), "start"))
        diagram.add_node(DiagramNode("b", "B"))
        return diagram

    def test_add_edge_requires_existing_endpoints(self, caplog):
        diagram = self._diagram()
        with caplog.at_level(logging.WARNING):
            diagram.add_edge(DiagramEdge("missing", "b"))
            diagram.add_edge(DiagramEdge("a", "missing"))
        assert diagram.edges == []
        assert "Source node missing not found" in caplog.text
        assert "Target node missing not found" in caplog.text

        edge = DiagramEdge("a", "b", "next")
        diagram.add_edge(edge)
        assert diagram.edges == [edge]

    def test_update_node_position(self):
        diagram = self._diagram()
        assert diagram.update_node_position("a", (10.0, 20.0)) is True
        assert diagram.nodes["a"].position == (10.0, 20.0)
        assert diagram.update_node_position("zzz", (0, 0)) is False

    def test_highlight_path_skips_unknown_nodes(self):
        diagram = self._diagram()
        diagram.highlight_path(["a", "ghost"])
        assert diagram.nodes["a"].properties == {"highlight": "#ff0000"}
        assert diagram.nodes["b"].properties == {}
        diagram.highlight_path(["b"], color="#00ff00")
        assert diagram.nodes["b"].properties["highlight"] == "#00ff00"

    def test_to_dict(self):
        diagram = self._diagram()
        diagram.add_edge(DiagramEdge("a", "b", "next", "undirected", {"w": 2}))
        diagram.properties["theme"] = "dark"
        assert diagram.to_dict() == {
            "type": "flowchart",
            "title": "Flow",
            "nodes": {
                "a": {
                    "id": "a",
                    "label": "A",
                    "position": (1, 2),
                    "node_type": "start",
                    "properties": {},
                },
                "b": {
                    "id": "b",
                    "label": "B",
                    "position": (0, 0),
                    "node_type": "default",
                    "properties": {},
                },
            },
            "edges": [
                {
                    "source": "a",
                    "target": "b",
                    "label": "next",
                    "edge_type": "undirected",
                    "properties": {"w": 2},
                }
            ],
            "properties": {"theme": "dark"},
        }


CONCEPTS = [
    ("active_inference", "Active Inference Overview", DiagramType.CONCEPT_MAP, 7, 7),
    ("free_energy_principle", "Free Energy Principle", DiagramType.FLOWCHART, 7, 7),
    (
        "bayesian_inference",
        "Bayesian Inference in Active Inference",
        DiagramType.FLOWCHART,
        7,
        6,
    ),
    (
        "hierarchical_models",
        "Hierarchical Active Inference",
        DiagramType.HIERARCHY,
        12,
        11,
    ),
    (
        "information_geometry",
        "Information Geometry in Active Inference",
        DiagramType.ARCHITECTURE,
        7,
        7,
    ),
    ("policy_selection", "Policy Selection Process", DiagramType.STATE_DIAGRAM, 7, 6),
    (
        "multi_scale_modeling",
        "Multi-Scale Active Inference",
        DiagramType.HIERARCHY,
        15,
        14,
    ),
]


@pytest.mark.parametrize("concept, title, dtype, n_nodes, n_edges", CONCEPTS)
def test_prebuilt_concept_diagrams(concept, title, dtype, n_nodes, n_edges):
    diagram = VisualizationEngine({}).get_concept_diagram(concept)
    assert diagram.title == title
    assert diagram.diagram_type is dtype
    assert len(diagram.nodes) == n_nodes
    # Every declared edge connects nodes that exist, so none were dropped.
    assert len(diagram.edges) == n_edges
    for edge in diagram.edges:
        assert edge.source in diagram.nodes and edge.target in diagram.nodes


def test_hierarchical_layout_details():
    diagram = ConceptDiagram().create_hierarchical_models_diagram()
    assert diagram.nodes["sensory_observations"].position == (150.0, 150)
    assert diagram.nodes["sensory_perceptions"].position == (250.0, 150)
    assert diagram.nodes["goal"].node_type == "level"
    chain = [(e.source, e.target) for e in diagram.edges if e.label == "influences"]
    assert chain == [
        ("sensory", "contextual"),
        ("contextual", "conceptual"),
        ("conceptual", "goal"),
    ]


def test_multi_scale_layout_details():
    diagram = ConceptDiagram().create_multi_scale_modeling_diagram()
    firing = diagram.nodes["microseconds_neural_firing"]
    assert (firing.label, firing.position, firing.node_type) == (
        "neural firing",
        (300, 100),
        "process",
    )
    assert diagram.nodes["hours_long-term_goals"].position == (420, 500)
    aggregates = [e for e in diagram.edges if e.label == "aggregates to"]
    assert [(e.source, e.target) for e in aggregates][-1] == ("minutes", "hours")


def test_overview_relationships():
    diagram = ConceptDiagram().create_active_inference_overview()
    labels = {(e.source, e.target): e.label for e in diagram.edges}
    assert labels[("perception", "free_energy")] == "minimizes"
    assert labels[("learning", "generative_model")] == "improves"
    assert diagram.nodes["free_energy"].node_type == "principle"


class TestVisualizationEngine:
    def test_unknown_concept_returns_none(self, caplog):
        engine = VisualizationEngine({"theme": "light"})
        assert engine.config == {"theme": "light"}
        with caplog.at_level(logging.WARNING):
            assert engine.get_concept_diagram("nope") is None
        assert "Concept diagram not found: nope" in caplog.text

    def test_create_and_export_diagram(self):
        engine = VisualizationEngine({})
        nodes = [DiagramNode("a", "A"), DiagramNode("b", "B")]
        edges = [DiagramEdge("a", "b", "to"), DiagramEdge("a", "ghost")]
        diagram = engine.create_diagram(DiagramType.TIMELINE, "Time", nodes, edges)

        assert engine.diagrams == {"Time": diagram}
        assert set(diagram.nodes) == {"a", "b"}
        assert [(e.source, e.target) for e in diagram.edges] == [("a", "b")]

        exported = engine.export_diagram("Time")
        assert exported == diagram.to_dict()
        assert exported["type"] == "timeline"

    def test_create_empty_diagram(self):
        engine = VisualizationEngine({})
        diagram = engine.create_diagram(DiagramType.CONCEPT_MAP, "Empty")
        assert diagram.nodes == {} and diagram.edges == []
        assert engine.export_diagram("Empty")["nodes"] == {}

    def test_export_errors(self, caplog):
        engine = VisualizationEngine({})
        engine.create_diagram(DiagramType.FLOWCHART, "F")
        with caplog.at_level(logging.WARNING):
            assert engine.export_diagram("missing") is None
            assert engine.export_diagram("F", format="svg") is None
        assert "Diagram not found: missing" in caplog.text
        assert "Export format not supported: svg" in caplog.text

    @pytest.mark.parametrize(
        "method",
        [
            "implement_diagram_rendering_engine",
            "create_concept_visualization_system",
            "implement_diagram_interaction_manager",
            "create_diagram_export_and_sharing_system",
            "implement_diagram_accessibility_features",
            "create_diagram_performance_optimization",
            "implement_diagram_validation_system",
            "create_diagram_integration_with_knowledge_graph",
            "implement_diagram_security_and_access_control",
            "create_diagram_analytics_and_insights",
        ],
    )
    def test_placeholder_hooks_return_none_and_log(self, method, caplog):
        engine = VisualizationEngine({})
        with caplog.at_level(logging.INFO):
            assert getattr(engine, method)() is None
        assert "placeholder" in caplog.text
        assert engine.diagrams == {}
