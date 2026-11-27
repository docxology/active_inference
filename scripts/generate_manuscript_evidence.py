"""Reproduce the paper's small local course, graph, and simulation examples."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from active_inference.knowledge.repository import (
    ContentType,
    KnowledgeRepository,
    KnowledgeRepositoryConfig,
)
from active_inference.platform.knowledge_graph import (
    KnowledgeEdge,
    KnowledgeGraphManager,
    KnowledgeNode,
    NodeType,
    RelationshipType,
)
from active_inference.research.simulations import ModelConfig, SimulationEngine

ROOT = Path(__file__).resolve().parents[1]
LESSONS = (
    ("prior", "Prior beliefs", []),
    ("observation", "Observation model", ["prior"]),
    ("posterior", "Posterior update", ["observation"]),
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def replay_course() -> dict[str, Any]:
    ordered = [name for name, _, _ in LESSONS]
    with TemporaryDirectory(prefix="active-inference-course-") as directory:
        root = Path(directory)
        lessons = root / "foundations"
        lessons.mkdir()
        metadata = root / "metadata"
        metadata.mkdir()
        for name, title, prerequisites in LESSONS:
            write_json(
                lessons / f"{name}.json",
                {
                    "id": name,
                    "title": title,
                    "description": f"Synthetic lesson on {title.lower()}",
                    "content_type": "foundation",
                    "difficulty": "beginner",
                    "prerequisites": prerequisites,
                    "tags": ["synthetic-course"],
                    "learning_objectives": [f"Describe {title.lower()}"],
                    "content": {"summary": title},
                    "metadata": {"fixture": True},
                },
            )
        write_json(
            metadata / "learning_paths.json",
            [
                {
                    "id": "in-order",
                    "name": "In order",
                    "description": "Prerequisites precede their lessons.",
                    "nodes": ordered,
                },
                {
                    "id": "reversed",
                    "name": "Reversed",
                    "description": "Ordering negative control.",
                    "nodes": list(reversed(ordered)),
                },
            ],
        )
        repository = KnowledgeRepository(
            KnowledgeRepositoryConfig(
                root_path=root, content_paths={ContentType.FOUNDATION: "foundations"}
            )
        )
        return {
            "search_ids": [node.id for node in repository.search("Posterior")],
            "ordered": repository.validate_learning_path("in-order"),
            "reversed": repository.validate_learning_path("reversed"),
            "prerequisites": repository.get_prerequisites_graph("posterior"),
        }


def replay_graph() -> dict[str, Any]:
    graph = KnowledgeGraphManager({"storage_backend": "memory"})
    for name, title, _ in LESSONS:
        graph.add_node(KnowledgeNode(name, title, NodeType.CONCEPT))
    for name, _, prerequisites in LESSONS:
        for prerequisite in prerequisites:
            graph.add_edge(
                KnowledgeEdge(prerequisite, name, RelationshipType.PREREQUISITE)
            )
    return {
        **graph.get_statistics(),
        "shortest_path": graph.find_shortest_path("prior", "posterior"),
        "missing_endpoint": graph.find_shortest_path("prior", "absent"),
    }


def replay_simulation() -> dict[str, Any]:
    engine = SimulationEngine({"mode": "local-teaching"})
    engine.register_model(ModelConfig(name="finite", time_horizon=8, time_step=0.5))
    inputs = {"n_states": 3, "n_observations": 4, "n_actions": 2}
    result = engine.run_simulation("finite", inputs)
    replay = engine.run_simulation("finite", inputs)
    fields = (
        "time_points",
        "state_beliefs",
        "observations",
        "actions",
        "free_energy",
        "accuracy",
    )
    return {
        **{key: result[key] for key in fields},
        **inputs,
        "steps": len(result["state_beliefs"]),
        "repeat_equal": all(result[key] == replay[key] for key in fields),
        "all_beliefs_normalized": all(
            abs(sum(row) - 1.0) <= 1e-12 for row in result["state_beliefs"]
        ),
    }


def draw_fixture(path: Path, simulation: dict[str, Any]) -> None:
    figure = Figure(figsize=(9, 3.3), layout="constrained")
    FigureCanvasAgg(figure)
    graph, beliefs = figure.subplots(1, 2)
    for position, (_, title, _) in enumerate(LESSONS):
        graph.text(
            0.5,
            2 - position,
            title,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#e6f0fa",
                "edgecolor": "#336699",
            },
        )
        if position:
            graph.annotate(
                "",
                xy=(0.5, 2.23 - position),
                xytext=(0.5, 2.77 - position),
                arrowprops={"arrowstyle": "->", "color": "#336699"},
            )
    graph.set(xlim=(0, 1), ylim=(-0.45, 2.45), title="Synthetic prerequisite graph")
    graph.axis("off")
    for state, values in enumerate(zip(*simulation["state_beliefs"])):
        beliefs.plot(
            range(1, len(values) + 1), values, "o-", label=f"State {state + 1}"
        )
    beliefs.set(
        title="Categorical belief replay",
        xlabel="Step",
        ylabel="Probability",
        ylim=(0, 1.04),
    )
    beliefs.legend(loc="best")
    figure.savefig(path, dpi=160)


def main() -> None:
    data = ROOT / "output" / "data"
    figures = ROOT / "output" / "figures"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    course = replay_course()
    graph = replay_graph()
    simulation = replay_simulation()
    if (
        course["search_ids"] != ["posterior"]
        or not course["ordered"]["valid"]
        or course["reversed"]["valid"]
        or len(course["reversed"]["issues"]) != 2
        or graph["total_nodes"] != 3
        or graph["total_edges"] != 2
        or graph["shortest_path"] != [name for name, _, _ in LESSONS]
        or graph["missing_endpoint"] is not None
        or simulation["steps"] != 8
        or not simulation["repeat_equal"]
        or not simulation["all_beliefs_normalized"]
    ):
        raise RuntimeError(
            "The replay no longer supports the manuscript's fixture claims."
        )
    evidence = {"course": course, "platform_graph": graph, "simulation": simulation}
    write_json(data / "teaching_fixture.json", evidence)
    filename = "teaching_fixture.png"
    draw_fixture(figures / filename, evidence["simulation"])
    write_json(
        figures / "figure_registry.json",
        {
            "fig:teaching-fixture": {
                "filename": filename,
                "caption": "A synthetic prerequisite graph and a seeded categorical belief replay.",
                "alt_text": (
                    "Arrows connect prior beliefs to the observation model and posterior update. "
                    "The line plot shows three state probabilities across eight replay steps."
                ),
            }
        },
    )


if __name__ == "__main__":
    main()
