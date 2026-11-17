"""Behavioural tests for the argparse-based ``active_inference.cli`` module.

Every test builds the CLI against a small knowledge base under ``tmp_path``,
persists conversations under ``tmp_path`` and replaces the Ollama client with
an in-memory fake, so no network or repository-tree writes happen.
"""

import argparse
import builtins
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from active_inference import cli as cli_mod
from active_inference.llm import ConversationManager, ModelManager, OllamaClient


def _node(
    node_id: str,
    title: str,
    content_type: str,
    description: str,
    *,
    difficulty: str = "beginner",
    prerequisites: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    objectives: Optional[List[str]] = None,
    content: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": node_id,
        "title": title,
        "content_type": content_type,
        "difficulty": difficulty,
        "description": description,
        "prerequisites": prerequisites or [],
        "tags": tags or [],
        "learning_objectives": objectives or [],
        "content": content if content is not None else {},
        "metadata": metadata or {},
    }


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _build_knowledge_base(root: Path) -> Path:
    _write(
        root / "foundations" / "entropy.json",
        _node(
            "info_theory_entropy",
            "Entropy Basics",
            "foundation",
            "Information entropy measures uncertainty",
            tags=["entropy", "information"],
            objectives=["Define entropy"],
            content={"sections": [{"title": "Intro"}, {"body": "no title"}]},
        ),
    )
    _write(
        root / "foundations" / "kl.json",
        _node(
            "kl_divergence",
            "KL Divergence",
            "foundation",
            "Relative entropy between distributions",
            difficulty="intermediate",
            prerequisites=["info_theory_entropy"],
        ),
    )
    _write(
        root / "implementations" / "basic.json",
        _node(
            "impl_basic",
            "Basic implementation",
            "implementation",
            "A minimal agent loop",
            tags=["python"],
            objectives=["Run an agent"],
            content={"sections": [{"title": "Setup"}, {"title": "Loop"}]},
        ),
    )
    _write(
        root / "implementations" / "untagged.json",
        _node(
            "impl_untagged",
            "Untagged implementation",
            "implementation",
            "No tags here",
            difficulty="advanced",
            content={"code": "pass"},
        ),
    )
    _write(
        root / "applications" / "robotics.json",
        _node(
            "app_robotics",
            "Robotics",
            "application",
            "Active Inference application engineering for robot arms",
            difficulty="advanced",
            tags=["robotics"],
            content={"sections": [{"title": "Control"}]},
            metadata={"domain": "engineering"},
        ),
    )
    _write(
        root / "applications" / "neuro.json",
        _node(
            "app_neuro",
            "Application neuroscience model",
            "application",
            "Cortical inference",
            difficulty="expert",
            content={"sections": []},
        ),
    )
    _write(
        root / "metadata" / "learning_paths.json",
        [
            {
                "id": "foundations_complete",
                "name": "Foundations Track",
                "description": "Core theory",
                "nodes": ["info_theory_entropy", "kl_divergence"],
                "estimated_hours": 6,
                "difficulty": "beginner",
            },
            {
                "id": "bad_order",
                "name": "Broken Track",
                "description": "Prerequisites out of order",
                "nodes": ["kl_divergence", "info_theory_entropy", "ghost_node"],
                "estimated_hours": 2,
                "difficulty": "intermediate",
            },
        ],
    )
    return root


class FakeLLMClient:
    """In-memory stand-in for :class:`OllamaClient`."""

    def __init__(
        self,
        responses: Optional[List[Any]] = None,
        models: Optional[List[str]] = None,
        info: Optional[Dict[str, Any]] = None,
        pull_ok: bool = True,
        init_error: Optional[Exception] = None,
        initialized: bool = False,
    ) -> None:
        self._is_initialized = initialized
        self.responses = list(responses or [])
        self.models = models or []
        self.info = info
        self.pull_ok = pull_ok
        self.init_error = init_error
        self.init_calls = 0
        self.generate_calls: List[tuple] = []
        self.info_requests: List[str] = []
        self.pulled: List[str] = []

    async def initialize(self) -> bool:
        self.init_calls += 1
        if self.init_error is not None:
            raise self.init_error
        self._is_initialized = True
        return True

    async def generate(self, prompt: str, model: Optional[str] = None) -> str:
        self.generate_calls.append((prompt, model))
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        return f"echo: {prompt}"

    async def list_models(self) -> List[str]:
        return list(self.models)

    async def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        self.info_requests.append(model)
        return self.info

    async def pull_model(self, model: str) -> bool:
        self.pulled.append(model)
        return self.pull_ok


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Patch component factories so the CLI only touches ``tmp_path``."""
    kb_root = _build_knowledge_base(tmp_path / "knowledge")
    conv_dir = tmp_path / "conversations"
    requested_roots: List[Path] = []
    real_config = cli_mod.KnowledgeRepositoryConfig

    def config_factory(root_path: Path, auto_index: bool = True):
        requested_roots.append(root_path)
        return real_config(root_path=kb_root, auto_index=auto_index)

    monkeypatch.setattr(cli_mod, "KnowledgeRepositoryConfig", config_factory)
    monkeypatch.setattr(
        cli_mod,
        "ConversationManager",
        lambda: ConversationManager(storage_path=conv_dir),
    )
    monkeypatch.setattr(
        cli_mod,
        "ModelManager",
        lambda: ModelManager(registry_path=tmp_path / "registry.json"),
    )
    return {
        "kb_root": kb_root,
        "conv_dir": conv_dir,
        "requested_roots": requested_roots,
    }


@pytest.fixture
def make_cli(env):
    def factory(llm: Optional[FakeLLMClient] = None) -> cli_mod.ActiveInferenceCLI:
        instance = cli_mod.ActiveInferenceCLI()
        if llm is not None:
            instance.llm_client = llm
        return instance

    return factory


def _saved_conversations(conv_dir: Path) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(conv_dir.glob("*.json"))]


# --------------------------------------------------------------------------
# Construction and top-level dispatch
# --------------------------------------------------------------------------


def test_init_wires_components_from_project_knowledge_dir(make_cli, env):
    cli = make_cli()

    assert env["requested_roots"] == [cli.project_root / "knowledge"]
    assert cli.project_root == Path(cli_mod.__file__).parent.parent.parent
    assert {n for n in cli.knowledge_repo._nodes} == {
        "info_theory_entropy",
        "kl_divergence",
        "impl_basic",
        "impl_untagged",
        "app_robotics",
        "app_neuro",
    }
    assert isinstance(cli.llm_client, OllamaClient)
    assert cli.llm_client._is_initialized is False
    assert cli.prompt_manager.get_template("research_question") is not None
    assert cli.conversation_manager.storage_path == env["conv_dir"]
    assert env["conv_dir"].is_dir()
    assert cli.visualization_engine is None


def test_init_failure_degrades_gracefully(env, monkeypatch, capsys):
    def broken_repo(config):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(cli_mod, "KnowledgeRepository", broken_repo)
    cli = cli_mod.ActiveInferenceCLI()
    out = capsys.readouterr().out

    assert "Warning: Could not initialize all components: disk gone" in out
    assert "Some features may not be available." in out
    assert cli.knowledge_repo is None
    assert cli.llm_client is None

    assert cli.run(["knowledge", "stats"]) == 1
    assert "Knowledge repository not available" in capsys.readouterr().out

    assert cli.run(["llm", "models", "list"]) == 1
    out = capsys.readouterr().out
    assert "LLM components not available" in out
    assert "Run: ollama serve" in out


def test_create_parser_parses_nested_commands(make_cli):
    parser = make_cli().create_parser()

    ns = parser.parse_args(["knowledge", "search", "entropy", "--limit", "3"])
    assert (ns.command, ns.knowledge_command, ns.query, ns.limit) == (
        "knowledge",
        "search",
        "entropy",
        3,
    )
    assert ns.type is None

    ns = parser.parse_args(["llm", "generate", "hi", "--template", "research_question"])
    assert ns.audience_level == "intermediate"
    assert ns.template == "research_question"

    ns = parser.parse_args(["research", "data", "backup", "ds-1"])
    assert (ns.command, ns.research_command, ns.data_command, ns.dataset_id) == (
        "research",
        "data",
        "backup",
        "ds-1",
    )


def test_run_without_command_prints_help(make_cli, capsys):
    assert make_cli().run([]) == 0
    out = capsys.readouterr().out
    assert "Active Inference Knowledge Environment" in out
    assert "ai-knowledge learn foundations" in out


def test_run_reads_sys_argv_when_args_none(make_cli, monkeypatch, capsys):
    cli = make_cli()
    monkeypatch.setattr(sys, "argv", ["ai-knowledge", "research", "simulations"])
    assert cli.run() == 0
    assert "Simulations coming soon!" in capsys.readouterr().out


def test_run_rejects_invalid_choice(make_cli, capsys):
    with pytest.raises(SystemExit) as excinfo:
        make_cli().run(["knowledge", "learn", "astrology"])
    assert excinfo.value.code == 2
    assert "invalid choice: 'astrology'" in capsys.readouterr().err


def test_run_converts_handler_exceptions_to_exit_code(make_cli, monkeypatch, capsys):
    cli = make_cli()

    def explode():
        raise RuntimeError("stats exploded")

    monkeypatch.setattr(cli.knowledge_repo, "get_statistics", explode)
    assert cli.run(["knowledge", "stats"]) == 1
    assert "Error: stats exploded" in capsys.readouterr().out


def test_dispatch_unknown_command(make_cli, capsys):
    result = make_cli()._dispatch_command(argparse.Namespace(command="bogus"))
    out = capsys.readouterr().out
    assert result == 1
    assert "Unknown command: bogus" in out
    assert "Available commands: knowledge, research" in out


@pytest.mark.parametrize(
    "argv, expected_code, expected_text",
    [
        (["ai-knowledge", "platform", "status"], 0, "Platform tools coming soon!"),
        (["ai-knowledge", "research"], 1, "Unknown research command: None"),
    ],
)
def test_main_exits_with_cli_status(
    env, monkeypatch, capsys, argv, expected_code, expected_text
):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli_mod.main()
    assert excinfo.value.code == expected_code
    assert expected_text in capsys.readouterr().out


# --------------------------------------------------------------------------
# Knowledge commands
# --------------------------------------------------------------------------


def test_knowledge_without_subcommand_is_error(make_cli, capsys):
    assert make_cli().run(["knowledge"]) == 1
    assert "Unknown knowledge command: None" in capsys.readouterr().out


def test_learn_foundations_lists_path_modules(make_cli, capsys):
    assert make_cli().run(["knowledge", "learn", "foundations"]) == 0
    out = capsys.readouterr().out
    assert "Starting Foundations Learning Track" in out
    assert "Learning Path: Foundations Track" in out
    assert "Estimated time: 6 hours" in out
    assert "Difficulty: beginner" in out
    assert " 1. Entropy Basics" in out
    assert " 2. KL Divergence" in out
    assert "Type: foundation | Difficulty: intermediate" in out
    assert "ai-knowledge search <topic>" in out


def test_learn_foundations_skips_missing_nodes(make_cli, capsys):
    cli = make_cli()
    cli.knowledge_repo._paths["foundations_complete"].nodes.insert(0, "ghost")
    assert cli.run(["knowledge", "learn", "foundations"]) == 0
    out = capsys.readouterr().out
    assert "ghost" not in out
    assert " 2. Entropy Basics" in out


def test_learn_foundations_without_path(make_cli, capsys):
    cli = make_cli()
    del cli.knowledge_repo._paths["foundations_complete"]
    assert cli.run(["knowledge", "learn", "foundations"]) == 0
    assert "Foundation learning path not found" in capsys.readouterr().out


def test_learn_handler_without_repository_prints_header_only(make_cli, capsys):
    cli = make_cli()
    cli.knowledge_repo = None
    assert cli._handle_learn_command(argparse.Namespace(track="foundations")) == 0
    out = capsys.readouterr().out
    assert "Starting Foundations Learning Track" in out
    assert "Learning Path" not in out


@pytest.mark.parametrize(
    "track, expected",
    [
        ("complete", "Starting Complete Learning Track"),
        ("research", "Learning track 'research' not implemented yet"),
        ("applications", "Learning track 'applications' not implemented yet"),
    ],
)
def test_learn_other_tracks(make_cli, capsys, track, expected):
    assert make_cli().run(["knowledge", "learn", track]) == 0
    assert expected in capsys.readouterr().out


def test_search_prints_matches_with_tags(make_cli, capsys):
    assert make_cli().run(["knowledge", "search", "entropy"]) == 0
    out = capsys.readouterr().out
    assert "Searching for: 'entropy'" in out
    assert "Found 2 results:" in out
    # Result order follows filesystem discovery, so check membership only.
    assert {"1.", "2."} == {
        line.split(" ", 1)[0] for line in out.splitlines() if line[:1].isdigit()
    }
    assert ". Entropy Basics" in out
    assert "Tags: entropy, information" in out
    assert ". KL Divergence" in out
    # KL divergence has no tags, so only one Tags line is printed.
    assert out.count("Tags:") == 1


def test_search_respects_limit(make_cli, capsys):
    assert make_cli().run(["knowledge", "search", "entropy", "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert "Found 1 results:" in out
    assert ("KL Divergence" in out) != ("Entropy Basics" in out)
    assert "2. " not in out


def test_search_without_results(make_cli, capsys):
    assert make_cli().run(["knowledge", "search", "quantum gravity"]) == 0
    out = capsys.readouterr().out
    assert "No results found." in out
    assert "Try adjusting your search terms" in out


def test_path_show_valid_path(make_cli, capsys):
    assert make_cli().run(["knowledge", "path", "show", "foundations_complete"]) == 0
    out = capsys.readouterr().out
    assert "Learning Path: Foundations Track" in out
    assert "Path validation: PASSED" in out
    assert "Prerequisites: None" in out
    assert "Prerequisites: info_theory_entropy" in out


def test_path_show_reports_validation_issues(make_cli, capsys):
    assert make_cli().run(["knowledge", "path", "show", "bad_order"]) == 0
    out = capsys.readouterr().out
    assert "Path validation: ISSUES FOUND" in out
    assert "- Unsatisfied prerequisite info_theory_entropy for kl_divergence" in out
    assert "- Missing node: ghost_node" in out
    assert "ghost_node" not in out.split("Path Modules:")[1]


def test_path_show_unknown_path(make_cli, capsys):
    assert make_cli().run(["knowledge", "path", "show", "nope"]) == 1
    assert "Learning path 'nope' not found" in capsys.readouterr().out


def test_path_list(make_cli, capsys):
    assert make_cli().run(["knowledge", "path", "list"]) == 0
    out = capsys.readouterr().out
    assert "Available Learning Paths:" in out
    assert "• Foundations Track" in out
    assert "Duration: 6h | Difficulty: beginner" in out
    assert "• Broken Track" in out
    assert "Duration: 2h | Difficulty: intermediate" in out


def test_path_without_subcommand_prints_nothing(make_cli, capsys):
    assert make_cli().run(["knowledge", "path"]) == 0
    assert capsys.readouterr().out == ""


def test_stats_summarises_repository(make_cli, capsys):
    assert make_cli().run(["knowledge", "stats"]) == 0
    out = capsys.readouterr().out
    assert "Total knowledge nodes: 6" in out
    assert "Total learning paths: 2" in out
    assert "foundation: 2" in out
    assert "implementation: 2" in out
    assert "application: 2" in out
    assert "beginner: 2" in out
    assert "advanced: 2" in out
    assert "expert: 1" in out
    assert "Popular Tags:" in out
    assert "entropy: 1" in out
    assert "more" not in out


def test_stats_truncates_long_tag_lists(make_cli, capsys):
    cli = make_cli()
    cli.knowledge_repo.add_knowledge_node(
        _node(
            "tag_heavy",
            "Tag heavy",
            "mathematics",
            "Lots of tags",
            tags=[f"t{i:02d}" for i in range(12)],
        )
    )
    assert cli.run(["knowledge", "stats"]) == 0
    out = capsys.readouterr().out
    assert "Total knowledge nodes: 7" in out
    # 4 fixture tags + 12 new ones: 10 shown, 6 summarised.
    assert "... and 6 more" in out


def test_stats_without_tags_omits_tag_section(make_cli, monkeypatch, capsys):
    cli = make_cli()
    monkeypatch.setattr(
        cli.knowledge_repo,
        "get_statistics",
        lambda: {
            "total_nodes": 0,
            "total_paths": 0,
            "content_types": {},
            "difficulties": {},
            "top_tags": [],
        },
    )
    assert cli.run(["knowledge", "stats"]) == 0
    out = capsys.readouterr().out
    assert "Total knowledge nodes: 0" in out
    assert "Popular Tags" not in out


@pytest.mark.parametrize(
    "handler, namespace",
    [
        ("_handle_search_command", argparse.Namespace(query="x", limit=1)),
        ("_handle_path_command", argparse.Namespace(path_command="list")),
        ("_handle_stats_command", argparse.Namespace()),
        ("_handle_implementations_command", argparse.Namespace(impl_command="list")),
        (
            "_handle_knowledge_applications_command",
            argparse.Namespace(app_command="list"),
        ),
    ],
)
def test_knowledge_handlers_require_repository(make_cli, capsys, handler, namespace):
    cli = make_cli()
    cli.knowledge_repo = None
    assert getattr(cli, handler)(namespace) == 1
    assert "Knowledge repository not available" in capsys.readouterr().out


def test_implementations_list_finds_implementation_nodes(make_cli, capsys):
    assert make_cli().run(["knowledge", "implementations", "list"]) == 0
    out = capsys.readouterr().out
    assert "Available Implementations:" in out
    assert ". Basic implementation" in out
    assert ". Untagged implementation" in out
    assert "1. " in out and "2. " in out
    assert "Type: implementation | Difficulty: advanced" in out
    assert "Tags: python" in out
    assert out.count("Tags:") == 1
    assert "Entropy Basics" not in out


def test_implementations_list_empty(make_cli, capsys):
    cli = make_cli()
    cli.knowledge_repo._nodes = {
        k: v for k, v in cli.knowledge_repo._nodes.items() if not k.startswith("impl_")
    }
    assert cli.run(["knowledge", "implementations", "list"]) == 0
    assert "No implementations found." in capsys.readouterr().out


def test_implementations_show(make_cli, capsys):
    assert make_cli().run(["knowledge", "implementations", "show", "impl_basic"]) == 0
    out = capsys.readouterr().out
    assert "Implementation: Basic implementation" in out
    assert "Tags: python" in out
    assert "• Setup" in out
    assert "• Loop" in out


def test_implementations_show_untagged_without_sections(make_cli, capsys):
    cli = make_cli()
    assert cli.run(["knowledge", "implementations", "show", "impl_untagged"]) == 0
    out = capsys.readouterr().out
    assert "Tags: None" in out
    assert "•" not in out


def test_implementations_show_untitled_section(make_cli, capsys):
    cli = make_cli()
    assert cli.run(["knowledge", "implementations", "show", "info_theory_entropy"]) == 0
    out = capsys.readouterr().out
    assert "• Intro" in out
    assert "• Untitled" in out


def test_implementations_show_missing(make_cli, capsys):
    assert make_cli().run(["knowledge", "implementations", "show", "nope"]) == 1
    assert "Implementation 'nope' not found" in capsys.readouterr().out


def test_implementations_validate_clean_node(make_cli, capsys):
    cli = make_cli()
    assert cli.run(["knowledge", "implementations", "validate", "impl_basic"]) == 0
    out = capsys.readouterr().out
    assert "Validation for 'impl_basic':" in out
    assert "Valid: True" in out
    assert "Issues:" not in out
    assert "Warnings:" not in out


def test_implementations_validate_reports_issues_and_warnings(make_cli, capsys):
    cli = make_cli()
    assert cli.run(["knowledge", "implementations", "validate", "kl_divergence"]) == 0
    out = capsys.readouterr().out
    assert "Valid: False" in out
    assert "Issues: Empty content" in out
    assert "Warnings: No learning objectives defined" in out


def test_implementations_validate_unknown_node(make_cli, capsys):
    assert make_cli().run(["knowledge", "implementations", "validate", "nope"]) == 0
    assert "Valid: False" in capsys.readouterr().out


def test_implementations_without_subcommand(make_cli, capsys):
    assert make_cli().run(["knowledge", "implementations"]) == 0
    assert capsys.readouterr().out == ""


def test_knowledge_applications_list(make_cli, capsys):
    assert make_cli().run(["knowledge", "applications", "list"]) == 0
    out = capsys.readouterr().out
    assert "Available Applications:" in out
    assert "Robotics" in out
    assert "Domain: engineering | Difficulty: advanced" in out
    assert "Application neuroscience model" in out
    assert "Domain: general | Difficulty: expert" in out


def test_knowledge_applications_list_empty(make_cli, capsys):
    cli = make_cli()
    cli.knowledge_repo._nodes = {
        k: v for k, v in cli.knowledge_repo._nodes.items() if not k.startswith("app_")
    }
    assert cli.run(["knowledge", "applications", "list"]) == 0
    assert "No applications found." in capsys.readouterr().out


def test_knowledge_applications_show_with_domain(make_cli, capsys):
    assert make_cli().run(["knowledge", "applications", "show", "app_robotics"]) == 0
    out = capsys.readouterr().out
    assert "Application: Robotics" in out
    assert "Domain: engineering" in out
    assert "• Control" in out


def test_knowledge_applications_show_defaults_domain(make_cli, capsys):
    assert make_cli().run(["knowledge", "applications", "show", "app_neuro"]) == 0
    out = capsys.readouterr().out
    assert "Domain: general" in out
    assert "Difficulty: expert" in out


def test_knowledge_applications_show_missing(make_cli, capsys):
    assert make_cli().run(["knowledge", "applications", "show", "nope"]) == 1
    assert "Application 'nope' not found" in capsys.readouterr().out


def test_knowledge_applications_domains(make_cli, capsys):
    assert make_cli().run(["knowledge", "applications", "domains"]) == 0
    out = capsys.readouterr().out
    assert "Applications by Domain:" in out
    assert "• Neuroscience:" in out
    assert "  - Application neuroscience model" in out
    assert "• Engineering:" in out
    assert "  - Robotics" in out
    assert "Psychology" not in out


def test_knowledge_applications_without_subcommand(make_cli, capsys):
    assert make_cli().run(["knowledge", "applications"]) == 0
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# Research, visualization, applications and platform placeholders
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["research", "experiments"], "Experiments coming soon!"),
        (["research", "simulations"], "Simulations coming soon!"),
        (["research", "analyze"], "Analysis tools coming soon!"),
        (["research", "benchmarks"], "Benchmarks coming soon!"),
        (["visualize", "concepts"], "Visualization tools coming soon!"),
        (["applications", "templates"], "Application framework coming soon!"),
        (["platform", "serve"], "Platform tools coming soon!"),
    ],
)
def test_placeholder_commands(make_cli, capsys, argv, expected):
    assert make_cli().run(argv) == 0
    assert expected in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["research", "data", "list"], "Available datasets: (none yet)"),
        (["research", "data", "collect"], "Data collection from various sources"),
        (["research", "data", "validate", "ds-7"], "Validating dataset: ds-7"),
        (["research", "data", "backup", "ds-7"], "Creating backup of dataset: ds-7"),
    ],
)
def test_research_data_commands(make_cli, capsys, argv, expected):
    assert make_cli().run(argv) == 0
    out = capsys.readouterr().out
    assert "Data management tools coming soon!" in out
    assert expected in out


def test_research_data_without_subcommand(make_cli, capsys):
    assert make_cli().run(["research", "data"]) == 0
    out = capsys.readouterr().out
    assert "Data management tools coming soon!" in out
    assert "dataset" not in out.split("storage.")[1]


def test_research_without_subcommand_is_error(make_cli, capsys):
    assert make_cli().run(["research"]) == 1
    assert "Unknown research command: None" in capsys.readouterr().out


# --------------------------------------------------------------------------
# LLM commands
# --------------------------------------------------------------------------


def test_llm_without_subcommand_is_error(make_cli, capsys):
    assert make_cli(FakeLLMClient()).run(["llm"]) == 1
    assert "Unknown LLM command: None" in capsys.readouterr().out


def test_llm_requires_prompt_manager(make_cli, capsys):
    cli = make_cli(FakeLLMClient())
    cli.prompt_manager = None
    assert cli.run(["llm", "chat", "hi"]) == 1
    assert "LLM components not available" in capsys.readouterr().out


def test_chat_single_message_persists_conversation(make_cli, env, capsys):
    llm = FakeLLMClient(responses=["Free energy is a bound."])
    cli = make_cli(llm)

    assert cli.run(["llm", "chat", "What is free energy?"]) == 0
    out = capsys.readouterr().out

    assert llm.init_calls == 1
    assert llm.generate_calls == [("What is free energy?", "gemma3:2b")]
    assert "Started new conversation" in out
    assert "Generating response for: What is free energy?" in out
    assert "Free energy is a bound." in out

    saved = _saved_conversations(env["conv_dir"])
    assert len(saved) == 1
    assert saved[0]["title"] == "AI Chat"
    assert [(m["role"], m["content"]) for m in saved[0]["messages"]] == [
        ("user", "What is free energy?"),
        ("assistant", "Free energy is a bound."),
    ]


@pytest.mark.parametrize(
    "template, conv_type",
    [
        ("explanation", "explanation"),
        ("research", "research"),
        ("implementation", "implementation"),
        ("education", "education"),
    ],
)
def test_chat_with_template(make_cli, env, capsys, template, conv_type):
    llm = FakeLLMClient(responses=["ok"], initialized=True)
    cli = make_cli(llm)

    argv = ["llm", "chat", "hello", "--template", template, "--model", "m:1"]
    assert cli.run(argv) == 0
    out = capsys.readouterr().out

    assert llm.init_calls == 0
    assert llm.generate_calls == [("hello", "m:1")]
    assert f"Started conversation with {template} template" in out

    conversations = cli.conversation_manager.list_conversations()
    assert len(conversations) == 1
    conv = conversations[0]
    assert conv.title == f"AI Chat - {template.title()}"
    assert conv.metadata["type"] == conv_type
    assert [m.role for m in conv.messages] == ["system", "user", "assistant"]


def test_chat_interactive_loop(make_cli, monkeypatch, capsys):
    llm = FakeLLMClient(responses=["first answer", RuntimeError("model crashed")])
    cli = make_cli(llm)
    inputs = iter(["", "  hello  ", "again", "QUIT", "never read"])
    prompts: List[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(inputs)

    monkeypatch.setattr(builtins, "input", fake_input)

    assert cli.run(["llm", "chat"]) == 0
    out = capsys.readouterr().out

    assert prompts == ["You: "] * 4
    assert "Interactive chat with gemma3:2b" in out
    assert "Assistant: first answer" in out
    assert "❌ Error: model crashed" in out
    assert "Goodbye!" in out
    assert llm.generate_calls == [("hello", "gemma3:2b"), ("again", "gemma3:2b")]

    conv = cli.conversation_manager.list_conversations()[0]
    assert [(m.role, m.content) for m in conv.messages] == [
        ("user", "hello"),
        ("assistant", "first answer"),
    ]


def test_chat_interactive_keyboard_interrupt(make_cli, monkeypatch, capsys):
    cli = make_cli(FakeLLMClient())

    def interrupted(prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", interrupted)
    assert cli.run(["llm", "chat"]) == 0
    assert "Chat interrupted. Goodbye!" in capsys.readouterr().out


def test_chat_failure_returns_error_code(make_cli, env, capsys):
    llm = FakeLLMClient(init_error=ConnectionError("ollama down"))
    cli = make_cli(llm)

    assert cli.run(["llm", "chat", "hi"]) == 1
    assert "Chat failed: ollama down" in capsys.readouterr().out
    assert llm.generate_calls == []
    assert _saved_conversations(env["conv_dir"]) == []


def test_generate_plain_prompt(make_cli, capsys):
    llm = FakeLLMClient(responses=["generated text"])
    cli = make_cli(llm)

    assert cli.run(["llm", "generate", "Short prompt", "--model", "tiny"]) == 0
    out = capsys.readouterr().out

    assert llm.init_calls == 1
    assert llm.generate_calls == [("Short prompt", "tiny")]
    assert "Generating with model: tiny" in out
    assert "Prompt: Short prompt\n" in out
    assert "generated text" in out


def test_generate_truncates_long_prompt_preview(make_cli, capsys):
    llm = FakeLLMClient(initialized=True)
    cli = make_cli(llm)
    prompt = "x" * 150

    assert cli.run(["llm", "generate", prompt]) == 0
    out = capsys.readouterr().out

    assert f"Prompt: {'x' * 100}..." in out
    assert llm.generate_calls == [(prompt, "gemma3:2b")]


@pytest.mark.parametrize(
    "template, extra, expected_fragments",
    [
        (
            "active_inference_explanation",
            ["--concept", "Markov blankets", "--audience-level", "expert"],
            ["Markov blankets", "expert", "comprehensive explanation"],
        ),
        (
            "active_inference_explanation",
            [],
            ["Active Inference", "intermediate"],
        ),
        (
            "research_question",
            ["--concept", "precision"],
            ["precision", "artificial intelligence", "5"],
        ),
        (
            "code_implementation",
            ["--concept", "belief updating"],
            ["belief updating", "Python", "Active Inference platform"],
        ),
    ],
)
def test_generate_with_prompt_templates(
    make_cli, capsys, template, extra, expected_fragments
):
    llm = FakeLLMClient(responses=["done"], initialized=True)
    cli = make_cli(llm)

    argv = ["llm", "generate", "ignored prompt", "--template", template, *extra]
    assert cli.run(argv) == 0
    out = capsys.readouterr().out

    sent_prompt, model = llm.generate_calls[0]
    assert model == "gemma3:2b"
    assert sent_prompt != "ignored prompt"
    for fragment in expected_fragments:
        assert fragment in sent_prompt
    assert "Prompt: " in out
    assert out.rstrip().endswith("done")


def test_generate_failure_returns_error_code(make_cli, capsys):
    llm = FakeLLMClient(responses=[TimeoutError("too slow")], initialized=True)
    assert make_cli(llm).run(["llm", "generate", "hi"]) == 1
    assert "Generation failed: too slow" in capsys.readouterr().out


def test_models_list(make_cli, capsys):
    llm = FakeLLMClient(models=["gemma3:2b", "llama3:8b"])
    assert make_cli(llm).run(["llm", "models", "list"]) == 0
    out = capsys.readouterr().out
    assert llm.init_calls == 1
    assert "Available Models:" in out
    assert "• gemma3:2b" in out
    assert "• llama3:8b" in out


def test_models_list_empty(make_cli, capsys):
    assert make_cli(FakeLLMClient(initialized=True)).run(["llm", "models", "list"]) == 0
    out = capsys.readouterr().out
    assert "No models available." in out
    assert "Recommended: gemma3:2b, gemma3:4b" in out


def test_models_info_found(make_cli, capsys):
    llm = FakeLLMClient(info={"family": "gemma", "size": "2b"}, initialized=True)
    assert make_cli(llm).run(["llm", "models", "info", "gemma3:2b"]) == 0
    out = capsys.readouterr().out
    assert llm.info_requests == ["gemma3:2b"]
    assert "Model Information: gemma3:2b" in out
    assert "family: gemma" in out
    assert "size: 2b" in out


def test_models_info_missing(make_cli, capsys):
    llm = FakeLLMClient(info=None, initialized=True)
    assert make_cli(llm).run(["llm", "models", "info", "ghost"]) == 0
    assert "Model ghost not found" in capsys.readouterr().out


def test_models_pull_success(make_cli, capsys):
    llm = FakeLLMClient(pull_ok=True, initialized=True)
    assert make_cli(llm).run(["llm", "models", "pull", "gemma3:4b"]) == 0
    out = capsys.readouterr().out
    assert llm.pulled == ["gemma3:4b"]
    assert "Pulling model: gemma3:4b" in out
    assert "Successfully pulled gemma3:4b" in out


def test_models_pull_failure_returns_error_code(make_cli, capsys):
    llm = FakeLLMClient(pull_ok=False, initialized=True)
    assert make_cli(llm).run(["llm", "models", "pull", "bad"]) == 1
    assert "Failed to pull bad" in capsys.readouterr().out


def test_models_init_failure_returns_error_code(make_cli, capsys):
    llm = FakeLLMClient(init_error=OSError("refused"))
    assert make_cli(llm).run(["llm", "models", "list"]) == 1
    assert "Model command failed: refused" in capsys.readouterr().out


def test_models_without_subcommand_only_initializes(make_cli, capsys):
    llm = FakeLLMClient()
    assert make_cli(llm).run(["llm", "models"]) == 0
    assert llm.init_calls == 1
    assert capsys.readouterr().out == ""


def test_conversation_list_empty(make_cli, capsys):
    assert make_cli(FakeLLMClient()).run(["llm", "conversation", "list"]) == 0
    out = capsys.readouterr().out
    assert "No conversations found." in out
    assert "ai-knowledge llm conversation create" in out


def test_conversation_create_list_export_delete(make_cli, env, capsys):
    cli = make_cli(FakeLLMClient())

    assert cli.run(["llm", "conversation", "create", "My Chat"]) == 0
    out = capsys.readouterr().out
    conv = cli.conversation_manager.list_conversations()[0]
    assert "Created conversation: My Chat" in out
    assert f"ID: {conv.id}" in out
    assert (env["conv_dir"] / f"{conv.id}.json").exists()

    cli.conversation_manager.add_message(conv.id, "user", "Hello there")
    typed = cli.conversation_manager.create_conversation(
        "Typed", metadata={"type": "research"}
    )

    assert cli.run(["llm", "conversation", "list"]) == 0
    out = capsys.readouterr().out
    assert f"• My Chat (ID: {conv.id[:8]}...)" in out
    assert "Messages: 1 | Updated:" in out
    assert f"• Typed (ID: {typed.id[:8]}...)" in out
    assert "Type: research" in out
    assert out.count("Type:") == 1

    assert cli.run(["llm", "conversation", "export", conv.id]) == 0
    out = capsys.readouterr().out
    assert "Conversation Export:" in out
    assert "# My Chat" in out
    assert "## USER" in out
    assert "Hello there" in out

    assert cli.run(["llm", "conversation", "delete", conv.id]) == 0
    out = capsys.readouterr().out
    assert f"Deleted conversation: {conv.id[:8]}..." in out
    assert not (env["conv_dir"] / f"{conv.id}.json").exists()
    assert cli.conversation_manager.get_conversation(conv.id) is None


def test_conversation_delete_and_export_missing(make_cli, capsys):
    cli = make_cli(FakeLLMClient())

    assert cli.run(["llm", "conversation", "delete", "missing-id"]) == 0
    assert "Conversation not found: missing-id" in capsys.readouterr().out

    assert cli.run(["llm", "conversation", "export", "missing-id"]) == 0
    assert "Conversation not found: missing-id" in capsys.readouterr().out


def test_conversation_without_subcommand(make_cli, capsys):
    assert make_cli(FakeLLMClient()).run(["llm", "conversation"]) == 0
    assert capsys.readouterr().out == ""
