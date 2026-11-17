"""Coverage tests for the documentation generator."""

import json
import subprocess
from pathlib import Path

import pytest

from active_inference.knowledge.repository import (
    KnowledgeRepository,
    KnowledgeRepositoryConfig,
)
from active_inference.tools.documentation import generator as gen_module
from active_inference.tools.documentation.generator import (
    DocumentationConfig,
    DocumentationGenerator,
)

LONG_DESCRIPTION = "Entropy measures uncertainty. " * 6


def _node(node_id, title, content_type, difficulty, prerequisites, tags, desc):
    return {
        "id": node_id,
        "title": title,
        "content_type": content_type,
        "difficulty": difficulty,
        "description": desc,
        "prerequisites": prerequisites,
        "tags": tags,
        "learning_objectives": ["understand"],
        "content": {"overview": "text"},
        "metadata": {},
    }


def build_knowledge_fixture(root: Path) -> Path:
    """Write a tiny knowledge repository (two nodes, two paths)."""
    (root / "foundations").mkdir(parents=True)
    (root / "mathematics").mkdir()
    (root / "metadata").mkdir()
    (root / "foundations" / "entropy.json").write_text(
        json.dumps(
            _node(
                "entropy",
                "Entropy Basics",
                "foundation",
                "beginner",
                [],
                ["information", "probability"],
                LONG_DESCRIPTION,
            )
        )
    )
    (root / "mathematics" / "kl.json").write_text(
        json.dumps(
            _node(
                "kl",
                "KL Divergence",
                "mathematics",
                "intermediate",
                ["entropy"],
                ["information"],
                "",
            )
        )
    )
    (root / "metadata" / "repository.json").write_text(
        json.dumps({"version": "1.0", "nodes": 2, "nested": {"skip": True}})
    )
    (root / "metadata" / "learning_paths.json").write_text(
        json.dumps(
            [
                {
                    "id": "basics",
                    "name": "Basics Path",
                    "description": "Start here.",
                    "nodes": ["entropy", "kl"],
                    "estimated_hours": 3,
                    "difficulty": "beginner",
                },
                {
                    "id": "broken",
                    "name": "Advanced Path",
                    "description": "Has problems.",
                    "nodes": ["kl", "ghost"],
                    "difficulty": "advanced",
                },
            ]
        )
    )
    return root


@pytest.fixture
def repo(tmp_path):
    root = build_knowledge_fixture(tmp_path / "knowledge")
    return KnowledgeRepository(KnowledgeRepositoryConfig(root_path=root))


@pytest.fixture
def generator(tmp_path):
    return DocumentationGenerator(tmp_path / "out")


@pytest.fixture
def fake_apidoc(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(gen_module.subprocess, "run", fake_run)
    return calls


def test_config_defaults_and_custom_extensions(tmp_path):
    default = DocumentationConfig(output_dir=tmp_path)
    assert "sphinx.ext.autodoc" in default.extensions
    assert len(default.extensions) == 8
    custom = DocumentationConfig(output_dir=tmp_path, extensions=["only.this"])
    assert custom.extensions == ["only.this"]


def test_init_creates_output_dir_without_templates(tmp_path):
    out = tmp_path / "nested" / "out"
    generator = DocumentationGenerator(str(out))
    assert out.is_dir()
    assert generator.output_dir == out
    assert generator.config.output_dir == out
    # The packaged templates directory does not exist
    assert generator.jinja_env is None


def test_init_builds_jinja_environment_for_existing_templates(tmp_path, monkeypatch):
    templates = tmp_path / "templates"
    templates.mkdir()

    class FakeLoader:
        def __init__(self, path):
            self.path = path

    class FakeEnvironment:
        def __init__(self, loader):
            self.loader = loader

    monkeypatch.setattr(gen_module, "Environment", FakeEnvironment)
    monkeypatch.setattr(gen_module, "FileSystemLoader", FakeLoader)
    config = DocumentationConfig(output_dir=tmp_path / "out", template_dir=templates)
    generator = DocumentationGenerator(tmp_path / "out", config)
    assert isinstance(generator.jinja_env, FakeEnvironment)
    assert generator.jinja_env.loader.path == templates


def test_generate_api_docs_runs_apidoc_and_writes_index(
    tmp_path, generator, fake_apidoc
):
    source = tmp_path / "src"
    (source / "mypkg").mkdir(parents=True)

    generator.generate_api_docs(source)

    api_dir = generator.output_dir / "api"
    cmd, kwargs = fake_apidoc[0]
    assert cmd[1:3] == ["-m", "sphinx.ext.apidoc"]
    assert cmd[cmd.index("-o") + 1] == str(api_dir)
    assert str(source) in cmd
    assert kwargs["cwd"] == generator.output_dir
    index = (api_dir / "index.rst").read_text()
    assert "mypkg/index" in index
    assert ":ref:`genindex`" in index
    assert not (api_dir / "docs").exists()


def test_copy_existing_docs(tmp_path, generator):
    source = tmp_path / "src"
    (source / "docs").mkdir(parents=True)
    (source / "docs" / "guide.md").write_text("# Guide\n")
    out = tmp_path / "api"
    out.mkdir()
    generator._copy_existing_docs(source, out)
    assert (out / "docs" / "guide.md").read_text() == "# Guide\n"


def test_generate_module_docs_logs_apidoc_warning(
    tmp_path, generator, monkeypatch, caplog
):
    monkeypatch.setattr(
        gen_module.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 2, "", "bad things"),
    )
    with caplog.at_level("WARNING"):
        generator._generate_module_docs(tmp_path, tmp_path / "api")
    assert "sphinx-apidoc completed with warnings: bad things" in caplog.text


def test_generate_module_docs_falls_back_to_manual(
    tmp_path_factory, generator, monkeypatch
):
    def boom(cmd, **kwargs):
        raise FileNotFoundError("no sphinx")

    monkeypatch.setattr(gen_module.subprocess, "run", boom)
    # Avoid tmp_path: its directory name starts with "test_", which the manual
    # generator treats as a test module path component.
    base = tmp_path_factory.mktemp("manual")
    source = base / "src"
    pkg = source / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "module.py").write_text('"""doc"""\n')
    (pkg / "test_skip.py").write_text("")
    out = base / "api"

    generator._generate_module_docs(source, out)

    doc = out / "src" / "pkg" / "module.rst"
    text = doc.read_text()
    assert "pkg.module\n==========" in text
    assert ".. automodule:: pkg.module" in text
    assert not (out / "src" / "pkg" / "__init__.rst").exists()
    assert not (out / "src" / "pkg" / "test_skip.rst").exists()


def test_manual_module_docs_logs_per_module_failure(
    tmp_path_factory, generator, caplog
):
    base = tmp_path_factory.mktemp("manualfail")
    source = base / "src"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "module.py").write_text("")
    out = base / "api"
    out.mkdir()
    (out / "src").write_text("a file blocks the directory")

    with caplog.at_level("WARNING"):
        generator._generate_manual_module_docs(source, out)
    assert "Could not generate docs for" in caplog.text


def test_package_index_skipped_without_package(tmp_path, generator):
    source = tmp_path / "flat"
    (source / "__pycache__").mkdir(parents=True)
    (source / "loose.py").write_text("")
    out = tmp_path / "api"
    out.mkdir()
    generator._generate_package_index(source, out)
    assert not (out / "index.rst").exists()


def test_generate_api_docs_reraises_errors(tmp_path, generator, fake_apidoc):
    with pytest.raises(FileNotFoundError):
        generator.generate_api_docs(tmp_path / "does-not-exist")


def test_generate_learning_paths(generator, repo):
    generator.generate_learning_paths(repo)
    paths_dir = generator.output_dir / "knowledge" / "learning_paths"

    basics = (paths_dir / "basics.rst").read_text()
    assert "Basics Path\n===========" in basics
    assert "**Difficulty:** Beginner" in basics
    assert "**Estimated Time:** 3 hours" in basics
    assert "   ../nodes/entropy\n   ../nodes/kl\n" in basics
    assert "Path validation: PASSED" in basics

    broken = (paths_dir / "broken.rst").read_text()
    assert "**Estimated Time:** Not specified hours" in broken
    assert "../nodes/ghost" not in broken
    assert "Path validation: ISSUES FOUND" in broken
    assert "- Unsatisfied prerequisite entropy for kl" in broken
    assert "- Missing node: ghost" in broken

    index = (paths_dir / "index.rst").read_text()
    # Sorted by name: "Advanced Path" before "Basics Path"
    assert index.index("   broken\n") < index.index("   basics\n")


def test_generate_concept_maps(generator, repo):
    generator.generate_concept_maps(repo)
    concepts = generator.output_dir / "knowledge" / "concepts"

    graph = json.loads((concepts / "knowledge_graph.json").read_text())
    assert graph["metadata"]["total_nodes"] == 2
    assert {n["id"] for n in graph["nodes"]} == {"entropy", "kl"}

    doc = (concepts / "knowledge_graph.rst").read_text()
    assert "contains 2 knowledge nodes" in doc
    assert "- **Total Learning Paths:** 2" in doc
    assert "Foundation (1 nodes)" in doc
    assert "Mathematics (1 nodes)" in doc
    assert "- **Entropy Basics**\n" in doc
    assert f"  {LONG_DESCRIPTION[:100]}...\n" in doc
    # Empty description gets no preview line
    assert "- **KL Divergence**\n\n" in doc

    index = (concepts / "index.rst").read_text()
    assert "Concept Maps" in index and "knowledge_graph" in index


def test_generate_statistics(generator, repo):
    generator.generate_statistics(repo)
    stats_dir = generator.output_dir / "knowledge" / "statistics"

    report = (stats_dir / "index.rst").read_text()
    assert "contains **2** knowledge nodes" in report
    assert "organized into **2** learning paths" in report
    assert "- **Foundation**: 1 nodes" in report
    assert "- **Intermediate**: 1 nodes" in report
    assert "- **information**: 2 nodes" in report
    assert "- **version**: 1.0" in report
    assert "- **nodes**: 2" in report
    assert "nested" not in report

    trends = (stats_dir / "trends.rst").read_text()
    assert "Trends and Analysis" in trends


def test_stats_report_without_tags_or_metadata(tmp_path, generator):
    stats = {
        "total_nodes": 0,
        "total_paths": 0,
        "content_types": {},
        "difficulties": {},
        "top_tags": [],
        "metadata": {},
    }
    generator._generate_stats_report(stats, tmp_path)
    report = (tmp_path / "index.rst").read_text()
    assert "Popular Topics" not in report
    assert "Additional Information" not in report


def test_generate_search_index(generator, repo):
    generator.generate_search_index(repo)
    text = (generator.output_dir / "knowledge" / "search" / "index.rst").read_text()
    assert "Search and Discovery" in text
    assert "difficulty:beginner" in text


def test_generate_all_docs(tmp_path, fake_apidoc):
    build_knowledge_fixture(tmp_path / "kb")
    source = tmp_path / "src"
    (source / "pkg").mkdir(parents=True)
    generator = DocumentationGenerator(tmp_path / "out")

    generator.generate_all_docs(source, tmp_path / "kb")

    out = generator.output_dir
    assert len(fake_apidoc) == 1
    assert (out / "api" / "index.rst").exists()
    assert (out / "knowledge" / "learning_paths" / "basics.rst").exists()
    assert (out / "knowledge" / "concepts" / "knowledge_graph.json").exists()
    assert (out / "knowledge" / "statistics" / "trends.rst").exists()
    assert (out / "knowledge" / "search" / "index.rst").exists()


def test_generate_all_docs_respects_disabled_sections(tmp_path, fake_apidoc):
    config = DocumentationConfig(
        output_dir=tmp_path / "out",
        api_docs_enabled=False,
        knowledge_docs_enabled=False,
    )
    generator = DocumentationGenerator(tmp_path / "out", config)
    generator.generate_all_docs(tmp_path / "src", tmp_path / "kb")
    assert fake_apidoc == []
    assert list(generator.output_dir.iterdir()) == []
