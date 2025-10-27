"""
Documentation Generator

Core class for generating comprehensive documentation for the Active Inference
Knowledge Environment. Handles API documentation, knowledge base documentation,
and various report generation.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import subprocess
import sys

from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader

import logging
from ...knowledge.repository import KnowledgeRepository

logger = logging.getLogger(__name__)


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation"""
    output_dir: Path
    template_dir: Optional[Path] = None
    api_docs_enabled: bool = True
    knowledge_docs_enabled: bool = True
    include_private: bool = False
    theme: str = "sphinx_rtd_theme"
    extensions: List[str] = None

    def __post_init__(self):
        if self.extensions is None:
            self.extensions = [
                'sphinx.ext.autodoc',
                'sphinx.ext.autosummary',
                'sphinx.ext.viewcode',
                'sphinx.ext.napoleon',
                'sphinx.ext.intersphinx',
                'sphinx.ext.mathjax',
                'myst_nb',
                'nbsphinx'
            ]


class DocumentationGenerator:
    """
    Comprehensive documentation generator for the Active Inference Knowledge Environment.

    Generates API documentation, knowledge base documentation, learning paths,
    concept maps, and various analysis reports.
    """

    def __init__(self, output_dir: Union[str, Path], config: Optional[DocumentationConfig] = None):
        """
        Initialize documentation generator

        Args:
            output_dir: Directory for generated documentation
            config: Documentation configuration
        """
        self.output_dir = Path(output_dir)
        self.config = config or DocumentationConfig(output_dir=self.output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up Jinja2 template environment
        self.template_dir = self.config.template_dir or Path(__file__).parent / "templates"
        if self.template_dir.exists():
            self.jinja_env = Environment(loader=FileSystemLoader(self.template_dir))
        else:
            self.jinja_env = None

        # Set up logging
        self.logger = setup_logger(__name__)

        self.logger.info(f"Documentation generator initialized for {self.output_dir}")

    def generate_api_docs(self, source_dir: Union[str, Path]) -> None:
        """
        Generate API documentation from source code

        Args:
            source_dir: Source code directory to document
        """
        source_path = Path(source_dir)
        api_output_dir = self.output_dir / "api"

        self.logger.info(f"Generating API documentation from {source_path}")

        try:
            # Create API documentation structure
            api_output_dir.mkdir(exist_ok=True)

            # Generate module documentation
            self._generate_module_docs(source_path, api_output_dir)

            # Generate package index
            self._generate_package_index(source_path, api_output_dir)

            # Copy any existing documentation
            self._copy_existing_docs(source_path, api_output_dir)

        except Exception as e:
            self.logger.error(f"Error generating API documentation: {e}")
            raise

    def _generate_module_docs(self, source_path: Path, output_dir: Path) -> None:
        """Generate documentation for individual modules"""
        # Use sphinx-apidoc to generate RST files
        try:
            cmd = [
                sys.executable, "-m", "sphinx.ext.apidoc",
                "-f", "-e", "-T",  # Force, separate modules, no table of contents
                "-o", str(output_dir),
                str(source_path),
                "**/test_*",
                "**/tests/**",
                "**/__pycache__/**"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.output_dir)

            if result.returncode == 0:
                self.logger.info("Module documentation generated successfully")
            else:
                self.logger.warning(f"sphinx-apidoc completed with warnings: {result.stderr}")

        except Exception as e:
            self.logger.error(f"Error running sphinx-apidoc: {e}")
            # Fallback to manual generation
            self._generate_manual_module_docs(source_path, output_dir)

    def _generate_manual_module_docs(self, source_path: Path, output_dir: Path) -> None:
        """Manual fallback for module documentation generation"""
        self.logger.info("Generating manual module documentation")

        # Find all Python modules
        modules = list(source_path.rglob("*.py"))
        modules = [m for m in modules if not any(part.startswith('__') for part in m.parts)]
        modules = [m for m in modules if not any(part.startswith('test_') for part in m.parts)]

        for module_path in modules:
            try:
                self._generate_single_module_doc(module_path, output_dir)
            except Exception as e:
                self.logger.warning(f"Could not generate docs for {module_path}: {e}")

    def _generate_single_module_doc(self, module_path: Path, output_dir: Path) -> None:
        """Generate documentation for a single module"""
        relative_path = module_path.relative_to(module_path.parent.parent.parent)
        doc_path = output_dir / relative_path.with_suffix('.rst')
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        module_name = str(relative_path.with_suffix('')).replace('/', '.')
        module_name = module_name.replace('src.', '')

        content = f"""
{module_name}
{'=' * len(module_name)}

.. automodule:: {module_name}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. autoclass:: {module_name}.*
   :members:
   :undoc-members:
   :show-inheritance:
"""

        with open(doc_path, 'w') as f:
            f.write(content)

    def _generate_package_index(self, source_path: Path, output_dir: Path) -> None:
        """Generate package index documentation"""
        index_path = output_dir / "index.rst"

        # Find main package
        main_package = None
        for item in source_path.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                main_package = item.name
                break

        if not main_package:
            return

        content = f"""
API Reference
=============

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   {main_package}/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""

        with open(index_path, 'w') as f:
            f.write(content)

    def _copy_existing_docs(self, source_path: Path, output_dir: Path) -> None:
        """Copy any existing documentation files"""
        docs_source = source_path / "docs"
        if docs_source.exists():
            shutil.copytree(docs_source, output_dir / "docs", dirs_exist_ok=True)

    def generate_learning_paths(self, repo: KnowledgeRepository) -> None:
        """
        Generate learning path documentation

        Args:
            repo: Knowledge repository instance
        """
        paths_dir = self.output_dir / "knowledge" / "learning_paths"
        paths_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Generating learning path documentation")

        # Get all learning paths
        paths = repo.get_learning_paths()

        for path in paths:
            self._generate_single_learning_path_doc(repo, path, paths_dir)

        # Generate learning paths index
        self._generate_learning_paths_index(paths, paths_dir)

    def _generate_single_learning_path_doc(self, repo: KnowledgeRepository, path, output_dir: Path) -> None:
        """Generate documentation for a single learning path"""
        doc_path = output_dir / f"{path.id}.rst"

        # Get path nodes
        nodes = []
        for node_id in path.nodes:
            node = repo.get_node(node_id)
            if node:
                nodes.append(node)

        # Generate content
        content = f"""
{path.name}
{'=' * len(path.name)}

{path.description}

**Difficulty:** {path.difficulty.value.title()}
**Estimated Time:** {path.estimated_hours or 'Not specified'} hours

.. toctree::
   :maxdepth: 2

"""

        for node in nodes:
            content += f"   ../nodes/{node.id}\n"

        content += f"""

Validation Report
-----------------

"""

        # Add validation information
        validation = repo.validate_learning_path(path.id)
        if validation['valid']:
            content += "✅ **Path validation: PASSED**\n\n"
        else:
            content += "⚠️  **Path validation: ISSUES FOUND**\n\n"
            for issue in validation['issues']:
                content += f"- {issue}\n"

        with open(doc_path, 'w') as f:
            f.write(content)

    def _generate_learning_paths_index(self, paths: List, output_dir: Path) -> None:
        """Generate index for all learning paths"""
        index_path = output_dir / "index.rst"

        content = """
Learning Paths
==============

Structured learning tracks for Active Inference and the Free Energy Principle.

.. toctree::
   :maxdepth: 2
   :caption: Available Learning Paths:

"""

        for path in sorted(paths, key=lambda p: p.name):
            content += f"   {path.id}\n"

        content += """

Search Learning Paths
---------------------

Use the search functionality to find learning paths by topic, difficulty, or content type.

.. seealso::

   For detailed information about individual knowledge nodes, see the :doc:`../nodes/index` section.
"""

        with open(index_path, 'w') as f:
            f.write(content)

    def generate_concept_maps(self, repo: KnowledgeRepository) -> None:
        """
        Generate concept map visualizations and documentation

        Args:
            repo: Knowledge repository instance
        """
        concepts_dir = self.output_dir / "knowledge" / "concepts"
        concepts_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Generating concept map documentation")

        # Generate concept dependency graphs
        self._generate_concept_graphs(repo, concepts_dir)

        # Generate concept index
        self._generate_concept_index(repo, concepts_dir)

    def _generate_concept_graphs(self, repo: KnowledgeRepository, output_dir: Path) -> None:
        """Generate concept dependency graphs"""
        # Get all nodes
        nodes = list(repo._nodes.values())

        # Generate prerequisite graph
        graph_data = repo.export_knowledge_graph(format='json')

        # Save graph data
        graph_file = output_dir / "knowledge_graph.json"
        with open(graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)

        # Generate graph documentation
        doc_path = output_dir / "knowledge_graph.rst"
        content = f"""
Knowledge Graph
===============

Interactive visualization of the knowledge dependency structure.

The knowledge graph contains {graph_data['metadata']['total_nodes']} knowledge nodes
and represents the prerequisite relationships and learning paths.

.. raw:: html

   <div id="knowledge-graph" style="width: 100%; height: 600px;">
   <p>Knowledge graph visualization will be rendered here.</p>
   </div>

   <script>
   // Graph visualization code would go here
   // This would typically use D3.js or similar library
   </script>

Graph Statistics
----------------

- **Total Nodes:** {graph_data['metadata']['total_nodes']}
- **Total Learning Paths:** {graph_data['metadata']['total_paths']}
- **Content Types:** {', '.join(graph_data['metadata']['content_types'])}
- **Difficulty Levels:** {', '.join(graph_data['metadata']['difficulties'])}

Node Types
----------

"""

        # Group nodes by content type
        content_types = {}
        for node_data in graph_data['nodes']:
            ct = node_data['content_type']
            if ct not in content_types:
                content_types[ct] = []
            content_types[ct].append(node_data)

        for content_type, nodes_list in content_types.items():
            content += f"""
{content_type.title()} ({len(nodes_list)} nodes)
{'~' * (len(content_type) + 20)}

"""

            for node_data in sorted(nodes_list, key=lambda x: x['title']):
                content += f"- **{node_data['title']}**\n"
                if node_data.get('description'):
                    content += f"  {node_data['description'][:100]}...\n"
                content += "\n"

        with open(doc_path, 'w') as f:
            f.write(content)

    def _generate_concept_index(self, repo: KnowledgeRepository, output_dir: Path) -> None:
        """Generate concept index documentation"""
        index_path = output_dir / "index.rst"

        content = """
Concept Maps
============

Visual representations of knowledge relationships and dependencies.

.. toctree::
   :maxdepth: 2

   knowledge_graph
   topic_maps
   learning_flow

Overview
--------

Concept maps help visualize:

- **Prerequisite relationships** between knowledge nodes
- **Learning path structures** and dependencies
- **Topic clusters** and related concepts
- **Knowledge gaps** and missing connections

.. seealso::

   For detailed node information, see the :doc:`../nodes/index` section.
"""

        with open(index_path, 'w') as f:
            f.write(content)

    def generate_statistics(self, repo: KnowledgeRepository) -> None:
        """
        Generate knowledge repository statistics

        Args:
            repo: Knowledge repository instance
        """
        stats_dir = self.output_dir / "knowledge" / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Generating statistics documentation")

        # Get comprehensive statistics
        stats = repo.get_statistics()

        # Generate statistics documentation
        self._generate_stats_report(stats, stats_dir)

        # Generate trends and analysis
        self._generate_trends_analysis(repo, stats, stats_dir)

    def _generate_stats_report(self, stats: Dict[str, Any], output_dir: Path) -> None:
        """Generate comprehensive statistics report"""
        report_path = output_dir / "index.rst"

        content = f"""
Knowledge Repository Statistics
==============================

Comprehensive analysis of the knowledge repository structure and content.

Overview
--------

The repository contains **{stats['total_nodes']}** knowledge nodes
organized into **{stats['total_paths']}** learning paths.

Content Distribution
--------------------

"""

        # Content type distribution
        content += "Content Types\n~~~~~~~~~~~~~\n\n"
        for content_type, count in stats['content_types'].items():
            content += f"- **{content_type.title()}**: {count} nodes\n"
        content += "\n"

        # Difficulty distribution
        content += "Difficulty Levels\n~~~~~~~~~~~~~~~~~\n\n"
        for difficulty, count in stats['difficulties'].items():
            content += f"- **{difficulty.title()}**: {count} nodes\n"
        content += "\n"

        # Popular tags
        if stats['top_tags']:
            content += "Popular Topics\n~~~~~~~~~~~~~~\n\n"
            for tag, count in stats['top_tags'][:15]:
                content += f"- **{tag}**: {count} nodes\n"
            content += "\n"

        # Metadata
        if stats.get('metadata'):
            content += "Additional Information\n~~~~~~~~~~~~~~~~~~~~~~\n\n"
            for key, value in stats['metadata'].items():
                if isinstance(value, (str, int, float)):
                    content += f"- **{key}**: {value}\n"
            content += "\n"

        content += """
Analysis
--------

This section provides insights into:

- **Coverage gaps** in specific topic areas
- **Learning path completeness** and validation status
- **Content quality metrics** and improvement opportunities
- **Usage patterns** and popular content areas

.. toctree::
   :maxdepth: 2

   trends
   quality_metrics
   usage_patterns
"""

        with open(report_path, 'w') as f:
            f.write(content)

    def _generate_trends_analysis(self, repo: KnowledgeRepository, stats: Dict[str, Any], output_dir: Path) -> None:
        """Generate trends and analysis documentation"""
        trends_path = output_dir / "trends.rst"

        content = """
Trends and Analysis
===================

Analysis of knowledge repository trends, patterns, and insights.

Content Growth
--------------

.. note::
   Trend analysis will be implemented in future versions.

Quality Metrics
---------------

.. note::
   Quality metrics analysis will be implemented in future versions.

Usage Patterns
--------------

.. note::
   Usage pattern analysis will be implemented in future versions.

Recommendations
---------------

Based on current repository analysis:

1. **Expand Coverage**: Consider adding more intermediate-level content
2. **Cross-linking**: Improve connections between related topics
3. **Validation**: Regular validation of learning path integrity
4. **Assessment**: Add more interactive assessments and exercises

"""

        with open(trends_path, 'w') as f:
            f.write(content)

    def generate_search_index(self, repo: KnowledgeRepository) -> None:
        """
        Generate search index for the knowledge repository

        Args:
            repo: Knowledge repository instance
        """
        search_dir = self.output_dir / "knowledge" / "search"
        search_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Generating search index")

        # Generate search interface documentation
        search_doc_path = search_dir / "index.rst"

        content = """
Search and Discovery
====================

Advanced search capabilities for the knowledge repository.

Search Features
---------------

- **Full-text search** across all knowledge nodes
- **Filter by content type** (foundations, mathematics, implementations, etc.)
- **Filter by difficulty level** (beginner, intermediate, advanced, expert)
- **Tag-based filtering** for topic-specific searches
- **Prerequisite-aware search** to find foundational concepts

Search Syntax
-------------

.. code-block:: text

   entropy                    # Search for exact term
   "free energy"             # Search for phrase
   entropy OR information    # Boolean search
   type:mathematics          # Filter by content type
   difficulty:beginner       # Filter by difficulty
   tag:probability           # Filter by tag

Examples
--------

.. code-block:: bash

   # Search for information theory concepts
   ai-knowledge search "information theory"

   # Find beginner-friendly implementations
   ai-knowledge search implementation --type implementation --difficulty beginner

   # Search within specific topics
   ai-knowledge search "Bayesian" --tag probability

Search Interface
----------------

The search interface provides:

- **Real-time results** as you type
- **Result previews** with descriptions and metadata
- **Breadcrumb navigation** to related concepts
- **Export capabilities** for search results

.. seealso::

   For command-line search, see :doc:`../cli/search`.
"""

        with open(search_doc_path, 'w') as f:
            f.write(content)

    def generate_all_docs(self, source_dir: Union[str, Path], repo_path: Union[str, Path]) -> None:
        """
        Generate all documentation types

        Args:
            source_dir: Source code directory
            repo_path: Knowledge repository path
        """
        self.logger.info("Generating comprehensive documentation")

        # Generate API documentation
        if self.config.api_docs_enabled:
            self.generate_api_docs(source_dir)

        # Generate knowledge documentation
        if self.config.knowledge_docs_enabled:
            # Initialize knowledge repository
            config = DocumentationConfig(output_dir=self.output_dir)
            repo_config = KnowledgeRepositoryConfig(
                root_path=Path(repo_path),
                auto_index=True
            )
            repo = KnowledgeRepository(repo_config)

            self.generate_learning_paths(repo)
            self.generate_concept_maps(repo)
            self.generate_statistics(repo)
            self.generate_search_index(repo)

        self.logger.info("Comprehensive documentation generation completed")
