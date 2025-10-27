"""
Tools - Documentation Generation

Automated documentation generation for Active Inference components and knowledge.
Provides tools for generating API documentation, knowledge base documentation,
and educational content from code and structured data.
"""

import logging
import inspect
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation"""
    output_format: str = "markdown"  # markdown, html, json
    include_examples: bool = True
    include_api_docs: bool = True
    include_tutorials: bool = False
    template_path: Optional[Path] = None


class DocumentationGenerator:
    """Generates documentation from code and data"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates: Dict[str, str] = {}

        logger.info("DocumentationGenerator initialized")

    def extract_function_docs(self, function: Callable) -> Dict[str, Any]:
        """Extract documentation from a function"""
        try:
            doc = {
                "name": function.__name__,
                "signature": str(inspect.signature(function)),
                "docstring": inspect.getdoc(function) or "",
                "module": getattr(function, "__module__", ""),
                "is_method": inspect.ismethod(function),
                "is_classmethod": inspect.isclass(function) if hasattr(inspect, 'isclass') else False
            }

            return doc

        except Exception as e:
            logger.error(f"Failed to extract docs from function {function}: {e}")
            return {}

    def extract_class_docs(self, cls: type) -> Dict[str, Any]:
        """Extract documentation from a class"""
        try:
            doc = {
                "name": cls.__name__,
                "docstring": inspect.getdoc(cls) or "",
                "module": getattr(cls, "__module__", ""),
                "methods": [],
                "attributes": []
            }

            # Extract methods
            for name, member in inspect.getmembers(cls):
                if inspect.isfunction(member) or inspect.ismethod(member):
                    if not name.startswith('_'):  # Skip private methods
                        method_doc = self.extract_function_docs(member)
                        if method_doc:
                            doc["methods"].append(method_doc)

            # Extract class attributes (this is simplified)
            if hasattr(cls, "__annotations__"):
                doc["attributes"] = list(cls.__annotations__.keys())

            return doc

        except Exception as e:
            logger.error(f"Failed to extract docs from class {cls}: {e}")
            return {}

    def generate_api_docs(self, module_name: str, output_path: Path) -> bool:
        """Generate API documentation for a module"""
        try:
            # Import module
            module = __import__(module_name, fromlist=[''])

            # Extract documentation
            api_docs = {
                "module": module_name,
                "classes": [],
                "functions": [],
                "timestamp": datetime.now().isoformat()
            }

            # Extract classes
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module_name:
                    class_doc = self.extract_class_docs(obj)
                    if class_doc:
                        api_docs["classes"].append(class_doc)

            # Extract functions
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and obj.__module__ == module_name:
                    if not name.startswith('_'):  # Skip private functions
                        func_doc = self.extract_function_docs(obj)
                        if func_doc:
                            api_docs["functions"].append(func_doc)

            # Generate output
            output_content = self._format_api_docs(api_docs)

            # Save to file
            with open(output_path, 'w') as f:
                f.write(output_content)

            logger.info(f"Generated API docs for {module_name} at {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate API docs for {module_name}: {e}")
            return False

    def _format_api_docs(self, api_docs: Dict[str, Any]) -> str:
        """Format API documentation as markdown"""
        lines = []

        lines.append(f"# API Documentation - {api_docs['module']}")
        lines.append("")
        lines.append(f"Generated on: {api_docs['timestamp']}")
        lines.append("")

        # Classes
        if api_docs["classes"]:
            lines.append("## Classes")
            lines.append("")

            for cls in api_docs["classes"]:
                lines.append(f"### {cls['name']}")
                lines.append("")

                if cls["docstring"]:
                    lines.append(cls["docstring"])
                    lines.append("")

                if cls["methods"]:
                    lines.append("#### Methods:")
                    lines.append("")

                    for method in cls["methods"]:
                        lines.append(f"**{method['name']}**{method['signature']}")
                        if method["docstring"]:
                            lines.append("")
                            lines.append(method["docstring"])
                        lines.append("")

        # Functions
        if api_docs["functions"]:
            lines.append("## Functions")
            lines.append("")

            for func in api_docs["functions"]:
                lines.append(f"### {func['name']}")
                lines.append(f"**Signature:** {func['signature']}")
                lines.append("")

                if func["docstring"]:
                    lines.append(func["docstring"])
                    lines.append("")

        return "\n".join(lines)


class KnowledgeDocBuilder:
    """Builds documentation from knowledge repository"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.documentation_generator = DocumentationGenerator(config.get("generator", {}))

        logger.info("KnowledgeDocBuilder initialized")

    def build_knowledge_docs(self, knowledge_nodes: Dict[str, Any], output_dir: Path) -> int:
        """Build documentation from knowledge nodes"""
        logger.info(f"Building knowledge documentation for {len(knowledge_nodes)} nodes")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        docs_generated = 0

        # Group nodes by type
        nodes_by_type = {}
        for node_id, node_data in knowledge_nodes.items():
            node_type = node_data.get("content_type", "unknown")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append((node_id, node_data))

        # Generate docs for each type
        for node_type, nodes in nodes_by_type.items():
            docs_content = self._generate_type_docs(node_type, nodes)

            output_file = output_dir / f"{node_type}_documentation.md"

            try:
                with open(output_file, 'w') as f:
                    f.write(docs_content)
                docs_generated += 1
                logger.debug(f"Generated docs for {node_type}: {len(nodes)} nodes")
            except Exception as e:
                logger.error(f"Failed to save docs for {node_type}: {e}")

        # Generate index
        index_content = self._generate_index_docs(nodes_by_type)
        index_file = output_dir / "index.md"

        try:
            with open(index_file, 'w') as f:
                f.write(index_content)
            docs_generated += 1
        except Exception as e:
            logger.error(f"Failed to save index docs: {e}")

        logger.info(f"Knowledge documentation built: {docs_generated} files")
        return docs_generated

    def _generate_type_docs(self, node_type: str, nodes: List) -> str:
        """Generate documentation for a specific node type"""
        lines = []

        lines.append(f"# {node_type.title()} Documentation")
        lines.append("")
        lines.append(f"Content type: {node_type}")
        lines.append(f"Number of nodes: {len(nodes)}")
        lines.append("")
        lines.append(f"Generated on: {datetime.now().isoformat()}")
        lines.append("")

        # Sort nodes by difficulty or title
        sorted_nodes = sorted(nodes, key=lambda x: (x[1].get("difficulty", ""), x[1].get("title", "")))

        for node_id, node_data in sorted_nodes:
            lines.append(f"## {node_data.get('title', node_id)}")
            lines.append("")

            if "description" in node_data:
                lines.append(node_data["description"])
                lines.append("")

            # Add metadata
            metadata_items = []
            if "difficulty" in node_data:
                metadata_items.append(f"Difficulty: {node_data['difficulty']}")
            if "prerequisites" in node_data and node_data["prerequisites"]:
                metadata_items.append(f"Prerequisites: {', '.join(node_data['prerequisites'])}")
            if "tags" in node_data and node_data["tags"]:
                metadata_items.append(f"Tags: {', '.join(node_data['tags'])}")

            if metadata_items:
                lines.append("**Metadata:** " + " | ".join(metadata_items))
                lines.append("")

            # Add learning objectives if available
            if "learning_objectives" in node_data and node_data["learning_objectives"]:
                lines.append("**Learning Objectives:**")
                for obj in node_data["learning_objectives"]:
                    lines.append(f"- {obj}")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _generate_index_docs(self, nodes_by_type: Dict[str, List]) -> str:
        """Generate index documentation"""
        lines = []

        lines.append("# Knowledge Base Documentation Index")
        lines.append("")
        lines.append(f"Generated on: {datetime.now().isoformat()}")
        lines.append("")

        total_nodes = sum(len(nodes) for nodes in nodes_by_type.values())
        lines.append(f"Total content nodes: {total_nodes}")
        lines.append("")

        lines.append("## Content Types")
        lines.append("")

        for node_type, nodes in nodes_by_type.items():
            lines.append(f"- **{node_type.title()}**: {len(nodes)} nodes")
            lines.append("")

        lines.append("## Navigation")
        lines.append("")
        lines.append("See individual documentation files for detailed content:")
        lines.append("")

        for node_type in nodes_by_type.keys():
            lines.append(f"- [{node_type.title()} Documentation]({node_type}_documentation.md)")
            lines.append("")

        return "\n".join(lines)

    def generate_tutorial_docs(self, tutorial_config: Dict[str, Any], output_path: Path) -> bool:
        """Generate tutorial documentation"""
        try:
            tutorial_content = self._build_tutorial_content(tutorial_config)

            with open(output_path, 'w') as f:
                f.write(tutorial_content)

            logger.info(f"Generated tutorial docs at {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate tutorial docs: {e}")
            return False

    def _build_tutorial_content(self, tutorial_config: Dict[str, Any]) -> str:
        """Build tutorial content from configuration"""
        lines = []

        title = tutorial_config.get("title", "Active Inference Tutorial")
        description = tutorial_config.get("description", "")

        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"Generated on: {datetime.now().isoformat()}")
        lines.append("")

        if description:
            lines.append(description)
            lines.append("")

        # Add sections
        sections = tutorial_config.get("sections", [])

        for i, section in enumerate(sections, 1):
            lines.append(f"## {i}. {section.get('title', 'Section')}")
            lines.append("")

            if "content" in section:
                lines.append(section["content"])
                lines.append("")

            # Add code examples
            if "code_example" in section:
                lines.append("**Code Example:**")
                lines.append("```python")
                lines.append(section["code_example"])
                lines.append("```")
                lines.append("")

            # Add exercises
            if "exercise" in section:
                lines.append("**Exercise:**")
                lines.append(section["exercise"])
                lines.append("")

        return "\n".join(lines)


class DocumentationManager:
    """Main documentation management system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generator = DocumentationGenerator(config.get("generator", {}))
        self.knowledge_builder = KnowledgeDocBuilder(config.get("knowledge", {}))

        logger.info("DocumentationManager initialized")

    def generate_module_docs(self, module_name: str, output_path: Path) -> bool:
        """Generate documentation for a specific module"""
        return self.generator.generate_api_docs(module_name, output_path)

    def generate_knowledge_docs(self, knowledge_nodes: Dict[str, Any], output_dir: Path) -> int:
        """Generate documentation from knowledge repository"""
        return self.knowledge_builder.build_knowledge_docs(knowledge_nodes, output_dir)

    def generate_tutorial(self, tutorial_config: Dict[str, Any], output_path: Path) -> bool:
        """Generate tutorial documentation"""
        return self.knowledge_builder.generate_tutorial_docs(tutorial_config, output_path)

    def generate_comprehensive_docs(self, knowledge_nodes: Dict[str, Any],
                                   modules: List[str], output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive documentation"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "knowledge_docs": 0,
            "api_docs": 0,
            "tutorials": 0,
            "total_files": 0
        }

        # Generate knowledge documentation
        knowledge_dir = output_dir / "knowledge"
        results["knowledge_docs"] = self.generate_knowledge_docs(knowledge_nodes, knowledge_dir)
        results["total_files"] += results["knowledge_docs"]

        # Generate API documentation for each module
        api_dir = output_dir / "api"
        api_dir.mkdir(exist_ok=True)

        for module_name in modules:
            output_file = api_dir / f"{module_name.replace('.', '_')}.md"
            if self.generate_module_docs(module_name, output_file):
                results["api_docs"] += 1

        results["total_files"] += results["api_docs"]

        # Generate index
        index_content = self._generate_comprehensive_index(results)
        index_file = output_dir / "README.md"

        try:
            with open(index_file, 'w') as f:
                f.write(index_content)
            results["total_files"] += 1
        except Exception as e:
            logger.error(f"Failed to generate comprehensive index: {e}")

        logger.info(f"Comprehensive documentation generated: {results}")
        return results

    def _generate_comprehensive_index(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive documentation index"""
        lines = []

        lines.append("# Active Inference Knowledge Environment - Documentation")
        lines.append("")
        lines.append(f"Generated on: {datetime.now().isoformat()}")
        lines.append("")
        lines.append("## Documentation Overview")
        lines.append("")
        lines.append(f"- **Knowledge Documentation**: {results['knowledge_docs']} files")
        lines.append(f"- **API Documentation**: {results['api_docs']} modules")
        lines.append(f"- **Tutorials**: {results['tutorials']} tutorials")
        lines.append(f"- **Total Files**: {results['total_files']} files")
        lines.append("")

        lines.append("## Knowledge Documentation")
        lines.append("")
        lines.append("Educational content organized by topic:")
        lines.append("")
        lines.append("- [Foundations Documentation](knowledge/foundations_documentation.md)")
        lines.append("- [Mathematics Documentation](knowledge/mathematics_documentation.md)")
        lines.append("- [Implementation Documentation](knowledge/implementation_documentation.md)")
        lines.append("- [Application Documentation](knowledge/application_documentation.md)")
        lines.append("")

        lines.append("## API Documentation")
        lines.append("")
        lines.append("Technical API documentation for all modules:")
        lines.append("")

        # This would be dynamically generated based on actual modules
        api_modules = [
            "active_inference.knowledge",
            "active_inference.research",
            "active_inference.visualization",
            "active_inference.applications",
            "active_inference.platform",
            "active_inference.tools"
        ]

        for module in api_modules:
            lines.append(f"- [{module} API](api/{module.replace('.', '_')}.md)")

        lines.append("")

        lines.append("## Getting Started")
        lines.append("")
        lines.append("1. Start with the [foundations documentation](knowledge/foundations_documentation.md)")
        lines.append("2. Explore the [API documentation](api/) for implementation details")
        lines.append("3. Check out [example applications](knowledge/application_documentation.md)")
        lines.append("")

        return "\n".join(lines)

