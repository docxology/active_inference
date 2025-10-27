#!/usr/bin/env python3
"""
Knowledge Documentation Generator

Generates comprehensive documentation for the knowledge repository,
including learning paths, concept maps, and interactive content.
This script is called by the Makefile during documentation builds.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from active_inference.tools.documentation.generator import DocumentationGenerator
from active_inference.knowledge.repository import KnowledgeRepository, KnowledgeRepositoryConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_knowledge_docs(output_dir: Path, repo_path: Path) -> None:
    """
    Generate comprehensive knowledge documentation

    Args:
        output_dir: Directory to output documentation
        repo_path: Path to knowledge repository
    """
    logger.info(f"Generating knowledge documentation in {output_dir}")

    try:
        # Initialize knowledge repository
        config = KnowledgeRepositoryConfig(
            root_path=repo_path,
            auto_index=True
        )
        repo = KnowledgeRepository(config)

        # Generate documentation
        doc_generator = DocumentationGenerator(output_dir)

        # Generate learning path documentation
        doc_generator.generate_learning_paths(repo)

        # Generate concept maps and diagrams
        doc_generator.generate_concept_maps(repo)

        # Generate knowledge statistics
        doc_generator.generate_statistics(repo)

        # Generate search index
        doc_generator.generate_search_index(repo)

        logger.info("Knowledge documentation generated successfully")

    except Exception as e:
        logger.error(f"Error generating knowledge documentation: {e}")
        raise


def generate_api_docs(output_dir: Path, source_dir: Path) -> None:
    """
    Generate API documentation from source code

    Args:
        output_dir: Directory to output API docs
        source_dir: Source code directory
    """
    logger.info(f"Generating API documentation in {output_dir}")

    try:
        doc_generator = DocumentationGenerator(output_dir)
        doc_generator.generate_api_docs(source_dir)
        logger.info("API documentation generated successfully")

    except Exception as e:
        logger.error(f"Error generating API documentation: {e}")
        raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive documentation for Active Inference Knowledge Environment"
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('docs'),
        help='Output directory for documentation'
    )

    parser.add_argument(
        '--repo-path',
        type=Path,
        default=Path('knowledge'),
        help='Path to knowledge repository'
    )

    parser.add_argument(
        '--source-dir',
        type=Path,
        default=Path('src'),
        help='Path to source code directory'
    )

    parser.add_argument(
        '--generate-api',
        action='store_true',
        help='Generate API documentation'
    )

    parser.add_argument(
        '--generate-knowledge',
        action='store_true',
        help='Generate knowledge repository documentation'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all documentation types'
    )

    args = parser.parse_args()

    try:
        # Ensure output directories exist
        args.output_dir.mkdir(parents=True, exist_ok=True)

        if args.all or args.generate_api:
            generate_api_docs(args.output_dir, args.source_dir)

        if args.all or args.generate_knowledge:
            generate_knowledge_docs(args.output_dir / 'knowledge', args.repo_path)

        logger.info("Documentation generation completed successfully")

    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
