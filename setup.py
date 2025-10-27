"""
Setup script for Active Inference Knowledge Environment

This script provides an interactive setup process for the Active Inference
Knowledge Environment, ensuring all dependencies are properly installed
and configured for optimal development and learning experience.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Optional


class SetupManager:
    """Interactive setup manager for the Active Inference Knowledge Environment"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.python_executable = sys.executable

    def run_command(self, command: List[str], description: str = "") -> bool:
        """Run a shell command and return success status"""
        try:
            print(f"🔧 {description}")
            print(f"   Command: {' '.join(command)}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"   ✅ Success")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed: {e}")
            print(f"   Error output: {e.stderr}")
            return False

    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            print(f"❌ Python {version.major}.{version.minor} detected")
            print("   Active Inference Knowledge Environment requires Python 3.9+")
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True

    def create_virtual_environment(self) -> bool:
        """Create and activate virtual environment"""
        venv_path = self.project_root / "venv"

        if venv_path.exists():
            print("✅ Virtual environment already exists")
            return True

        return self.run_command(
            [self.python_executable, "-m", "venv", "venv"],
            "Creating virtual environment"
        )

    def activate_and_install_requirements(self) -> bool:
        """Install project requirements"""
        if not self.requirements_file.exists():
            print("❌ requirements.txt not found")
            return False

        # Determine activation script based on platform
        if platform.system() == "Windows":
            activate_script = "venv\\Scripts\\activate"
            pip_command = "python -m pip"
        else:
            activate_script = "venv/bin/activate"
            pip_command = "pip"

        # Install requirements
        commands = [
            ["source", activate_script, "&&", pip_command, "install", "--upgrade", "pip"],
            ["source", activate_script, "&&", pip_command, "install", "-r", "requirements.txt"]
        ]

        for i, command in enumerate(commands):
            success = self.run_command(
                command,
                f"Installing requirements (step {i+1}/2)"
            )
            if not success:
                return False

        return True

    def install_pre_commit_hooks(self) -> bool:
        """Install pre-commit hooks"""
        return self.run_command(
            ["source", "venv/bin/activate", "&&", "pre-commit", "install"],
            "Installing pre-commit hooks"
        )

    def initialize_knowledge_repository(self) -> bool:
        """Initialize the knowledge repository"""
        try:
            from src.active_inference.knowledge import KnowledgeRepository, KnowledgeRepositoryConfig

            config = KnowledgeRepositoryConfig(
                root_path=self.project_root / "knowledge",
                auto_index=True
            )

            repo = KnowledgeRepository(config)
            stats = repo.get_statistics()

            print("✅ Knowledge repository initialized"            print(f"   📚 Nodes: {stats['total_nodes']}")
            print(f"   📖 Paths: {stats['total_paths']}")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize knowledge repository: {e}")
            return False

    def run_tests(self) -> bool:
        """Run the test suite"""
        return self.run_command(
            ["source", "venv/bin/activate", "&&", "python", "-m", "pytest", "tests/", "-v"],
            "Running test suite"
        )

    def generate_documentation(self) -> bool:
        """Generate project documentation"""
        return self.run_command(
            ["source", "venv/bin/activate", "&&", "make", "docs"],
            "Generating documentation"
        )

    def create_sample_content(self) -> bool:
        """Create sample knowledge content for testing"""
        sample_dir = self.project_root / "knowledge" / "samples"
        sample_dir.mkdir(exist_ok=True)

        sample_content = {
            "id": "getting_started",
            "title": "Getting Started with Active Inference",
            "content_type": "tutorial",
            "difficulty": "beginner",
            "description": "A gentle introduction to Active Inference concepts",
            "prerequisites": [],
            "tags": ["tutorial", "beginner", "introduction"],
            "learning_objectives": [
                "Understand basic Active Inference concepts",
                "Run your first Active Inference simulation",
                "Navigate the knowledge environment"
            ],
            "content": {
                "overview": "Welcome to the Active Inference Knowledge Environment! This tutorial will guide you through your first steps.",
                "what_is_active_inference": "Active Inference is a theoretical framework that explains how biological and artificial systems can act to minimize surprise and maintain their integrity.",
                "key_concepts": [
                    "Free Energy Principle",
                    "Generative Models",
                    "Variational Inference",
                    "Policy Selection"
                ]
            }
        }

        sample_file = sample_dir / "getting_started.json"
        import json
        with open(sample_file, 'w') as f:
            json.dump(sample_content, f, indent=2)

        print(f"✅ Created sample content: {sample_file}")
        return True

    def interactive_setup(self) -> bool:
        """Run interactive setup process"""
        print("🚀 Active Inference Knowledge Environment Setup")
        print("=" * 50)
        print()

        # Check Python version
        if not self.check_python_version():
            return False

        print()

        # Create virtual environment
        if not self.create_virtual_environment():
            return False

        print()

        # Install requirements
        if not self.activate_and_install_requirements():
            return False

        print()

        # Initialize knowledge repository
        if not self.initialize_knowledge_repository():
            print("⚠️  Knowledge repository initialization failed, but continuing...")

        print()

        # Create sample content
        self.create_sample_content()

        print()

        # Install pre-commit hooks
        if not self.install_pre_commit_hooks():
            print("⚠️  Pre-commit hooks installation failed, but continuing...")

        print()

        # Run basic tests
        if not self.run_tests():
            print("⚠️  Some tests failed, but setup is complete")

        print()

        # Generate documentation
        if not self.generate_documentation():
            print("⚠️  Documentation generation failed, but continuing...")

        print()
        print("🎉 Setup Complete!")
        print()
        print("Next steps:")
        print("  1. Start learning: ai-knowledge learn foundations")
        print("  2. Explore content: ai-knowledge search <topic>")
        print("  3. Run platform: make serve")
        print("  4. View docs: open docs/_build/index.html")
        print()
        print("For help: make help")

        return True

    def quick_setup(self) -> bool:
        """Run quick setup without interactive prompts"""
        print("⚡ Quick Setup Mode")
        print()

        steps = [
            ("Python version check", self.check_python_version),
            ("Virtual environment", self.create_virtual_environment),
            ("Install requirements", self.activate_and_install_requirements),
            ("Sample content", self.create_sample_content),
        ]

        for description, step_func in steps:
            if not step_func():
                return False
            print()

        print("✅ Quick setup complete!")
        return True


def main():
    """Main setup entry point"""
    setup = SetupManager()

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = setup.quick_setup()
    else:
        success = setup.interactive_setup()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
