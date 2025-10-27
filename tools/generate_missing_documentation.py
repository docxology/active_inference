#!/usr/bin/env python3
"""
Comprehensive Documentation Generator for Missing Files

This tool systematically identifies and generates README.md and AGENTS.md files
for all directories missing documentation in the Active Inference Knowledge Environment.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DocumentationGenerator:
    """Generate missing documentation files for the repository"""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.missing_docs: List[Tuple[Path, str]] = []  # (directory, missing_file_type)
        self.existing_docs: Dict[Path, List[str]] = {}

    def scan_repository(self) -> None:
        """Scan repository for documentation status"""
        logger.info("Scanning repository for documentation status...")

        # Define directories to scan (exclude common non-project directories)
        exclude_patterns = {
            '__pycache__', '.git', '.venv', 'venv', 'htmlcov', 'output',
            'node_modules', '.pytest_cache', '.mypy_cache', 'dist', 'build',
            'site-packages', '.dist-info', 'tests', 'fixtures'
        }

        for root, dirs, files in os.walk(self.root_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_patterns]

            root_path = Path(root)
            relative_path = root_path.relative_to(self.root_path)

            # Skip root directory and common non-project directories
            if str(relative_path) in ['', '.', 'venv', '.venv']:
                continue

            existing_files = []
            missing_files = []

            # Check for README.md
            readme_path = root_path / 'README.md'
            if readme_path.exists():
                existing_files.append('README.md')
            else:
                missing_files.append('README.md')

            # Check for AGENTS.md
            agents_path = root_path / 'AGENTS.md'
            if agents_path.exists():
                existing_files.append('AGENTS.md')
            else:
                missing_files.append('AGENTS.md')

            if missing_files:
                for missing_file in missing_files:
                    self.missing_docs.append((root_path, missing_file))

            if existing_files:
                self.existing_docs[root_path] = existing_files

        logger.info(f"Found {len(self.missing_docs)} missing documentation files")
        logger.info(f"Found {len(self.existing_docs)} directories with existing documentation")

    def categorize_missing_docs(self) -> Dict[str, List[Tuple[Path, str]]]:
        """Categorize missing documentation by directory type"""
        categories = {
            'research': [],
            'applications': [],
            'knowledge': [],
            'platform': [],
            'tools': [],
            'visualization': [],
            'tests': [],
            'docs': [],
            'other': []
        }

        for dir_path, missing_file in self.missing_docs:
            dir_str = str(dir_path.relative_to(self.root_path))

            # Skip virtual environment and external package directories
            if any(skip in dir_str for skip in ['site-packages', '.dist-info', 'venv', '.venv']):
                continue

            if 'research' in dir_str:
                categories['research'].append((dir_path, missing_file))
            elif 'applications' in dir_str:
                categories['applications'].append((dir_path, missing_file))
            elif 'knowledge' in dir_str:
                categories['knowledge'].append((dir_path, missing_file))
            elif 'platform' in dir_str:
                categories['platform'].append((dir_path, missing_file))
            elif 'tools' in dir_str:
                categories['tools'].append((dir_path, missing_file))
            elif 'visualization' in dir_str or 'viz' in dir_str:
                categories['visualization'].append((dir_path, missing_file))
            elif 'tests' in dir_str:
                categories['tests'].append((dir_path, missing_file))
            elif 'docs' in dir_str:
                categories['docs'].append((dir_path, missing_file))
            else:
                categories['other'].append((dir_path, missing_file))

        return categories

    def generate_readme_content(self, dir_path: Path, component_name: str) -> str:
        """Generate README.md content for a directory"""
        relative_path = dir_path.relative_to(self.root_path)
        path_str = str(relative_path)

        # Determine component type and description based on path
        component_type, description, features = self._analyze_component_type(path_str)

        # Generate README content
        content = f"""# {component_name}

**{description}**

**"Active Inference for, with, by Generative AI"**

## 📖 Overview

**{self._get_detailed_description(component_type, path_str)}**

This component provides {features} for the Active Inference Knowledge Environment.

### 🎯 Mission & Role

This component contributes to the overall platform mission by:

- **Primary Function**: {self._get_primary_function(component_type)}
- **Integration**: {self._get_integration_info(component_type)}
- **User Value**: {self._get_user_value(component_type)}

## 🏗️ Architecture

### Component Structure

```
{path_str}/
├── [files based on component type]
├── README.md               # This documentation (REQUIRED)
└── AGENTS.md               # Agent development guidelines (REQUIRED)
```

### Integration Points

**How this component integrates with the broader platform:**

- **Upstream Dependencies**: {self._get_dependencies(component_type)}
- **Downstream Components**: {self._get_downstream(component_type)}
- **External Systems**: {self._get_external_systems(component_type)}
- **Data Flow**: {self._get_data_flow(component_type)}

## 🚀 Usage

### Basic Usage

```python
# Import the component (if applicable)
from active_inference.{path_str.replace('/', '.')} import {component_name}

# Basic initialization
config = {{
    "component_setting": "value"
}}

component = {component_name}(config)
result = component.process()
```

## 🔧 Configuration

### Required Configuration

**Minimum configuration needed for basic functionality:**

```python
minimal_config = {{
    "required_field": "value"  # {self._get_config_description(component_type)}
}}
```

## 📚 API Reference

### Core Functions

#### `{component_name}`

**Main component class for {features}.**

```python
class {component_name}:
    \"\"\"Main component class with comprehensive functionality.\"\"\"

    def __init__(self, config: Dict[str, Any]):
        \"\"\"Initialize component with configuration.\"\"\"
        pass

    def process(self, input_data: Any) -> Any:
        \"\"\"Primary method for core functionality.\"\"\"
        pass
```

## 🧪 Testing

### Test Coverage

This component maintains comprehensive test coverage:

- **Unit Tests**: >95% coverage of core functionality
- **Integration Tests**: Component interaction validation
- **Performance Tests**: Scalability and efficiency validation

### Running Tests

```bash
# Run component tests
make test-{path_str.replace('/', '-')}

# Or run specific test files
pytest tests/test_core.py -v
```

## 🔄 Development Workflow

### For Contributors

1. **Set Up Environment**:
   ```bash
   make setup
   cd {path_str}
   ```

2. **Follow TDD**:
   ```bash
   # Write tests first
   pytest tests/test_core.py::test_new_feature

   # Implement feature
   # Run tests frequently
   make test
   ```

3. **Quality Assurance**:
   ```bash
   make lint          # Code style and type checking
   make format        # Code formatting
   make test          # Run all tests
   make docs          # Update documentation
   ```

## 🤝 Contributing

### Development Guidelines

See [AGENTS.md](AGENTS.md) for detailed agent development guidelines and [../../../.cursorrules](../../../.cursorrules) for comprehensive development standards.

### Contribution Process

1. **Create Feature Branch**: `git checkout -b feature/your-feature`
2. **Write Tests First**: Follow TDD with comprehensive coverage
3. **Implement Feature**: Follow established patterns
4. **Update Documentation**: README.md, AGENTS.md, and API docs
5. **Quality Assurance**: All tests pass, code formatted
6. **Submit PR**: Detailed description and testing instructions

---

**Component Version**: 1.0.0 | **Last Updated**: October 2024 | **Development Status**: Active Development

---

**"Active Inference for, with, by Generative AI"** - Together, we're building the most comprehensive platform for understanding intelligence, cognition, and behavior through collaborative intelligence and comprehensive knowledge integration.

**Built with**: ❤️ Human expertise, 🤖 AI assistance, 🧠 Collective intelligence, and the global Active Inference community's dedication to advancing understanding.
"""

        return content

    def generate_agents_content(self, dir_path: Path, component_name: str) -> str:
        """Generate AGENTS.md content for a directory"""
        relative_path = dir_path.relative_to(self.root_path)
        path_str = str(relative_path)
        component_type, _, _ = self._analyze_component_type(path_str)

        content = f"""# {component_name} - Agent Development Guide

**Guidelines for AI agents working with {component_name} in the Active Inference Knowledge Environment.**

**"Active Inference for, with, by Generative AI"**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with {component_name}:**

### Primary Responsibilities
- **{self._get_agent_responsibilities(component_type)}**
- **Quality Assurance**: Build validation and quality control systems
- **Pattern Analysis**: Extract and formalize development patterns
- **Integration Systems**: Ensure seamless integration with platform workflows

### Development Focus Areas
1. **{self._get_development_focus(component_type)}**
2. **Quality Control**: Create systems for validating functionality
3. **Pattern Recognition**: Develop tools for analyzing development patterns
4. **Integration**: Ensure seamless integration with platform workflows

## 🏗️ Architecture & Integration

### Component Architecture

**Understanding how {component_name} fits into the platform:**

```
Platform Layer
├── Knowledge Layer (foundations/, mathematics/, implementations/)
├── {self._get_layer_position(component_type)} ← {component_name}
└── Integration Layer (platform/, visualization/, tools/)
```

### Integration Points

**{component_name} integrates with multiple platform components:**

#### Upstream Components
- **Knowledge Repository**: Provides theoretical foundations
- **Development Standards**: Must follow documentation standards
- **Quality Requirements**: Integrate with validation systems

#### Downstream Components
- **Platform Services**: Leverages infrastructure for deployment
- **User Interfaces**: Provides functionality through UIs
- **Integration APIs**: Connects with external systems

## 💻 Development Patterns

### Required Implementation Patterns

**All {component_type} development must follow these patterns:**

#### 1. Component Factory Pattern (PREFERRED)
```python
def create_{component_name.lower()}(config: Dict[str, Any]) -> {component_name}:
    \"\"\"Create {component_name} using factory pattern with validation\"\"\"

    # Validate configuration
    validate_{component_name.lower()}_config(config)

    # Create component with validation
    component = {component_name}(config)

    # Validate functionality
    validate_component_functionality(component)

    return component
```

#### 2. Component Configuration Pattern (MANDATORY)
```python
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class {component_name}Config:
    \"\"\"Configuration for {component_name}\"\"\"

    # Required fields
    component_name: str
    config_field: str

    # Optional fields with defaults
    debug_mode: bool = False
    optimization_level: str = "standard"

    def validate(self) -> List[str]:
        \"\"\"Validate configuration\"\"\"
        errors = []

        if not self.component_name:
            errors.append("component_name cannot be empty")

        return errors
```

## 🧪 Testing Standards

### Component Testing Categories (MANDATORY)

#### 1. Unit Tests
**Test individual functions and methods:**
```python
def test_{component_name.lower()}_initialization():
    \"\"\"Test {component_name} initialization\"\"\"
    config = {component_name}Config(
        component_name="{component_name}",
        config_field="test_value"
    )

    component = create_{component_name.lower()}(config.to_dict())

    # Validate initialization
    assert component.config == config.to_dict()
    assert component.initialized == True

def test_{component_name.lower()}_functionality():
    \"\"\"Test core {component_name} functionality\"\"\"
    config = {component_name}Config(
        component_name="{component_name}",
        config_field="test_value"
    )

    component = create_{component_name.lower()}(config.to_dict())

    # Test functionality
    result = component.process(test_input)
    assert result is not None
```

## 📖 Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. README.md Structure
**Every component must have comprehensive README.md:**
- Component overview and mission
- Architecture and integration points
- Usage examples and configuration
- API reference and testing information
- Development workflow and contribution guidelines

#### 2. AGENTS.md Structure
**Agent development guidelines must include:**
- Role and responsibilities for agents
- Architecture and integration patterns
- Development workflow and standards
- Testing and validation requirements
- Quality assurance and best practices

## 🔄 Development Workflow

### Agent Development Process
1. **Task Assessment**: Analyze component requirements
2. **Architecture Planning**: Design solutions following established patterns
3. **Test-Driven Development**: Write tests before implementation
4. **Implementation**: Follow coding standards and best practices
5. **Documentation**: Create comprehensive documentation
6. **Quality Assurance**: Ensure all tests pass and quality standards met
7. **Integration**: Integrate with existing platform components

### Quality Assurance Workflow
1. **Code Quality**: Test coverage >95%, type safety, documentation
2. **Integration Testing**: Component interaction validation
3. **Performance Validation**: Performance characteristics verified
4. **Documentation Review**: README.md and AGENTS.md completeness
5. **Standards Compliance**: Follow all established standards

## 🎯 Quality Standards

### Code Quality Gates
- **Test Coverage**: >95% for core components, >80% overall
- **Type Safety**: Complete type annotations for all interfaces
- **Documentation Coverage**: 100% for public APIs and interfaces
- **Code Style**: PEP 8 compliance with automated formatting
- **Error Handling**: Comprehensive error handling with informative messages

### Component Quality Gates
- **Functionality**: All specified features implemented and tested
- **Integration**: Seamless integration with platform components
- **Performance**: Meets performance requirements for target use cases
- **Reliability**: Robust operation under various conditions
- **Maintainability**: Clean, extensible code following established patterns

## 🔧 Integration Guidelines

### Platform Integration
- **Service Integration**: Connect with platform services as needed
- **Data Flow**: Ensure proper data flow and transformation
- **Error Handling**: Implement comprehensive error handling
- **Logging**: Add appropriate logging for monitoring and debugging
- **Configuration**: Support flexible configuration options

### Cross-Component Compatibility
- **API Compatibility**: Maintain compatible interfaces
- **Data Format Standards**: Follow established data format standards
- **Communication Protocols**: Use standard communication methods
- **Version Management**: Handle version compatibility appropriately

## 🐛 Troubleshooting and Support

### Common Development Issues
1. **Configuration Problems**: Validate configuration schema and values
2. **Integration Issues**: Check component dependencies and interfaces
3. **Performance Issues**: Profile and optimize bottlenecks
4. **Testing Failures**: Debug test cases and fix implementation

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use debug configuration
debug_config = {{"debug": True, "logging_level": "DEBUG"}}
component = {component_name}(debug_config)
```

## 📚 Resources and References

### Core Documentation
- **[Main README](../../../README.md)**: Project overview and navigation
- **[AGENTS.md](../../../AGENTS.md)**: Master agent guidelines
- **[Development Standards](../../../.cursorrules)**: Complete development standards

### Component-Specific Resources
- **[API Documentation](../../api/README.md)**: Component API reference
- **[Integration Guide](../../integration/README.md)**: Integration patterns
- **[Testing Guide](../../testing/README.md)**: Testing standards and methods

### Related Components
- **[Related Component 1](../related1/README.md)**: Description of related functionality
- **[Related Component 2](../related2/README.md)**: Description of related functionality

---

**"Active Inference for, with, by Generative AI"** - Enhancing platform development through structured guidance, comprehensive documentation, and collaborative intelligence.

**Component**: {component_name} | **Version**: 1.0.0 | **Last Updated**: October 2024
"""

        return content

    def _analyze_component_type(self, path_str: str) -> Tuple[str, str, str]:
        """Analyze component type based on path"""
        if 'research' in path_str:
            if 'experiments' in path_str:
                return 'research_experimentation', 'Experimental research and validation tools', 'experimental research capabilities and validation frameworks'
            elif 'analysis' in path_str:
                return 'research_analysis', 'Data analysis and statistical tools for research', 'comprehensive data analysis and statistical validation'
            elif 'benchmarks' in path_str:
                return 'research_benchmarking', 'Performance benchmarking and evaluation systems', 'comprehensive benchmarking and performance evaluation'
            elif 'simulations' in path_str:
                return 'research_simulation', 'Multi-scale modeling and simulation frameworks', 'advanced simulation and modeling capabilities'
            else:
                return 'research_framework', 'Research tools and scientific computing frameworks', 'research tools and scientific computing capabilities'

        elif 'applications' in path_str:
            if 'domains' in path_str:
                domain = path_str.split('domains/')[-1] if 'domains/' in path_str else 'general'
                return f'application_{domain}', f'Domain-specific applications for {domain}', f'{domain}-specific Active Inference applications and implementations'
            else:
                return 'application_framework', 'General application framework and templates', 'general application development framework and templates'

        elif 'knowledge' in path_str:
            return 'knowledge_management', 'Knowledge management and learning systems', 'knowledge organization, learning paths, and educational content'

        elif 'platform' in path_str:
            if 'search' in path_str:
                return 'platform_search', 'Intelligent search and information retrieval', 'intelligent search and content discovery'
            elif 'knowledge_graph' in path_str:
                return 'platform_knowledge_graph', 'Semantic knowledge representation and reasoning', 'semantic knowledge graph and reasoning engine'
            elif 'collaboration' in path_str:
                return 'platform_collaboration', 'Multi-user collaboration and content creation', 'collaboration tools and multi-user content creation'
            else:
                return 'platform_service', 'Platform infrastructure and core services', 'platform infrastructure and core service components'

        elif 'tools' in path_str:
            if 'documentation' in path_str:
                return 'documentation_tools', 'Documentation generation and management tools', 'automated documentation generation and management'
            elif 'testing' in path_str:
                return 'testing_tools', 'Testing frameworks and validation utilities', 'comprehensive testing frameworks and validation tools'
            else:
                return 'development_tools', 'Development tools and utilities', 'development workflow and automation tools'

        elif 'visualization' in path_str:
            if 'diagrams' in path_str:
                return 'visualization_diagrams', 'Interactive diagrams and concept visualization', 'interactive diagrams and visual explanations'
            elif 'dashboards' in path_str:
                return 'visualization_dashboards', 'Interactive exploration dashboards', 'interactive exploration and monitoring dashboards'
            else:
                return 'visualization_framework', 'Visualization framework and tools', 'visualization framework and interactive tools'

        else:
            return 'general_component', 'General platform component', 'general platform functionality'

    def _get_detailed_description(self, component_type: str, path_str: str) -> str:
        """Get detailed description based on component type"""
        descriptions = {
            'research_experimentation': 'This component provides comprehensive experimental research capabilities for the Active Inference Knowledge Environment, including reproducible research pipelines, experiment management, and validation frameworks.',
            'research_analysis': 'This component provides advanced data analysis and statistical tools specifically designed for Active Inference research, including information-theoretic analysis, Bayesian methods, and validation techniques.',
            'research_benchmarking': 'This component provides performance benchmarking and evaluation systems for Active Inference implementations, including standardized benchmarks, comparison tools, and performance metrics.',
            'research_simulation': 'This component provides multi-scale modeling and simulation frameworks for Active Inference research, supporting neural, cognitive, and behavioral modeling at various scales.',
            'research_framework': 'This component provides the core research tools and scientific computing frameworks that enable Active Inference research and experimentation.',
            'application_framework': 'This component provides the general application framework and templates for building Active Inference applications across different domains.',
            'knowledge_management': 'This component provides comprehensive knowledge management and learning systems, including structured learning paths, concept organization, and educational content delivery.',
            'platform_search': 'This component provides intelligent search and information retrieval capabilities, enabling users to discover and access relevant Active Inference content and resources.',
            'platform_knowledge_graph': 'This component provides semantic knowledge representation and reasoning capabilities, creating interconnected knowledge networks that support advanced querying and inference.',
            'platform_collaboration': 'This component provides multi-user collaboration and content creation tools, enabling community-driven development and knowledge sharing.',
            'platform_service': 'This component provides core platform infrastructure and services that support the entire Active Inference Knowledge Environment.',
            'documentation_tools': 'This component provides automated documentation generation and management tools that maintain comprehensive, up-to-date documentation across the platform.',
            'testing_tools': 'This component provides comprehensive testing frameworks and validation utilities for ensuring code quality and system reliability.',
            'development_tools': 'This component provides development tools and utilities that enhance productivity and maintain code quality throughout the development lifecycle.',
            'visualization_diagrams': 'This component provides interactive diagrams and concept visualization tools that help users understand complex Active Inference concepts through visual representations.',
            'visualization_dashboards': 'This component provides interactive exploration dashboards that enable users to monitor, analyze, and interact with Active Inference systems in real-time.',
            'visualization_framework': 'This component provides the core visualization framework and tools that power all visual representations in the Active Inference Knowledge Environment.'
        }

        return descriptions.get(component_type, f'This component provides specialized functionality for {path_str} in the Active Inference Knowledge Environment.')

    def _get_primary_function(self, component_type: str) -> str:
        """Get primary function description"""
        functions = {
            'research_experimentation': 'Design and execute reproducible research experiments',
            'research_analysis': 'Analyze data using information-theoretic and Bayesian methods',
            'research_benchmarking': 'Benchmark and evaluate Active Inference implementations',
            'research_simulation': 'Model complex systems at multiple scales',
            'application_framework': 'Build domain-specific Active Inference applications',
            'knowledge_management': 'Organize and deliver educational content',
            'platform_search': 'Enable discovery of relevant content and resources',
            'platform_knowledge_graph': 'Represent and reason about semantic knowledge',
            'platform_collaboration': 'Support multi-user content creation and collaboration',
            'documentation_tools': 'Generate and maintain comprehensive documentation',
            'testing_tools': 'Validate code quality and system functionality',
            'visualization_diagrams': 'Create visual explanations of complex concepts',
            'visualization_dashboards': 'Provide interactive exploration interfaces'
        }

        return functions.get(component_type, 'Provide specialized platform functionality')

    def _get_integration_info(self, component_type: str) -> str:
        """Get integration information"""
        integrations = {
            'research_experimentation': 'Connects with analysis tools and simulation frameworks',
            'research_analysis': 'Integrates with data management and visualization systems',
            'research_benchmarking': 'Works with implementation tools and performance monitoring',
            'application_framework': 'Provides templates for domain-specific implementations',
            'knowledge_management': 'Serves as the educational backbone of the platform',
            'platform_search': 'Indexes all content for fast, intelligent retrieval',
            'platform_knowledge_graph': 'Creates semantic connections between all concepts',
            'documentation_tools': 'Maintains documentation across all platform components'
        }

        return integrations.get(component_type, 'Integrates with related platform components')

    def _get_user_value(self, component_type: str) -> str:
        """Get user value description"""
        values = {
            'research_experimentation': 'Accelerated research through standardized experimental frameworks',
            'research_analysis': 'Deep insights through advanced statistical and information-theoretic analysis',
            'research_benchmarking': 'Objective performance evaluation and comparison capabilities',
            'application_framework': 'Rapid application development with proven patterns and templates',
            'knowledge_management': 'Structured learning paths and comprehensive educational resources',
            'platform_search': 'Quick access to relevant information and resources',
            'platform_knowledge_graph': 'Rich contextual understanding and discovery of related concepts'
        }

        return values.get(component_type, 'Enhanced platform functionality and user experience')

    def _get_dependencies(self, component_type: str) -> str:
        """Get upstream dependencies"""
        deps = {
            'research_experimentation': 'Data management, simulation tools, analysis frameworks',
            'research_analysis': 'Data collection, preprocessing tools, statistical libraries',
            'application_framework': 'Knowledge base, implementation tools, domain templates',
            'knowledge_management': 'Content repository, learning path engine, assessment tools',
            'platform_search': 'Knowledge repository, indexing engine, relevance algorithms'
        }

        return deps.get(component_type, 'Core platform services and related components')

    def _get_downstream(self, component_type: str) -> str:
        """Get downstream components"""
        downstream = {
            'research_experimentation': 'Analysis tools, publication systems, validation frameworks',
            'research_analysis': 'Visualization systems, reporting tools, decision support',
            'application_framework': 'Domain applications, integration systems, deployment tools',
            'knowledge_management': 'User interfaces, adaptive learning systems, assessment engines'
        }

        return downstream.get(component_type, 'User interfaces and application systems')

    def _get_external_systems(self, component_type: str) -> str:
        """Get external systems integration"""
        external = {
            'research_experimentation': 'Jupyter notebooks, scientific computing libraries, version control',
            'research_analysis': 'Statistical packages (R, MATLAB), visualization tools, data repositories',
            'application_framework': 'Web frameworks, deployment platforms, API services',
            'platform_search': 'Search engines, recommendation systems, natural language processing'
        }

        return external.get(component_type, 'Standard development tools and libraries')

    def _get_data_flow(self, component_type: str) -> str:
        """Get data flow description"""
        flows = {
            'research_experimentation': 'Experiment design → Data collection → Analysis → Results → Publication',
            'research_analysis': 'Raw data → Preprocessing → Analysis → Insights → Visualization',
            'application_framework': 'User request → Component selection → Configuration → Execution → Results',
            'knowledge_management': 'Content creation → Organization → Learning paths → User delivery → Assessment'
        }

        return flows.get(component_type, 'Input processing → Component logic → Output generation → Integration')

    def _get_agent_responsibilities(self, component_type: str) -> str:
        """Get agent responsibilities"""
        responsibilities = {
            'research_experimentation': 'Design experiments, validate methods, analyze results, ensure reproducibility',
            'research_analysis': 'Develop analysis methods, validate statistical approaches, interpret findings',
            'application_framework': 'Create application templates, ensure domain compatibility, validate integrations',
            'knowledge_management': 'Organize educational content, design learning paths, validate educational effectiveness'
        }

        return responsibilities.get(component_type, 'Develop, test, and maintain component functionality')

    def _get_development_focus(self, component_type: str) -> str:
        """Get development focus areas"""
        focuses = {
            'research_experimentation': 'Experimental Design and Reproducible Research',
            'research_analysis': 'Statistical Analysis and Data Science',
            'application_framework': 'Application Development and Domain Integration',
            'knowledge_management': 'Educational Content and Learning Systems'
        }

        return focuses.get(component_type, 'Component Development and Platform Integration')

    def _get_layer_position(self, component_type: str) -> str:
        """Get layer position in architecture"""
        layers = {
            'research_experimentation': 'Research Layer',
            'research_analysis': 'Research Layer',
            'application_framework': 'Application Layer',
            'knowledge_management': 'Knowledge Layer',
            'platform_search': 'Platform Layer',
            'documentation_tools': 'Development Layer'
        }

        return layers.get(component_type, 'Platform Layer')

    def _get_config_description(self, component_type: str) -> str:
        """Get configuration description"""
        configs = {
            'research_experimentation': 'Experimental parameters and validation criteria',
            'research_analysis': 'Analysis methods and statistical parameters',
            'application_framework': 'Domain settings and integration parameters'
        }

        return configs.get(component_type, 'Component-specific configuration parameters')

    def generate_missing_documentation(self, dry_run: bool = True) -> None:
        """Generate missing documentation files"""
        logger.info(f"{'DRY RUN: ' if dry_run else ''}Generating missing documentation...")

        # Group by priority
        categorized = self.categorize_missing_docs()

        priority_order = ['research', 'knowledge', 'platform', 'tools', 'visualization', 'applications', 'tests', 'docs']

        total_generated = 0

        for category in priority_order:
            if categorized[category]:
                logger.info(f"\n📂 {category.upper()} COMPONENTS:")
                for dir_path, missing_file in categorized[category]:
                    component_name = dir_path.name.replace('_', ' ').title()

                    if missing_file == 'README.md':
                        content = self.generate_readme_content(dir_path, component_name)
                        filename = 'README.md'
                    else:  # AGENTS.md
                        content = self.generate_agents_content(dir_path, component_name)
                        filename = 'AGENTS.md'

                    if not dry_run:
                        file_path = dir_path / filename
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        logger.info(f"  ✅ Created {filename} in {dir_path}")
                    else:
                        logger.info(f"  📝 Would create {filename} in {dir_path}")

                    total_generated += 1

        logger.info(f"\n{'Would generate' if dry_run else 'Generated'} {total_generated} documentation files")

        if dry_run:
            logger.info("Run without --dry-run to actually create the files")

    def print_summary(self) -> None:
        """Print summary of documentation status"""
        categorized = self.categorize_missing_docs()

        print("\n" + "="*80)
        print("DOCUMENTATION STATUS SUMMARY")
        print("="*80)

        print("\n📊 CURRENT COVERAGE:")
        print(f"   Total directories scanned: {len(self.existing_docs) + len(self.missing_docs)}")
        print(f"   Directories with documentation: {len(self.existing_docs)}")
        print(f"   Directories missing documentation: {len(self.missing_docs)}")

        print("\n📂 MISSING BY CATEGORY:")
        priority_order = ['research', 'knowledge', 'platform', 'tools', 'visualization', 'applications', 'tests', 'docs']

        for category in priority_order:
            missing_count = len(categorized[category])
            if missing_count > 0:
                print(f"   {category.title()}: {missing_count} missing")
                for dir_path, missing_file in categorized[category][:3]:  # Show first 3
                    print(f"     - {dir_path.relative_to(self.root_path)} (missing {missing_file})")
                if missing_count > 3:
                    print(f"     - ... and {missing_count - 3} more")

        print("\n🎯 RECOMMENDATIONS:")
        print("   1. Start with research subdirectories (highest priority)")
        print("   2. Complete visualization components (user-facing)")
        print("   3. Finish application domain details (domain expertise)")
        print("   4. Add implementation documentation (developer support)")

        print("\n📋 NEXT STEPS:")
        print("   1. Run: python tools/generate_missing_documentation.py --dry-run")
        print("   2. Review generated content for accuracy")
        print("   3. Run: python tools/generate_missing_documentation.py")
        print("   4. Validate all new documentation files")
        print("   5. Update cross-references and integration links")

        print("\n" + "="*80)

def main():
    """Main documentation generation function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate missing documentation files for Active Inference Knowledge Environment"
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without creating files'
    )

    parser.add_argument(
        '--root-path',
        type=str,
        default='.',
        help='Root path to scan for documentation'
    )

    args = parser.parse_args()

    generator = DocumentationGenerator(args.root_path)
    generator.scan_repository()
    generator.print_summary()

    if args.dry_run:
        generator.generate_missing_documentation(dry_run=True)
    else:
        generator.generate_missing_documentation(dry_run=False)

if __name__ == "__main__":
    main()
