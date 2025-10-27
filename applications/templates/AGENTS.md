# Templates - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Templates module of the Active Inference Knowledge Environment. It outlines template development workflows, quality standards, and best practices for creating reusable Active Inference implementation templates.

## Templates Module Overview

The Templates module offers a comprehensive collection of implementation templates covering various Active Inference applications, system architectures, and integration patterns. This module accelerates development by providing proven, well-documented starting points that developers can customize and extend for their specific needs.

## Core Responsibilities

### Template Development
- **Create Templates**: Develop reusable implementation templates
- **Pattern Documentation**: Document established Active Inference patterns
- **Example Implementation**: Provide working code examples
- **Customization Support**: Enable easy template customization
- **Quality Assurance**: Ensure template quality and correctness

### Template Maintenance
- **Update Templates**: Keep templates current with latest developments
- **Bug Fixes**: Address issues and improve template reliability
- **Performance Optimization**: Optimize template performance
- **Documentation Updates**: Maintain current documentation
- **Community Support**: Support template users and contributors

### Quality Control
- **Code Review**: Ensure code quality and correctness
- **Testing Validation**: Validate template functionality
- **Performance Testing**: Ensure acceptable performance
- **Documentation Review**: Maintain high-quality documentation
- **User Feedback**: Incorporate user feedback and improvements

## Development Workflows

### Template Creation Process
1. **Identify Need**: Recognize common implementation patterns or use cases
2. **Research Requirements**: Understand requirements and constraints
3. **Design Architecture**: Design template architecture and structure
4. **Implementation**: Implement template following best practices
5. **Testing**: Develop comprehensive test suite
6. **Documentation**: Create detailed documentation and examples
7. **Review**: Submit for peer review and validation
8. **Release**: Release template with proper documentation

### Template Enhancement Process
1. **Usage Analysis**: Analyze how templates are being used
2. **Identify Improvements**: Find areas for enhancement
3. **Design Changes**: Design improvements while maintaining compatibility
4. **Implementation**: Implement enhancements
5. **Testing**: Test changes thoroughly
6. **Documentation**: Update documentation
7. **Release**: Release enhanced version

### Quality Assurance Process
1. **Code Review**: Review code for quality and correctness
2. **Testing**: Run comprehensive test suite
3. **Performance Testing**: Validate performance characteristics
4. **Documentation Testing**: Verify documentation accuracy
5. **Integration Testing**: Test template integration
6. **User Testing**: Validate usability and functionality

## Quality Standards

### Code Quality
- **Correctness**: All templates must implement Active Inference correctly
- **Efficiency**: Templates should be computationally efficient
- **Modularity**: Templates should be modular and extensible
- **Documentation**: Code should be well-documented
- **Testing**: Comprehensive test coverage required

### Template Quality
- **Usability**: Templates should be easy to use and understand
- **Completeness**: Templates should provide complete implementations
- **Flexibility**: Templates should support customization
- **Documentation**: Templates should have comprehensive documentation
- **Examples**: Working examples should be provided

### Documentation Quality
- **Clarity**: Documentation should be clear and accessible
- **Completeness**: All features should be documented
- **Accuracy**: Documentation should be technically accurate
- **Examples**: Practical examples should be included
- **Maintenance**: Documentation should be current

## Implementation Patterns

### Template Base Class
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class BaseTemplate(ABC):
    """Base class for Active Inference templates"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize template with configuration"""
        self.config = config
        self.template_name = self.__class__.__name__
        self.version = "1.0.0"
        self.components: Dict[str, Any] = {}
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure template logging"""
        self.logger = logging.getLogger(f"template.{self.template_name}")
        self.logger.info(f"Template {self.template_name} initialized")

    @abstractmethod
    def create_project_structure(self) -> Dict[str, Any]:
        """Create project directory structure"""
        pass

    @abstractmethod
    def generate_code_files(self) -> List[Path]:
        """Generate template code files"""
        pass

    @abstractmethod
    def generate_config_files(self) -> List[Path]:
        """Generate configuration files"""
        pass

    @abstractmethod
    def generate_documentation(self) -> List[Path]:
        """Generate documentation files"""
        pass

    def create_project(self, project_config: Dict[str, Any]) -> 'Project':
        """Create complete project from template"""
        self.logger.info(f"Creating project from {self.template_name}")

        # Validate project configuration
        self.validate_project_config(project_config)

        # Create project structure
        project_structure = self.create_project_structure()

        # Generate code files
        code_files = self.generate_code_files()

        # Generate configuration files
        config_files = self.generate_config_files()

        # Generate documentation
        docs = self.generate_documentation()

        # Create project object
        project = Project(
            name=project_config['name'],
            template=self.template_name,
            config=project_config,
            structure=project_structure,
            files=code_files + config_files + docs,
            created_at=self.get_timestamp()
        )

        self.logger.info(f"Project {project_config['name']} created successfully")
        return project

    def validate_project_config(self, config: Dict[str, Any]) -> None:
        """Validate project configuration"""
        required_fields = ['name', 'description', 'author']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required field '{field}' missing in project config")

        # Additional validation based on template type
        self.validate_template_specific_config(config)

    @abstractmethod
    def validate_template_specific_config(self, config: Dict[str, Any]) -> None:
        """Validate template-specific configuration"""
        pass

    def get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def export_template_info(self) -> Dict[str, Any]:
        """Export template information"""
        return {
            'name': self.template_name,
            'version': self.version,
            'description': self.get_description(),
            'parameters': self.get_parameters(),
            'requirements': self.get_requirements(),
            'examples': self.get_examples()
        }

    @abstractmethod
    def get_description(self) -> str:
        """Get template description"""
        pass

    @abstractmethod
    def get_parameters(self) -> List[Dict[str, Any]]:
        """Get template parameters"""
        pass

    @abstractmethod
    def get_requirements(self) -> List[str]:
        """Get template requirements"""
        pass

    @abstractmethod
    def get_examples(self) -> List[Dict[str, Any]]:
        """Get template examples"""
        pass
```

### Project Generation Framework
```python
from dataclasses import dataclass, field
from typing import Dict, Any, List
from pathlib import Path
import shutil
import json

@dataclass
class Project:
    """Represents a generated project"""
    name: str
    template: str
    config: Dict[str, Any]
    structure: Dict[str, Any]
    files: List[Path]
    created_at: str
    version: str = "1.0.0"

    def save_to_directory(self, output_path: Path) -> None:
        """Save project to directory"""
        output_path.mkdir(parents=True, exist_ok=True)

        # Create project structure
        self._create_directories(output_path)

        # Copy files
        for file_path in self.files:
            if file_path.exists():
                relative_path = file_path.relative_to(file_path.parent.parent)
                destination = output_path / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, destination)

        # Create project metadata
        self._create_project_metadata(output_path)

    def _create_directories(self, base_path: Path) -> None:
        """Create project directory structure"""
        for directory in self.structure.get('directories', []):
            (base_path / directory).mkdir(parents=True, exist_ok=True)

    def _create_project_metadata(self, output_path: Path) -> None:
        """Create project metadata file"""
        metadata = {
            'project_name': self.name,
            'template': self.template,
            'config': self.config,
            'created_at': self.created_at,
            'version': self.version,
            'structure': self.structure
        }

        metadata_file = output_path / 'project.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

class TemplateManager:
    """Manager for template operations"""

    def __init__(self, templates_path: Path = None):
        """Initialize template manager"""
        if templates_path is None:
            templates_path = Path(__file__).parent
        self.templates_path = templates_path
        self.templates: Dict[str, BaseTemplate] = {}
        self.load_templates()

    def load_templates(self) -> None:
        """Load available templates"""
        templates_dir = self.templates_path
        if not templates_dir.exists():
            return

        # Import and register templates
        for template_file in templates_dir.rglob('*.py'):
            if template_file.name.startswith('template_'):
                self._load_template_from_file(template_file)

    def _load_template_from_file(self, template_file: Path) -> None:
        """Load template from Python file"""
        try:
            # Dynamic import
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"template_{template_file.stem}",
                template_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find template class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    hasattr(attr, 'create_project') and
                    attr.__name__.endswith('Template')):
                    self.register_template(attr_name.lower().replace('template', ''), attr)

        except Exception as e:
            logger.error(f"Failed to load template from {template_file}: {e}")

    def register_template(self, name: str, template_class: type) -> None:
        """Register a template"""
        self.templates[name] = template_class
        logger.info(f"Registered template: {name}")

    def get_template(self, name: str) -> Optional[BaseTemplate]:
        """Get template by name"""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List available templates"""
        return list(self.templates.keys())

    def create_project_from_template(self, template_name: str, project_config: Dict[str, Any]) -> Project:
        """Create project from template"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        return template.create_project(project_config)
```

### Configuration Management
```python
from typing import Dict, Any, Optional, Union
import json
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class TemplateParameter:
    """Template parameter definition"""
    name: str
    type: str  # string, int, float, bool, list, dict
    description: str
    required: bool = True
    default: Any = None
    constraints: Optional[Dict[str, Any]] = None
    examples: List[Any] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []

@dataclass
class TemplateConfig:
    """Template configuration"""
    name: str
    description: str
    version: str
    parameters: List[TemplateParameter]
    requirements: List[str]
    examples: List[Dict[str, Any]]

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against template requirements"""
        errors = []

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in config:
                errors.append(f"Required parameter '{param.name}' missing")
            elif param.name in config:
                errors.extend(self._validate_parameter_value(param, config[param.name]))

        return errors

    def _validate_parameter_value(self, param: TemplateParameter, value: Any) -> List[str]:
        """Validate individual parameter value"""
        errors = []

        # Type validation
        if not self._validate_type(value, param.type):
            errors.append(f"Parameter '{param.name}' has invalid type")

        # Constraint validation
        if param.constraints:
            errors.extend(self._validate_constraints(value, param.constraints))

        return errors

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate parameter type"""
        type_mapping = {
            'string': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict
        }

        expected_class = type_mapping.get(expected_type)
        if expected_class:
            return isinstance(value, expected_class)

        return True  # Unknown type, allow

    def _validate_constraints(self, value: Any, constraints: Dict[str, Any]) -> List[str]:
        """Validate parameter constraints"""
        errors = []

        if 'min' in constraints and isinstance(value, (int, float)):
            if value < constraints['min']:
                errors.append(f"Value {value} below minimum {constraints['min']}")

        if 'max' in constraints and isinstance(value, (int, float)):
            if value > constraints['max']:
                errors.append(f"Value {value} above maximum {constraints['max']}")

        if 'options' in constraints and value not in constraints['options']:
            errors.append(f"Value {value} not in allowed options {constraints['options']}")

        return errors

    def to_json(self) -> str:
        """Export configuration as JSON"""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_data: str) -> 'TemplateConfig':
        """Create configuration from JSON"""
        data = json.loads(json_data)
        parameters = [TemplateParameter(**p) for p in data['parameters']]
        return cls(
            name=data['name'],
            description=data['description'],
            version=data['version'],
            parameters=parameters,
            requirements=data['requirements'],
            examples=data['examples']
        )
```

## Testing Guidelines

### Template Testing
- **Functionality Testing**: Test all template functionality
- **Customization Testing**: Test template customization capabilities
- **Integration Testing**: Test template integration with other components
- **Performance Testing**: Validate template performance
- **Documentation Testing**: Verify documentation examples work

### Code Quality Testing
- **Code Review**: Ensure code follows established standards
- **Style Compliance**: Verify code style compliance
- **Type Checking**: Validate type annotations and usage
- **Documentation**: Ensure comprehensive docstrings
- **Error Handling**: Test error handling and edge cases

### User Experience Testing
- **Ease of Use**: Test template usability
- **Setup Process**: Validate setup and configuration process
- **Documentation Clarity**: Test documentation clarity
- **Example Completeness**: Verify examples are complete and working

## Performance Considerations

### Template Generation Performance
- **Generation Speed**: Ensure templates generate quickly
- **Resource Usage**: Minimize resource usage during generation
- **Scalability**: Support generation of complex projects
- **Caching**: Implement caching for frequently used templates

### Runtime Performance
- **Code Efficiency**: Ensure generated code is efficient
- **Memory Usage**: Optimize memory usage of generated systems
- **Scalability**: Support scaling of generated applications
- **Resource Management**: Proper resource management in templates

## Maintenance and Evolution

### Template Updates
- **Version Management**: Maintain template versions
- **Backward Compatibility**: Ensure compatibility with existing projects
- **Migration Support**: Provide migration guides for breaking changes
- **Deprecation**: Handle deprecated template features

### Community Integration
- **User Feedback**: Incorporate user feedback and suggestions
- **Usage Analytics**: Track template usage patterns
- **Improvement Cycles**: Regular template improvement cycles
- **Community Templates**: Support community-contributed templates

## Common Challenges and Solutions

### Challenge: Template Complexity
**Solution**: Provide templates at different complexity levels and clear progression paths.

### Challenge: Customization Difficulty
**Solution**: Design flexible template architectures with clear customization points.

### Challenge: Documentation Maintenance
**Solution**: Establish processes for keeping documentation synchronized with code.

### Challenge: Quality Assurance
**Solution**: Implement comprehensive testing and validation processes.

## Getting Started as an Agent

### Development Setup
1. **Study Existing Templates**: Review current template implementations
2. **Understand Patterns**: Learn established template patterns
3. **Practice Customization**: Practice modifying and extending templates
4. **Test Thoroughly**: Develop comprehensive testing skills

### Contribution Process
1. **Identify Template Needs**: Find gaps in current template offerings
2. **Design Template Structure**: Create detailed template design
3. **Implement Template**: Follow TDD with comprehensive testing
4. **Document Completely**: Provide detailed documentation and examples
5. **Review and Validate**: Ensure quality and correctness
6. **Community Review**: Submit for community feedback

### Learning Resources
- **Template Patterns**: Study software template and scaffolding patterns
- **Active Inference Implementations**: Review existing Active Inference code
- **Configuration Management**: Learn about configuration systems
- **Code Generation**: Understand code generation techniques
- **Testing Strategies**: Master comprehensive testing approaches

## Related Documentation

- **[Templates README](./README.md)**: Templates module overview
- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Applications AGENTS.md](../AGENTS.md)**: Applications module guidelines
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution processes
- **[Knowledge Repository](../../knowledge/)**: Theoretical foundations

---

*"Active Inference for, with, by Generative AI"* - Accelerating development through comprehensive templates and proven implementation patterns.

