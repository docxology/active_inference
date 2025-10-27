# Prompt Templates - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Prompt Templates module of the Active Inference Knowledge Environment. It outlines prompt template development, implementation patterns, and best practices for creating and maintaining structured guidance templates.

## Prompt Templates Module Overview

The Prompt Templates module serves as the central repository for structured prompt templates that guide AI agents in performing specialized tasks across the Active Inference Knowledge Environment. These templates ensure consistent, high-quality outputs by providing detailed instructions, requirements, and quality standards for various development, maintenance, and enhancement activities.

## Directory Structure

```
tools/prompts/
├── documentation_audit_prompt.md          # Repository documentation auditing
├── knowledge_base_audit_prompt.md         # Knowledge content validation
├── knowledge_base_enhancement_prompt.md   # Content enhancement strategies
├── knowledge_base_maintenance_prompt.md   # Content maintenance workflows
└── learning_path_development_prompt.md   # Learning curriculum development
```

## Core Responsibilities

### Prompt Template Development
- **Template Creation**: Develop comprehensive prompt templates for various tasks
- **Quality Assurance**: Ensure templates produce consistent, high-quality outputs
- **Template Maintenance**: Keep templates current with platform evolution
- **Integration**: Ensure templates integrate seamlessly with agent workflows

### Template Quality Management
- **Validation Systems**: Create validation frameworks for template outputs
- **Performance Monitoring**: Track template effectiveness and usage patterns
- **User Feedback Integration**: Incorporate feedback for template improvement
- **Version Management**: Manage template versions and evolution

### Template Organization
- **Categorization**: Organize templates by function and application area
- **Cross-Referencing**: Maintain relationships between related templates
- **Accessibility**: Ensure templates are easily discoverable and usable
- **Documentation**: Maintain comprehensive template documentation

## Development Workflows

### Template Creation Process
1. **Requirements Analysis**: Identify specific tasks requiring structured guidance
2. **Template Research**: Study existing templates and identify improvement opportunities
3. **Structure Design**: Design comprehensive template structure and content
4. **Implementation**: Create detailed prompt templates following established patterns
5. **Testing**: Validate template effectiveness and output quality
6. **Integration**: Integrate templates with platform template system
7. **Documentation**: Create comprehensive usage documentation and examples

### Template Enhancement Process
1. **Usage Analysis**: Analyze template usage patterns and effectiveness
2. **Feedback Collection**: Gather user feedback on template performance
3. **Improvement Design**: Design enhancements based on analysis and feedback
4. **Implementation**: Update templates with improvements
5. **Validation**: Test enhanced templates for effectiveness
6. **Deployment**: Deploy improved templates to production

## Quality Standards

### Template Quality Standards
- **Clarity**: Instructions must be clear, unambiguous, and comprehensive
- **Completeness**: Templates must cover all aspects of the target task
- **Consistency**: Uniform structure and terminology across all templates
- **Effectiveness**: Templates must produce high-quality, useful outputs
- **Maintainability**: Templates must be easy to understand and modify

### Content Quality Standards
- **Accuracy**: All information and instructions must be accurate and current
- **Relevance**: Content must be relevant to the target task and context
- **Comprehensiveness**: Templates must provide sufficient detail for task completion
- **Usability**: Templates must be easy to use and understand
- **Validation**: Templates must include validation and quality checkpoints

## Implementation Patterns

### Prompt Template Structure Pattern
```python
class BasePromptTemplate:
    """Base class for prompt templates"""

    def __init__(self, template_name: str, category: str, version: str = "1.0.0"):
        """Initialize prompt template"""
        self.template_name = template_name
        self.category = category
        self.version = version
        self.content = self.load_template_content()
        self.metadata = self.load_template_metadata()

    def load_template_content(self) -> str:
        """Load template content from file"""
        template_path = self.get_template_path()
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_template_metadata(self) -> Dict[str, Any]:
        """Load template metadata"""
        return {
            'name': self.template_name,
            'category': self.category,
            'version': self.version,
            'created_date': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'author': 'Active Inference Community',
            'description': self.get_template_description(),
            'tags': self.get_template_tags(),
            'quality_score': self.calculate_quality_score()
        }

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute template with given parameters"""
        # Validate parameters
        self.validate_parameters(parameters)

        # Format template with parameters
        formatted_prompt = self.format_template(parameters)

        # Execute with language model
        execution_result = self.execute_with_llm(formatted_prompt)

        # Validate output
        validation_result = self.validate_output(execution_result)

        return {
            'prompt': formatted_prompt,
            'result': execution_result,
            'validation': validation_result,
            'metadata': self.metadata
        }

    def format_template(self, parameters: Dict[str, Any]) -> str:
        """Format template with parameters"""
        formatted_content = self.content

        # Replace parameter placeholders
        for key, value in parameters.items():
            placeholder = f"{{{key}}}"
            formatted_content = formatted_content.replace(placeholder, str(value))

        return formatted_content

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate template parameters"""
        required_params = self.get_required_parameters()
        optional_params = self.get_optional_parameters()

        # Check required parameters
        for param in required_params:
            if param not in parameters:
                raise TemplateParameterError(f"Missing required parameter: {param}")

        # Validate parameter types and values
        for param, value in parameters.items():
            if param in required_params or param in optional_params:
                self.validate_parameter_value(param, value)

    def validate_output(self, output: Any) -> Dict[str, Any]:
        """Validate template execution output"""
        validation_result = {
            'is_valid': True,
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }

        # Perform quality validation
        quality_issues = self.check_output_quality(output)
        if quality_issues:
            validation_result['is_valid'] = False
            validation_result['issues'] = quality_issues

        # Calculate quality score
        validation_result['quality_score'] = self.calculate_output_quality_score(output)

        # Generate recommendations
        validation_result['recommendations'] = self.generate_improvement_recommendations(output)

        return validation_result

    @abstractmethod
    def get_template_description(self) -> str:
        """Get template description"""
        pass

    @abstractmethod
    def get_template_tags(self) -> List[str]:
        """Get template tags"""
        pass

    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        """Get required template parameters"""
        pass

    @abstractmethod
    def get_optional_parameters(self) -> List[str]:
        """Get optional template parameters"""
        pass

    @abstractmethod
    def validate_parameter_value(self, parameter: str, value: Any) -> None:
        """Validate individual parameter value"""
        pass

    @abstractmethod
    def check_output_quality(self, output: Any) -> List[str]:
        """Check output quality and identify issues"""
        pass

    @abstractmethod
    def calculate_output_quality_score(self, output: Any) -> float:
        """Calculate quality score for output"""
        pass

    @abstractmethod
    def generate_improvement_recommendations(self, output: Any) -> List[str]:
        """Generate recommendations for output improvement"""
        pass
```

### Template Registry Pattern
```python
class PromptTemplateRegistry:
    """Registry for managing prompt templates"""

    def __init__(self):
        """Initialize template registry"""
        self.templates: Dict[str, BasePromptTemplate] = {}
        self.categories: Dict[str, List[str]] = {}
        self.load_templates()

    def load_templates(self) -> None:
        """Load all available templates"""
        template_files = self.discover_template_files()

        for template_file in template_files:
            try:
                template = self.create_template_from_file(template_file)
                self.register_template(template)
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")

    def discover_template_files(self) -> List[Path]:
        """Discover all template files in the prompts directory"""
        prompts_dir = Path(__file__).parent / 'prompts'
        return list(prompts_dir.glob('*.md'))

    def create_template_from_file(self, template_file: Path) -> BasePromptTemplate:
        """Create template instance from file"""
        # Parse template file and create appropriate template class
        template_config = self.parse_template_file(template_file)

        if template_config['category'] == 'documentation':
            return DocumentationTemplate(template_config)
        elif template_config['category'] == 'knowledge':
            return KnowledgeTemplate(template_config)
        elif template_config['category'] == 'learning':
            return LearningTemplate(template_config)
        else:
            return BasePromptTemplate(**template_config)

    def register_template(self, template: BasePromptTemplate) -> None:
        """Register template in registry"""
        template_key = f"{template.category}.{template.template_name}"
        self.templates[template_key] = template

        if template.category not in self.categories:
            self.categories[template.category] = []
        self.categories[template.category].append(template.template_name)

        logger.info(f"Registered template: {template_key}")

    def get_template(self, category: str, template_name: str) -> Optional[BasePromptTemplate]:
        """Get template by category and name"""
        template_key = f"{category}.{template_name}"
        return self.templates.get(template_key)

    def get_templates_by_category(self, category: str) -> List[BasePromptTemplate]:
        """Get all templates in a category"""
        return [self.templates[f"{category}.{name}"] for name in self.categories.get(category, [])]

    def search_templates(self, query: str) -> List[BasePromptTemplate]:
        """Search templates by query"""
        matching_templates = []

        for template in self.templates.values():
            if (query.lower() in template.template_name.lower() or
                query.lower() in template.get_template_description().lower() or
                any(query.lower() in tag.lower() for tag in template.get_template_tags())):
                matching_templates.append(template)

        return matching_templates

    def validate_all_templates(self) -> Dict[str, Any]:
        """Validate all registered templates"""
        validation_results = {
            'total_templates': len(self.templates),
            'valid_templates': 0,
            'invalid_templates': 0,
            'template_results': {}
        }

        for template_key, template in self.templates.items():
            try:
                # Validate template structure and content
                template_validation = self.validate_template_structure(template)
                validation_results['template_results'][template_key] = template_validation

                if template_validation['is_valid']:
                    validation_results['valid_templates'] += 1
                else:
                    validation_results['invalid_templates'] += 1

            except Exception as e:
                validation_results['template_results'][template_key] = {
                    'is_valid': False,
                    'error': str(e)
                }
                validation_results['invalid_templates'] += 1

        return validation_results

    def parse_template_file(self, template_file: Path) -> Dict[str, Any]:
        """Parse template file and extract configuration"""
        # Parse markdown file and extract template metadata and content
        content = template_file.read_text(encoding='utf-8')

        # Extract metadata from frontmatter or headers
        metadata = self.extract_template_metadata(content)

        return {
            'template_name': metadata.get('name', template_file.stem),
            'category': metadata.get('category', 'general'),
            'version': metadata.get('version', '1.0.0'),
            'content': content,
            'file_path': str(template_file)
        }

    def extract_template_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from template content"""
        metadata = {}

        # Look for YAML frontmatter
        if content.startswith('---'):
            end_idx = content.find('---', 3)
            if end_idx > 0:
                frontmatter = content[3:end_idx]
                metadata = yaml.safe_load(frontmatter)

        # Extract from headers if no frontmatter
        if not metadata:
            lines = content.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                if line.startswith('# '):
                    metadata['title'] = line[2:].strip()
                elif line.startswith('**Category:**'):
                    metadata['category'] = line.split('**Category:**')[1].strip()
                elif line.startswith('**Version:**'):
                    metadata['version'] = line.split('**Version:**')[1].strip()

        return metadata
```

### Template Validation Pattern
```python
class TemplateValidator:
    """Comprehensive template validation system"""

    def __init__(self):
        """Initialize template validator"""
        self.validation_rules = self.load_validation_rules()
        self.quality_metrics = self.load_quality_metrics()

    def validate_template(self, template: BasePromptTemplate) -> Dict[str, Any]:
        """Validate template structure and content"""
        validation_result = {
            'is_valid': True,
            'structure_validation': {},
            'content_validation': {},
            'quality_validation': {},
            'overall_score': 0.0,
            'issues': [],
            'recommendations': []
        }

        # Structure validation
        structure_result = self.validate_template_structure(template)
        validation_result['structure_validation'] = structure_result

        if not structure_result['is_valid']:
            validation_result['is_valid'] = False
            validation_result['issues'].extend(structure_result['issues'])

        # Content validation
        content_result = self.validate_template_content(template)
        validation_result['content_validation'] = content_result

        if not content_result['is_valid']:
            validation_result['is_valid'] = False
            validation_result['issues'].extend(content_result['issues'])

        # Quality validation
        quality_result = self.validate_template_quality(template)
        validation_result['quality_validation'] = quality_result
        validation_result['overall_score'] = quality_result['quality_score']

        # Generate recommendations
        validation_result['recommendations'] = self.generate_validation_recommendations(
            structure_result,
            content_result,
            quality_result
        )

        return validation_result

    def validate_template_structure(self, template: BasePromptTemplate) -> Dict[str, Any]:
        """Validate template structure"""
        structure_result = {
            'is_valid': True,
            'required_sections': [],
            'missing_sections': [],
            'issues': []
        }

        # Check required sections
        required_sections = [
            'Mission & Objectives',
            'Task Requirements',
            'Implementation Guidelines',
            'Quality Standards'
        ]

        content = template.content

        for section in required_sections:
            if section.lower() in content.lower():
                structure_result['required_sections'].append(section)
            else:
                structure_result['missing_sections'].append(section)
                structure_result['is_valid'] = False
                structure_result['issues'].append(f"Missing required section: {section}")

        # Check for clear instructions
        if 'clear instructions' not in content.lower():
            structure_result['is_valid'] = False
            structure_result['issues'].append("Template should include clear instructions")

        return structure_result

    def validate_template_content(self, template: BasePromptTemplate) -> Dict[str, Any]:
        """Validate template content quality"""
        content_result = {
            'is_valid': True,
            'clarity_score': 0.0,
            'completeness_score': 0.0,
            'consistency_score': 0.0,
            'issues': []
        }

        content = template.content

        # Clarity validation
        clarity_indicators = ['clear', 'specific', 'detailed', 'step-by-step', 'example']
        clarity_score = sum(1 for indicator in clarity_indicators if indicator in content.lower())
        content_result['clarity_score'] = clarity_score / len(clarity_indicators)

        # Completeness validation
        completeness_indicators = ['requirements', 'objectives', 'validation', 'quality', 'standards']
        completeness_score = sum(1 for indicator in completeness_indicators if indicator in content.lower())
        content_result['completeness_score'] = completeness_score / len(completeness_indicators)

        # Consistency validation
        consistency_issues = self.check_consistency_issues(content)
        content_result['consistency_score'] = 1.0 if not consistency_issues else 0.5
        content_result['issues'].extend(consistency_issues)

        # Overall content validation
        if (content_result['clarity_score'] < 0.6 or
            content_result['completeness_score'] < 0.6 or
            content_result['consistency_score'] < 0.8):
            content_result['is_valid'] = False

        return content_result

    def validate_template_quality(self, template: BasePromptTemplate) -> Dict[str, Any]:
        """Validate overall template quality"""
        quality_result = {
            'quality_score': 0.0,
            'effectiveness_metrics': {},
            'improvement_areas': []
        }

        # Calculate overall quality score
        content_validation = self.validate_template_content(template)
        structure_validation = self.validate_template_structure(template)

        # Weighted quality score
        quality_score = (
            content_validation['clarity_score'] * 0.3 +
            content_validation['completeness_score'] * 0.3 +
            content_validation['consistency_score'] * 0.2 +
            (1.0 if structure_validation['is_valid'] else 0.0) * 0.2
        )

        quality_result['quality_score'] = quality_score

        # Identify improvement areas
        if content_validation['clarity_score'] < 0.7:
            quality_result['improvement_areas'].append('clarity')
        if content_validation['completeness_score'] < 0.7:
            quality_result['improvement_areas'].append('completeness')
        if content_validation['consistency_score'] < 0.8:
            quality_result['improvement_areas'].append('consistency')
        if not structure_validation['is_valid']:
            quality_result['improvement_areas'].append('structure')

        return quality_result

    def check_consistency_issues(self, content: str) -> List[str]:
        """Check for consistency issues in template content"""
        issues = []

        # Check for consistent terminology
        if 'README.md' in content and 'readme.md' in content.lower():
            issues.append("Inconsistent capitalization of README.md")

        # Check for consistent formatting
        markdown_headers = content.count('\n#') + content.count('\n##') + content.count('\n###')
        if markdown_headers == 0:
            issues.append("No markdown headers found - consider adding structure")

        return issues

    def generate_validation_recommendations(self, structure_result: Dict[str, Any],
                                          content_result: Dict[str, Any],
                                          quality_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for template improvement"""
        recommendations = []

        # Structure recommendations
        for section in structure_result['missing_sections']:
            recommendations.append(f"Add required section: {section}")

        # Content recommendations
        if content_result['clarity_score'] < 0.7:
            recommendations.append("Improve clarity by adding specific examples and detailed instructions")

        if content_result['completeness_score'] < 0.7:
            recommendations.append("Add missing requirements, validation criteria, and quality standards")

        if content_result['consistency_score'] < 0.8:
            recommendations.append("Ensure consistent terminology and formatting throughout template")

        # Quality recommendations
        for area in quality_result['improvement_areas']:
            recommendations.append(f"Improve {area} to enhance overall template quality")

        return recommendations

    def load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for templates"""
        return {
            'required_sections': [
                'Mission & Objectives',
                'Task Requirements',
                'Implementation Guidelines',
                'Quality Standards'
            ],
            'clarity_indicators': [
                'clear', 'specific', 'detailed', 'step-by-step', 'example'
            ],
            'completeness_indicators': [
                'requirements', 'objectives', 'validation', 'quality', 'standards'
            ],
            'quality_thresholds': {
                'clarity': 0.6,
                'completeness': 0.6,
                'consistency': 0.8,
                'overall': 0.75
            }
        }

    def load_quality_metrics(self) -> Dict[str, Any]:
        """Load quality metrics for template evaluation"""
        return {
            'weights': {
                'clarity': 0.3,
                'completeness': 0.3,
                'consistency': 0.2,
                'structure': 0.2
            },
            'scoring_criteria': {
                'excellent': 0.9,
                'good': 0.75,
                'acceptable': 0.6,
                'needs_improvement': 0.4
            }
        }
```

## Testing Guidelines

### Template Testing Categories
1. **Structure Testing**: Validate template structure and required sections
2. **Content Testing**: Test template content quality and completeness
3. **Execution Testing**: Test template execution and output validation
4. **Integration Testing**: Test template integration with platform systems
5. **Performance Testing**: Validate template execution performance

### Template Testing Requirements
- **Structure Validation**: Ensure templates follow required structure
- **Content Quality**: Validate content clarity, completeness, and consistency
- **Parameter Handling**: Test parameter validation and formatting
- **Output Validation**: Ensure output meets quality standards
- **Error Handling**: Test error conditions and recovery

## Performance Considerations

### Template Execution Performance
- **Loading Speed**: Fast template loading and initialization
- **Parameter Processing**: Efficient parameter validation and formatting
- **Memory Usage**: Minimal memory footprint for template operations
- **Caching**: Cache frequently used templates for performance
- **Parallel Execution**: Support parallel template execution when possible

### Quality Monitoring
- **Output Quality Tracking**: Monitor quality of template-generated outputs
- **Usage Analytics**: Track template usage patterns and effectiveness
- **Feedback Integration**: Incorporate user feedback for improvement
- **Version Performance**: Monitor performance across template versions

## Getting Started as an Agent

### Template Development Setup
1. **Study Existing Templates**: Understand current template implementations and patterns
2. **Learn Template Standards**: Master template structure and quality requirements
3. **Set Up Development Environment**: Configure environment for template development
4. **Test Template Framework**: Ensure template testing framework works correctly
5. **Understand Integration Points**: Learn how templates integrate with platform

### Template Development Process
1. **Identify Template Needs**: Analyze platform requirements for new templates
2. **Design Template Structure**: Create comprehensive template structure
3. **Implement Template Content**: Write detailed, high-quality template content
4. **Add Validation Systems**: Implement comprehensive validation and quality checks
5. **Testing and Integration**: Thoroughly test and integrate templates
6. **Documentation**: Create comprehensive template documentation

### Quality Assurance
1. **Structure Validation**: Ensure templates follow required structure
2. **Content Quality**: Validate content clarity and completeness
3. **Functionality Testing**: Test template execution and output quality
4. **Integration Testing**: Validate template integration with platform
5. **Performance Validation**: Ensure templates meet performance requirements

## Common Challenges and Solutions

### Challenge: Template Clarity
**Solution**: Use clear, specific language with concrete examples and step-by-step instructions.

### Challenge: Template Completeness
**Solution**: Include all necessary requirements, validation criteria, and quality standards.

### Challenge: Template Consistency
**Solution**: Maintain consistent structure, terminology, and formatting across all templates.

### Challenge: Template Effectiveness
**Solution**: Regularly validate template outputs and incorporate feedback for improvement.

### Challenge: Template Maintenance
**Solution**: Implement version control and systematic review processes for template updates.

## Collaboration Guidelines

### Work with Other Agents
- **Platform Agents**: Ensure templates integrate with platform infrastructure
- **Knowledge Agents**: Validate template content for educational effectiveness
- **Implementation Agents**: Test templates with practical implementation scenarios
- **Quality Agents**: Maintain template quality standards and validation
- **User Experience Agents**: Ensure templates are user-friendly and effective

### Community Engagement
- **Template Usage Feedback**: Collect and analyze template usage and effectiveness
- **Improvement Suggestions**: Encourage suggestions for template enhancement
- **Best Practices**: Share template development and usage best practices
- **Quality Standards**: Maintain and evolve template quality standards

## Related Documentation

- **[Main Tools README](../README.md)**: Development tools overview and usage
- **[Tools AGENTS.md](../AGENTS.md)**: Development tools agent guidelines
- **[Prompt Templates README](./README.md)**: Prompt templates module documentation
- **[Documentation Tools](../documentation/README.md)**: Documentation generation tools
- **[Platform Integration](../../platform/README.md)**: Platform integration guidelines
- **[Quality Standards](../../applications/best_practices/)**: Quality standards and best practices

---

*"Active Inference for, with, by Generative AI"* - Enhancing platform development through structured guidance, comprehensive templates, and collaborative intelligence.
