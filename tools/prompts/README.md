# Prompt Templates

Comprehensive collection of prompt templates and guidelines for AI agents working within the Active Inference Knowledge Environment. These templates provide structured guidance for various development, maintenance, and enhancement tasks across all platform components.

## Overview

The Prompt Templates module contains carefully crafted prompt templates that guide AI agents in performing specific tasks within the Active Inference Knowledge Environment. These templates ensure consistent, high-quality outputs across documentation, knowledge management, learning path development, and platform maintenance activities.

## Directory Structure

```
tools/prompts/
├── documentation_audit_prompt.md          # Comprehensive repository documentation auditing
├── knowledge_base_audit_prompt.md         # Knowledge base content validation and auditing
├── knowledge_base_enhancement_prompt.md   # Knowledge content enhancement and improvement
├── knowledge_base_maintenance_prompt.md   # Knowledge base maintenance and updates
└── learning_path_development_prompt.md   # Learning path creation and curriculum development
```

## Prompt Categories

### Documentation Templates
**Files**: `documentation_audit_prompt.md`

Comprehensive prompt for performing systematic documentation audits across the entire repository. Includes detailed workflows for:
- Repository structure analysis
- Documentation gap identification
- Content enhancement strategies
- Quality assurance validation
- Cross-reference integration

**Usage**:
```python
from tools.prompts.documentation_audit_prompt import DocumentationAuditPrompt

# Load documentation audit template
audit_prompt = DocumentationAuditPrompt.load()

# Execute documentation audit
audit_results = audit_prompt.execute_audit(repository_path=".")
```

### Knowledge Base Templates
**Files**: `knowledge_base_audit_prompt.md`, `knowledge_base_enhancement_prompt.md`, `knowledge_base_maintenance_prompt.md`

Specialized prompts for knowledge base management including:
- Content accuracy validation
- Educational quality assessment
- Knowledge gap analysis
- Content enhancement strategies
- Maintenance workflow automation

**Usage**:
```python
from tools.prompts.knowledge_base_audit_prompt import KnowledgeBaseAuditPrompt

# Load knowledge audit template
kb_audit_prompt = KnowledgeBaseAuditPrompt.load()

# Audit knowledge base content
validation_results = kb_audit_prompt.validate_content(knowledge_path="./knowledge")
```

### Learning Path Templates
**Files**: `learning_path_development_prompt.md`

Structured prompts for creating comprehensive learning paths and curricula including:
- Learning objective definition
- Prerequisite mapping
- Content sequencing
- Assessment integration
- Educational effectiveness validation

**Usage**:
```python
from tools.prompts.learning_path_development_prompt import LearningPathPrompt

# Load learning path template
path_prompt = LearningPathPrompt.load()

# Develop new learning path
learning_path = path_prompt.create_learning_path(
    domain="active_inference",
    difficulty="intermediate",
    target_audience="researchers"
)
```

## Template Standards

### Prompt Structure Requirements
All prompt templates follow a standardized structure:

```markdown
# Template Title

**Brief description and purpose**

## 🎯 Mission & Objectives

Clear statement of template purpose and intended outcomes.

## 📋 Task Requirements

Detailed requirements and specifications for the task.

## 🏗️ Implementation Guidelines

Step-by-step implementation guidance and best practices.

## 📊 Quality Standards

Quality criteria and validation requirements.

## 📞 Support & Validation

Integration points and validation procedures.
```

### Template Quality Standards
- **Clarity**: Clear, unambiguous instructions and requirements
- **Completeness**: Comprehensive coverage of all task aspects
- **Consistency**: Uniform structure and terminology across templates
- **Adaptability**: Templates adaptable to different contexts and requirements
- **Validation**: Built-in validation and quality checks

## Usage Examples

### Documentation Audit Example
```python
# Load documentation audit prompt
audit_prompt = load_prompt_template('documentation_audit_prompt.md')

# Configure audit parameters
audit_config = {
    'repository_root': '.',
    'audit_depth': 'comprehensive',
    'include_diagrams': True,
    'quality_threshold': 0.95
}

# Execute audit
audit_results = execute_prompt_template(audit_prompt, audit_config)

# Process results
for directory, status in audit_results['documentation_status'].items():
    if status['missing_docs']:
        print(f"Missing documentation in: {directory}")
```

### Knowledge Base Enhancement Example
```python
# Load knowledge enhancement prompt
enhancement_prompt = load_prompt_template('knowledge_base_enhancement_prompt.md')

# Configure enhancement parameters
enhancement_config = {
    'domain': 'active_inference',
    'content_type': 'mathematical_foundations',
    'target_difficulty': 'intermediate',
    'include_examples': True,
    'validation_required': True
}

# Execute enhancement
enhanced_content = execute_prompt_template(enhancement_prompt, enhancement_config)

# Validate enhancement
validation_result = validate_knowledge_content(enhanced_content)
```

## Template Development

### Creating New Templates
When developing new prompt templates:

1. **Analyze Requirements**: Understand the specific task and requirements
2. **Study Existing Templates**: Learn from established template patterns
3. **Design Template Structure**: Create clear, comprehensive template structure
4. **Include Quality Gates**: Add validation and quality requirements
5. **Test Template Effectiveness**: Validate template produces desired outcomes
6. **Document Template Usage**: Provide clear usage instructions and examples

### Template Categories
Organize templates by functional categories:
- **Development Templates**: For software development and implementation
- **Documentation Templates**: For content creation and documentation
- **Analysis Templates**: For research, analysis, and validation
- **Management Templates**: For project management and coordination

## Integration

### Platform Integration
Templates integrate with the broader platform through:
- **Template Registry**: Centralized template management and discovery
- **Execution Engine**: Standardized template execution framework
- **Validation System**: Automated quality validation for template outputs
- **Version Control**: Template versioning and change management

### Agent Integration
Templates designed for seamless agent integration:
- **Context Awareness**: Templates adapt to current context and requirements
- **Progressive Enhancement**: Templates can be enhanced based on execution results
- **Error Recovery**: Built-in error handling and recovery mechanisms
- **Feedback Loops**: Templates incorporate feedback for continuous improvement

## Quality Assurance

### Template Validation
- **Structure Validation**: Ensure templates follow required structure
- **Content Validation**: Verify template content quality and completeness
- **Functionality Testing**: Test template execution and output quality
- **Integration Testing**: Validate template integration with platform

### Performance Monitoring
- **Execution Monitoring**: Track template execution performance
- **Output Quality**: Monitor quality of template-generated outputs
- **Usage Analytics**: Analyze template usage patterns and effectiveness
- **Improvement Tracking**: Track template improvements over time

## Best Practices

### Template Design
1. **Clear Objectives**: Define clear, measurable objectives for each template
2. **Structured Guidance**: Provide step-by-step guidance and requirements
3. **Quality Gates**: Include validation and quality assurance checkpoints
4. **Flexibility**: Design templates to be adaptable to different contexts
5. **Documentation**: Provide comprehensive documentation and examples

### Template Usage
1. **Context Understanding**: Understand the context before using templates
2. **Parameter Configuration**: Properly configure template parameters
3. **Output Validation**: Always validate template outputs
4. **Feedback Integration**: Provide feedback for template improvement
5. **Version Awareness**: Use appropriate template versions for tasks

## Contributing

### Adding New Templates
1. **Identify Need**: Analyze platform needs for new template categories
2. **Design Template**: Create comprehensive template following standards
3. **Implement Template**: Develop template with proper structure and validation
4. **Test Template**: Thoroughly test template functionality and outputs
5. **Document Template**: Provide comprehensive documentation and examples
6. **Integration**: Integrate template with platform template system

### Template Review Process
1. **Functionality Review**: Validate template functionality and completeness
2. **Quality Review**: Ensure template meets quality standards
3. **Integration Review**: Verify template integration with platform
4. **Documentation Review**: Validate template documentation completeness
5. **Performance Review**: Ensure template performance meets requirements

## Related Documentation

- **[Main Tools README](../README.md)**: Development tools overview and usage
- **[Tools AGENTS.md](../AGENTS.md)**: Development tools agent guidelines
- **[Documentation Tools](../documentation/README.md)**: Documentation generation tools
- **[Platform Integration](../../platform/README.md)**: Platform integration guidelines
- **[Knowledge Management](../../knowledge/README.md)**: Knowledge base management

---

*"Active Inference for, with, by Generative AI"* - Enhancing platform development through structured guidance, comprehensive templates, and collaborative intelligence.


