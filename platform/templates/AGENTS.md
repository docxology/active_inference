# Platform Templates - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working with platform templates in the Active Inference Knowledge Environment.

## Module Overview

The Platform Templates module manages HTML templates for the web platform, providing user interface rendering, layout structure, and content presentation. This module ensures consistent, accessible, and responsive web interfaces for the Active Inference Knowledge Environment.

## Core Responsibilities

### Template Development
- **Template Creation**: Create and maintain HTML templates for platform pages
- **Layout Management**: Ensure consistent layout and navigation structure
- **Content Rendering**: Implement dynamic content rendering and presentation
- **Responsive Design**: Create mobile-friendly and accessible templates

### Template Quality
- **Accessibility**: Ensure templates meet accessibility standards
- **Performance**: Optimize template rendering and loading performance
- **Consistency**: Maintain consistent styling and behavior across templates
- **Documentation**: Document template usage and customization options

## Development Workflows

### Template Development Process
1. **Requirements Analysis**: Identify template requirements and use cases
2. **Design Structure**: Design template layout and component structure
3. **Implementation**: Create templates following established patterns
4. **Styling**: Apply CSS styling and responsive design
5. **Testing**: Test template rendering and functionality
6. **Documentation**: Document template usage and examples

### Template Rendering Workflow
```python
from flask import Flask, render_template
from typing import Dict, Any

class TemplateRenderer:
    """Render platform templates with context"""

    def __init__(self, app: Flask):
        """Initialize template renderer"""
        self.app = app
        self.template_helpers = self.load_template_helpers()

    def render_knowledge_page(self, content_id: str) -> str:
        """Render knowledge content page"""
        
        # Load knowledge content
        content = self.load_knowledge_content(content_id)
        
        # Load related content
        related = self.load_related_content(content_id)
        
        # Prepare template context
        context = {
            "content": content,
            "related_content": related,
            "breadcrumbs": self.generate_breadcrumbs(content),
            "navigation": self.get_navigation_context()
        }
        
        # Render template
        return render_template('knowledge.html', **context)

    def prepare_template_context(self, page_type: str, **kwargs) -> Dict[str, Any]:
        """Prepare context for template rendering"""
        
        base_context = {
            "platform_name": "Active Inference Knowledge Environment",
            "navigation": self.get_navigation_context(),
            "footer": self.get_footer_context()
        }
        
        # Add page-specific context
        if page_type == "knowledge":
            base_context.update(self.prepare_knowledge_context(**kwargs))
        elif page_type == "learning_path":
            base_context.update(self.prepare_learning_path_context(**kwargs))
        
        return base_context
```

## Implementation Patterns

### Template Inheritance Pattern
```html
<!-- base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}{% endblock %}</title>
    {% block extra_head %}{% endblock %}
</head>
<body>
    {% block navigation %}{% endblock %}
    {% block content %}{% endblock %}
    {% block footer %}{% endblock %}
    {% block scripts %}{% endblock %}
</body>
</html>

<!-- page.html -->
{% extends "base.html" %}

{% block title %}
    {{ page_title }} - Active Inference Knowledge Environment
{% endblock %}

{% block content %}
    <main>
        <h1>{{ page_title }}</h1>
        <div class="content">{{ content }}</div>
    </main>
{% endblock %}
```

### Template Component Pattern
```html
<!-- macros.html -->
{% macro knowledge_card(node) %}
    <article class="knowledge-card">
        <h3>{{ node.title }}</h3>
        <div class="metadata">
            <span class="difficulty">{{ node.difficulty }}</span>
            <span class="time">{{ node.estimated_time }} min</span>
        </div>
        <p>{{ node.description }}</p>
        <a href="/knowledge/{{ node.id }}">Learn More</a>
    </article>
{% endmacro %}

<!-- usage.html -->
{% import 'macros.html' as macros %}

{% for node in knowledge_nodes %}
    {{ macros.knowledge_card(node) }}
{% endfor %}
```

## Quality Standards

### Template Quality Requirements
- **Accessibility**: WCAG 2.1 AA compliance
- **Performance**: Fast rendering and loading times
- **Responsiveness**: Mobile-friendly and responsive design
- **Browser Compatibility**: Cross-browser compatibility

### Code Quality
- **Semantic HTML**: Use appropriate HTML elements
- **Clean Structure**: Well-organized and readable templates
- **Documentation**: Inline comments for complex sections
- **Testing**: Template rendering and functionality tests

## Testing Guidelines

### Template Testing Requirements
1. **Rendering Tests**: Test template rendering with various contexts
2. **Component Tests**: Test individual template components
3. **Responsive Tests**: Test responsive design across devices
4. **Accessibility Tests**: Validate accessibility compliance

## Related Documentation

- **[Platform Templates README](./README.md)**: Template overview and usage
- **[Platform README](../README.md)**: Platform services documentation
- **[Platform AGENTS.md](../AGENTS.md)**: Agent guidelines for platform development

---

*"Active Inference for, with, by Generative AI"* - Providing intuitive web interface through effective template design and implementation.

