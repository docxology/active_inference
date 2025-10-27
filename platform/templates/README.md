# Platform Templates

**HTML templates for the Active Inference Knowledge Environment web platform.**

## 📖 Overview

This directory contains Jinja2 HTML templates that render the web interface for the Active Inference Knowledge Environment platform. These templates provide the user interface for accessing knowledge content, exploring learning paths, and interacting with platform features.

## 🎯 Purpose

The platform templates provide:

- **Web Interface**: User-facing web pages and components
- **Layout Structure**: Consistent layout and navigation across the platform
- **Content Rendering**: Dynamic rendering of knowledge content and data
- **Error Handling**: User-friendly error pages and messages

## 📁 Directory Structure

```
platform/templates/
├── base.html          # Base template with common layout
├── index.html         # Home page template
├── error.html         # Error page template
└── README.md          # This file
```

## 📄 Core Templates

### base.html

Base template that provides the common structure for all platform pages including:

- **HTML Structure**: DOCTYPE, head, and body elements
- **Common Styles**: CSS and styling includes
- **Navigation**: Platform navigation and menu structure
- **Footer**: Common footer with links and information
- **Block Definitions**: Jinja2 block definitions for content injection

#### Template Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Active Inference Knowledge Environment{% endblock %}</title>
    {% block styles %}{% endblock %}
</head>
<body>
    {% block navigation %}
        <!-- Navigation content -->
    {% endblock %}
    
    {% block content %}
        <!-- Page content -->
    {% endblock %}
    
    {% block footer %}
        <!-- Footer content -->
    {% endblock %}
    
    {% block scripts %}{% endblock %}
</body>
</html>
```

### index.html

Home page template that extends the base template and provides:

- **Welcome Content**: Platform introduction and overview
- **Quick Access**: Links to key knowledge areas
- **Learning Paths**: Display of available learning paths
- **Search Interface**: Quick search functionality
- **Recent Content**: Showcase of recent knowledge updates

### error.html

Error page template for handling errors gracefully with:

- **Error Message**: Clear error message display
- **Error Details**: Contextual error information
- **Helpful Links**: Navigation to useful pages
- **Error Recovery**: Suggestions for resolving issues

## 🚀 Usage

### Rendering Templates

```python
from flask import Flask, render_template

app = Flask(__name__, template_folder='platform/templates')

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html', 
                         title="Active Inference Knowledge Environment")

@app.route('/knowledge/<path:content_id>')
def knowledge_content(content_id):
    """Render knowledge content page"""
    content = load_knowledge_content(content_id)
    return render_template('knowledge.html', 
                         content=content)
```

### Template Variables

Templates can receive various context variables:

```python
# Base template variables
{
    'title': 'Page Title',
    'description': 'Page description',
    'keywords': ['tag1', 'tag2']
}

# Index page variables
{
    'recent_content': [...],
    'featured_learning_paths': [...],
    'statistics': {...}
}

# Knowledge content variables
{
    'content': {...},
    'related_content': [...],
    'breadcrumbs': [...]
}
```

## 🔧 Template Development

### Creating New Templates

1. **Extend Base Template**: Use base.html for common structure
2. **Define Blocks**: Override appropriate Jinja2 blocks
3. **Add Content**: Implement template-specific content
4. **Test Rendering**: Verify template rendering and styling

### Template Example

```html
{% extends "base.html" %}

{% block title %}
    {{ content.title }} - Active Inference Knowledge Environment
{% endblock %}

{% block content %}
    <article>
        <header>
            <h1>{{ content.title }}</h1>
            <div class="metadata">
                <span>Difficulty: {{ content.difficulty }}</span>
                <span>Reading Time: {{ content.metadata.estimated_reading_time }} min</span>
            </div>
        </header>
        
        <section class="content">
            {% for section in content.content %}
                <h2>{{ section.title }}</h2>
                <div>{{ section.body|markdown }}</div>
            {% endfor %}
        </section>
        
        {% if content.interactive_exercises %}
            <section class="exercises">
                <h2>Interactive Exercises</h2>
                {% for exercise in content.interactive_exercises %}
                    <div class="exercise">{{ exercise|markdown }}</div>
                {% endfor %}
            </section>
        {% endif %}
    </article>
{% endblock %}
```

## 🎨 Styling and Design

### CSS Integration

Templates can include custom CSS:

```html
{% block styles %}
    <link rel="stylesheet" href="{{ url_for('static', filename='css/custom.css') }}">
    <style>
        /* Template-specific styles */
    </style>
{% endblock %}
```

### Responsive Design

Templates should be responsive and mobile-friendly:

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```

## 📚 Related Documentation

- **[Platform README](../README.md)**: Platform services overview
- **[Platform AGENTS.md](../AGENTS.md)**: Agent guidelines for platform development
- **[Flask Documentation](https://flask.palletsprojects.com/)**: Flask template engine documentation

---

*"Active Inference for, with, by Generative AI"* - Providing intuitive web interface through Dexterity template design and implementation.

