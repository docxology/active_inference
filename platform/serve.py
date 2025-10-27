"""
Platform Server for Active Inference Knowledge Environment

Provides a web interface for accessing the knowledge repository, running
simulations, and exploring Active Inference concepts interactively.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.exceptions import HTTPException

from active_inference.knowledge import KnowledgeRepository, KnowledgeRepositoryConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
           template_folder=Path(__file__).parent.parent / "platform" / "templates",
           static_folder=Path(__file__).parent.parent / "platform" / "static")

# Initialize knowledge repository
project_root = Path(__file__).parent.parent
knowledge_config = KnowledgeRepositoryConfig(
    root_path=project_root / "knowledge",
    auto_index=True
)
knowledge_repo = KnowledgeRepository(knowledge_config)


@app.route('/')
def index():
    """Main platform homepage"""
    try:
        # Get repository statistics
        stats = knowledge_repo.get_statistics()

        # Get available learning paths
        learning_paths = knowledge_repo.get_learning_paths()

        return render_template('index.html',
                             stats=stats,
                             learning_paths=learning_paths,
                             title="Active Inference Knowledge Environment")
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        return render_template('error.html', error=str(e))


@app.route('/learn')
def learn():
    """Learning interface"""
    try:
        # Get learning paths by difficulty
        beginner_paths = knowledge_repo.get_learning_paths(difficulty="beginner")
        intermediate_paths = knowledge_repo.get_learning_paths(difficulty="intermediate")
        advanced_paths = knowledge_repo.get_learning_paths(difficulty="advanced")

        return render_template('learn.html',
                             beginner_paths=beginner_paths,
                             intermediate_paths=intermediate_paths,
                             advanced_paths=advanced_paths,
                             title="Learning Paths")
    except Exception as e:
        logger.error(f"Error loading learning paths: {e}")
        return render_template('error.html', error=str(e))


@app.route('/learn/<path_id>')
def learning_path(path_id: str):
    """Display specific learning path"""
    try:
        path = knowledge_repo.get_learning_path(path_id)
        if not path:
            return render_template('error.html', error=f"Learning path '{path_id}' not found"), 404

        # Validate the path
        validation = knowledge_repo.validate_learning_path(path_id)

        # Get path nodes with details
        path_nodes = []
        for node_id in path.nodes:
            node = knowledge_repo.get_node(node_id)
            if node:
                path_nodes.append(node)

        return render_template('learning_path.html',
                             path=path,
                             path_nodes=path_nodes,
                             validation=validation,
                             title=f"Learning Path: {path.name}")
    except Exception as e:
        logger.error(f"Error loading learning path {path_id}: {e}")
        return render_template('error.html', error=str(e))


@app.route('/search')
def search():
    """Search interface"""
    query = request.args.get('q', '')
    content_type = request.args.get('type', '')
    difficulty = request.args.get('difficulty', '')

    try:
        # Build search filters
        content_types = []
        if content_type:
            # Map string to enum (simplified for web interface)
            type_mapping = {
                'foundation': 'foundation',
                'mathematics': 'mathematics',
                'implementation': 'implementation',
                'application': 'application'
            }
            content_types = [type_mapping.get(content_type, '')]

        difficulties = []
        if difficulty:
            difficulties = [difficulty]

        # Perform search
        results = knowledge_repo.search(
            query=query,
            limit=50
        )

        # Apply additional filters if needed
        if content_types or difficulties:
            filtered_results = []
            for node in results:
                if content_types and node.content_type.value not in content_types:
                    continue
                if difficulties and node.difficulty.value not in difficulties:
                    continue
                filtered_results.append(node)
            results = filtered_results

        return render_template('search.html',
                             query=query,
                             results=results,
                             total_results=len(results),
                             title=f"Search Results: {query}" if query else "Search")
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        return render_template('error.html', error=str(e))


@app.route('/knowledge/<node_id>')
def knowledge_node(node_id: str):
    """Display specific knowledge node"""
    try:
        node = knowledge_repo.get_node(node_id)
        if not node:
            return render_template('error.html', error=f"Knowledge node '{node_id}' not found"), 404

        # Get prerequisite graph
        prereq_graph = knowledge_repo.get_prerequisites_graph(node_id)

        return render_template('knowledge_node.html',
                             node=node,
                             prereq_graph=prereq_graph,
                             title=node.title)
    except Exception as e:
        logger.error(f"Error loading knowledge node {node_id}: {e}")
        return render_template('error.html', error=str(e))


@app.route('/api/knowledge/graph')
def api_knowledge_graph():
    """API endpoint for knowledge graph data"""
    try:
        graph = knowledge_repo.export_knowledge_graph(format='json')
        return jsonify(graph)
    except Exception as e:
        logger.error(f"Error exporting knowledge graph: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/knowledge/stats')
def api_knowledge_stats():
    """API endpoint for repository statistics"""
    try:
        stats = knowledge_repo.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search')
def api_search():
    """API endpoint for search functionality"""
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))

        results = knowledge_repo.search(query=query, limit=limit)

        # Convert to JSON-serializable format
        results_data = []
        for node in results:
            results_data.append({
                'id': node.id,
                'title': node.title,
                'content_type': node.content_type.value,
                'difficulty': node.difficulty.value,
                'description': node.description,
                'tags': node.tags,
                'learning_objectives': node.learning_objectives
            })

        return jsonify({
            'query': query,
            'total_results': len(results_data),
            'results': results_data
        })
    except Exception as e:
        logger.error(f"Error in API search: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/visualize')
def visualize():
    """Visualization interface"""
    try:
        return render_template('visualize.html',
                             title="Visualization Tools")
    except Exception as e:
        logger.error(f"Error loading visualization: {e}")
        return render_template('error.html', error=str(e))


@app.route('/research')
def research():
    """Research tools interface"""
    try:
        return render_template('research.html',
                             title="Research Tools")
    except Exception as e:
        logger.error(f"Error loading research tools: {e}")
        return render_template('error.html', error=str(e))


@app.route('/about')
def about():
    """About page"""
    try:
        return render_template('about.html',
                             title="About")
    except Exception as e:
        logger.error(f"Error loading about page: {e}")
        return render_template('error.html', error=str(e))


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return render_template('error.html', error="Internal server error"), 500


def create_sample_templates():
    """Create sample HTML templates if they don't exist"""
    templates_dir = Path(__file__).parent / "templates"

    if not templates_dir.exists():
        templates_dir.mkdir()

    # Create basic templates
    templates = {
        'base.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}{{ title }}{% endblock %} - Active Inference</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-brain"></i> Active Inference Knowledge
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/learn"><i class="fas fa-graduation-cap"></i> Learn</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/search"><i class="fas fa-search"></i> Search</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/visualize"><i class="fas fa-eye"></i> Visualize</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/research"><i class="fas fa-microscope"></i> Research</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>

    <footer class="bg-light text-center py-3 mt-5">
        <div class="container">
            <p class="mb-0">&copy; 2024 Active Inference Community. Built with generative AI assistance.</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
        ''',

        'index.html': '''
{% extends "base.html" %}
{% block content %}
<div class="row">
    <div class="col-lg-8 mx-auto text-center">
        <h1 class="display-4 mb-4">
            <i class="fas fa-brain text-primary"></i><br>
            Active Inference Knowledge Environment
        </h1>
        <p class="lead mb-4">
            A comprehensive integrated platform for Active Inference & Free Energy Principle
            education, research, visualization, and application development.
        </p>
        <div class="row g-3 justify-content-center">
            <div class="col-md-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-book fa-2x text-primary mb-3"></i>
                        <h5 class="card-title">📚 Knowledge</h5>
                        <p class="card-text">{{ stats.total_nodes }} nodes, {{ stats.total_paths }} paths</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-microscope fa-2x text-success mb-3"></i>
                        <h5 class="card-title">🔬 Research</h5>
                        <p class="card-text">Tools & frameworks</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-eye fa-2x text-info mb-3"></i>
                        <h5 class="card-title">👁️ Visualize</h5>
                        <p class="card-text">Interactive diagrams</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-cogs fa-2x text-warning mb-3"></i>
                        <h5 class="card-title">🛠️ Applications</h5>
                        <p class="card-text">Real-world examples</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

{% if learning_paths %}
<div class="row mt-5">
    <div class="col-12">
        <h2 class="mb-4">🚀 Learning Paths</h2>
        <div class="row g-3">
            {% for path in learning_paths %}
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-body">
                        <h5 class="card-title">{{ path.name }}</h5>
                        <p class="card-text">{{ path.description }}</p>
                        <div class="d-flex justify-content-between align-items-center">
                            <small class="text-muted">
                                <i class="fas fa-clock"></i> {{ path.estimated_hours }}h
                                <i class="fas fa-signal ms-2"></i> {{ path.difficulty.value }}
                            </small>
                            <a href="/learn/{{ path.id }}" class="btn btn-primary btn-sm">Start</a>
                        </div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</div>
{% endif %}

<div class="row mt-5">
    <div class="col-12 text-center">
        <h3>Ready to Start Learning?</h3>
        <p class="mb-3">Begin your Active Inference journey with our structured learning paths.</p>
        <a href="/learn" class="btn btn-primary btn-lg me-2">
            <i class="fas fa-graduation-cap"></i> Start Learning
        </a>
        <a href="/search" class="btn btn-outline-primary btn-lg">
            <i class="fas fa-search"></i> Explore Content
        </a>
    </div>
</div>
{% endblock %}
        ''',

        'error.html': '''
{% extends "base.html" %}
{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="alert alert-danger text-center">
            <h4 class="alert-heading">
                <i class="fas fa-exclamation-triangle"></i> Error
            </h4>
            <p>{{ error }}</p>
            <hr>
            <a href="/" class="btn btn-primary">
                <i class="fas fa-home"></i> Back to Home
            </a>
        </div>
    </div>
</div>
{% endblock %}
        '''
    }

    for filename, content in templates.items():
        template_file = templates_dir / filename
        if not template_file.exists():
            with open(template_file, 'w') as f:
                f.write(content.strip())


def main():
    """Main server entry point"""
    # Create sample templates
    create_sample_templates()

    # Start the server
    print("🚀 Starting Active Inference Knowledge Platform")
    print("📚 Knowledge repository loaded"    print(f"   - Nodes: {len(knowledge_repo._nodes)}")
    print(f"   - Paths: {len(knowledge_repo._paths)}")
    print()
    print("🌐 Web interface available at: http://localhost:5000")
    print("📖 API endpoints available at: http://localhost:5000/api/")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"\n❌ Server error: {e}")


if __name__ == "__main__":
    main()
