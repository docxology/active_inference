# Active Inference Knowledge Environment
# Comprehensive build and development system

.PHONY: help setup clean test docs serve install-deps lint format check-all

# Default target
help:
	@echo "Active Inference Knowledge Environment"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup       - Set up development environment"
	@echo "  install-deps - Install all dependencies"
	@echo "  test        - Run all tests"
	@echo "  docs        - Generate documentation"
	@echo "  serve       - Start the knowledge platform"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  clean       - Clean build artifacts"
	@echo "  check-all   - Run all quality checks"
	@echo ""
	@echo "Learning paths:"
	@echo "  learn-foundations - Start foundations learning track"
	@echo "  learn-research    - Start research learning track"
	@echo "  learn-applications - Start applications learning track"
	@echo ""
	@echo "Research tools:"
	@echo "  run-experiments   - Run experiment suite"
	@echo "  run-simulations   - Run simulation benchmarks"
	@echo "  analyze-results   - Analyze experimental results"

# Environment setup
setup:
	@echo "Setting up Active Inference Knowledge Environment..."
	@python3 -m venv venv
	@source venv/bin/activate && pip install --upgrade pip
	@make install-deps
	@make docs
	@echo "Setup complete! Run 'make serve' to start the platform."

install-deps:
	@echo "Installing dependencies..."
	@source venv/bin/activate && pip install -r requirements.txt
	@npm install -g jupyterlab matplotlib plotly dash streamlit
	@echo "Dependencies installed."

# Testing
test:
	@echo "Running tests..."
	@source venv/bin/activate && python -m pytest tests/ -v --cov=src/ --cov-report=html

test-unit:
	@echo "Running unit tests..."
	@source venv/bin/activate && python -m pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	@source venv/bin/activate && python -m pytest tests/integration/ -v

test-knowledge:
	@echo "Testing knowledge repository..."
	@source venv/bin/activate && python -m pytest tests/knowledge/ -v

# Documentation
docs:
	@echo "Generating documentation..."
	@source venv/bin/activate && sphinx-build docs/ docs/_build/
	@mkdir -p documentation/knowledge
	@python tools/documentation/generate_knowledge_docs.py

docs-serve:
	@echo "Serving documentation locally..."
	@source venv/bin/activate && sphinx-autobuild docs/ docs/_build/

# Platform services
serve:
	@echo "Starting Active Inference Knowledge Platform..."
	@source venv/bin/activate && python platform/serve.py

serve-knowledge:
	@echo "Starting knowledge repository server..."
	@source venv/bin/activate && python platform/knowledge_server.py

serve-visualization:
	@echo "Starting visualization server..."
	@source venv/bin/activate && python platform/visualization_server.py

# Code quality
lint:
	@echo "Running linting checks..."
	@source venv/bin/activate && flake8 src/ tests/
	@source venv/bin/activate && mypy src/

format:
	@echo "Formatting code..."
	@source venv/bin/activate && black src/ tests/
	@source venv/bin/activate && isort src/ tests/

check-all: lint test
	@echo "All quality checks passed!"

# Learning paths
learn-foundations:
	@echo "Starting Foundations Learning Track..."
	@source venv/bin/activate && python tools/learning/foundations_track.py

learn-research:
	@echo "Starting Research Learning Track..."
	@source venv/bin/activate && python tools/learning/research_track.py

learn-applications:
	@echo "Starting Applications Learning Track..."
	@source venv/bin/activate && python tools/learning/applications_track.py

# Research tools
run-experiments:
	@echo "Running experiment suite..."
	@source venv/bin/activate && python research/experiments/run_suite.py

run-simulations:
	@echo "Running simulation benchmarks..."
	@source venv/bin/activate && python research/simulations/run_benchmarks.py

analyze-results:
	@echo "Analyzing experimental results..."
	@source venv/bin/activate && python research/analysis/analyze_results.py

# Knowledge management
knowledge-index:
	@echo "Indexing knowledge repository..."
	@source venv/bin/activate && python platform/knowledge_graph/index_knowledge.py

knowledge-search:
	@echo "Starting knowledge search interface..."
	@source venv/bin/activate && python platform/search/search_interface.py

# Visualization tools
visualize-concepts:
	@echo "Starting concept visualization..."
	@source venv/bin/activate && python visualization/diagrams/concept_explorer.py

visualize-models:
	@echo "Starting model comparison visualization..."
	@source venv/bin/activate && python visualization/comparative/model_comparison.py

# Application tools
generate-templates:
	@echo "Generating application templates..."
	@source venv/bin/activate && python applications/templates/generate_templates.py

run-case-studies:
	@echo "Running case study examples..."
	@source venv/bin/activate && python applications/case_studies/run_examples.py

# Deployment
deploy-local:
	@echo "Deploying locally..."
	@docker-compose up -d

deploy-production:
	@echo "Deploying to production..."
	@ansible-playbook deployment/production.yml

# Clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf docs/_build/
	@rm -rf .coverage htmlcov/
	@rm -rf __pycache__ **/__pycache__/
	@rm -rf *.pyc **/*.pyc
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."

# Development utilities
shell:
	@echo "Starting development shell..."
	@source venv/bin/activate && python

jupyter:
	@echo "Starting Jupyter Lab..."
	@source venv/bin/activate && jupyter lab

monitor:
	@echo "Starting monitoring dashboard..."
	@source venv/bin/activate && python tools/monitoring/dashboard.py
