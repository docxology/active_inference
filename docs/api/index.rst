API Reference
=============

Comprehensive API documentation for the Active Inference Knowledge Environment.
This section provides detailed reference for all public classes, functions, and
modules.

.. toctree::
   :maxdepth: 2
   :caption: API Sections:

   knowledge
   research
   visualization
   applications
   platform
   tools

Overview
--------

The API is organized into the following main modules:

🧠 **Knowledge Module**
   Educational content and learning path management

🔬 **Research Module**
   Experiment framework and analysis tools

👁️ **Visualization Module**
   Interactive diagrams and dashboards

🛠️ **Applications Module**
   Templates and case studies

🚀 **Platform Module**
   Infrastructure and deployment tools

📋 **Tools Module**
   Utilities and development tools

Quick Start
-----------

Basic usage examples:

.. code-block:: python

   from active_inference import KnowledgeRepository, ResearchFramework

   # Initialize knowledge repository
   repo = KnowledgeRepository()
   nodes = repo.search("free energy")

   # Run experiments
   research = ResearchFramework()
   results = research.run_experiment("perception_test")

Module Structure
----------------

Each module follows consistent patterns:

- **Configuration classes** for setup and customization
- **Core classes** implementing main functionality
- **Utility functions** for common operations
- **Exception classes** for error handling

Code Examples
-------------

Interactive examples throughout the documentation:

.. code-block:: python

   # Complete example with error handling
   try:
       repo = KnowledgeRepository()
       path = repo.get_learning_path("foundations")
       print(f"Path: {path.name}")
   except Exception as e:
       logger.error(f"Error: {e}")

Testing
-------

All modules include comprehensive test suites:

.. code-block:: bash

   # Run all tests
   make test

   # Run specific module tests
   python -m pytest tests/unit/test_knowledge.py

   # Generate coverage report
   python -m pytest --cov=active_inference

Best Practices
--------------

When using the API:

1. **Initialize properly** - Use configuration objects
2. **Handle exceptions** - Implement proper error handling
3. **Use type hints** - Leverage full type information
4. **Follow patterns** - Use established design patterns
5. **Test thoroughly** - Write comprehensive tests

Version Information
-------------------

Current API version: **0.1.0**

Breaking changes are documented in the changelog.

.. seealso::

   For learning resources, see :doc:`../knowledge/index`.
   For research tools, see :doc:`../research/index`.
   For deployment guides, see :doc:`../platform/index`.

