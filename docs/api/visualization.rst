Visualization API Reference
===========================

The Visualization API provides access to interactive diagrams, animations, dashboards, and comparative analysis tools for Active Inference concepts and research data.

.. automodule:: active_inference.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Core Classes
------------

DiagramGenerator
~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.visualization.diagrams.DiagramGenerator
   :members:
   :undoc-members:
   :show-inheritance:

AnimationEngine
~~~~~~~~~~~~~~~

.. autoclass:: active_inference.visualization.animations.AnimationEngine
   :members:
   :undoc-members:
   :show-inheritance:

DashboardBuilder
~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.visualization.dashboards.DashboardBuilder
   :members:
   :undoc-members:
   :show-inheritance:

ComparativeAnalyzer
~~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.visualization.comparative.ComparativeAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. automodule:: active_inference.visualization.utils
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

VisualizationConfig
~~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.visualization.models.VisualizationConfig
   :members:
   :undoc-members:
   :show-inheritance:

AnimationParameters
~~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.visualization.models.AnimationParameters
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Creating Diagrams
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.visualization import DiagramGenerator

   # Initialize diagram generator
   diagrams = DiagramGenerator()

   # Create architecture diagram
   config = {
       "diagram_type": "system_architecture",
       "components": ["sensory_input", "neural_processing", "motor_output"],
       "connections": [("sensory_input", "neural_processing"), ("neural_processing", "motor_output")]
   }

   diagram = diagrams.create_diagram(config)
   diagram.save("architecture_diagram.png")

Building Animations
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.visualization import AnimationEngine

   # Initialize animation engine
   animator = AnimationEngine()

   # Configure belief update animation
   config = {
       "animation_type": "belief_dynamics",
       "time_steps": 100,
       "initial_beliefs": [0.5, 0.3, 0.2],
       "evidence_stream": generate_evidence_data()
   }

   animation = animator.create_animation(config)
   animation.save("belief_updates.gif")

Creating Dashboards
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.visualization import DashboardBuilder

   # Initialize dashboard builder
   dashboard = DashboardBuilder()

   # Configure research dashboard
   config = {
       "title": "Active Inference Research Dashboard",
       "panels": [
           {"type": "performance_metrics", "data": performance_data},
           {"type": "belief_evolution", "data": belief_data},
           {"type": "parameter_sensitivity", "data": sensitivity_data}
       ],
       "layout": "grid_2x2"
   }

   research_dashboard = dashboard.create_dashboard(config)
   research_dashboard.save("research_dashboard.html")

Comparative Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.visualization import ComparativeAnalyzer

   # Initialize comparative analyzer
   comparator = ComparativeAnalyzer()

   # Configure comparison
   config = {
       "comparison_type": "model_performance",
       "models": ["baseline", "active_inference_v1", "active_inference_v2"],
       "metrics": ["accuracy", "efficiency", "robustness"],
       "datasets": ["dataset_a", "dataset_b", "dataset_c"]
   }

   comparison = comparator.create_comparison(config)
   comparison.save("model_comparison.html")

Interactive Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.visualization import InteractiveBuilder

   # Create interactive belief exploration
   interactive = InteractiveBuilder()

   config = {
       "visualization_type": "interactive_beliefs",
       "parameters": {
           "prior_range": [0.0, 1.0],
           "likelihood_range": [0.0, 1.0],
           "evidence_levels": [1, 5, 10, 20]
       },
       "real_time_updates": True
   }

   belief_explorer = interactive.create_interactive(config)
   belief_explorer.save("belief_explorer.html")

API Reference
-------------

Complete API documentation for all visualization functionality.

.. seealso::

   For visualization guides, see :doc:`../visualization/index`.
   For diagram creation, see :doc:`../visualization/diagrams/index`.
   For animation tools, see :doc:`../visualization/animations/index`.

