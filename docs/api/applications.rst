Applications API Reference
==========================

The Applications API provides access to implementation templates, case studies, integration patterns, and best practices for deploying Active Inference in real-world applications.

.. automodule:: active_inference.applications
   :members:
   :undoc-members:
   :show-inheritance:

Core Classes
------------

TemplateManager
~~~~~~~~~~~~~~~

.. autoclass:: active_inference.applications.templates.TemplateManager
   :members:
   :undoc-members:
   :show-inheritance:

CaseStudyAnalyzer
~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.applications.case_studies.CaseStudyAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

IntegrationManager
~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.applications.integrations.IntegrationManager
   :members:
   :undoc-members:
   :show-inheritance:

BestPracticesGuide
~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.applications.best_practices.BestPracticesGuide
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. automodule:: active_inference.applications.utils
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

ApplicationTemplate
~~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.applications.models.ApplicationTemplate
   :members:
   :undoc-members:
   :show-inheritance:

IntegrationConfig
~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.applications.models.IntegrationConfig
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Using Application Templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.applications import TemplateManager

   # Initialize template manager
   templates = TemplateManager()

   # Get available templates
   available_templates = templates.list_templates()
   print(f"Available templates: {len(available_templates)}")

   # Load neural control template
   neural_template = templates.get_template("neural_control_system")

   # Customize template for specific use case
   config = {
       "domain": "robotics",
       "sensors": ["camera", "imu", "lidar"],
       "actuators": ["servo_motors", "wheels"],
       "control_frequency": 100  # Hz
   }

   customized_template = templates.customize_template(neural_template, config)

   # Generate application code
   application_code = templates.generate_code(customized_template)
   print("Application code generated")

Analyzing Case Studies
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.applications import CaseStudyAnalyzer

   # Initialize case study analyzer
   analyzer = CaseStudyAnalyzer()

   # Load case studies for specific domain
   robotics_studies = analyzer.get_case_studies("robotics")

   # Analyze success patterns
   patterns = analyzer.extract_patterns(robotics_studies)
   print(f"Identified {len(patterns)} success patterns")

   # Generate implementation recommendations
   recommendations = analyzer.generate_recommendations(patterns)
   print("Implementation recommendations generated")

Managing Integrations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.applications import IntegrationManager

   # Initialize integration manager
   integrations = IntegrationManager()

   # Configure external API integration
   api_config = {
       "api_type": "rest",
       "endpoint": "https://api.external-service.com/v1",
       "authentication": "bearer_token",
       "data_format": "json"
   }

   integration = integrations.create_integration(api_config)

   # Test integration
   test_result = integrations.test_integration(integration)
   if test_result["success"]:
       print("Integration test passed")
   else:
       print(f"Integration test failed: {test_result['error']}")

Applying Best Practices
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.applications import BestPracticesGuide

   # Initialize best practices guide
   guide = BestPracticesGuide()

   # Get best practices for neural control
   neural_practices = guide.get_best_practices("neural_control")

   # Get scalability guidelines
   scalability_guide = guide.get_scalability_guide("distributed_systems")

   # Generate implementation checklist
   checklist = guide.generate_checklist("production_deployment")
   print(f"Implementation checklist has {len(checklist)} items")

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.applications import PerformanceOptimizer

   # Initialize performance optimizer
   optimizer = PerformanceOptimizer()

   # Analyze current application performance
   performance_report = optimizer.analyze_performance(current_app)

   # Get optimization recommendations
   recommendations = optimizer.get_recommendations(performance_report)

   # Apply optimizations
   optimized_app = optimizer.apply_optimizations(current_app, recommendations)

   print("Application optimized for performance")

API Reference
-------------

Complete API documentation for all application functionality.

.. seealso::

   For application guides, see :doc:`../applications/index`.
   For templates, see :doc:`../applications/templates/index`.
   For case studies, see :doc:`../applications/case_studies/index`.
