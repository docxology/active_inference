Tools API Reference
===================

The Tools API provides access to development utilities, documentation generation, testing frameworks, orchestration components, and automation tools.

.. automodule:: active_inference.tools
   :members:
   :undoc-members:
   :show-inheritance:

Core Classes
------------

DocumentationGenerator
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.tools.documentation.DocumentationGenerator
   :members:
   :undoc-members:
   :show-inheritance:

TestRunner
~~~~~~~~~~

.. autoclass:: active_inference.tools.testing.TestRunner
   :members:
   :undoc-members:
   :show-inheritance:

Orchestrator
~~~~~~~~~~~~

.. autoclass:: active_inference.tools.orchestrators.Orchestrator
   :members:
   :undoc-members:
   :show-inheritance:

ContentValidator
~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.tools.utilities.ContentValidator
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. automodule:: active_inference.tools.utils
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

ToolConfig
~~~~~~~~~~

.. autoclass:: active_inference.tools.models.ToolConfig
   :members:
   :undoc-members:
   :show-inheritance:

ValidationResult
~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.tools.models.ValidationResult
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Documentation Generation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.tools import DocumentationGenerator

   # Initialize documentation generator
   doc_gen = DocumentationGenerator()

   # Generate API documentation
   config = {
       "source_path": "src/active_inference",
       "output_path": "docs/api",
       "docstring_style": "google",
       "include_private": False
   }

   doc_gen.generate_api_docs(config)
   print("API documentation generated")

   # Generate knowledge documentation
   knowledge_config = {
       "knowledge_path": "knowledge/",
       "output_path": "docs/knowledge/",
       "include_metadata": True,
       "cross_references": True
   }

   doc_gen.generate_knowledge_docs(knowledge_config)

Testing and Validation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.tools import TestRunner, ContentValidator

   # Initialize test runner
   test_runner = TestRunner()

   # Run comprehensive test suite
   results = test_runner.run_full_suite()

   # Generate test report
   report = test_runner.generate_report(results)
   print(f"Tests passed: {report['passed']}/{report['total']}")

   # Initialize content validator
   validator = ContentValidator()

   # Validate knowledge content
   knowledge_files = ["knowledge/foundations/*.json"]
   validation_results = validator.validate_knowledge_content(knowledge_files)

   for result in validation_results:
       if not result["valid"]:
           print(f"Validation error in {result['file']}: {result['errors']}")

Workflow Orchestration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.tools import Orchestrator

   # Initialize orchestrator
   orchestrator = Orchestrator()

   # Define workflow
   workflow = {
       "name": "content_update_workflow",
       "steps": [
           {"name": "validate_content", "tool": "ContentValidator"},
           {"name": "generate_docs", "tool": "DocumentationGenerator"},
           {"name": "run_tests", "tool": "TestRunner"},
           {"name": "deploy_docs", "tool": "DeploymentManager"}
       ],
       "dependencies": {
           "generate_docs": ["validate_content"],
           "run_tests": ["generate_docs"],
           "deploy_docs": ["run_tests"]
       }
   }

   # Execute workflow
   result = orchestrator.execute_workflow(workflow)

   if result["success"]:
       print("Workflow completed successfully")
   else:
       print(f"Workflow failed at step: {result['failed_step']}")

Content Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.tools import ContentAnalyzer

   # Initialize content analyzer
   analyzer = ContentAnalyzer()

   # Analyze knowledge base completeness
   analysis_config = {
       "content_paths": ["knowledge/"],
       "analysis_types": ["completeness", "quality", "cross_references"],
       "output_format": "detailed"
   }

   analysis_results = analyzer.analyze_content(analysis_config)

   # Generate improvement recommendations
   recommendations = analyzer.generate_recommendations(analysis_results)

   print(f"Content analysis complete. {len(recommendations)} recommendations generated")

   # Export analysis report
   analyzer.export_report(analysis_results, "content_analysis_report.json")

Metadata Management
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.tools import MetadataManager

   # Initialize metadata manager
   metadata_mgr = MetadataManager()

   # Update metadata for knowledge content
   update_config = {
       "content_type": "knowledge_nodes",
       "fields_to_update": ["last_reviewed", "version"],
       "batch_size": 50
   }

   updated_count = metadata_mgr.batch_update_metadata(update_config)
   print(f"Updated metadata for {updated_count} knowledge nodes")

   # Validate metadata completeness
   validation_config = {
       "required_fields": ["id", "title", "content", "metadata"],
       "content_types": ["foundations", "mathematics", "implementations"]
   }

   validation_results = metadata_mgr.validate_metadata_completeness(validation_config)

   if validation_results["complete"]:
       print("All metadata requirements satisfied")
   else:
       print(f"Missing metadata for: {validation_results['missing']}")

API Reference
-------------

Complete API documentation for all tools functionality.

.. seealso::

   For development tools, see :doc:`../../tools/README.md`.
   For testing frameworks, see :doc:`../../tests/README.md`.
   For documentation tools, see :doc:`documentation/README.md`.

