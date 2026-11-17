Knowledge API Reference
========================

The Knowledge API provides comprehensive access to the Active Inference knowledge repository, enabling programmatic interaction with educational content, learning paths, and knowledge management functionality.

.. automodule:: active_inference.knowledge
   :members:
   :undoc-members:
   :show-inheritance:

Core Classes
------------

KnowledgeRepository
~~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.knowledge.repository.KnowledgeRepository
   :members:
   :undoc-members:
   :show-inheritance:

LearningPathManager
~~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.knowledge.learning_paths.LearningPathManager
   :members:
   :undoc-members:
   :show-inheritance:

ContentValidator
~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.knowledge.validation.ContentValidator
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. automodule:: active_inference.knowledge.utils
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

LearningObjective
~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.knowledge.models.LearningObjective
   :members:
   :undoc-members:
   :show-inheritance:

KnowledgeNode
~~~~~~~~~~~~~

.. autoclass:: active_inference.knowledge.models.KnowledgeNode
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Knowledge Access
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.knowledge import KnowledgeRepository

   # Initialize repository
   repo = KnowledgeRepository()

   # Search for content
   results = repo.search("free energy principle")
   print(f"Found {len(results)} knowledge nodes")

   # Get specific node
   node = repo.get_node("free_energy_basics")
   print(f"Node title: {node.title}")

Learning Path Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.knowledge import LearningPathManager

   # Initialize manager
   path_manager = LearningPathManager()

   # Get available paths
   paths = path_manager.list_paths()
   print(f"Available learning paths: {len(paths)}")

   # Get specific path
   foundations_path = path_manager.get_path("foundations_complete")
   print(f"Path duration: {foundations_path.estimated_hours} hours")

Content Validation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.knowledge import ContentValidator

   # Initialize validator
   validator = ContentValidator()

   # Validate knowledge node
   node = {"id": "test_node", "title": "Test Node", "content": "..."}
   validation_result = validator.validate_node(node)

   if validation_result["valid"]:
       print("Node is valid")
   else:
       print(f"Validation errors: {validation_result['errors']}")

API Reference
-------------

Complete API documentation for all knowledge management functionality.

.. seealso::

   For user guides, see :doc:`../knowledge/index`.
   For implementation examples, see :doc:`../knowledge/implementations/index`.

