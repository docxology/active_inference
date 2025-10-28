Platform API Reference
======================

The Platform API provides access to infrastructure services, deployment tools, knowledge graph operations, search functionality, and collaboration features.

.. automodule:: active_inference.platform
   :members:
   :undoc-members:
   :show-inheritance:

Core Classes
------------

KnowledgeGraph
~~~~~~~~~~~~~~

.. autoclass:: active_inference.platform.knowledge_graph.KnowledgeGraph
   :members:
   :undoc-members:
   :show-inheritance:

SearchEngine
~~~~~~~~~~~~

.. autoclass:: active_inference.platform.search.SearchEngine
   :members:
   :undoc-members:
   :show-inheritance:

CollaborationHub
~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.platform.collaboration.CollaborationHub
   :members:
   :undoc-members:
   :show-inheritance:

DeploymentManager
~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.platform.deployment.DeploymentManager
   :members:
   :undoc-members:
   :show-inheritance:

WebServer
~~~~~~~~~

.. autoclass:: active_inference.platform.server.WebServer
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. automodule:: active_inference.platform.utils
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

ServiceConfig
~~~~~~~~~~~~~

.. autoclass:: active_inference.platform.models.ServiceConfig
   :members:
   :undoc-members:
   :show-inheritance:

DeploymentConfig
~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.platform.models.DeploymentConfig
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Knowledge Graph Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.platform import KnowledgeGraph

   # Initialize knowledge graph
   kg = KnowledgeGraph()

   # Add knowledge node
   node_data = {
       "id": "free_energy_principle",
       "title": "Free Energy Principle",
       "content": "The Free Energy Principle...",
       "tags": ["theory", "foundations"],
       "relationships": ["active_inference", "predictive_coding"]
   }

   kg.add_node(node_data)

   # Query knowledge graph
   results = kg.query("MATCH (n) WHERE n.tags CONTAINS 'theory' RETURN n")
   print(f"Found {len(results)} theoretical concepts")

   # Find related concepts
   related = kg.find_related("free_energy_principle", max_depth=2)
   print(f"Related concepts: {len(related)}")

Search Functionality
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.platform import SearchEngine

   # Initialize search engine
   search = SearchEngine()

   # Index content
   documents = [
       {"id": "doc1", "title": "Active Inference Basics", "content": "..."},
       {"id": "doc2", "title": "Free Energy Principle", "content": "..."}
   ]

   search.index_documents(documents)

   # Perform search
   query = "free energy principle"
   results = search.search(query, limit=10)

   print(f"Search results for '{query}': {len(results)} documents")

   # Advanced search with filters
   advanced_results = search.search(
       query="active inference",
       filters={"difficulty": "beginner", "content_type": "tutorial"},
       sort_by="relevance"
   )

Collaboration Features
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.platform import CollaborationHub

   # Initialize collaboration hub
   collab = CollaborationHub()

   # Create collaborative document
   doc_id = collab.create_document({
       "title": "Research Paper Draft",
       "content": "Initial draft...",
       "authors": ["researcher1", "researcher2"],
       "tags": ["research", "active_inference"]
   })

   # Add collaborators
   collab.add_collaborator(doc_id, "researcher3", "editor")

   # Real-time editing simulation
   collab.update_document(doc_id, {
       "content": "Updated draft with new section...",
       "last_modified": "2024-01-15T10:30:00Z"
   })

   # Get document history
   history = collab.get_document_history(doc_id)
   print(f"Document has {len(history)} revisions")

Deployment Management
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.platform import DeploymentManager

   # Initialize deployment manager
   deployer = DeploymentManager()

   # Configure deployment
   config = {
       "environment": "production",
       "services": ["knowledge_graph", "search_engine", "web_server"],
       "scaling": {"min_instances": 2, "max_instances": 10},
       "monitoring": {"enable_metrics": True, "enable_logging": True}
   }

   # Deploy platform
   deployment = deployer.deploy(config)

   # Monitor deployment
   status = deployer.get_deployment_status(deployment["id"])
   print(f"Deployment status: {status['state']}")

   # Scale services
   deployer.scale_service("knowledge_graph", instances=5)

Web Server Operations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.platform import WebServer

   # Initialize web server
   server = WebServer()

   # Configure server
   config = {
       "host": "0.0.0.0",
       "port": 8000,
       "ssl_enabled": True,
       "cors_origins": ["https://example.com"],
       "rate_limiting": {"requests_per_minute": 100}
   }

   # Start server
   server.start(config)

   # Add API routes
   @server.app.route("/api/knowledge/search")
   def search_knowledge():
       query = request.args.get("q")
       results = server.knowledge_graph.search(query)
       return jsonify(results)

   # Health check
   health_status = server.health_check()
   print(f"Server health: {health_status}")

API Reference
-------------

Complete API documentation for all platform functionality.

.. seealso::

   For platform guides, see :doc:`../platform/index`.
   For deployment, see :doc:`../platform/deployment/index`.
   For knowledge graph, see :doc:`../platform/knowledge_graph/index`.

