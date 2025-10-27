Research API Reference
=======================

The Research API provides access to experimental frameworks, simulation tools, analysis methods, and benchmarking capabilities for Active Inference research.

.. automodule:: active_inference.research
   :members:
   :undoc-members:
   :show-inheritance:

Core Classes
------------

ExperimentFramework
~~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.research.experiments.ExperimentFramework
   :members:
   :undoc-members:
   :show-inheritance:

SimulationEngine
~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.research.simulations.SimulationEngine
   :members:
   :undoc-members:
   :show-inheritance:

AnalysisTools
~~~~~~~~~~~~~

.. autoclass:: active_inference.research.analysis.AnalysisTools
   :members:
   :undoc-members:
   :show-inheritance:

BenchmarkSuite
~~~~~~~~~~~~~~

.. autoclass:: active_inference.research.benchmarks.BenchmarkSuite
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. automodule:: active_inference.research.utils
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

ExperimentConfig
~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.research.models.ExperimentConfig
   :members:
   :undoc-members:
   :show-inheritance:

SimulationParameters
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.research.models.SimulationParameters
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Running Experiments
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.research import ExperimentFramework

   # Initialize experiment framework
   experiments = ExperimentFramework()

   # Configure experiment
   config = {
       "experiment_type": "perception_test",
       "duration": 1000,
       "parameters": {"noise_level": 0.1}
   }

   # Run experiment
   results = experiments.run_experiment(config)
   print(f"Experiment completed with {len(results)} data points")

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.research import AnalysisTools

   # Initialize analysis tools
   analysis = AnalysisTools()

   # Load experimental data
   data = analysis.load_data("experiment_results.csv")

   # Perform information-theoretic analysis
   entropy = analysis.calculate_entropy(data)
   mutual_info = analysis.calculate_mutual_information(data["input"], data["output"])

   print(f"Data entropy: {entropy}")
   print(f"Mutual information: {mutual_info}")

Running Simulations
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from active_inference.research import SimulationEngine

   # Initialize simulation engine
   simulator = SimulationEngine()

   # Configure simulation parameters
   params = {
       "model_type": "active_inference_agent",
       "time_steps": 10000,
       "environment": "dynamic_maze",
       "learning_rate": 0.01
   }

   # Run simulation
   simulation_results = simulator.run_simulation(params)

   # Analyze results
   performance_metrics = simulator.analyze_results(simulation_results)
   print(f"Simulation performance: {performance_metrics}")

Benchmarking
~~~~~~~~~~~~

.. code-block:: python

   from active_inference.research import BenchmarkSuite

   # Initialize benchmark suite
   benchmarks = BenchmarkSuite()

   # Run comprehensive benchmarks
   results = benchmarks.run_full_suite()

   # Generate comparison report
   report = benchmarks.generate_comparison_report(results)
   print("Benchmark results generated")

   # Export results
   benchmarks.export_results(results, "benchmark_report.json")

API Reference
-------------

Complete API documentation for all research functionality.

.. seealso::

   For research guides, see :doc:`../research/index`.
   For analysis methods, see :doc:`../research/analysis/index`.
   For simulation tools, see :doc:`../research/simulations/index`.
