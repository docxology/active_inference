"""
Active Inference Knowledge Environment

A comprehensive integrated platform for Active Inference & Free Energy Principle
education, research, visualization, and application development.

This package provides modular, well-documented components for learning, research,
and development in the Active Inference framework.
"""

from pathlib import Path

__version__ = "0.1.0"
__author__ = "Active Inference Community"
__email__ = "community@activeinference.org"

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Core module imports (with optional imports for missing dependencies)
try:
    from .knowledge import (
        KnowledgeRepository,
        KnowledgeRepositoryConfig,
        KnowledgeNode,
        LearningPath,
        ContentType,
        DifficultyLevel,
        KnowledgeNodeSchema,
        Foundations,
        Mathematics,
        Implementations,
        Applications
    )
    _HAS_KNOWLEDGE = True
except ImportError:
    KnowledgeRepository = None
    KnowledgeRepositoryConfig = None
    KnowledgeNode = None
    LearningPath = None
    ContentType = None
    DifficultyLevel = None
    KnowledgeNodeSchema = None
    Foundations = None
    Mathematics = None
    Implementations = None
    Applications = None
    _HAS_KNOWLEDGE = False

try:
    from .research import (
        ResearchFramework,
        ExperimentManager,
        ExperimentConfig,
        SimulationEngine,
        ModelRunner,
        AnalysisTools,
        StatisticalAnalysis,
        BenchmarkSuite,
        PerformanceMetrics,
        DataManager,
        DataMetadata,
        DataCollectionConfig,
        DataSecurityLevel,
        DataFormat
    )
    _HAS_RESEARCH = True
except ImportError:
    ResearchFramework = None
    ExperimentManager = None
    ExperimentConfig = None
    SimulationEngine = None
    ModelRunner = None
    AnalysisTools = None
    StatisticalAnalysis = None
    BenchmarkSuite = None
    PerformanceMetrics = None
    DataManager = None
    DataMetadata = None
    DataCollectionConfig = None
    DataSecurityLevel = None
    DataFormat = None
    _HAS_RESEARCH = False

try:
    from .visualization import VisualizationEngine
    _HAS_VISUALIZATION = True
except ImportError:
    VisualizationEngine = None
    _HAS_VISUALIZATION = False

try:
    from .applications import ApplicationFramework
    _HAS_APPLICATIONS = True
except ImportError:
    ApplicationFramework = None
    _HAS_APPLICATIONS = False

try:
    from .tools import Orchestrator, Utilities
    _HAS_TOOLS = True
except ImportError:
    Orchestrator = None
    Utilities = None
    _HAS_TOOLS = False

try:
    from .platform import Platform
    _HAS_PLATFORM = True
except ImportError:
    Platform = None
    _HAS_PLATFORM = False

try:
    from .llm import (
        OllamaClient,
        LLMConfig,
        PromptTemplate,
        PromptBuilder,
        PromptManager,
        ModelManager,
        ModelInfo,
        ModelRegistry,
        ConversationManager,
        Conversation
    )
    _HAS_LLM = True
except ImportError:
    OllamaClient = None
    LLMConfig = None
    PromptTemplate = None
    PromptBuilder = None
    PromptManager = None
    ModelManager = None
    ModelInfo = None
    ModelRegistry = None
    ConversationManager = None
    Conversation = None
    _HAS_LLM = False

__all__ = [
    # Knowledge module
    "KnowledgeRepository",
    "KnowledgeRepositoryConfig",
    "KnowledgeNode",
    "LearningPath",
    "ContentType",
    "DifficultyLevel",
    "KnowledgeNodeSchema",
    "Foundations",
    "Mathematics",
    "Implementations",
    "Applications",
    # Research module
    "ResearchFramework",
    "ExperimentManager",
    "ExperimentConfig",
    "SimulationEngine",
    "ModelRunner",
    "AnalysisTools",
    "StatisticalAnalysis",
    "BenchmarkSuite",
    "PerformanceMetrics",
    "DataManager",
    "DataMetadata",
    "DataCollectionConfig",
    "DataSecurityLevel",
    "DataFormat",
    # Visualization module
    "VisualizationEngine",
    # Applications module
    "ApplicationFramework",
    # Tools module
    "Orchestrator",
    "Utilities",
    # Platform module
    "Platform",
    # LLM module
    "OllamaClient",
    "LLMConfig",
    "PromptTemplate",
    "PromptBuilder",
    "PromptManager",
    "ModelManager",
    "ModelInfo",
    "ModelRegistry",
    "ConversationManager",
    "Conversation",
    # Configuration
    "PROJECT_ROOT",
    # Status flags
    "_HAS_KNOWLEDGE",
    "_HAS_RESEARCH",
    "_HAS_VISUALIZATION",
    "_HAS_APPLICATIONS",
    "_HAS_TOOLS",
    "_HAS_PLATFORM",
    "_HAS_LLM",
]
