"""
Base Orchestrator Module

Provides the foundation for research process orchestration in Active Inference.
This thin orchestration layer coordinates different research stages and roles,
providing a unified interface for research workflow management.

The base orchestrator defines the common interface and functionality that
all specialized orchestrators inherit from, ensuring consistency and
interoperability across the research framework.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ResearchStage(Enum):
    """Research process stages"""
    IDEATION = "ideation"
    HYPOTHESIS = "hypothesis"
    DESIGN = "design"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    DOCUMENTATION = "documentation"
    PUBLICATION = "publication"
    GRANT_WRITING = "grant_writing"


class ResearchRole(Enum):
    """Research roles and expertise levels"""
    INTERN = "intern"
    PHD_STUDENT = "phd_student"
    POSTDOC = "postdoc"
    PRINCIPAL_INVESTIGATOR = "principal_investigator"
    COLLABORATOR = "collaborator"
    REVIEWER = "reviewer"


@dataclass
class OrchestratorConfig:
    """Configuration for research orchestrators"""
    name: str
    role: ResearchRole
    stages: List[ResearchStage]
    output_dir: Path = field(default_factory=lambda: Path("./research_output"))
    enable_logging: bool = True
    enable_validation: bool = True
    enable_caching: bool = True
    max_parallel_processes: int = 4
    timeout_minutes: int = 60

    def __post_init__(self):
        """Ensure output directory exists"""
        self.output_dir.mkdir(parents=True, exist_ok=True)


class BaseOrchestrator:
    """
    Base orchestrator for research process coordination.

    This class provides the foundation for all specialized research orchestrators,
    offering common functionality for workflow management, result tracking,
    and cross-stage coordination.

    The orchestrator follows a thin architecture pattern, coordinating existing
    research components without duplicating their functionality.
    """

    def __init__(self, config: OrchestratorConfig):
        """
        Initialize the base orchestrator.

        Args:
            config: Orchestrator configuration settings
        """
        self.config = config
        self.research_state: Dict[str, Any] = {}
        self.active_processes: Dict[str, Any] = {}
        self.results_cache: Dict[str, Any] = {}

        if config.enable_logging:
            self._setup_logging()

        logger.info(f"Initialized {self.__class__.__name__} for role: {config.role.value}")

    def _setup_logging(self) -> None:
        """Setup logging for the orchestrator"""
        log_file = self.config.output_dir / f"{self.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def coordinate_stage(self, stage: ResearchStage, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate a specific research stage.

        Args:
            stage: The research stage to coordinate
            context: Context information for the stage

        Returns:
            Stage results and metadata
        """
        logger.info(f"Coordinating stage: {stage.value}")

        if not self._validate_stage(stage):
            raise ValueError(f"Invalid stage: {stage}")

        # Pre-stage validation
        if self.config.enable_validation:
            self._validate_context(stage, context)

        # Execute stage coordination
        try:
            results = self._execute_stage(stage, context)

            # Post-stage validation
            if self.config.enable_validation:
                results = self._validate_results(stage, results)

            # Cache results if enabled
            if self.config.enable_caching:
                self._cache_results(stage, context, results)

            # Update research state
            self._update_research_state(stage, results)

            logger.info(f"Stage {stage.value} completed successfully")
            return results

        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}")
            self._handle_stage_error(stage, context, e)
            raise

    def _validate_stage(self, stage: ResearchStage) -> bool:
        """Validate that the stage is supported"""
        return stage in self.config.stages

    def _validate_context(self, stage: ResearchStage, context: Dict[str, Any]) -> None:
        """Validate stage context"""
        required_fields = self._get_required_context_fields(stage)
        for field in required_fields:
            if field not in context:
                raise ValueError(f"Missing required context field: {field}")

    def _get_required_context_fields(self, stage: ResearchStage) -> List[str]:
        """Get required context fields for a stage"""
        # Base implementation - override in subclasses
        return []

    def _execute_stage(self, stage: ResearchStage, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific research stage.

        This method should be overridden by subclasses to implement
        stage-specific coordination logic.
        """
        raise NotImplementedError("Subclasses must implement _execute_stage")

    def _validate_results(self, stage: ResearchStage, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate stage results"""
        # Base validation - can be extended by subclasses
        if not isinstance(results, dict):
            raise ValueError("Results must be a dictionary")

        # Add metadata
        results['_metadata'] = {
            'stage': stage.value,
            'timestamp': datetime.now().isoformat(),
            'orchestrator': self.config.name,
            'role': self.config.role.value
        }

        return results

    def _cache_results(self, stage: ResearchStage, context: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Cache results for potential reuse"""
        cache_key = self._generate_cache_key(stage, context)
        self.results_cache[cache_key] = {
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_cache_key(self, stage: ResearchStage, context: Dict[str, Any]) -> str:
        """Generate cache key for results"""
        # Simple key generation based on stage and context hash
        context_str = json.dumps(context, sort_keys=True)
        return f"{stage.value}_{hash(context_str)}"

    def _update_research_state(self, stage: ResearchStage, results: Dict[str, Any]) -> None:
        """Update the overall research state"""
        self.research_state[stage.value] = {
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'results_summary': self._summarize_results(results)
        }

    def _summarize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of results"""
        # Base implementation - extract key metrics
        summary = {
            'result_keys': list(results.keys()),
            'metadata_available': '_metadata' in results
        }

        if '_metadata' in results:
            summary['metadata'] = results['_metadata']

        return summary

    def _handle_stage_error(self, stage: ResearchStage, context: Dict[str, Any], error: Exception) -> None:
        """Handle errors in stage execution"""
        error_info = {
            'stage': stage.value,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context_summary': self._summarize_context(context)
        }

        self.research_state[f"{stage.value}_error"] = error_info
        logger.error(f"Stage error in {stage.value}: {error}")

    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of context"""
        return {
            'context_keys': list(context.keys()),
            'context_size': len(context)
        }

    def get_research_state(self) -> Dict[str, Any]:
        """Get current research state"""
        return self.research_state.copy()

    def get_stage_results(self, stage: ResearchStage) -> Optional[Dict[str, Any]]:
        """Get results for a specific stage"""
        return self.research_state.get(stage.value, {}).get('results')

    def validate_workflow(self, stages: List[ResearchStage]) -> Dict[str, Any]:
        """
        Validate a complete research workflow.

        Args:
            stages: List of stages in workflow order

        Returns:
            Validation results and recommendations
        """
        validation = {
            'valid': True,
            'issues': [],
            'recommendations': [],
            'missing_stages': [],
            'redundant_stages': []
        }

        # Check for missing required stages
        required_stages = {ResearchStage.IDEATION, ResearchStage.EXECUTION, ResearchStage.ANALYSIS}
        provided_stages = set(stages)

        missing = required_stages - provided_stages
        if missing:
            validation['missing_stages'] = [stage.value for stage in missing]
            validation['issues'].append(f"Missing required stages: {missing}")

        # Check for stage dependencies
        stage_dependencies = self._get_stage_dependencies()
        for i, stage in enumerate(stages):
            if stage in stage_dependencies:
                required_deps = stage_dependencies[stage]
                for dep in required_deps:
                    if dep not in stages[:i]:
                        validation['issues'].append(f"Stage {stage.value} requires {dep.value} but it's not before it")

        # Generate recommendations
        validation['recommendations'] = self._generate_workflow_recommendations(stages)

        if validation['issues']:
            validation['valid'] = False

        return validation

    def _get_stage_dependencies(self) -> Dict[ResearchStage, List[ResearchStage]]:
        """Get stage dependencies"""
        return {
            ResearchStage.HYPOTHESIS: [ResearchStage.IDEATION],
            ResearchStage.DESIGN: [ResearchStage.IDEATION, ResearchStage.HYPOTHESIS],
            ResearchStage.EXECUTION: [ResearchStage.DESIGN],
            ResearchStage.ANALYSIS: [ResearchStage.EXECUTION],
            ResearchStage.VALIDATION: [ResearchStage.ANALYSIS],
            ResearchStage.DOCUMENTATION: [ResearchStage.ANALYSIS, ResearchStage.VALIDATION],
            ResearchStage.PUBLICATION: [ResearchStage.DOCUMENTATION, ResearchStage.VALIDATION]
        }

    def _generate_workflow_recommendations(self, stages: List[ResearchStage]) -> List[str]:
        """Generate workflow recommendations"""
        recommendations = []

        # Check for optimal stage ordering
        if stages and stages[0] != ResearchStage.IDEATION:
            recommendations.append("Consider starting with ideation stage")

        # Check for missing validation
        has_analysis = ResearchStage.ANALYSIS in stages
        has_validation = ResearchStage.VALIDATION in stages

        if has_analysis and not has_validation:
            recommendations.append("Consider adding validation stage after analysis")

        # Check for documentation
        has_publication = ResearchStage.PUBLICATION in stages
        has_documentation = ResearchStage.DOCUMENTATION in stages

        if has_publication and not has_documentation:
            recommendations.append("Consider adding documentation stage before publication")

        return recommendations

    def export_research_summary(self, output_path: Optional[Path] = None) -> Path:
        """
        Export comprehensive research summary.

        Args:
            output_path: Path for output file (optional)

        Returns:
            Path to exported summary
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.config.output_dir / f"research_summary_{timestamp}.json"

        summary = {
            'orchestrator_config': {
                'name': self.config.name,
                'role': self.config.role.value,
                'stages': [stage.value for stage in self.config.stages]
            },
            'research_state': self.research_state,
            'workflow_validation': self.validate_workflow([ResearchStage(s) for s in self.research_state.keys() if not s.endswith('_error')]),
            'export_timestamp': datetime.now().isoformat(),
            'active_inference_version': '0.1.0'
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Research summary exported to: {output_path}")
        return output_path

    def get_capabilities(self) -> Dict[str, Any]:
        """Get orchestrator capabilities and supported features"""
        return {
            'supported_stages': [stage.value for stage in self.config.stages],
            'research_role': self.config.role.value,
            'features': [
                'stage_coordination',
                'result_caching',
                'workflow_validation',
                'research_summary_export'
            ],
            'integrations': [
                'experiments',
                'simulations',
                'analysis',
                'benchmarks'
            ]
        }
