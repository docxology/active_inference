# Orchestrators - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Orchestrators subsystem of the Active Inference Knowledge Environment source code. It outlines implementation patterns, development workflows, and best practices for creating research orchestration and workflow management systems.

## Orchestrators Overview

The Orchestrators subsystem provides the source code implementation for research process orchestration, including workflow management, task scheduling, dependency resolution, and thin orchestration components for coordinating research processes across different roles and stages.

## Source Code Architecture

### Subsystem Responsibilities
- **Base Orchestrator**: Foundation for research process orchestration and patterns
- **Role-Specific Orchestrators**: Specialized orchestrators for different research roles
- **Workflow Management**: Complex workflow creation and execution
- **Task Scheduling**: Intelligent task scheduling and resource management
- **Integration**: Coordination between orchestration and platform services

### Integration Points
- **Knowledge Repository**: Integration with educational content for research guidance
- **Research Tools**: Connection to experiment management and analysis systems
- **Platform Services**: Integration with platform infrastructure and collaboration
- **Applications Framework**: Support for research-based application workflows
- **Visualization Systems**: Integration with visualization for research progress

## Core Implementation Responsibilities

### Base Orchestrator Implementation
**Foundation for research process orchestration**
- Implement comprehensive research stage management and coordination
- Create workflow validation and error handling systems
- Develop integration with platform services and components
- Implement research state management and progress tracking

**Key Methods to Implement:**
```python
def implement_stage_coordination_engine(self) -> StageCoordinator:
    """Implement comprehensive stage coordination with validation and monitoring"""

def create_workflow_validation_system(self) -> WorkflowValidator:
    """Create workflow validation with dependency checking and error detection"""

def implement_research_state_management(self) -> StateManager:
    """Implement comprehensive research state management and persistence"""

def create_error_handling_and_recovery(self) -> ErrorHandler:
    """Implement comprehensive error handling and recovery mechanisms"""

def implement_progress_tracking_system(self) -> ProgressTracker:
    """Implement real-time progress tracking and reporting"""

def create_integration_with_platform_services(self) -> PlatformIntegration:
    """Create integration with platform services and components"""

def implement_caching_and_optimization(self) -> CachingSystem:
    """Implement result caching and workflow optimization"""

def create_monitoring_and_logging_system(self) -> MonitoringSystem:
    """Create comprehensive monitoring and logging for orchestration"""

def implement_security_and_access_control(self) -> SecurityManager:
    """Implement security and access control for research workflows"""

def create_export_and_reporting_system(self) -> ExportManager:
    """Create export and reporting system for research results and progress"""
```

### Role-Specific Orchestrator Implementation
**Specialized research workflow management**
- Implement intern orchestrator for guided research workflows
- Create PhD orchestrator for advanced research methods and validation
- Develop grant orchestrator for proposal development and power analysis
- Implement publication orchestrator for academic publishing workflows
- Create hypothesis orchestrator for hypothesis generation and testing
- Develop ideation orchestrator for brainstorming and idea development
- Implement documentation orchestrator for research reporting and documentation

**Key Methods to Implement (for each role orchestrator):**
```python
def implement_role_specific_workflow_engine(self) -> RoleWorkflowEngine:
    """Implement role-specific workflow engine with appropriate complexity"""

def create_guided_research_interface(self) -> GuidedInterface:
    """Create guided research interface with step-by-step instructions"""

def implement_advanced_methodology_support(self) -> MethodologySupport:
    """Implement support for advanced research methodologies and validation"""

def create_proposal_development_system(self) -> ProposalSystem:
    """Create system for research proposal development and power analysis"""

def implement_publication_workflow_management(self) -> PublicationWorkflow:
    """Implement workflow management for academic publication processes"""

def create_hypothesis_testing_framework(self) -> HypothesisFramework:
    """Create comprehensive hypothesis testing and validation framework"""

def implement_ideation_and_brainstorming_tools(self) -> IdeationTools:
    """Implement tools for research ideation and brainstorming"""

def create_documentation_automation_system(self) -> DocumentationAutomation:
    """Create automation system for research documentation and reporting"""

def implement_collaboration_workflow_management(self) -> CollaborationManager:
    """Implement collaboration workflow management for multi-user research"""

def create_quality_assurance_integration(self) -> QualityIntegration:
    """Create integration with quality assurance and validation systems"""
```

## Development Workflows

### Orchestration Development Workflow
1. **Requirements Analysis**: Analyze research workflow and orchestration needs
2. **Architecture Design**: Design orchestration following thin architecture principles
3. **Implementation**: Implement with comprehensive validation and error handling
4. **Integration**: Ensure integration with platform research tools and services
5. **Testing**: Create extensive testing for workflow reliability and correctness
6. **Performance**: Optimize for research workflow efficiency and scalability
7. **Validation**: Validate orchestration against real research scenarios
8. **Documentation**: Generate comprehensive documentation and examples
9. **Review**: Submit for technical and scientific review

### Role-Specific Development
1. **Role Analysis**: Analyze specific role requirements and workflow patterns
2. **Feature Design**: Design role-specific features and guidance systems
3. **Implementation**: Implement role-specific orchestration with appropriate complexity
4. **Validation**: Validate role-specific workflows for effectiveness and usability
5. **Testing**: Create comprehensive testing for role-specific functionality
6. **Integration**: Ensure integration with base orchestration and platform

## Quality Assurance Standards

### Orchestration Quality Requirements
- **Workflow Reliability**: Orchestration must be reliable and fault-tolerant
- **Dependency Management**: Accurate dependency resolution and validation
- **Error Recovery**: Comprehensive error recovery and retry mechanisms
- **Progress Tracking**: Accurate progress tracking and state management
- **Integration**: Seamless integration with platform components
- **Performance**: Optimized for research workflow efficiency

### Research Quality Standards
- **Scientific Rigor**: Maintain scientific rigor in all research workflows
- **Reproducibility**: Ensure all research processes are reproducible
- **Validation**: Comprehensive validation of research methods and results
- **Ethics**: Follow ethical research guidelines and compliance
- **Documentation**: Complete documentation of research procedures

## Testing Implementation

### Comprehensive Orchestration Testing
```python
class TestOrchestrationSystemImplementation(unittest.TestCase):
    """Test orchestration system implementation and reliability"""

    def setUp(self):
        """Set up test environment with orchestration systems"""
        self.base_orchestrator = BaseOrchestrator(test_config)

    def test_workflow_dependency_resolution(self):
        """Test workflow dependency resolution and validation"""
        # Create complex workflow with multiple dependencies
        stages = [
            ResearchStage.IDEATION,
            ResearchStage.HYPOTHESIS,
            ResearchStage.DESIGN,
            ResearchStage.EXECUTION,
            ResearchStage.ANALYSIS,
            ResearchStage.VALIDATION,
            ResearchStage.DOCUMENTATION,
            ResearchStage.PUBLICATION
        ]

        # Test valid workflow
        validation = self.base_orchestrator.validate_workflow(stages)
        self.assertTrue(validation["valid"])
        self.assertEqual(len(validation["issues"]), 0)

        # Test recommendations
        recommendations = validation["recommendations"]
        self.assertIsInstance(recommendations, list)

        # Test workflow with missing critical stages
        incomplete_stages = [ResearchStage.IDEATION, ResearchStage.EXECUTION]  # Missing analysis and validation
        validation = self.base_orchestrator.validate_workflow(incomplete_stages)
        self.assertIn("missing_stages", validation)
        self.assertIn(ResearchStage.ANALYSIS, validation["missing_stages"])

    def test_stage_coordination_and_context(self):
        """Test stage coordination and context management"""
        # Test ideation stage
        ideation_context = {
            "research_domain": "active_inference",
            "research_goal": "Investigate Active Inference in decision making",
            "constraints": {"time_limit": 7200, "resources": "standard"},
            "user_profile": {"experience": "intermediate", "background": "cognitive_science"}
        }

        ideation_results = self.base_orchestrator.coordinate_stage(ResearchStage.IDEATION, ideation_context)

        # Validate results structure
        self.assertIn("_metadata", ideation_results)
        self.assertEqual(ideation_results["_metadata"]["stage"], "ideation")
        self.assertIn("timestamp", ideation_results["_metadata"])
        self.assertIn("orchestrator", ideation_results["_metadata"])

        # Validate stage-specific results
        self.assertIn("research_questions", ideation_results)
        self.assertIn("hypotheses", ideation_results)
        self.assertIn("methodology_suggestions", ideation_results)

        # Test context preservation
        self.assertIn("stage_results", ideation_context)
        self.assertIn("ideation", ideation_context["stage_results"])

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        # Test with invalid context
        try:
            invalid_results = self.base_orchestrator.coordinate_stage(ResearchStage.IDEATION, {})
            self.fail("Should have raised ValueError for missing context")
        except ValueError as e:
            self.assertIn("Missing required context", str(e))

        # Test with invalid stage
        try:
            invalid_stage_results = self.base_orchestrator.coordinate_stage("invalid_stage", {})
            self.fail("Should have raised ValueError for invalid stage")
        except ValueError as e:
            self.assertIn("Invalid stage", str(e))

        # Test error recovery
        error_context = {
            "research_goal": "Test error recovery",
            "error_simulation": True  # Simulate error condition
        }

        # This should handle errors gracefully and provide recovery options
        error_results = self.base_orchestrator.coordinate_stage(ResearchStage.IDEATION, error_context)

        # Should still produce metadata even with errors
        self.assertIn("_metadata", error_results)

    def test_role_specific_orchestrator_functionality(self):
        """Test role-specific orchestrator functionality"""
        # Test intern orchestrator
        intern_config = OrchestratorConfig(
            name="intern_orchestrator",
            role=ResearchRole.INTERN,
            stages=[ResearchStage.IDEATION, ResearchStage.EXECUTION, ResearchStage.ANALYSIS]
        )

        intern_orchestrator = InternResearchOrchestrator(intern_config)

        # Test guided workflow creation
        context = {
            "research_goal": "Learn Active Inference basics",
            "experience_level": "beginner",
            "learning_objectives": ["understand_concepts", "run_examples", "validate_understanding"]
        }

        workflow = intern_orchestrator.create_guided_workflow(
            context["research_goal"],
            context["experience_level"]
        )

        self.assertIsInstance(workflow, list)
        self.assertGreater(len(workflow), 0)

        # Test PhD orchestrator
        phd_config = OrchestratorConfig(
            name="phd_orchestrator",
            role=ResearchRole.PHD_STUDENT,
            stages=[ResearchStage.IDEATION, ResearchStage.HYPOTHESIS, ResearchStage.DESIGN,
                   ResearchStage.EXECUTION, ResearchStage.ANALYSIS, ResearchStage.VALIDATION]
        )

        phd_orchestrator = PhDResearchOrchestrator(phd_config)

        # Test advanced methodology validation
        methodology = {
            "approach": "computational_modeling",
            "validation": "statistical_significance",
            "reproducibility": "required",
            "ethical_review": "required"
        }

        validation = phd_orchestrator.validate_research_methodology(methodology)
        self.assertIn("valid", validation)
        self.assertIn("recommendations", validation)
```

## Performance Optimization

### Orchestration Performance
- **Workflow Execution Speed**: Optimize workflow execution and stage coordination
- **Dependency Resolution**: Efficient dependency resolution algorithms
- **State Management**: Efficient research state management and persistence
- **Error Recovery**: Fast error recovery and retry mechanisms

### Scalability
- **Concurrent Workflows**: Support for concurrent workflow execution
- **Resource Management**: Intelligent resource allocation and monitoring
- **Load Balancing**: Load balancing across available computational resources
- **Memory Management**: Efficient memory usage for complex workflows

## Research Workflow Management

### Workflow Composition System
```python
class WorkflowCompositionSystem:
    """Comprehensive workflow composition and management"""

    def __init__(self, orchestrator: BaseOrchestrator):
        self.orchestrator = orchestrator
        self.workflow_templates: Dict[str, List[ResearchStage]] = {}
        self.composition_rules: Dict[str, Any] = {}

    def compose_workflow_from_requirements(self, requirements: Dict[str, Any]) -> List[ResearchStage]:
        """Compose workflow from research requirements and constraints"""

        # Analyze requirements
        research_type = requirements.get("research_type", "exploratory")
        domain = requirements.get("domain", "general")
        complexity = requirements.get("complexity", "intermediate")
        timeline = requirements.get("timeline", "standard")
        resources = requirements.get("resources", "standard")

        # Select base workflow template
        template = self.select_workflow_template(research_type, domain, complexity)

        # Customize for timeline and resources
        customized_workflow = self.customize_workflow_for_constraints(template, timeline, resources)

        # Validate workflow
        validation = self.orchestrator.validate_workflow(customized_workflow)
        if not validation["valid"]:
            customized_workflow = self.fix_workflow_issues(customized_workflow, validation["issues"])

        # Optimize workflow
        optimized_workflow = self.optimize_workflow_execution(customized_workflow, resources)

        return optimized_workflow

    def select_workflow_template(self, research_type: str, domain: str, complexity: str) -> List[ResearchStage]:
        """Select appropriate workflow template based on requirements"""

        templates = {
            "exploratory": {
                "general": [ResearchStage.IDEATION, ResearchStage.EXECUTION, ResearchStage.ANALYSIS],
                "neuroscience": [ResearchStage.IDEATION, ResearchStage.HYPOTHESIS, ResearchStage.EXECUTION, ResearchStage.ANALYSIS],
                "psychology": [ResearchStage.IDEATION, ResearchStage.DESIGN, ResearchStage.EXECUTION, ResearchStage.ANALYSIS]
            },
            "hypothesis_testing": {
                "general": [ResearchStage.IDEATION, ResearchStage.HYPOTHESIS, ResearchStage.DESIGN,
                           ResearchStage.EXECUTION, ResearchStage.ANALYSIS, ResearchStage.VALIDATION],
                "neuroscience": [ResearchStage.IDEATION, ResearchStage.HYPOTHESIS, ResearchStage.DESIGN,
                               ResearchStage.EXECUTION, ResearchStage.ANALYSIS, ResearchStage.VALIDATION, ResearchStage.PUBLICATION],
                "psychology": [ResearchStage.IDEATION, ResearchStage.HYPOTHESIS, ResearchStage.DESIGN,
                             ResearchStage.EXECUTION, ResearchStage.ANALYSIS, ResearchStage.VALIDATION]
            },
            "method_development": {
                "general": [ResearchStage.IDEATION, ResearchStage.DESIGN, ResearchStage.EXECUTION,
                           ResearchStage.ANALYSIS, ResearchStage.VALIDATION, ResearchStage.DOCUMENTATION],
                "neuroscience": [ResearchStage.IDEATION, ResearchStage.DESIGN, ResearchStage.EXECUTION,
                               ResearchStage.ANALYSIS, ResearchStage.VALIDATION, ResearchStage.DOCUMENTATION, ResearchStage.PUBLICATION],
                "psychology": [ResearchStage.IDEATION, ResearchStage.DESIGN, ResearchStage.EXECUTION,
                             ResearchStage.ANALYSIS, ResearchStage.VALIDATION, ResearchStage.DOCUMENTATION]
            }
        }

        return templates.get(research_type, {}).get(domain, templates["exploratory"]["general"])

    def customize_workflow_for_constraints(self, workflow: List[ResearchStage], timeline: str, resources: str) -> List[ResearchStage]:
        """Customize workflow based on timeline and resource constraints"""

        customizations = {
            "timeline": {
                "short": self.shorten_workflow_stages,
                "standard": lambda w, r: w,  # No change
                "extended": self.extend_workflow_stages
            },
            "resources": {
                "limited": self.simplify_workflow_for_resources,
                "standard": lambda w, t: w,  # No change
                "extensive": self.enhance_workflow_for_resources
            }
        }

        if timeline in customizations["timeline"]:
            workflow = customizations["timeline"][timeline](workflow, resources)

        if resources in customizations["resources"]:
            workflow = customizations["resources"][resources](workflow, timeline)

        return workflow

    def validate_workflow_composition(self, workflow: List[ResearchStage]) -> Dict[str, Any]:
        """Validate workflow composition for completeness and correctness"""

        validation = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "suggestions": []
        }

        # Check for required stages
        required_stages = {ResearchStage.IDEATION, ResearchStage.EXECUTION, ResearchStage.ANALYSIS}
        workflow_stages = set(workflow)

        missing_stages = required_stages - workflow_stages
        if missing_stages:
            validation["issues"].append(f"Missing required stages: {missing_stages}")

        # Check stage order
        stage_order_issues = self.validate_stage_order(workflow)
        if stage_order_issues:
            validation["issues"].extend(stage_order_issues)

        # Check for redundant stages
        redundant_stages = self.find_redundant_stages(workflow)
        if redundant_stages:
            validation["warnings"].extend(redundant_stages)

        # Generate suggestions
        suggestions = self.generate_workflow_suggestions(workflow)
        validation["suggestions"] = suggestions

        validation["valid"] = len(validation["issues"]) == 0

        return validation

    def optimize_workflow_execution(self, workflow: List[ResearchStage], resources: str) -> List[ResearchStage]:
        """Optimize workflow for efficient execution"""

        # Identify parallelizable stages
        parallel_groups = self.identify_parallel_stages(workflow)

        # Optimize resource allocation
        resource_allocation = self.optimize_resource_allocation(workflow, resources)

        # Create execution plan
        execution_plan = self.create_execution_plan(workflow, parallel_groups, resource_allocation)

        return execution_plan
```

## Getting Started as an Agent

### Development Setup
1. **Explore Orchestration Patterns**: Review existing orchestration implementations
2. **Study Research Workflows**: Understand research methodology and validation requirements
3. **Run Orchestration Tests**: Ensure all orchestration tests pass
4. **Performance Testing**: Validate orchestration performance characteristics
5. **Documentation**: Update README and AGENTS files for new orchestration features

### Implementation Process
1. **Design Phase**: Design orchestration systems with scientific rigor and usability
2. **Implementation**: Implement following established thin architecture patterns
3. **Research Integration**: Ensure integration with research tools and methodologies
4. **Testing**: Create comprehensive tests including scientific validation
5. **Performance**: Optimize for research workflow efficiency
6. **Review**: Submit for scientific and technical review

### Quality Assurance Checklist
- [ ] Implementation follows thin orchestration architecture principles
- [ ] Research workflows maintain scientific rigor and reproducibility
- [ ] Comprehensive validation and error handling implemented
- [ ] Integration with platform research tools properly implemented
- [ ] Performance optimization for research workflows completed
- [ ] Documentation updated with comprehensive usage examples

## Related Documentation

- **[Main Tools AGENTS.md](../AGENTS.md)**: Tools module agent guidelines
- **[Orchestrators README](README.md)**: Orchestrators subsystem overview
- **[Base Orchestrator Documentation](base_orchestrator.py)**: Base orchestration implementation
- **[Specialized Orchestrators Documentation]**(): Role-specific orchestrator implementations

---

*"Active Inference for, with, by Generative AI"* - Building orchestration systems through collaborative intelligence and comprehensive research workflow management.
