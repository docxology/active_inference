"""
LLM Prompt System

Flexible modular prompt composition system for creating sophisticated prompts
with templates, builders, and dynamic content generation.

This module provides:
- Template-based prompt generation
- Dynamic content insertion
- Context-aware prompt building
- Integration with knowledge and research systems
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import logging


@dataclass
class PromptTemplate:
    """Template for generating structured prompts"""

    name: str
    description: str
    template: str
    variables: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        """Format template with provided variables"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise ValueError(f"Missing required variable: {missing_var}")

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate that all required variables are provided"""
        missing = []
        for var in self.variables:
            if var not in variables:
                missing.append(var)
        return missing

    def get_example_prompt(self, example_name: str = None) -> Optional[str]:
        """Get an example prompt using template examples"""
        if not self.examples:
            return None

        if example_name:
            example = next((ex for ex in self.examples if ex.get('name') == example_name), None)
        else:
            example = self.examples[0]

        if example:
            return self.format(**example.get('variables', {}))

        return None


class PromptVariable:
    """Represents a variable in a prompt template"""

    def __init__(
        self,
        name: str,
        var_type: str = "string",
        description: str = "",
        required: bool = True,
        default: Any = None,
        validator: Optional[Callable] = None
    ):
        self.name = name
        self.var_type = var_type
        self.description = description
        self.required = required
        self.default = default
        self.validator = validator

    def validate(self, value: Any) -> bool:
        """Validate the variable value"""
        if self.validator:
            return self.validator(value)
        return True


@dataclass
class PromptSection:
    """A section of a prompt with specific purpose and formatting"""

    name: str
    content: str
    position: int = 0
    separator: str = "\n\n"

    def format(self) -> str:
        """Format the section content"""
        return f"{self.name.upper()}:\n{self.content}"


class PromptBuilder:
    """Builder for creating complex prompts with multiple sections"""

    def __init__(self):
        self.sections: List[PromptSection] = []
        self.variables: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def add_section(self, name: str, content: str, position: int = None, separator: str = "\n\n") -> 'PromptBuilder':
        """Add a section to the prompt"""
        section = PromptSection(name, content, position or len(self.sections), separator)

        if position is not None:
            self.sections.insert(position, section)
            # Update positions for sections after insertion
            for i in range(position + 1, len(self.sections)):
                self.sections[i].position = i
        else:
            self.sections.append(section)

        return self

    def set_variable(self, name: str, value: Any) -> 'PromptBuilder':
        """Set a variable value"""
        self.variables[name] = value
        return self

    def set_variables(self, variables: Dict[str, Any]) -> 'PromptBuilder':
        """Set multiple variables"""
        self.variables.update(variables)
        return self

    def set_metadata(self, key: str, value: Any) -> 'PromptBuilder':
        """Set metadata"""
        self.metadata[key] = value
        return self

    def build(self) -> str:
        """Build the final prompt"""
        # Sort sections by position
        sorted_sections = sorted(self.sections, key=lambda s: s.position)

        # Apply variable substitution to each section
        formatted_sections = []
        for section in sorted_sections:
            content = section.content
            # Replace variables in content
            for var_name, var_value in self.variables.items():
                placeholder = f"{{{var_name}}}"
                content = content.replace(placeholder, str(var_value))
            formatted_sections.append(content)

        # Join sections with separators
        result = ""
        for i, section_content in enumerate(formatted_sections):
            if i > 0:
                result += sorted_sections[i].separator
            result += section_content

        return result

    def clear(self) -> 'PromptBuilder':
        """Clear all sections and variables"""
        self.sections.clear()
        self.variables.clear()
        self.metadata.clear()
        return self


class PromptManager:
    """Manager for prompt templates and prompt generation"""

    def __init__(self, templates_path: Optional[Path] = None):
        self.templates: Dict[str, PromptTemplate] = {}
        self.templates_path = templates_path or Path(__file__).parent / "templates"
        self.logger = logging.getLogger(f"active_inference.llm.{self.__class__.__name__}")

        # Load default templates
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load default prompt templates"""

        # Active Inference explanation template
        ai_template = PromptTemplate(
            name="active_inference_explanation",
            description="Template for explaining Active Inference concepts",
            template="""Explain the concept of {concept} in the context of Active Inference.

CONTEXT:
{context}

AUDIENCE LEVEL:
{audience_level}

KEY POINTS TO COVER:
{key_points}

Provide a {response_type} that includes:
1. Clear definition
2. Mathematical formulation (if applicable)
3. Practical examples
4. Connection to Free Energy Principle

Focus on making the explanation accessible while maintaining scientific accuracy.""",
            variables=["concept", "context", "audience_level", "key_points", "response_type"],
            examples=[
                {
                    "name": "beginner_entropy",
                    "variables": {
                        "concept": "entropy",
                        "context": "information theory foundations",
                        "audience_level": "beginner",
                        "key_points": "uncertainty, information content, probability distributions",
                        "response_type": "comprehensive explanation"
                    }
                }
            ]
        )

        # Research question template
        research_template = PromptTemplate(
            name="research_question",
            description="Template for generating research questions in Active Inference",
            template="""Generate research questions related to {topic} in Active Inference.

DOMAIN:
{domain}

CURRENT KNOWLEDGE:
{current_knowledge}

RESEARCH GAP:
{research_gap}

METHODOLOGICAL APPROACH:
{methodology}

Generate {num_questions} research questions that:
1. Address the identified gap
2. Are methodologically feasible
3. Have potential for significant impact
4. Connect to existing Active Inference literature

Format each question with:
- The research question
- Rationale
- Potential methodology
- Expected contribution""",
            variables=["topic", "domain", "current_knowledge", "research_gap", "methodology", "num_questions"],
            examples=[
                {
                    "name": "multi_agent_systems",
                    "variables": {
                        "topic": "multi-agent Active Inference",
                        "domain": "artificial intelligence",
                        "current_knowledge": "single agent models well established",
                        "research_gap": "coordination and communication between agents",
                        "methodology": "simulation and theoretical analysis",
                        "num_questions": "5"
                    }
                }
            ]
        )

        # Code implementation template
        code_template = PromptTemplate(
            name="code_implementation",
            description="Template for generating code implementations",
            template="""Implement {algorithm} in {language} for the Active Inference framework.

PROBLEM DESCRIPTION:
{problem_description}

REQUIREMENTS:
{requirements}

CONSTRAINTS:
{constraints}

EXISTING CODEBASE CONTEXT:
{codebase_context}

Provide a complete, well-documented implementation that includes:
1. Function/class definition with proper signatures
2. Comprehensive docstrings
3. Input validation and error handling
4. Example usage
5. Unit tests
6. Integration with existing Active Inference patterns

Follow these coding standards:
- Type hints for all parameters and return values
- Comprehensive error handling
- Clear, descriptive variable names
- Modular, reusable design
- Performance considerations""",
            variables=["algorithm", "language", "problem_description", "requirements", "constraints", "codebase_context"],
            examples=[
                {
                    "name": "variational_inference",
                    "variables": {
                        "algorithm": "variational inference",
                        "language": "Python",
                        "problem_description": "Implement variational inference for approximate Bayesian computation",
                        "requirements": "numpy, scipy, type hints, comprehensive testing",
                        "constraints": "numerical stability, memory efficiency",
                        "codebase_context": "Active Inference knowledge environment"
                    }
                }
            ]
        )

        # Add templates
        self.templates[ai_template.name] = ai_template
        self.templates[research_template.name] = research_template
        self.templates[code_template.name] = code_template

    def add_template(self, template: PromptTemplate) -> None:
        """Add a prompt template"""
        self.templates[template.name] = template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name"""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())

    def generate_prompt(
        self,
        template_name: str,
        variables: Dict[str, Any],
        validate: bool = True
    ) -> str:
        """Generate a prompt using a template"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        if validate:
            missing_vars = template.validate_variables(variables)
            if missing_vars:
                raise ValueError(f"Missing required variables: {missing_vars}")

        return template.format(**variables)

    def create_custom_template(
        self,
        name: str,
        description: str,
        template: str,
        variables: List[str] = None,
        examples: List[Dict[str, Any]] = None
    ) -> PromptTemplate:
        """Create a custom prompt template"""
        return PromptTemplate(
            name=name,
            description=description,
            template=template,
            variables=variables or [],
            examples=examples or []
        )

    def save_template(self, template: PromptTemplate, file_path: Optional[Path] = None) -> None:
        """Save template to file"""
        if not file_path:
            file_path = self.templates_path / f"{template.name}.json"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": template.name,
            "description": template.description,
            "template": template.template,
            "variables": template.variables,
            "examples": template.examples,
            "metadata": template.metadata
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_template(self, file_path: Path) -> PromptTemplate:
        """Load template from file"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        template = PromptTemplate(
            name=data["name"],
            description=data["description"],
            template=data["template"],
            variables=data.get("variables", []),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {})
        )

        # Add to templates dictionary
        self.templates[template.name] = template

        return template

    def get_template_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a template"""
        template = self.get_template(name)
        if not template:
            return None

        return {
            "name": template.name,
            "description": template.description,
            "variables": template.variables,
            "examples_count": len(template.examples),
            "metadata": template.metadata
        }


# Pre-configured prompt builders for common use cases
class ActiveInferencePromptBuilder(PromptBuilder):
    """Specialized prompt builder for Active Inference content"""

    def __init__(self):
        super().__init__()

    def add_concept_explanation(
        self,
        concept: str,
        context: str = "",
        audience_level: str = "intermediate",
        include_math: bool = True
    ) -> 'ActiveInferencePromptBuilder':
        """Add concept explanation section"""
        content = f"Explain {concept} in Active Inference context"
        if context:
            content += f"\n\nContext: {context}"
        if audience_level:
            content += f"\n\nAudience level: {audience_level}"
        if include_math:
            content += "\n\nInclude mathematical formulation where applicable."

        return self.add_section("concept_explanation", content)

    def add_learning_objective(
        self,
        objectives: List[str],
        difficulty: str = "intermediate"
    ) -> 'ActiveInferencePromptBuilder':
        """Add learning objectives section"""
        content = f"Learning Objectives ({difficulty.capitalize()} level):\n"
        for i, obj in enumerate(objectives, 1):
            content += f"{i}. {obj}\n"

        return self.add_section("learning_objectives", content)

    def add_research_question(
        self,
        topic: str,
        domain: str = "artificial_intelligence",
        methodology: str = "theoretical_analysis"
    ) -> 'ActiveInferencePromptBuilder':
        """Add research question section"""
        content = f"Generate research questions about {topic} in {domain} using {methodology} approaches."
        return self.add_section("research_questions", content)

    def add_code_implementation(
        self,
        algorithm: str,
        language: str = "Python",
        requirements: List[str] = None
    ) -> 'ActiveInferencePromptBuilder':
        """Add code implementation section"""
        requirements = requirements or []
        content = f"Implement {algorithm} in {language}"
        if requirements:
            content += f"\n\nRequirements: {', '.join(requirements)}"

        return self.add_section("code_implementation", content)


class EducationalPromptBuilder(PromptBuilder):
    """Specialized prompt builder for educational content"""

    def __init__(self):
        super().__init__()

    def add_learning_objective(
        self,
        objectives: List[str],
        difficulty: str = "intermediate"
    ) -> 'EducationalPromptBuilder':
        """Add learning objectives section"""
        content = f"Learning Objectives ({difficulty} level):\n" + "\n".join(f"- {obj}" for obj in objectives)
        return self.add_section("learning_objectives", content)

    def add_prerequisite_check(
        self,
        prerequisites: List[str]
    ) -> 'EducationalPromptBuilder':
        """Add prerequisite knowledge section"""
        content = "Ensure explanation builds on these prerequisite concepts:\n" + "\n".join(f"- {prereq}" for prereq in prerequisites)
        return self.add_section("prerequisites", content)

    def add_assessment_criteria(
        self,
        criteria: List[str]
    ) -> 'EducationalPromptBuilder':
        """Add assessment criteria section"""
        content = "Content should enable assessment of:\n" + "\n".join(f"- {criterion}" for criterion in criteria)
        return self.add_section("assessment", content)
