Contributing to Active Inference Knowledge Environment
========================================================

Thank you for your interest in contributing to the Active Inference Knowledge Environment! This document provides comprehensive guidelines for contributing to the project, whether you're adding educational content, improving the platform, fixing bugs, or enhancing documentation.

.. contents::
   :local:
   :depth: 2

Getting Started
---------------

Before contributing, please familiarize yourself with the project structure and guidelines:

1. **Read the Documentation**: Start with :doc:`README` and :doc:`AGENTS`
2. **Understand the Architecture**: Review the platform components in :doc:`platform/index`
3. **Set Up Development Environment**: Follow the setup instructions
4. **Join the Community**: Connect with other contributors

Types of Contributions
----------------------

Content Contributions
~~~~~~~~~~~~~~~~~~~~~

**Educational Content**
   Add or improve knowledge nodes, learning paths, or educational materials.

**Research Integration**
   Contribute research papers, implementations, or analysis methods.

**Documentation**
   Improve user guides, API documentation, or educational materials.

Code Contributions
~~~~~~~~~~~~~~~~~~

**Platform Development**
   Enhance platform services, APIs, or infrastructure.

**Tools & Utilities**
   Create development tools, testing frameworks, or automation scripts.

**Bug Fixes**
   Identify and fix issues in the codebase.

**Performance Improvements**
   Optimize algorithms, reduce resource usage, or improve scalability.

**Security Enhancements**
   Improve security measures and vulnerability prevention.

Contribution Workflow
---------------------

1. **Choose an Issue**
   - Check the `issues <https://github.com/docxology/active_inference/issues>`_ for open tasks
   - Look for issues labeled ``good first issue`` or ``help wanted``

2. **Fork and Clone**
   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/active_inference.git
      cd active_inference
      git checkout -b feature/your-feature-name

3. **Set Up Development Environment**
   .. code-block:: bash

      make setup
      make test  # Ensure everything works

4. **Make Changes**
   - Follow the coding standards in :doc:`AGENTS`
   - Write tests for new functionality
   - Update documentation as needed

5. **Test Your Changes**
   .. code-block:: bash

      make test
      make docs
      make check-all

6. **Submit a Pull Request**
   - Push your changes to your fork
   - Create a pull request with a clear description
   - Reference any related issues

Content Contribution Guidelines
-------------------------------

Knowledge Nodes
~~~~~~~~~~~~~~~

Knowledge nodes must follow the established JSON schema:

.. code-block:: json

   {
     "id": "unique_identifier",
     "title": "Human-readable title",
     "content_type": "foundation|mathematics|implementation|application",
     "difficulty": "beginner|intermediate|advanced|expert",
     "description": "Clear, concise description",
     "prerequisites": ["prerequisite_node_ids"],
     "tags": ["relevant", "tags", "for", "search"],
     "learning_objectives": ["measurable", "outcomes"],
     "content": {
       "overview": "High-level summary",
       "mathematical_definition": "Formal treatment (if applicable)",
       "examples": "Practical examples",
       "interactive_exercises": "Hands-on activities"
     },
     "metadata": {
       "estimated_reading_time": 15,
       "author": "Content creator",
       "last_updated": "2024-10-27",
       "version": "1.0"
     }
   }

**Quality Requirements:**
- **Accuracy**: Mathematically and conceptually correct
- **Clarity**: Accessible language with progressive disclosure
- **Completeness**: Cover important aspects without overwhelming
- **Examples**: Include practical, working examples
- **References**: Support claims with appropriate references

Learning Paths
~~~~~~~~~~~~~~

Learning paths organize knowledge nodes into structured curricula:

.. code-block:: json

   {
     "id": "learning_path_id",
     "title": "Learning Path Title",
     "description": "Clear description of the learning journey",
     "difficulty": "beginner_to_advanced",
     "estimated_hours": 40,
     "tracks": [
       {
         "id": "track_id",
         "title": "Track Title",
         "nodes": ["node_id_1", "node_id_2"],
         "estimated_hours": 8
       }
     ]
   }

**Design Principles:**
- **Progressive Complexity**: Start simple, build to advanced concepts
- **Prerequisite Validation**: Ensure proper knowledge sequencing
- **Assessment Integration**: Include formative and summative assessments
- **Multiple Pathways**: Support different learning styles and goals

Code Contribution Guidelines
----------------------------

Development Standards
~~~~~~~~~~~~~~~~~~~~~

All code contributions must follow established patterns:

**Python Code Standards**
   - **Type Hints**: Complete type annotations for all interfaces
   - **Documentation**: Comprehensive docstrings following Google style
   - **Testing**: Test-driven development with >95% coverage
   - **Style**: Black formatting and isort import organization

**Testing Requirements**
   - **Unit Tests**: Test individual functions and methods
   - **Integration Tests**: Test component interactions
   - **Performance Tests**: Validate performance characteristics
   - **Documentation Tests**: Test code examples in documentation

**Example Test Structure:**
   .. code-block:: python

      import pytest
      from active_inference.knowledge import KnowledgeRepository

      class TestKnowledgeRepository:
          """Test cases for KnowledgeRepository"""

          @pytest.fixture
          def repo(self):
              """Create test repository"""
              return KnowledgeRepository(test_mode=True)

          def test_initialization(self, repo):
              """Test repository initialization"""
              assert repo.config is not None
              assert repo.initialized is True

          def test_node_retrieval(self, repo):
              """Test knowledge node retrieval"""
              node = repo.get_node("test_node_id")
              assert node is not None
              assert node.id == "test_node_id"

Platform Development
~~~~~~~~~~~~~~~~~~~~

**Service Architecture**
   Follow the established service pattern for platform components:

   .. code-block:: python

      from abc import ABC, abstractmethod
      from typing import Dict, Any

      class BasePlatformService(ABC):
          """Base pattern for platform services"""

          def __init__(self, config: Dict[str, Any]):
              self.config = config
              self.initialize_service()

          @abstractmethod
          def initialize_service(self) -> None:
              """Initialize service-specific components"""
              pass

          @abstractmethod
          def create_endpoints(self) -> None:
              """Create service API endpoints"""
              pass

**API Design**
   - **RESTful Design**: Follow REST principles for HTTP APIs
   - **Versioning**: Include API versioning in URLs
   - **Error Handling**: Consistent error response formats
   - **Documentation**: Complete API documentation with examples

Documentation Guidelines
------------------------

Documentation Structure
~~~~~~~~~~~~~~~~~~~~~~~

**README.md Files**
   Every component must have a comprehensive README.md:

   .. code-block:: markdown

      # Component Name

      Brief description of what this component does.

      ## Overview

      Detailed explanation of purpose, scope, and functionality.

      ## Architecture

      How the component fits into the larger system.

      ## Usage

      ### Basic Usage

      ```python
      # Code examples
      ```

      ### Advanced Usage

      ```python
      # Advanced examples
      ```

      ## Configuration

      Required and optional configuration parameters.

      ## API Reference

      Complete API documentation.

      ## Testing

      How to test this component.

      ## Contributing

      How to contribute to this component.

**AGENTS.md Files**
   Agent development guidelines for each component:

   - **Role and Responsibilities**: What agents should do
   - **Architecture & Integration**: How component fits into system
   - **Development Patterns**: Required implementation patterns
   - **Testing Standards**: Component testing requirements
   - **Quality Standards**: Code and documentation quality gates

Quality Assurance
-----------------

Testing Requirements
~~~~~~~~~~~~~~~~~~~~

**Automated Testing**
   - **CI/CD Pipeline**: All changes must pass automated tests
   - **Code Coverage**: Maintain >95% coverage for core components
   - **Integration Testing**: Test component interactions
   - **Performance Testing**: Validate performance requirements

**Manual Testing**
   - **User Acceptance**: Test with intended user groups
   - **Cross-Platform**: Test on supported platforms
   - **Accessibility**: Test accessibility features
   - **Documentation**: Test documentation accuracy

Review Process
~~~~~~~~~~~~~~

**Code Review Checklist**
   - [ ] **Functionality**: Code implements specified requirements
   - [ ] **Testing**: Comprehensive test coverage
   - [ ] **Documentation**: Complete documentation updates
   - [ ] **Style**: Follows coding standards
   - [ ] **Performance**: No performance regressions
   - [ ] **Security**: Secure coding practices

**Content Review Checklist**
   - [ ] **Accuracy**: Technical and conceptual accuracy
   - [ ] **Clarity**: Clear, accessible presentation
   - [ ] **Completeness**: Comprehensive coverage
   - [ ] **Examples**: Working examples and demonstrations
   - [ ] **References**: Appropriate supporting references

**Documentation Review Checklist**
   - [ ] **Completeness**: All features documented
   - [ ] **Accuracy**: Documentation matches implementation
   - [ ] **Clarity**: Clear explanations and examples
   - [ ] **Navigation**: Easy to navigate and search
   - [ ] **Consistency**: Consistent style and terminology

Community Guidelines
--------------------

Communication
~~~~~~~~~~~~~

**Respectful Interaction**
   - Be respectful and constructive in all interactions
   - Focus on technical merit and improvement
   - Provide helpful, actionable feedback
   - Acknowledge contributions from all participants

**Issue Reporting**
   - Use clear, descriptive issue titles
   - Provide steps to reproduce bugs
   - Include relevant system information
   - Suggest potential solutions when possible

**Pull Request Guidelines**
   - Write clear, descriptive PR titles
   - Provide detailed descriptions of changes
   - Reference related issues
   - Keep PRs focused on single concerns
   - Respond promptly to review feedback

Code of Conduct
~~~~~~~~~~~~~~~

See :doc:`code_of_conduct` for detailed community standards.

Recognition
~~~~~~~~~~~

**Contributor Recognition**
   - All contributors listed in project acknowledgments
   - Significant contributions highlighted in release notes
   - Community recognition for outstanding contributions

**Credit Attribution**
   - Maintain proper attribution for all contributions
   - Respect intellectual property rights
   - Follow licensing requirements

Getting Help
------------

**Support Resources**
   - **Documentation**: Comprehensive guides and references
   - **Community**: Discussion forums and community support
   - **Issues**: Bug reports and feature requests
   - **Discussions**: General discussion and Q&A

**Contact Information**
   - **GitHub Issues**: Technical issues and bug reports
   - **GitHub Discussions**: General discussion and support
   - **Documentation**: Self-service support through guides

License
-------

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.

See :doc:`license` for complete license text.

.. seealso::

   :doc:`README`
      Project overview and getting started

   :doc:`AGENTS`
      Agent development guidelines

   :doc:`platform/index`
      Platform architecture and development

   :doc:`code_of_conduct`
      Community standards and conduct guidelines
