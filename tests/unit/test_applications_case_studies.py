"""
Tests for Applications - Case Studies Module

Unit tests for case studies and example applications functionality,
ensuring proper case study management, code generation, and example
application execution.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from active_inference.applications.case_studies import (
    ApplicationDomain,
    CaseStudy,
    ExampleApplications,
    CaseStudyManager
)


class TestApplicationDomain:
    """Test cases for ApplicationDomain enum"""

    def test_application_domain_values(self):
        """Test application domain enum values"""
        assert ApplicationDomain.ROBOTICS.value == "robotics"
        assert ApplicationDomain.DECISION_MAKING.value == "decision_making"
        assert ApplicationDomain.PERCEPTION.value == "perception"
        assert ApplicationDomain.MOTOR_CONTROL.value == "motor_control"
        assert ApplicationDomain.SOCIAL_COGNITION.value == "social_cognition"
        assert ApplicationDomain.CLINICAL.value == "clinical"
        assert ApplicationDomain.EDUCATION.value == "education"

    def test_application_domain_count(self):
        """Test correct number of application domains"""
        domains = list(ApplicationDomain)
        assert len(domains) == 7


class TestCaseStudy:
    """Test cases for CaseStudy dataclass"""

    def test_case_study_creation(self):
        """Test creating case study instances"""
        study = CaseStudy(
            id="test_study",
            title="Test Case Study",
            domain=ApplicationDomain.ROBOTICS,
            description="A test case study for robotics",
            difficulty="intermediate",
            implementation_files=["robot.py", "controller.py"],
            requirements=["numpy", "scipy"],
            learning_objectives=["Learn robotics", "Understand control"],
            key_concepts=["robotics", "control", "active_inference"]
        )

        assert study.id == "test_study"
        assert study.title == "Test Case Study"
        assert study.domain == ApplicationDomain.ROBOTICS
        assert study.description == "A test case study for robotics"
        assert study.difficulty == "intermediate"
        assert study.implementation_files == ["robot.py", "controller.py"]
        assert study.requirements == ["numpy", "scipy"]
        assert study.learning_objectives == ["Learn robotics", "Understand control"]
        assert study.key_concepts == ["robotics", "control", "active_inference"]


class TestExampleApplications:
    """Test cases for ExampleApplications class"""

    def test_initialization(self):
        """Test example applications initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            assert examples.examples_dir == Path(temp_dir)
            assert isinstance(examples.case_studies, dict)
            assert len(examples.case_studies) > 0

            # Check that expected case studies are loaded
            expected_studies = ["perceptual_inference", "decision_making", "motor_control"]
            for study_id in expected_studies:
                assert study_id in examples.case_studies

    def test_get_case_study_existing(self):
        """Test getting existing case study"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            # Test getting existing study
            study = examples.get_case_study("perceptual_inference")
            assert study is not None
            assert isinstance(study, CaseStudy)
            assert study.id == "perceptual_inference"
            assert study.domain == ApplicationDomain.PERCEPTION
            assert study.difficulty == "intermediate"

    def test_get_case_study_nonexistent(self):
        """Test getting non-existent case study"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            study = examples.get_case_study("nonexistent_study")
            assert study is None

    def test_list_case_studies_all(self):
        """Test listing all case studies"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            studies = examples.list_case_studies()

            assert isinstance(studies, list)
            assert len(studies) == len(examples.case_studies)

            # Check that all studies are CaseStudy instances
            for study in studies:
                assert isinstance(study, CaseStudy)

            # Check that studies are sorted by title
            titles = [study.title for study in studies]
            assert titles == sorted(titles)

    def test_list_case_studies_by_domain(self):
        """Test listing case studies by domain"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            # Test filtering by domain
            perception_studies = examples.list_case_studies(domain=ApplicationDomain.PERCEPTION)
            assert isinstance(perception_studies, list)
            assert len(perception_studies) == 1
            assert perception_studies[0].domain == ApplicationDomain.PERCEPTION

            # Test filtering by non-existent domain
            clinical_studies = examples.list_case_studies(domain=ApplicationDomain.CLINICAL)
            assert isinstance(clinical_studies, list)
            assert len(clinical_studies) == 0

    def test_list_case_studies_by_difficulty(self):
        """Test listing case studies by difficulty"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            # Test filtering by difficulty
            intermediate_studies = examples.list_case_studies(difficulty="intermediate")
            assert isinstance(intermediate_studies, list)
            assert len(intermediate_studies) == 2  # perceptual_inference and motor_control

            for study in intermediate_studies:
                assert study.difficulty == "intermediate"

            # Test filtering by advanced difficulty
            advanced_studies = examples.list_case_studies(difficulty="advanced")
            assert isinstance(advanced_studies, list)
            assert len(advanced_studies) == 1  # decision_making
            assert advanced_studies[0].difficulty == "advanced"

    def test_list_case_studies_combined_filters(self):
        """Test listing case studies with combined filters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            # Test combined domain and difficulty filters
            studies = examples.list_case_studies(
                domain=ApplicationDomain.DECISION_MAKING,
                difficulty="advanced"
            )

            assert isinstance(studies, list)
            assert len(studies) == 1
            assert studies[0].domain == ApplicationDomain.DECISION_MAKING
            assert studies[0].difficulty == "advanced"

    def test_generate_example_code_perceptual_inference(self):
        """Test generating perceptual inference example code"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            code = examples.generate_example_code("perceptual_inference")

            assert isinstance(code, str)
            assert len(code) > 0

            # Check that code contains expected elements
            assert "PerceptualInferenceModel" in code
            assert "class PerceptualInferenceModel" in code
            assert "def infer" in code
            assert "def update_model" in code
            assert "import numpy as np" in code
            assert "Predicted category:" in code  # From example usage

    def test_generate_example_code_decision_making(self):
        """Test generating decision making example code"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            code = examples.generate_example_code("decision_making")

            assert isinstance(code, str)
            assert len(code) > 0

            # Check that code contains expected elements
            assert "DecisionMakingModel" in code
            assert "class DecisionMakingModel" in code
            assert "def compute_expected_free_energy" in code
            assert "def select_action" in code
            assert "def simulate_decision_process" in code
            assert "import numpy as np" in code
            assert "Average expected FE:" in code  # From example usage

    def test_generate_example_code_motor_control(self):
        """Test generating motor control example code"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            code = examples.generate_example_code("motor_control")

            assert isinstance(code, str)
            assert len(code) > 0

            # Check that code contains expected elements
            assert "MotorControlModel" in code
            assert "class MotorControlModel" in code
            assert "def forward_model" in code
            assert "def inverse_model" in code
            assert "def control_step" in code
            assert "def simulate_reaching_task" in code
            assert "import numpy as np" in code
            assert "Final error:" in code  # From example usage

    def test_generate_example_code_nonexistent(self):
        """Test generating code for non-existent case study"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            code = examples.generate_example_code("nonexistent_study")

            assert isinstance(code, str)
            assert "not found" in code
            assert "Implementation pending" in code

    def test_case_study_content_validation(self):
        """Test that case studies contain valid content"""
        with tempfile.TemporaryDirectory() as temp_dir:
            examples = ExampleApplications(temp_dir)

            for study in examples.case_studies.values():
                assert isinstance(study, CaseStudy)
                assert len(study.id) > 0
                assert len(study.title) > 0
                assert len(study.description) > 0
                assert study.difficulty in ["beginner", "intermediate", "advanced"]
                assert isinstance(study.implementation_files, list)
                assert len(study.implementation_files) > 0
                assert isinstance(study.requirements, list)
                assert len(study.requirements) > 0
                assert isinstance(study.learning_objectives, list)
                assert len(study.learning_objectives) > 0
                assert isinstance(study.key_concepts, list)
                assert len(study.key_concepts) > 0


class TestCaseStudyManager:
    """Test cases for CaseStudyManager class"""

    def test_initialization(self):
        """Test case study manager initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CaseStudyManager(temp_dir)

            assert manager.examples_dir == Path(temp_dir)
            assert hasattr(manager, 'example_applications')
            assert isinstance(manager.example_applications, ExampleApplications)

    def test_get_case_study(self):
        """Test getting case study through manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CaseStudyManager(temp_dir)

            # Test getting existing study
            study = manager.get_case_study("perceptual_inference")
            assert study is not None
            assert isinstance(study, CaseStudy)
            assert study.id == "perceptual_inference"

            # Test getting non-existent study
            nonexistent_study = manager.get_case_study("nonexistent")
            assert nonexistent_study is None

    def test_list_case_studies(self):
        """Test listing case studies through manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CaseStudyManager(temp_dir)

            # Test listing all studies
            studies = manager.list_case_studies()
            assert isinstance(studies, list)
            assert len(studies) > 0

            # Test listing with filters
            perception_studies = manager.list_case_studies(domain=ApplicationDomain.PERCEPTION)
            assert isinstance(perception_studies, list)
            assert len(perception_studies) == 1

    def test_run_example(self):
        """Test running example case study"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CaseStudyManager(temp_dir)

            # Test running existing example
            result = manager.run_example("perceptual_inference")

            assert isinstance(result, dict)
            assert result["study_id"] == "perceptual_inference"
            assert "title" in result
            assert "code" in result
            assert "requirements" in result
            assert "learning_objectives" in result
            assert len(result["code"]) > 0

            # Test running non-existent example
            nonexistent_result = manager.run_example("nonexistent")
            assert isinstance(nonexistent_result, dict)
            assert "error" in nonexistent_result
            assert "not found" in nonexistent_result["error"]

    def test_example_applications_integration(self):
        """Test integration between manager and example applications"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CaseStudyManager(temp_dir)

            # Verify that manager delegates to example applications correctly
            assert manager.example_applications.examples_dir == Path(temp_dir) / "examples"

            # Test that manager methods work through delegation
            study = manager.get_case_study("decision_making")
            assert study is not None
            assert study.domain == ApplicationDomain.DECISION_MAKING

            studies = manager.list_case_studies(difficulty="intermediate")
            assert len(studies) >= 1

    def test_manager_configuration_handling(self):
        """Test manager configuration handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with custom directory
            manager = CaseStudyManager(temp_dir)

            assert manager.examples_dir == Path(temp_dir)
            assert manager.example_applications.examples_dir == Path(temp_dir) / "examples"


if __name__ == "__main__":
    pytest.main([__file__])
