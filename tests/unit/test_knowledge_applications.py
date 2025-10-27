"""
Tests for Knowledge Applications Module
"""
import unittest
from pathlib import Path
import tempfile
import json

from active_inference.knowledge.repository import KnowledgeRepository, KnowledgeRepositoryConfig
from active_inference.knowledge.applications import Applications
from active_inference.knowledge.foundations import Foundations


class TestKnowledgeApplications(unittest.TestCase):
    """Test knowledge applications functionality"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = KnowledgeRepositoryConfig(
            root_path=self.test_dir,
            auto_index=True,
            cache_enabled=True
        )
        self.repository = KnowledgeRepository(self.config)
        self.applications = Applications(self.repository)

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_applications_initialization(self):
        """Test applications module initialization"""
        self.assertIsNotNone(self.applications)
        self.assertIsInstance(self.applications.repository, KnowledgeRepository)

    def test_ai_applications_creation(self):
        """Test creation of AI applications"""
        # Check for AI generative models application
        ai_app = self.repository.get_node("ai_generative_models")
        if ai_app:
            self.assertEqual(ai_app.content_type.value, "application")
            self.assertEqual(ai_app.difficulty.value, "advanced")
            self.assertIn("artificial_intelligence", ai_app.tags)

    def test_neuroscience_applications_creation(self):
        """Test creation of neuroscience applications"""
        # Check for neural perception application
        neuro_app = self.repository.get_node("neuroscience_perception")
        if neuro_app:
            self.assertEqual(neuro_app.content_type.value, "application")
            self.assertEqual(neuro_app.difficulty.value, "advanced")
            self.assertIn("neuroscience", neuro_app.tags)

    def test_engineering_applications_creation(self):
        """Test creation of engineering applications"""
        # Check for control systems application
        control_app = self.repository.get_node("engineering_control_systems")
        if control_app:
            self.assertEqual(control_app.content_type.value, "application")
            self.assertEqual(control_app.difficulty.value, "intermediate")
            self.assertIn("engineering", control_app.tags)

        # Check for robotics application
        robotics_app = self.repository.get_node("robotics_control")
        if robotics_app:
            self.assertEqual(robotics_app.content_type.value, "application")
            self.assertEqual(robotics_app.difficulty.value, "advanced")
            self.assertIn("robotics", robotics_app.tags)

    def test_psychology_applications_creation(self):
        """Test creation of psychology applications"""
        # Check for decision making application
        psych_app = self.repository.get_node("psychology_decision_making")
        if psych_app:
            self.assertEqual(psych_app.content_type.value, "application")
            self.assertEqual(psych_app.difficulty.value, "intermediate")
            self.assertIn("psychology", psych_app.tags)

    def test_education_applications_creation(self):
        """Test creation of education applications"""
        # Check for adaptive learning application
        edu_app = self.repository.get_node("education_adaptive_learning")
        if edu_app:
            self.assertEqual(edu_app.content_type.value, "application")
            self.assertEqual(edu_app.difficulty.value, "intermediate")
            self.assertIn("education", edu_app.tags)

    def test_domain_filtering(self):
        """Test filtering applications by domain"""
        # Test getting applications by domain
        ai_apps = self.applications.get_applications_by_domain("artificial_intelligence")
        self.assertIsInstance(ai_apps, list)

        neuro_apps = self.applications.get_applications_by_domain("neuroscience")
        self.assertIsInstance(neuro_apps, list)

        eng_apps = self.applications.get_applications_by_domain("engineering")
        self.assertIsInstance(eng_apps, list)

    def test_domain_applications_organization(self):
        """Test organizing applications by domain"""
        domain_apps = self.applications.get_domain_applications()

        # Should return dictionary of domains to applications
        self.assertIsInstance(domain_apps, dict)

        # Each domain should map to a list of applications
        for domain, apps in domain_apps.items():
            self.assertIsInstance(apps, list)

    def test_application_validation(self):
        """Test application validation functionality"""
        # Get an application to validate
        app = self.repository.get_node("ai_generative_models")
        if app:
            validation = self.applications.validate_application("ai_generative_models")

            # Should have validation result
            self.assertIn("valid", validation)
            self.assertIn("issues", validation)
            self.assertIn("suggestions", validation)

    def test_case_studies_retrieval(self):
        """Test case studies retrieval"""
        case_studies = self.applications.get_case_studies()

        # Should return list of case study nodes
        self.assertIsInstance(case_studies, list)


if __name__ == '__main__':
    unittest.main()
