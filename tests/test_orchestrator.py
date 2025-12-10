import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock external dependencies before importing application code
sys.modules["strands"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["boto3"] = MagicMock()
sys.modules["frontmatter"] = MagicMock()
sys.modules["bedrock_agentcore"] = MagicMock()
sys.modules["bedrock_agentcore.memory"] = MagicMock()
sys.modules["bedrock_agentcore.memory.integrations"] = MagicMock()
sys.modules["bedrock_agentcore.memory.integrations.strands"] = MagicMock()
sys.modules["bedrock_agentcore.memory.integrations.strands.config"] = MagicMock()
sys.modules["bedrock_agentcore.memory.integrations.strands.session_manager"] = MagicMock()

from arch_sparring_agent.orchestrator import ReviewOrchestrator


class TestReviewOrchestrator(unittest.TestCase):
    def setUp(self):
        self.mock_check_access = patch(
            "arch_sparring_agent.orchestrator.check_model_access"
        ).start()
        self.mock_check_access.return_value = True

        self.mock_get_arn = patch(
            "arch_sparring_agent.orchestrator.get_inference_profile_arn"
        ).start()
        self.mock_get_arn.return_value = "arn:aws:bedrock:..."

        self.mock_setup_policies = patch(
            "arch_sparring_agent.orchestrator.setup_architecture_review_policies"
        ).start()
        self.mock_setup_policies.return_value = "policy-engine-id"

        # Mock Agent creators
        self.mock_create_req = patch(
            "arch_sparring_agent.orchestrator.create_requirements_agent"
        ).start()
        self.mock_create_arch = patch(
            "arch_sparring_agent.orchestrator.create_architecture_agent"
        ).start()
        self.mock_create_quest = patch(
            "arch_sparring_agent.orchestrator.create_question_agent"
        ).start()
        self.mock_create_spar = patch(
            "arch_sparring_agent.orchestrator.create_sparring_agent"
        ).start()
        self.mock_create_rev = patch("arch_sparring_agent.orchestrator.create_review_agent").start()

        # Mock Runner functions
        self.mock_run_questions = patch("arch_sparring_agent.orchestrator.run_questions").start()
        self.mock_run_sparring = patch("arch_sparring_agent.orchestrator.run_sparring").start()
        self.mock_gen_review = patch("arch_sparring_agent.orchestrator.generate_review").start()

        # Mock Agents
        self.mock_req_agent = MagicMock()
        self.mock_create_req.return_value = self.mock_req_agent
        self.mock_req_agent.return_value = "Requirements Summary"

        self.mock_arch_agent = MagicMock()
        self.mock_create_arch.return_value = self.mock_arch_agent
        self.mock_arch_agent.return_value = "Architecture Summary"

    def tearDown(self):
        patch.stopall()

    def test_init(self):
        """Test initialization of ReviewOrchestrator."""
        orch = ReviewOrchestrator(documents_dir="docs", templates_dir="tmpl", diagrams_dir="diag")

        self.mock_check_access.assert_called()
        self.mock_setup_policies.assert_called()
        self.mock_create_req.assert_called()
        self.mock_create_arch.assert_called()
        self.mock_create_quest.assert_called()
        self.mock_create_spar.assert_called()
        self.mock_create_rev.assert_called()

        self.assertFalse(orch.ci_mode)

    def test_run_review(self):
        """Test the run_review method flow."""
        orch = ReviewOrchestrator(documents_dir="docs", templates_dir="tmpl", diagrams_dir="diag")

        # Setup return values for runners
        self.mock_run_questions.return_value = "Questions Context"
        self.mock_run_sparring.return_value = "Sparring Context"
        self.mock_gen_review.return_value = "Final Review"

        result = orch.run_review()

        # Verify Requirements Phase
        self.mock_req_agent.assert_called()

        # Verify Architecture Phase
        self.mock_arch_agent.assert_called()

        # Verify Questions Phase
        self.mock_run_questions.assert_called_with(
            orch.question_agent, "Requirements Summary", "Architecture Summary"
        )

        # Verify Sparring Phase
        self.mock_run_sparring.assert_called()

        # Verify Final Review
        self.mock_gen_review.assert_called()

        self.assertEqual(result["review"], "Final Review")
        self.assertEqual(result["requirements_summary"], "Requirements Summary")
        self.assertEqual(result["architecture_summary"], "Architecture Summary")
        self.assertEqual(result["gaps"], "Questions Context")
        self.assertEqual(result["risks"], "Sparring Context")

    def test_init_fails_if_no_access(self):
        """Test that init raises RuntimeError if model access check fails."""
        self.mock_check_access.return_value = False

        with self.assertRaises(RuntimeError):
            ReviewOrchestrator(documents_dir="docs", templates_dir="tmpl", diagrams_dir="diag")


if __name__ == "__main__":
    unittest.main()
