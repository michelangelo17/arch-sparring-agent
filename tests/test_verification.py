"""Tests for the service-defaults verification step."""

import unittest
from unittest.mock import MagicMock, patch

from arch_sparring_agent.infra import SharedConfig
from arch_sparring_agent.orchestrator import ReviewOrchestrator

FAKE_CONFIG = SharedConfig(
    gateway_id="gw-123",
    gateway_arn="arn:aws:bedrock-agentcore:eu-central-1:111111111111:gateway/gw-123",
    policy_engine_id="pe-456",
    region="eu-central-1",
)


class TestVerifyAgainstDefaults(unittest.TestCase):
    def setUp(self):
        self.mock_standard_model = MagicMock(name="standard_model")

        def fake_create_model(model_name, reasoning=False, reasoning_level="low"):
            return self.mock_standard_model

        patch(
            "arch_sparring_agent.orchestrator.create_model", side_effect=fake_create_model
        ).start()
        patch("arch_sparring_agent.orchestrator.create_requirements_agent").start()
        patch("arch_sparring_agent.orchestrator.create_architecture_agent").start()
        patch("arch_sparring_agent.orchestrator.create_question_agent").start()
        patch("arch_sparring_agent.orchestrator.create_sparring_agent").start()
        patch("arch_sparring_agent.orchestrator.create_review_agent").start()

    def tearDown(self):
        patch.stopall()

    def _make_orchestrator(self):
        return ReviewOrchestrator(
            documents_dir="docs",
            templates_dir="tmpl",
            diagrams_dir="diag",
            shared_config=FAKE_CONFIG,
        )

    def test_passthrough_when_no_features_not_found(self):
        orch = self._make_orchestrator()
        findings = "### Components\n- DynamoDB\n\n### Features Verified\n- Encryption: via default"
        result = orch._verify_against_defaults(findings)
        self.assertEqual(result, findings)

    @patch("strands.Agent")
    def test_calls_verifier_when_features_not_found_present(self, mock_agent_cls):
        orch = self._make_orchestrator()
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_agent.return_value = "verified findings"

        findings = "### Features Verified\n- X\n\n### Features Not Found\n- Encryption at rest"
        result = orch._verify_against_defaults(findings)

        mock_agent_cls.assert_called_once()
        self.assertEqual(result, "verified findings")

    @patch("strands.Agent")
    def test_returns_original_on_verifier_failure(self, mock_agent_cls):
        orch = self._make_orchestrator()
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_agent.side_effect = RuntimeError("model error")

        findings = "### Features Not Found\n- Something"
        result = orch._verify_against_defaults(findings)

        self.assertEqual(result, findings)


class TestVerificationIntegrationInRunReview(unittest.TestCase):
    """Ensure _verify_against_defaults is called during run_review."""

    def setUp(self):
        self.mock_standard_model = MagicMock(name="standard_model")

        def fake_create_model(model_name, reasoning=False, reasoning_level="low"):
            return self.mock_standard_model

        patch(
            "arch_sparring_agent.orchestrator.create_model", side_effect=fake_create_model
        ).start()

        patch("arch_sparring_agent.orchestrator.extract_requirements").start().side_effect = (
            lambda c, m: c
        )

        self.mock_extract_arch = patch(
            "arch_sparring_agent.orchestrator.extract_architecture_findings"
        ).start()
        self.mock_extract_arch.side_effect = lambda c, m: f"### Features Not Found\n- item from {c}"

        patch("arch_sparring_agent.orchestrator.extract_phase_findings").start().side_effect = (
            lambda c, p, m: c
        )

        mock_req = MagicMock()
        mock_req.return_value = "req summary"
        patch(
            "arch_sparring_agent.orchestrator.create_requirements_agent"
        ).start().return_value = mock_req

        mock_arch = MagicMock()
        mock_arch.return_value = "arch summary"
        patch(
            "arch_sparring_agent.orchestrator.create_architecture_agent"
        ).start().return_value = mock_arch

        patch("arch_sparring_agent.orchestrator.create_question_agent").start()
        patch("arch_sparring_agent.orchestrator.create_sparring_agent").start()
        patch("arch_sparring_agent.orchestrator.create_review_agent").start()
        patch("arch_sparring_agent.orchestrator.run_questions").start().return_value = "q"
        patch("arch_sparring_agent.orchestrator.run_sparring").start().return_value = "s"
        patch("arch_sparring_agent.orchestrator.generate_review").start().return_value = "review"

    def tearDown(self):
        patch.stopall()

    @patch.object(ReviewOrchestrator, "_verify_against_defaults")
    def test_verify_called_during_run_review(self, mock_verify):
        mock_verify.side_effect = lambda x: x.replace("Not Found", "Verified (via service default)")

        orch = ReviewOrchestrator(
            documents_dir="docs",
            templates_dir="tmpl",
            diagrams_dir="diag",
            shared_config=FAKE_CONFIG,
        )
        orch.run_review()

        mock_verify.assert_called_once()
        call_arg = mock_verify.call_args[0][0]
        self.assertIn("Features Not Found", call_arg)


if __name__ == "__main__":
    unittest.main()
