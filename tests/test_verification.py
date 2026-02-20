"""Tests for the service-defaults verification step."""

import unittest
from unittest.mock import MagicMock, patch

from arch_sparring_agent.orchestrator import ReviewOrchestrator


class TestVerifyAgainstDefaults(unittest.TestCase):
    def _make_orchestrator(self):
        return ReviewOrchestrator(
            requirements_agent=MagicMock(),
            architecture_agent=MagicMock(),
            question_agent=MagicMock(),
            sparring_agent=MagicMock(),
            review_agent=MagicMock(),
            standard_model=MagicMock(name="standard_model"),
        )

    def test_passthrough_when_no_features_not_found(self):
        orch = self._make_orchestrator()
        findings = "### Components\n- DynamoDB\n\n### Features Verified\n- Encryption: via default"
        result = orch._verify_against_defaults(findings)
        self.assertEqual(result, findings)

    @patch("arch_sparring_agent.orchestrator.Agent")
    def test_calls_verifier_when_features_not_found_present(self, mock_agent_cls):
        orch = self._make_orchestrator()
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_agent.return_value = "verified findings"

        findings = "### Features Verified\n- X\n\n### Features Not Found\n- Encryption at rest"
        result = orch._verify_against_defaults(findings)

        mock_agent_cls.assert_called_once()
        self.assertEqual(result, "verified findings")

    @patch("arch_sparring_agent.orchestrator.Agent")
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

        patch("arch_sparring_agent.orchestrator.run_questions").start().return_value = "q"
        patch("arch_sparring_agent.orchestrator.run_sparring").start().return_value = "s"
        patch("arch_sparring_agent.orchestrator.run_review").start().return_value = "review"

    def tearDown(self):
        patch.stopall()

    @patch.object(ReviewOrchestrator, "_verify_against_defaults")
    def test_verify_called_during_run_review(self, mock_verify):
        mock_verify.side_effect = lambda x: x.replace("Not Found", "Verified (via service default)")

        mock_req = MagicMock()
        mock_req.return_value = "req summary"
        mock_arch = MagicMock()
        mock_arch.return_value = "arch summary"

        orch = ReviewOrchestrator(
            requirements_agent=mock_req,
            architecture_agent=mock_arch,
            question_agent=MagicMock(),
            sparring_agent=MagicMock(),
            review_agent=MagicMock(),
            standard_model=MagicMock(name="standard_model"),
        )
        orch.run_review()

        mock_verify.assert_called_once()
        call_arg = mock_verify.call_args[0][0]
        self.assertIn("Features Not Found", call_arg)


if __name__ == "__main__":
    unittest.main()
