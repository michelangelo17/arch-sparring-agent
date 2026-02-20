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


class TestReviewOrchestratorCreate(unittest.TestCase):
    """Tests for the ``create()`` classmethod factory path."""

    def setUp(self):
        self.mock_reasoning_model = MagicMock(name="reasoning_model")
        self.mock_standard_model = MagicMock(name="standard_model")

        def fake_create_model(model_name, reasoning=False, reasoning_level="low"):
            return self.mock_reasoning_model if reasoning else self.mock_standard_model

        self.mock_create_model = patch(
            "arch_sparring_agent.orchestrator.create_model", side_effect=fake_create_model
        ).start()

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

    def tearDown(self):
        patch.stopall()

    def test_create_builds_all_agents(self):
        orch = ReviewOrchestrator.create(
            documents_dir="docs",
            templates_dir="tmpl",
            diagrams_dir="diag",
            shared_config=FAKE_CONFIG,
        )

        self.mock_create_req.assert_called()
        self.mock_create_arch.assert_called()
        self.mock_create_quest.assert_called()
        self.mock_create_spar.assert_called()
        self.mock_create_rev.assert_called()

        self.assertIsNotNone(orch.standard_model)

    def test_reasoning_level_off_uses_standard_model_for_all(self):
        ReviewOrchestrator.create(
            documents_dir="docs",
            templates_dir="tmpl",
            diagrams_dir="diag",
            shared_config=FAKE_CONFIG,
            reasoning_level="off",
        )

        calls = self.mock_create_model.call_args_list
        self.assertEqual(len(calls), 1)
        _, kwargs = calls[0]
        self.assertFalse(kwargs.get("reasoning", False))

    def test_model_name_passed_to_create_model(self):
        ReviewOrchestrator.create(
            documents_dir="docs",
            templates_dir="tmpl",
            diagrams_dir="diag",
            shared_config=FAKE_CONFIG,
            model_name="opus-4.6",
        )

        first_call = self.mock_create_model.call_args_list[0]
        self.assertEqual(first_call[0][0], "opus-4.6")


class TestReviewOrchestratorRunReview(unittest.TestCase):
    """Tests that exercise ``run_review`` via the lean DI constructor."""

    def setUp(self):
        self.mock_standard_model = MagicMock(name="standard_model")

        self.mock_req_agent = MagicMock()
        self.mock_req_agent.return_value = "Requirements Summary"

        self.mock_arch_agent = MagicMock()
        self.mock_arch_agent.return_value = "Architecture Summary"

        self.mock_quest_agent = MagicMock()
        self.mock_spar_agent = MagicMock()
        self.mock_rev_agent = MagicMock()

        self.mock_extract_req = patch(
            "arch_sparring_agent.orchestrator.extract_requirements"
        ).start()
        self.mock_extract_req.side_effect = lambda content, model_id: f"[extracted] {content}"

        self.mock_extract_arch = patch(
            "arch_sparring_agent.orchestrator.extract_architecture_findings"
        ).start()
        self.mock_extract_arch.side_effect = lambda content, model_id: f"[extracted] {content}"

        self.mock_extract_phase = patch(
            "arch_sparring_agent.orchestrator.extract_phase_findings"
        ).start()
        self.mock_extract_phase.side_effect = (
            lambda content, phase, model_id: f"[extracted:{phase}] {content}"
        )

        self.mock_run_questions = patch("arch_sparring_agent.orchestrator.run_questions").start()
        self.mock_run_sparring = patch("arch_sparring_agent.orchestrator.run_sparring").start()
        self.mock_gen_review = patch("arch_sparring_agent.orchestrator.run_review").start()

    def tearDown(self):
        patch.stopall()

    def _make_orchestrator(self, **kwargs) -> ReviewOrchestrator:
        defaults = {
            "requirements_agent": self.mock_req_agent,
            "architecture_agent": self.mock_arch_agent,
            "question_agent": self.mock_quest_agent,
            "sparring_agent": self.mock_spar_agent,
            "review_agent": self.mock_rev_agent,
            "standard_model": self.mock_standard_model,
        }
        defaults.update(kwargs)
        return ReviewOrchestrator(**defaults)

    def test_run_review(self):
        orch = self._make_orchestrator()

        self.mock_run_questions.return_value = "Questions Context"
        self.mock_run_sparring.return_value = "Sparring Context"
        self.mock_gen_review.return_value = "Final Review"

        result = orch.run_review()

        self.mock_req_agent.assert_called()
        self.mock_arch_agent.assert_called()

        self.mock_extract_req.assert_called_once_with("Requirements Summary", orch.standard_model)
        self.mock_extract_arch.assert_called_once_with("Architecture Summary", orch.standard_model)

        self.mock_run_questions.assert_called_with(
            orch.question_agent,
            "[extracted] Requirements Summary",
            "[extracted] Architecture Summary",
        )

        self.mock_run_sparring.assert_called()
        sparring_call_args = self.mock_run_sparring.call_args
        self.assertEqual(sparring_call_args[0][1], "[extracted] Architecture Summary")
        self.assertEqual(sparring_call_args[0][2], "[extracted:Q&A] Questions Context")

        self.mock_gen_review.assert_called()

        self.assertEqual(result["review"], "Final Review")
        self.assertEqual(result["requirements_summary"], "Requirements Summary")
        self.assertEqual(result["requirements_findings"], "[extracted] Requirements Summary")
        self.assertEqual(result["architecture_summary"], "Architecture Summary")
        self.assertEqual(result["architecture_findings"], "[extracted] Architecture Summary")
        self.assertEqual(result["gaps"], "Questions Context")
        self.assertEqual(result["gaps_findings"], "[extracted:Q&A] Questions Context")
        self.assertEqual(result["risks"], "Sparring Context")
        self.assertEqual(result["risks_findings"], "[extracted:Sparring] Sparring Context")

    def test_output_fn_callback(self):
        captured = []
        orch = self._make_orchestrator(output_fn=captured.append)

        self.mock_run_questions.return_value = "Q Context"
        self.mock_run_sparring.return_value = "S Context"
        self.mock_gen_review.return_value = "Final"

        orch.run_review()

        self.assertTrue(len(captured) > 0)
        self.assertEqual(captured[0], "=" * 60)


if __name__ == "__main__":
    unittest.main()
