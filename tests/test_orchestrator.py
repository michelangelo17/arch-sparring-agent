import unittest
from unittest.mock import MagicMock, patch

from arch_sparring_agent.exceptions import ConfigurationError
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

        # Mock create_model to return distinguishable fake models
        self.mock_reasoning_model = MagicMock(name="reasoning_model")
        self.mock_standard_model = MagicMock(name="standard_model")

        def fake_create_model(model_id, reasoning=False, reasoning_level="low"):
            return self.mock_reasoning_model if reasoning else self.mock_standard_model

        self.mock_create_model = patch(
            "arch_sparring_agent.orchestrator.create_model", side_effect=fake_create_model
        ).start()

        # Mock context condenser functions (passthrough for short content)
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
        """Test the run_review method flow with context condensation."""
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

        # Verify context condenser was called with the standard model
        self.mock_extract_req.assert_called_once_with("Requirements Summary", orch.standard_model)
        self.mock_extract_arch.assert_called_once_with("Architecture Summary", orch.standard_model)

        # Verify Questions Phase receives extracted findings
        self.mock_run_questions.assert_called_with(
            orch.question_agent,
            "[extracted] Requirements Summary",
            "[extracted] Architecture Summary",
        )

        # Verify Sparring Phase receives extracted findings
        self.mock_run_sparring.assert_called()
        sparring_call_args = self.mock_run_sparring.call_args
        self.assertEqual(sparring_call_args[0][1], "[extracted] Requirements Summary")
        self.assertEqual(sparring_call_args[0][2], "[extracted] Architecture Summary")
        self.assertEqual(sparring_call_args[0][3], "[extracted:Q&A] Questions Context")

        # Verify Final Review receives extracted findings
        self.mock_gen_review.assert_called()

        # Verify result contains both raw and extracted versions
        self.assertEqual(result["review"], "Final Review")
        self.assertEqual(result["requirements_summary"], "Requirements Summary")
        self.assertEqual(result["requirements_findings"], "[extracted] Requirements Summary")
        self.assertEqual(result["architecture_summary"], "Architecture Summary")
        self.assertEqual(result["architecture_findings"], "[extracted] Architecture Summary")
        self.assertEqual(result["gaps"], "Questions Context")
        self.assertEqual(result["gaps_findings"], "[extracted:Q&A] Questions Context")
        self.assertEqual(result["risks"], "Sparring Context")
        self.assertEqual(result["risks_findings"], "[extracted:Sparring] Sparring Context")

    def test_init_fails_if_no_access(self):
        """Test that init raises ConfigurationError if model access check fails."""
        self.mock_check_access.return_value = False

        with self.assertRaises(ConfigurationError):
            ReviewOrchestrator(documents_dir="docs", templates_dir="tmpl", diagrams_dir="diag")

    def test_init_fails_if_policy_setup_fails(self):
        """Test that init raises ConfigurationError if policy engine setup fails."""
        self.mock_setup_policies.return_value = None

        with self.assertRaises(ConfigurationError) as ctx:
            ReviewOrchestrator(documents_dir="docs", templates_dir="tmpl", diagrams_dir="diag")

        self.assertIn("Policy Engine setup failed", str(ctx.exception))
        self.assertIn("--skip-policy-check", str(ctx.exception))

    def test_init_allows_skip_policy_check(self):
        """Test that skip_policy_check=True allows init to succeed without policies."""
        self.mock_setup_policies.return_value = None

        orch = ReviewOrchestrator(
            documents_dir="docs",
            templates_dir="tmpl",
            diagrams_dir="diag",
            skip_policy_check=True,
        )

        self.assertIsNone(orch.policy_engine_id)

    def test_output_fn_callback(self):
        """Test that output_fn callback is called for captured output."""
        captured = []
        orch = ReviewOrchestrator(
            documents_dir="docs",
            templates_dir="tmpl",
            diagrams_dir="diag",
            output_fn=captured.append,
        )

        self.mock_run_questions.return_value = "Q Context"
        self.mock_run_sparring.return_value = "S Context"
        self.mock_gen_review.return_value = "Final"

        orch.run_review()

        # output_fn should have been called for each _capture()
        self.assertTrue(len(captured) > 0)
        # First capture should be the divider
        self.assertEqual(captured[0], "=" * 60)

    def test_reasoning_level_off_uses_standard_model_for_all(self):
        """When reasoning_level='off', all agents should use the standard model."""
        ReviewOrchestrator(
            documents_dir="docs",
            templates_dir="tmpl",
            diagrams_dir="diag",
            reasoning_level="off",
        )

        # create_model should have been called with reasoning=False only (no reasoning=True call)
        calls = self.mock_create_model.call_args_list
        # Should only have one call: standard_model (reasoning=False)
        self.assertEqual(len(calls), 1)
        _, kwargs = calls[0]
        self.assertFalse(kwargs.get("reasoning", False))

        # All agents that normally use reasoning_model should use standard_model instead
        # Architecture agent should receive the standard model
        arch_call = self.mock_create_arch.call_args
        self.assertEqual(arch_call[1].get("model") or arch_call[0][2], self.mock_standard_model)

        # Review agent should receive the standard model
        rev_call = self.mock_create_rev.call_args
        self.assertEqual(rev_call[0][0], self.mock_standard_model)


if __name__ == "__main__":
    unittest.main()
