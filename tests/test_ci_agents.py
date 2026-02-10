"""Unit tests for arch_sparring_agent.agents.ci_agents."""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure strands etc. are mocked before importing ci_agents (conftest does this for pytest)
for mod in (
    "strands",
    "strands.types",
    "strands.types.exceptions",
    "botocore",
    "botocore.exceptions",
    "boto3",
):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from arch_sparring_agent.agents import ci_agents  # noqa: E402
from arch_sparring_agent.agents.ci_agents import (  # noqa: E402
    create_ci_question_agent,
    create_ci_review_agent,
    create_ci_sparring_agent,
    generate_ci_review,
    run_ci_questions,
    run_ci_sparring,
)


class TestCreateCiQuestionAgent(unittest.TestCase):
    def test_creates_agent(self):
        with patch.object(ci_agents, "Agent", MagicMock()) as mock_agent:
            create_ci_question_agent("test-model")
            mock_agent.assert_called_once()
            args, kwargs = mock_agent.call_args
            self.assertEqual(kwargs["name"], "QuestionAgent")
            self.assertEqual(kwargs["model"], "test-model")
            self.assertEqual(kwargs["tools"], [])


class TestRunCiQuestions(unittest.TestCase):
    def test_calls_agent_with_arch_findings_and_returns_string(self):
        agent = MagicMock()
        agent.return_value = "mock response"
        result = run_ci_questions(agent, "some findings")
        self.assertEqual(result, "mock response")
        agent.assert_called_once()
        call_args = agent.call_args[0][0]
        self.assertIn("ARCHITECTURE FINDINGS:", call_args)
        self.assertIn("some findings", call_args)


class TestCreateCiSparringAgent(unittest.TestCase):
    def test_creates_agent(self):
        with patch.object(ci_agents, "Agent", MagicMock()) as mock_agent:
            create_ci_sparring_agent("test-model")
            mock_agent.assert_called_once()
            args, kwargs = mock_agent.call_args
            self.assertEqual(kwargs["name"], "SparringAgent")
            self.assertEqual(kwargs["model"], "test-model")
            self.assertEqual(kwargs["tools"], [])


class TestRunCiSparring(unittest.TestCase):
    def test_calls_agent_with_qa_findings_and_returns_string(self):
        agent = MagicMock()
        agent.return_value = "mock response"
        result = run_ci_sparring(agent, "qa findings")
        self.assertEqual(result, "mock response")
        agent.assert_called_once()
        call_args = agent.call_args[0][0]
        self.assertIn("CONFIRMED GAPS:", call_args)
        self.assertIn("qa findings", call_args)


class TestCreateCiReviewAgent(unittest.TestCase):
    def test_creates_agent(self):
        with patch.object(ci_agents, "Agent", MagicMock()) as mock_agent:
            create_ci_review_agent("test-model")
            mock_agent.assert_called_once()
            args, kwargs = mock_agent.call_args
            self.assertEqual(kwargs["name"], "ReviewAgent")
            self.assertEqual(kwargs["model"], "test-model")
            self.assertEqual(kwargs["tools"], [])


class TestGenerateCiReview(unittest.TestCase):
    def test_calls_agent_and_returns_string(self):
        agent = MagicMock()
        agent.return_value = "mock review"
        result = generate_ci_review(agent, "qa findings", "sparring findings")
        self.assertEqual(result, "mock review")
        agent.assert_called_once()
        call_args = agent.call_args[0][0]
        self.assertIn("CONFIRMED GAPS:", call_args)
        self.assertIn("qa findings", call_args)
        self.assertIn("ASSESSED RISKS:", call_args)
        self.assertIn("sparring findings", call_args)
