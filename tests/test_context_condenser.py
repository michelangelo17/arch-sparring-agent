import unittest
from unittest.mock import MagicMock, patch

from arch_sparring_agent.config import (
    CONDENSER_CHUNK_SIZE as CHUNK_SIZE,
)
from arch_sparring_agent.config import (
    CONDENSER_MAX_CHUNKS as MAX_CHUNKS,
)
from arch_sparring_agent.config import (
    CONDENSER_PASSTHROUGH_THRESHOLD as PASSTHROUGH_THRESHOLD,
)
from arch_sparring_agent.context_condenser import (
    _extract,
    chunked_extract,
    extract_architecture_findings,
    extract_phase_findings,
    extract_requirements,
)
from tests.conftest import FakeClientError, FakeContextWindowOverflow, FakeMaxTokensReached


class TestPassthrough(unittest.TestCase):
    """Test that short content passes through without LLM calls."""

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_short_content_passes_through(self, mock_agent_cls):
        short = "- Requirement 1\n- Requirement 2"
        self.assertTrue(len(short) <= PASSTHROUGH_THRESHOLD)

        result = extract_requirements(short, "test-model")
        self.assertEqual(result, short)
        mock_agent_cls.assert_not_called()

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_exact_threshold_passes_through(self, mock_agent_cls):
        content = "x" * PASSTHROUGH_THRESHOLD
        result = _extract(content, "prompt", "test-model")
        self.assertEqual(result, content)
        mock_agent_cls.assert_not_called()

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_one_over_threshold_triggers_extraction(self, mock_agent_cls):
        content = "x" * (PASSTHROUGH_THRESHOLD + 1)
        mock_agent = MagicMock()
        mock_agent.return_value = "extracted output"
        mock_agent_cls.return_value = mock_agent

        result = _extract(content, "prompt", "test-model")
        self.assertEqual(result, "extracted output")
        mock_agent_cls.assert_called_once()


class TestExtraction(unittest.TestCase):
    """Test that extraction calls the agent correctly."""

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_extract_requirements_calls_agent(self, mock_agent_cls):
        content = "A" * (PASSTHROUGH_THRESHOLD + 100)
        mock_agent = MagicMock()
        mock_agent.return_value = "### Functional Requirements\n- Req 1"
        mock_agent_cls.return_value = mock_agent

        result = extract_requirements(content, "test-model")

        mock_agent_cls.assert_called_once()
        kwargs = mock_agent_cls.call_args[1]
        self.assertEqual(kwargs["name"], "FindingsExtractor")
        self.assertEqual(kwargs["model"], "test-model")
        self.assertIn("requirement", kwargs["system_prompt"].lower())
        self.assertEqual(result, "### Functional Requirements\n- Req 1")

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_extract_architecture_findings_calls_agent(self, mock_agent_cls):
        content = "B" * (PASSTHROUGH_THRESHOLD + 100)
        mock_agent = MagicMock()
        mock_agent.return_value = "### Components\n- API Gateway"
        mock_agent_cls.return_value = mock_agent

        result = extract_architecture_findings(content, "test-model")

        kwargs = mock_agent_cls.call_args[1]
        self.assertIn("Features Verified", kwargs["system_prompt"])
        self.assertIn("Features Not Found", kwargs["system_prompt"])
        self.assertEqual(result, "### Components\n- API Gateway")

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_extract_phase_findings_includes_phase_name(self, mock_agent_cls):
        content = "C" * (PASSTHROUGH_THRESHOLD + 100)
        mock_agent = MagicMock()
        mock_agent.return_value = "### Confirmed Gaps\n- Gap 1"
        mock_agent_cls.return_value = mock_agent

        result = extract_phase_findings(content, "Sparring", "test-model")

        kwargs = mock_agent_cls.call_args[1]
        self.assertIn("Sparring", kwargs["system_prompt"])
        self.assertEqual(result, "### Confirmed Gaps\n- Gap 1")


class TestChunkedFallback(unittest.TestCase):
    """Test the chunked extraction fallback path."""

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_context_overflow_triggers_chunked_fallback(self, mock_agent_cls):
        content = "D" * (PASSTHROUGH_THRESHOLD + 100)

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock = MagicMock()
            if call_count[0] == 1:
                # First call (single-pass extraction) raises overflow
                mock.side_effect = FakeContextWindowOverflow("overflow")
                return mock
            else:
                # Subsequent calls (chunked) succeed
                mock.return_value = f"chunk {call_count[0]} extracted"
                return mock

        mock_agent_cls.side_effect = side_effect

        result = _extract(content, "prompt", "test-model")
        # Should have fallen back to chunked extraction
        self.assertIn("chunk", result)

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_max_tokens_triggers_chunked_fallback(self, mock_agent_cls):
        content = "E" * (PASSTHROUGH_THRESHOLD + 100)

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock = MagicMock()
            if call_count[0] == 1:
                mock.side_effect = FakeMaxTokensReached("max tokens")
                return mock
            else:
                mock.return_value = f"chunk {call_count[0]} result"
                return mock

        mock_agent_cls.side_effect = side_effect

        result = _extract(content, "prompt", "test-model")
        self.assertIn("chunk", result)

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_client_error_validation_triggers_fallback(self, mock_agent_cls):
        content = "F" * (PASSTHROUGH_THRESHOLD + 100)

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock = MagicMock()
            if call_count[0] == 1:
                mock.side_effect = FakeClientError("ValidationException")
                return mock
            else:
                mock.return_value = f"chunk {call_count[0]} result"
                return mock

        mock_agent_cls.side_effect = side_effect

        result = _extract(content, "prompt", "test-model")
        self.assertIn("chunk", result)

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_client_error_non_token_reraises(self, mock_agent_cls):
        content = "G" * (PASSTHROUGH_THRESHOLD + 100)

        mock_agent = MagicMock()
        mock_agent.side_effect = FakeClientError("AccessDeniedException")
        mock_agent_cls.return_value = mock_agent

        with self.assertRaises(FakeClientError):
            _extract(content, "prompt", "test-model")

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_unrelated_exception_reraises(self, mock_agent_cls):
        content = "H" * (PASSTHROUGH_THRESHOLD + 100)

        mock_agent = MagicMock()
        mock_agent.side_effect = ValueError("something else")
        mock_agent_cls.return_value = mock_agent

        with self.assertRaises(ValueError):
            _extract(content, "prompt", "test-model")


class TestChunkedExtract(unittest.TestCase):
    """Test chunked_extract directly."""

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_chunks_large_content(self, mock_agent_cls):
        content = "I" * (CHUNK_SIZE * 3)
        mock_agent = MagicMock()
        mock_agent.return_value = "chunk summary"
        mock_agent_cls.return_value = mock_agent

        chunked_extract(content, "prompt", "test-model")

        # 1 extractor reused for all chunks (combined output is short → no merger)
        self.assertEqual(mock_agent_cls.call_count, 1)
        # Called once per chunk (3 chunks)
        self.assertEqual(mock_agent.call_count, 3)

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_max_chunks_limit(self, mock_agent_cls):
        content = "J" * (CHUNK_SIZE * (MAX_CHUNKS + 5))
        mock_agent = MagicMock()
        mock_agent.return_value = "chunk summary"
        mock_agent_cls.return_value = mock_agent

        chunked_extract(content, "prompt", "test-model")

        # 1 extractor + 1 merger (agent instances are reused, not created per chunk)
        self.assertLessEqual(mock_agent_cls.call_count, 2)

    @patch("arch_sparring_agent.context_condenser.Agent")
    def test_failed_chunk_produces_placeholder(self, mock_agent_cls):
        content = "K" * (CHUNK_SIZE + 100)

        call_count = [0]

        def agent_call_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise FakeClientError("ServiceUnavailable")
            return "ok"

        mock_agent = MagicMock()
        mock_agent.side_effect = agent_call_side_effect
        mock_agent_cls.return_value = mock_agent

        result = chunked_extract(content, "prompt", "test-model")
        self.assertIn("could not be processed", result)


if __name__ == "__main__":
    unittest.main()
