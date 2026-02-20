from unittest.mock import MagicMock, patch

import pytest

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


@patch("arch_sparring_agent.context_condenser.Agent")
def test_short_content_passes_through(mock_agent_cls):
    short = "- Requirement 1\n- Requirement 2"
    assert len(short) <= PASSTHROUGH_THRESHOLD

    result = extract_requirements(short, "test-model")
    assert result == short
    mock_agent_cls.assert_not_called()


@patch("arch_sparring_agent.context_condenser.Agent")
def test_exact_threshold_passes_through(mock_agent_cls):
    content = "x" * PASSTHROUGH_THRESHOLD
    result = _extract(content, "prompt", "test-model")
    assert result == content
    mock_agent_cls.assert_not_called()


@patch("arch_sparring_agent.context_condenser.Agent")
def test_one_over_threshold_triggers_extraction(mock_agent_cls):
    content = "x" * (PASSTHROUGH_THRESHOLD + 1)
    mock_agent = MagicMock()
    mock_agent.return_value = "extracted output"
    mock_agent_cls.return_value = mock_agent

    result = _extract(content, "prompt", "test-model")
    assert result == "extracted output"
    mock_agent_cls.assert_called_once()


@patch("arch_sparring_agent.context_condenser.Agent")
def test_extract_requirements_calls_agent(mock_agent_cls):
    content = "A" * (PASSTHROUGH_THRESHOLD + 100)
    mock_agent = MagicMock()
    mock_agent.return_value = "### Functional Requirements\n- Req 1"
    mock_agent_cls.return_value = mock_agent

    result = extract_requirements(content, "test-model")

    mock_agent_cls.assert_called_once()
    kwargs = mock_agent_cls.call_args[1]
    assert kwargs["name"] == "FindingsExtractor"
    assert kwargs["model"] == "test-model"
    assert "requirement" in kwargs["system_prompt"].lower()
    assert result == "### Functional Requirements\n- Req 1"


@patch("arch_sparring_agent.context_condenser.Agent")
def test_extract_architecture_findings_calls_agent(mock_agent_cls):
    content = "B" * (PASSTHROUGH_THRESHOLD + 100)
    mock_agent = MagicMock()
    mock_agent.return_value = "### Components\n- API Gateway"
    mock_agent_cls.return_value = mock_agent

    result = extract_architecture_findings(content, "test-model")

    kwargs = mock_agent_cls.call_args[1]
    assert "Features Verified" in kwargs["system_prompt"]
    assert "Features Not Found" in kwargs["system_prompt"]
    assert result == "### Components\n- API Gateway"


@patch("arch_sparring_agent.context_condenser.Agent")
def test_extract_phase_findings_includes_phase_name(mock_agent_cls):
    content = "C" * (PASSTHROUGH_THRESHOLD + 100)
    mock_agent = MagicMock()
    mock_agent.return_value = "### Confirmed Gaps\n- Gap 1"
    mock_agent_cls.return_value = mock_agent

    result = extract_phase_findings(content, "Sparring", "test-model")

    kwargs = mock_agent_cls.call_args[1]
    assert "Sparring" in kwargs["system_prompt"]
    assert result == "### Confirmed Gaps\n- Gap 1"


@patch("arch_sparring_agent.context_condenser.Agent")
def test_context_overflow_triggers_chunked_fallback(mock_agent_cls):
    content = "D" * (PASSTHROUGH_THRESHOLD + 100)

    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        mock = MagicMock()
        if call_count[0] == 1:
            mock.side_effect = FakeContextWindowOverflow("overflow")
            return mock
        else:
            mock.return_value = f"chunk {call_count[0]} extracted"
            return mock

    mock_agent_cls.side_effect = side_effect

    result = _extract(content, "prompt", "test-model")
    assert "chunk" in result


@patch("arch_sparring_agent.context_condenser.Agent")
def test_max_tokens_triggers_chunked_fallback(mock_agent_cls):
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
    assert "chunk" in result


@patch("arch_sparring_agent.context_condenser.Agent")
def test_client_error_validation_triggers_fallback(mock_agent_cls):
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
    assert "chunk" in result


@patch("arch_sparring_agent.context_condenser.Agent")
def test_client_error_non_token_reraises(mock_agent_cls):
    content = "G" * (PASSTHROUGH_THRESHOLD + 100)

    mock_agent = MagicMock()
    mock_agent.side_effect = FakeClientError("AccessDeniedException")
    mock_agent_cls.return_value = mock_agent

    with pytest.raises(FakeClientError):
        _extract(content, "prompt", "test-model")


@patch("arch_sparring_agent.context_condenser.Agent")
def test_unrelated_exception_reraises(mock_agent_cls):
    content = "H" * (PASSTHROUGH_THRESHOLD + 100)

    mock_agent = MagicMock()
    mock_agent.side_effect = ValueError("something else")
    mock_agent_cls.return_value = mock_agent

    with pytest.raises(ValueError):
        _extract(content, "prompt", "test-model")


@patch("arch_sparring_agent.context_condenser.Agent")
def test_chunks_large_content(mock_agent_cls):
    content = "I" * (CHUNK_SIZE * 3)
    mock_agent = MagicMock()
    mock_agent.return_value = "chunk summary"
    mock_agent_cls.return_value = mock_agent

    chunked_extract(content, "prompt", "test-model")

    assert mock_agent_cls.call_count == 1
    assert mock_agent.call_count == 3


@patch("arch_sparring_agent.context_condenser.Agent")
def test_max_chunks_limit(mock_agent_cls):
    content = "J" * (CHUNK_SIZE * (MAX_CHUNKS + 5))
    mock_agent = MagicMock()
    mock_agent.return_value = "chunk summary"
    mock_agent_cls.return_value = mock_agent

    chunked_extract(content, "prompt", "test-model")

    assert mock_agent_cls.call_count <= 2


@patch("arch_sparring_agent.context_condenser.Agent")
def test_failed_chunk_produces_placeholder(mock_agent_cls):
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
    assert "could not be processed" in result
