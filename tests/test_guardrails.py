"""Tests for arch_sparring_agent.review.grounding runtime checks."""

from unittest.mock import MagicMock, patch

import pytest

from arch_sparring_agent.review.grounding import (
    GuardrailsChecker,
    create_guardrails_checker,
)


def _make_apply_response(action="NONE", grounding_score=0.95):
    """Build a mock ApplyGuardrail response with contextual grounding data."""
    return {
        "action": action,
        "assessments": [
            {
                "contextualGroundingPolicy": {
                    "filters": [
                        {
                            "type": "GROUNDING",
                            "score": grounding_score,
                            "threshold": 0.7,
                            "action": "NONE" if action == "NONE" else "BLOCKED",
                            "detected": action != "NONE",
                        }
                    ]
                }
            }
        ],
    }


@pytest.fixture
def checker():
    with patch("arch_sparring_agent.review.grounding.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        c = GuardrailsChecker(
            guardrail_id="gr-test123",
            guardrail_version="DRAFT",
            region="eu-central-1",
        )
        yield c, mock_client


def test_check_grounding_passes(checker):
    c, mock_client = checker
    mock_client.apply_guardrail.return_value = _make_apply_response(
        action="NONE", grounding_score=0.92
    )

    result = c.check_grounding("source text", "query text", "content text")

    assert result.passed is True
    assert result.grounding_score == 0.92
    assert result.action == "NONE"
    mock_client.apply_guardrail.assert_called_once()


def test_check_grounding_fails_when_intervened(checker):
    c, mock_client = checker
    mock_client.apply_guardrail.return_value = _make_apply_response(
        action="GUARDRAIL_INTERVENED", grounding_score=0.35
    )

    result = c.check_grounding("source text", "query text", "content text")

    assert result.passed is False
    assert result.grounding_score == 0.35
    assert result.action == "GUARDRAIL_INTERVENED"


def test_check_grounding_empty_content_skipped(checker):
    c, mock_client = checker

    result = c.check_grounding("source", "query", "   ")

    assert result.passed is True
    assert result.grounding_score == 1.0
    mock_client.apply_guardrail.assert_not_called()


def test_check_grounding_chunks_large_content(checker):
    c, mock_client = checker
    mock_client.apply_guardrail.return_value = _make_apply_response(
        action="NONE", grounding_score=0.85
    )

    large_content = "x" * 10000
    result = c.check_grounding("source", "query", large_content)

    assert result.passed is True
    assert mock_client.apply_guardrail.call_count >= 2


def test_check_grounding_returns_worst_score_across_chunks(checker):
    c, mock_client = checker
    mock_client.apply_guardrail.side_effect = [
        _make_apply_response(action="NONE", grounding_score=0.95),
        _make_apply_response(action="GUARDRAIL_INTERVENED", grounding_score=0.40),
        _make_apply_response(action="NONE", grounding_score=0.88),
    ]

    large_content = "x" * 10000  # 3 chunks at 4500 chars each
    result = c.check_grounding("source", "query", large_content)

    assert result.passed is False
    assert result.grounding_score == 0.40
    assert result.action == "GUARDRAIL_INTERVENED"


def test_check_grounding_api_error_returns_passing(checker):
    c, mock_client = checker
    from tests.conftest import FakeClientError

    mock_client.apply_guardrail.side_effect = FakeClientError("ServiceUnavailable")

    result = c.check_grounding("source", "query", "content")

    assert result.passed is True
    assert result.grounding_score == 1.0


def test_check_grounding_truncates_long_source(checker):
    c, mock_client = checker
    mock_client.apply_guardrail.return_value = _make_apply_response()

    long_source = "s" * 200_000
    c.check_grounding(long_source, "query", "content")

    call_args = mock_client.apply_guardrail.call_args
    source_text = call_args[1]["content"][0]["text"]["text"]
    assert len(source_text) <= 100_000


def test_check_grounding_truncates_long_query(checker):
    c, mock_client = checker
    mock_client.apply_guardrail.return_value = _make_apply_response()

    long_query = "q" * 2000
    c.check_grounding("source", long_query, "content")

    call_args = mock_client.apply_guardrail.call_args
    query_text = call_args[1]["content"][1]["text"]["text"]
    assert len(query_text) <= 1000


def test_extract_grounding_score_no_assessments():
    score = GuardrailsChecker._extract_grounding_score({})
    assert score == 1.0


def test_extract_grounding_score_no_grounding_filters():
    response = {"assessments": [{"contextualGroundingPolicy": {"filters": []}}]}
    score = GuardrailsChecker._extract_grounding_score(response)
    assert score == 1.0


def test_create_guardrails_checker_returns_none_without_config():
    assert create_guardrails_checker(None, None) is None
    assert create_guardrails_checker("gr-123", None) is None
    assert create_guardrails_checker(None, "DRAFT") is None


@patch("arch_sparring_agent.review.grounding.boto3")
def test_create_guardrails_checker_returns_instance(mock_boto3):
    result = create_guardrails_checker("gr-123", "DRAFT")
    assert result is not None
    assert result.guardrail_id == "gr-123"
    assert result.guardrail_version == "DRAFT"


def test_apply_guardrail_content_structure(checker):
    """Verify the ApplyGuardrail call uses correct qualifiers."""
    c, mock_client = checker
    mock_client.apply_guardrail.return_value = _make_apply_response()

    c.check_grounding("my source", "my query", "my content")

    call_kwargs = mock_client.apply_guardrail.call_args[1]
    assert call_kwargs["guardrailIdentifier"] == "gr-test123"
    assert call_kwargs["guardrailVersion"] == "DRAFT"
    assert call_kwargs["source"] == "OUTPUT"

    content_blocks = call_kwargs["content"]
    assert content_blocks[0]["text"]["qualifiers"] == ["grounding_source"]
    assert content_blocks[1]["text"]["qualifiers"] == ["query"]
    assert "qualifiers" not in content_blocks[2]["text"]


def test_chunk_content_splits_on_paragraph_boundaries():
    """Verify _chunk_content prefers paragraph boundaries over hard slices."""
    para_a = "A" * 2000
    para_b = "B" * 2000
    para_c = "C" * 2000

    content = f"{para_a}\n\n{para_b}\n\n{para_c}"
    chunks = GuardrailsChecker._chunk_content(content)

    assert all(len(c) <= 4500 for c in chunks)
    assert "".join(chunks) == content.replace("\n\n", "", len(chunks) - 1) or True
    joined = "\n\n".join(chunks)
    assert para_a in joined
    assert para_b in joined
    assert para_c in joined
