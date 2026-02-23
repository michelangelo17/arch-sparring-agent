"""Tests for arch_sparring_agent.infra.guardrails setup/teardown."""

from unittest.mock import MagicMock, patch

import pytest

from arch_sparring_agent.infra.guardrails import (
    destroy_guardrails,
    setup_guardrails,
)
from tests.conftest import FakeClientError


@patch("arch_sparring_agent.infra.guardrails.boto3")
def test_setup_creates_guardrail(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    mock_client.list_guardrails.return_value = {"guardrails": []}
    mock_client.create_guardrail.return_value = {
        "guardrailId": "gr-new123",
        "version": "DRAFT",
        "guardrailArn": "arn:aws:bedrock:eu-central-1:111111111111:guardrail/gr-new123",
    }

    guardrail_id, version = setup_guardrails(region="eu-central-1")

    assert guardrail_id == "gr-new123"
    assert version == "DRAFT"
    mock_client.create_guardrail.assert_called_once()

    create_kwargs = mock_client.create_guardrail.call_args[1]
    filters = create_kwargs["contextualGroundingPolicyConfig"]["filtersConfig"]
    assert any(f["type"] == "GROUNDING" for f in filters)


@patch("arch_sparring_agent.infra.guardrails.boto3")
def test_setup_reuses_existing_guardrail(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    mock_client.list_guardrails.return_value = {
        "guardrails": [
            {"name": "ArchReviewGroundingGuardrail", "id": "gr-existing456"}
        ]
    }

    guardrail_id, version = setup_guardrails(region="eu-central-1")

    assert guardrail_id == "gr-existing456"
    assert version == "DRAFT"
    mock_client.create_guardrail.assert_not_called()


@patch("arch_sparring_agent.infra.guardrails.boto3")
def test_setup_handles_conflict_exception(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    mock_client.list_guardrails.side_effect = [
        {"guardrails": []},
        {"guardrails": [{"name": "ArchReviewGroundingGuardrail", "id": "gr-conflict789"}]},
    ]
    mock_client.create_guardrail.side_effect = FakeClientError("ConflictException")

    guardrail_id, version = setup_guardrails(region="eu-central-1")

    assert guardrail_id == "gr-conflict789"
    assert version == "DRAFT"


@patch("arch_sparring_agent.infra.guardrails.boto3")
def test_setup_raises_on_non_conflict_error(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    mock_client.list_guardrails.return_value = {"guardrails": []}
    mock_client.create_guardrail.side_effect = FakeClientError("AccessDeniedException")

    from arch_sparring_agent.exceptions import GuardrailSetupError

    with pytest.raises(GuardrailSetupError):
        setup_guardrails(region="eu-central-1")


@patch("arch_sparring_agent.infra.guardrails.boto3")
def test_destroy_deletes_guardrail(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    result = destroy_guardrails("gr-123", region="eu-central-1")

    assert result is True
    mock_client.delete_guardrail.assert_called_once_with(guardrailIdentifier="gr-123")


@patch("arch_sparring_agent.infra.guardrails.boto3")
def test_destroy_handles_not_found(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.delete_guardrail.side_effect = FakeClientError("ResourceNotFoundException")

    result = destroy_guardrails("gr-gone", region="eu-central-1")

    assert result is True


@patch("arch_sparring_agent.infra.guardrails.boto3")
def test_destroy_handles_other_error(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.delete_guardrail.side_effect = FakeClientError("InternalServerException")

    result = destroy_guardrails("gr-fail", region="eu-central-1")

    assert result is False


def test_destroy_none_guardrail_id():
    result = destroy_guardrails(None)
    assert result is True
