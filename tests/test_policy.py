"""Tests for arch_sparring_agent.infra.policy."""

from unittest.mock import MagicMock, patch

import pytest

from arch_sparring_agent.exceptions import PolicySetupError
from arch_sparring_agent.infra.policy import (
    _build_policy_specs,
    _wait_for_policy_active,
    create_policy,
    destroy_policy_engine,
    setup_architecture_review_policies,
    setup_policy_engine,
)
from tests.conftest import FakeClientError


@patch("arch_sparring_agent.infra.policy.boto3")
def test_setup_policy_engine_creates_new(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.list_policy_engines.return_value = {"policyEngines": []}
    mock_client.create_policy_engine.return_value = {"policyEngineId": "pe-new123"}

    engine_id = setup_policy_engine(region="us-east-1")

    assert engine_id == "pe-new123"
    mock_client.create_policy_engine.assert_called_once()


@patch("arch_sparring_agent.infra.policy.boto3")
def test_setup_policy_engine_reuses_existing(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.list_policy_engines.return_value = {
        "policyEngines": [{"policyEngineId": "pe-exist456", "name": "ArchReviewPolicyEngine"}]
    }

    engine_id = setup_policy_engine(region="us-east-1")

    assert engine_id == "pe-exist456"
    mock_client.create_policy_engine.assert_not_called()


@patch("arch_sparring_agent.infra.policy.boto3")
def test_setup_policy_engine_raises_on_error(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.list_policy_engines.side_effect = FakeClientError("AccessDenied")

    with pytest.raises(PolicySetupError):
        setup_policy_engine(region="us-east-1")


@patch("arch_sparring_agent.infra.policy.time")
def test_wait_for_policy_active_returns_true(mock_time):
    mock_client = MagicMock()
    mock_client.get_policy.return_value = {"status": "ACTIVE"}

    result = _wait_for_policy_active(mock_client, "pe-1", "pol-1", "TestPolicy")

    assert result is True


@patch("arch_sparring_agent.infra.policy.time")
def test_wait_for_policy_active_returns_false_on_fail(mock_time):
    mock_client = MagicMock()
    mock_client.get_policy.return_value = {
        "status": "FAILED",
        "statusReasons": ["Cedar syntax error"],
    }

    result = _wait_for_policy_active(mock_client, "pe-1", "pol-1", "TestPolicy")

    assert result is False


@patch("arch_sparring_agent.infra.policy.time")
def test_wait_for_policy_active_timeout(mock_time):
    mock_client = MagicMock()
    mock_client.get_policy.return_value = {"status": "CREATING"}

    result = _wait_for_policy_active(mock_client, "pe-1", "pol-1", "TestPolicy")

    assert result is False


@patch("arch_sparring_agent.infra.policy._wait_for_policy_active", return_value=True)
@patch("arch_sparring_agent.infra.policy.boto3")
def test_create_policy_success(mock_boto3, mock_wait):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.create_policy.return_value = {"policyId": "pol-new"}

    result = create_policy("pe-1", "TestPolicy", "permit(...);", region="us-east-1")

    assert result == "pol-new"
    mock_client.create_policy.assert_called_once()


@patch("arch_sparring_agent.infra.policy._wait_for_policy_active", return_value=True)
@patch("arch_sparring_agent.infra.policy.boto3")
def test_create_policy_handles_conflict(mock_boto3, mock_wait):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.create_policy.side_effect = FakeClientError("ConflictException")
    mock_client.list_policies.return_value = {
        "policies": [{"policyId": "pol-exist", "name": "TestPolicy"}]
    }

    result = create_policy("pe-1", "TestPolicy", "permit(...);", region="us-east-1")

    assert result == "pol-exist"
    mock_client.update_policy.assert_called_once()


@patch("arch_sparring_agent.infra.policy.boto3")
def test_create_policy_returns_none_on_error(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.create_policy.side_effect = FakeClientError("InternalServerError")

    result = create_policy("pe-1", "TestPolicy", "permit(...);", region="us-east-1")

    assert result is None


def test_build_policy_specs_returns_expected_count():
    specs = _build_policy_specs("arn:aws:bedrock-agentcore:us-east-1:111:gateway/gw-1")

    assert len(specs) == 4
    names = [s.name for s in specs]
    assert "RequirementsAgentToolRestrictions" in names
    assert "ArchitectureAgentToolRestrictions" in names
    assert "ReviewAgentToolRestrictions" in names
    assert "DefaultDenyUnknownAgents" in names


def test_build_policy_specs_embeds_gateway_arn():
    arn = "arn:aws:bedrock-agentcore:us-east-1:111:gateway/gw-test"
    specs = _build_policy_specs(arn)

    for spec in specs:
        assert arn in spec.cedar


@patch("arch_sparring_agent.infra.policy.time")
@patch("arch_sparring_agent.infra.policy.boto3")
def test_destroy_policy_engine_success(mock_boto3, mock_time):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.list_policies.return_value = {
        "policies": [{"policyId": "pol-1", "name": "Policy1"}]
    }

    result = destroy_policy_engine("pe-123", region="us-east-1")

    assert result is True
    mock_client.delete_policy.assert_called_once()
    mock_client.delete_policy_engine.assert_called_once_with(policyEngineId="pe-123")


@patch("arch_sparring_agent.infra.policy.boto3")
def test_destroy_policy_engine_handles_error(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.list_policies.side_effect = FakeClientError("InternalServerError")

    result = destroy_policy_engine("pe-123", region="us-east-1")

    assert result is False


@patch("arch_sparring_agent.infra.policy.associate_gateway_with_policy_engine")
@patch("arch_sparring_agent.infra.policy.create_policy")
@patch("arch_sparring_agent.infra.policy.setup_policy_engine")
@patch("arch_sparring_agent.infra.policy.setup_gateway")
def test_setup_architecture_review_policies_wires_together(
    mock_setup_gw, mock_setup_pe, mock_create_pol, mock_assoc
):
    mock_setup_gw.return_value = (
        "arn:aws:bedrock-agentcore:us-east-1:111:gateway/gw-1",
        "gw-1",
    )
    mock_setup_pe.return_value = "pe-1"
    mock_create_pol.return_value = "pol-id"

    engine_id = setup_architecture_review_policies(region="us-east-1")

    assert engine_id == "pe-1"
    assert mock_create_pol.call_count == 4
    mock_assoc.assert_called_once()


@patch("arch_sparring_agent.infra.policy.create_policy")
@patch("arch_sparring_agent.infra.policy.setup_policy_engine")
def test_setup_architecture_review_policies_raises_on_failures(mock_setup_pe, mock_create_pol):
    mock_setup_pe.return_value = "pe-1"
    mock_create_pol.return_value = None

    with pytest.raises(PolicySetupError, match="could not be activated"):
        setup_architecture_review_policies(
            region="us-east-1",
            gateway_arn="arn:aws:bedrock-agentcore:us-east-1:111:gateway/gw-1",
        )
