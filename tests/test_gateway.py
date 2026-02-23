"""Tests for arch_sparring_agent.infra.gateway."""

from unittest.mock import MagicMock, patch

import pytest

from arch_sparring_agent.exceptions import PolicySetupError
from arch_sparring_agent.infra.gateway import (
    _find_gateway_by_name,
    _wait_for_iam_propagation,
    associate_gateway_with_policy_engine,
    destroy_gateway,
    list_gateways,
    setup_gateway,
)
from tests.conftest import FakeClientError


@patch("arch_sparring_agent.infra.gateway.boto3")
def test_list_gateways_returns_items(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.list_gateways.return_value = {
        "items": [{"gatewayId": "gw-1", "name": "TestGateway"}]
    }

    result = list_gateways(region="us-east-1")

    assert len(result) == 1
    assert result[0]["gatewayId"] == "gw-1"


@patch("arch_sparring_agent.infra.gateway.boto3")
def test_list_gateways_returns_empty_on_error(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.list_gateways.side_effect = FakeClientError("AccessDenied")

    result = list_gateways(region="us-east-1")

    assert result == []


@patch("arch_sparring_agent.infra.gateway.boto3")
def test_find_gateway_by_name_found(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    mock_client.list_gateways.return_value = {
        "items": [{"gatewayId": "gw-abc", "name": "ArchReviewGateway"}]
    }
    mock_client.get_caller_identity.return_value = {"Account": "123456789012"}

    arn, gw_id, url = _find_gateway_by_name("archreviewgateway", region="us-east-1")

    assert gw_id == "gw-abc"
    assert arn is not None
    assert "gw-abc" in arn


@patch("arch_sparring_agent.infra.gateway.boto3")
def test_find_gateway_by_name_not_found(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.list_gateways.return_value = {"items": []}

    arn, gw_id, url = _find_gateway_by_name("NonExistent", region="us-east-1")

    assert arn is None
    assert gw_id is None
    assert url is None


@patch("arch_sparring_agent.infra.gateway.time")
def test_wait_for_iam_propagation_succeeds_first_try(mock_time):
    mock_client = MagicMock()
    mock_client.get_gateway.return_value = {"status": "ACTIVE"}
    mock_time.monotonic.side_effect = [0.0, 0.0, 1.0]

    result = _wait_for_iam_propagation(mock_client, "gw-123", timeout=30)

    assert result is True
    mock_client.get_gateway.assert_called_once_with(gatewayIdentifier="gw-123")


@patch("arch_sparring_agent.infra.gateway.time")
def test_wait_for_iam_propagation_retries_on_access_denied(mock_time):
    mock_client = MagicMock()
    mock_client.get_gateway.side_effect = [
        FakeClientError("AccessDeniedException"),
        {"status": "ACTIVE"},
    ]
    mock_time.monotonic.side_effect = [0.0, 1.0, 2.0, 3.0]

    result = _wait_for_iam_propagation(mock_client, "gw-123", timeout=30)

    assert result is True
    assert mock_client.get_gateway.call_count == 2


@patch("arch_sparring_agent.infra.gateway.time")
def test_wait_for_iam_propagation_times_out(mock_time):
    mock_client = MagicMock()
    mock_client.get_gateway.side_effect = FakeClientError("AccessDeniedException")
    call_count = [0]

    def advancing_monotonic():
        call_count[0] += 1
        return float(call_count[0] * 50)

    mock_time.monotonic.side_effect = advancing_monotonic

    result = _wait_for_iam_propagation(mock_client, "gw-123", timeout=30)

    assert result is False


@patch("arch_sparring_agent.infra.gateway.boto3")
def test_setup_gateway_reuses_existing(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    mock_client.list_gateways.return_value = {
        "items": [{"gatewayId": "gw-exist", "name": "ArchReviewGateway"}]
    }
    mock_client.get_caller_identity.return_value = {"Account": "123456789012"}

    arn, gw_id = setup_gateway(region="us-east-1")

    assert gw_id == "gw-exist"
    assert arn is not None


@patch("arch_sparring_agent.infra.gateway._create_gateway")
@patch("arch_sparring_agent.infra.gateway._find_gateway_by_name")
def test_setup_gateway_raises_policy_setup_error(mock_find, mock_create):
    mock_find.return_value = (None, None, None)
    mock_create.side_effect = FakeClientError("InternalServerError")

    with pytest.raises(PolicySetupError):
        setup_gateway(region="us-east-1")


@patch("arch_sparring_agent.infra.gateway.boto3")
def test_associate_gateway_with_policy_engine_success(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.get_caller_identity.return_value = {"Account": "123456789012"}
    mock_client.get_gateway.return_value = {
        "name": "TestGateway",
        "roleArn": "arn:aws:iam::role/test",
        "protocolType": "MCP",
        "authorizerType": "CUSTOM_JWT",
        "authorizerConfiguration": {"jwtIssuer": "https://example.com"},
    }

    result = associate_gateway_with_policy_engine("gw-123", "pe-456", region="us-east-1")

    assert result is True
    mock_client.update_gateway.assert_called_once()


@patch("arch_sparring_agent.infra.gateway.boto3")
def test_associate_gateway_failure(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.get_caller_identity.return_value = {"Account": "123456789012"}
    mock_client.get_gateway.side_effect = FakeClientError("InternalServerError")

    result = associate_gateway_with_policy_engine("gw-123", "pe-456", region="us-east-1")

    assert result is False


@patch("arch_sparring_agent.infra.gateway.boto3")
def test_destroy_gateway_success(mock_boto3):
    import sys

    mock_gw_client = MagicMock()
    sys.modules[
        "bedrock_agentcore_starter_toolkit.operations.gateway.client"
    ].GatewayClient.return_value = mock_gw_client

    result = destroy_gateway("gw-123", region="us-east-1")

    assert result is True


@patch("arch_sparring_agent.infra.gateway.boto3")
def test_destroy_gateway_handles_error(mock_boto3):
    import sys

    mock_gw_client = MagicMock()
    mock_gw_client.cleanup_gateway.side_effect = FakeClientError("NotFound")
    sys.modules[
        "bedrock_agentcore_starter_toolkit.operations.gateway.client"
    ].GatewayClient.return_value = mock_gw_client

    result = destroy_gateway("gw-123", region="us-east-1")

    assert result is False
