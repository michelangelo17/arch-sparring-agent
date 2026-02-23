"""Gateway setup and management for policy enforcement."""

from __future__ import annotations

import logging
import time
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ..config import DEFAULT_REGION, IAM_PROPAGATION_TIMEOUT
from ..exceptions import AWS_ERRORS, PolicySetupError
from . import build_gateway_arn, get_account_id

logger = logging.getLogger(__name__)


def list_gateways(region: str = DEFAULT_REGION) -> list[dict[str, Any]]:
    """List available Gateways using boto3 directly."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        response = client.list_gateways()
        return response.get("items", [])
    except AWS_ERRORS as e:
        logger.warning("Could not list Gateways: %s", e)
        return []


def _find_gateway_by_name(
    gateway_name: str, region: str = DEFAULT_REGION
) -> tuple[str | None, str | None, str | None]:
    """Find an existing gateway by name (case-insensitive).

    Returns:
        Tuple of (gateway_arn, gateway_id, gateway_url).
    """
    gateways = list_gateways(region=region)
    gateway_name_lower = gateway_name.lower()

    for gw in gateways:
        gw_name = gw.get("name", "")
        gw_id = gw.get("gatewayId", "")

        if gw_name.lower() != gateway_name_lower:
            continue

        gateway_arn = build_gateway_arn(region, get_account_id(region), gw_id)

        return gateway_arn, gw_id, None

    return None, None, None


def _wait_for_iam_propagation(
    client, gateway_id: str, timeout: int = IAM_PROPAGATION_TIMEOUT
) -> bool:
    """Wait for IAM permissions to propagate using exponential backoff.

    Polls the gateway with exponential backoff until the call succeeds
    or the timeout is reached.

    Returns:
        True if IAM propagated successfully, False on timeout.
    """
    logger.debug("Waiting for IAM propagation (timeout: %ds)...", timeout)
    start = time.monotonic()
    delay = 1.0  # Start with 1 second
    max_delay = 16.0

    while (time.monotonic() - start) < timeout:
        try:
            client.get_gateway(gatewayIdentifier=gateway_id)
            elapsed = time.monotonic() - start
            logger.debug("IAM propagation confirmed after %.1fs", elapsed)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("AccessDeniedException", "UnauthorizedException"):
                logger.debug("IAM not yet propagated, retrying in %.1fs...", delay)
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            else:
                # Non-IAM error -- propagation may already be done
                logger.debug("Gateway check returned non-IAM error: %s", e)
                return True
        except BotoCoreError as e:
            logger.debug("Gateway check returned non-IAM error: %s", e)
            return True

    logger.warning("IAM propagation timed out after %ds. Continuing anyway (best-effort).", timeout)
    return False


def associate_gateway_with_policy_engine(
    gateway_id: str,
    policy_engine_id: str,
    enforcement_mode: str = "ENFORCE",
    region: str = DEFAULT_REGION,
) -> bool:
    """Associate a Gateway with a Policy Engine. Mode: ENFORCE or LOG_ONLY."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        account_id = get_account_id(region)

        gateway = client.get_gateway(gatewayIdentifier=gateway_id)
        policy_engine_arn = (
            f"arn:aws:bedrock-agentcore:{region}:{account_id}:policy-engine/{policy_engine_id}"
        )

        update_params = {
            "gatewayIdentifier": gateway_id,
            "name": gateway.get("name"),
            "roleArn": gateway.get("roleArn"),
            "protocolType": gateway.get("protocolType"),
            "authorizerType": gateway.get("authorizerType"),
            "policyEngineConfiguration": {
                "arn": policy_engine_arn,
                "mode": enforcement_mode,
            },
        }

        # Required for CUSTOM_JWT authorizer type
        if gateway.get("authorizerConfiguration"):
            update_params["authorizerConfiguration"] = gateway.get("authorizerConfiguration")

        client.update_gateway(**update_params)
        logger.info("Associated Policy Engine with Gateway")
        logger.debug("Gateway ID: %s", gateway_id)
        logger.debug("Policy Engine ARN: %s", policy_engine_arn)
        logger.debug("Enforcement mode: %s", enforcement_mode)
        return True
    except AWS_ERRORS as e:
        logger.warning("Could not associate Gateway with Policy Engine: %s", e)
        logger.warning("You may need to associate them manually via the AWS Console.")
        return False


def _create_gateway(gateway_name: str, region: str) -> tuple[str, str]:
    """Create a new Gateway with Cognito auth and IAM permissions.

    Returns:
        Tuple of (gateway_arn, gateway_id).
    """
    from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

    client = GatewayClient(region_name=region)

    try:
        logger.info("Creating OAuth authorization server...")
        cognito_response = client.create_oauth_authorizer_with_cognito(gateway_name)

        if isinstance(cognito_response, str):
            raise PolicySetupError(f"Unexpected cognito response: {cognito_response[:100]}")

        authorizer_config = cognito_response.get("authorizer_config") or cognito_response.get(
            "authorizerConfig"
        )
        if not authorizer_config:
            raise PolicySetupError(
                f"Missing authorizer_config. Keys: {list(cognito_response.keys())}"
            )
        logger.info("Authorization server created")

        logger.info("Creating MCP Gateway...")
        gateway = client.create_mcp_gateway(
            name=gateway_name,
            role_arn=None,
            authorizer_config=authorizer_config,
            enable_semantic_search=False,
        )

        if isinstance(gateway, str):
            gateway = {"gatewayId": gateway}

        observability = gateway.get("observability", {})
        if observability.get("status") == "error":
            logger.warning(
                "Gateway created but observability (X-Ray tracing) could not be enabled. "
                "To fix, run: aws xray update-trace-segment-destination "
                "--destination CloudWatchLogs --region %s",
                region,
            )

    except AWS_ERRORS as create_error:
        if "already exists" in str(create_error).lower():
            gateway_arn, gateway_id, _ = _find_gateway_by_name(gateway_name, region)
            if gateway_id and gateway_arn:
                logger.info("Using existing Gateway: %s", gateway_name)
                return gateway_arn, gateway_id
        raise

    logger.info("Gateway created")
    logger.info("Configuring IAM permissions...")
    client.fix_iam_permissions(gateway)

    agentcore_client = boto3.client("bedrock-agentcore-control", region_name=region)
    gateway_id = gateway.get("gatewayId") or gateway.get("id")
    if gateway_id:
        _wait_for_iam_propagation(agentcore_client, gateway_id)

    gateway_arn = gateway.get("gatewayArn")
    if not gateway_arn and gateway_id:
        gateway_arn = build_gateway_arn(region, get_account_id(region), gateway_id)

    logger.info("Gateway setup complete: %s", gateway_name)
    logger.debug("Gateway ID: %s", gateway_id)

    return gateway_arn, gateway_id


def setup_gateway(
    region: str = DEFAULT_REGION,
    gateway_name: str = "ArchReviewGateway",
) -> tuple[str, str]:
    """Create or retrieve a Gateway for policy enforcement.

    Returns:
        Tuple of (gateway_arn, gateway_id).

    Raises:
        PolicySetupError: If the gateway cannot be created or found.
    """
    try:
        from bedrock_agentcore_starter_toolkit.operations.gateway.client import (
            GatewayClient,  # noqa: F401
        )
    except ImportError as e:
        raise PolicySetupError(
            "bedrock-agentcore-starter-toolkit not installed. "
            "Run: pip install bedrock-agentcore-starter-toolkit"
        ) from e

    try:
        gateway_arn, gateway_id, _ = _find_gateway_by_name(gateway_name, region)
        if gateway_id and gateway_arn:
            logger.info("Using existing Gateway: %s", gateway_name)
            logger.debug("Gateway ID: %s", gateway_id)
            return gateway_arn, gateway_id

        logger.info("Creating Gateway: %s...", gateway_name)
        return _create_gateway(gateway_name, region)

    except PolicySetupError:
        raise
    except AWS_ERRORS as e:
        raise PolicySetupError(f"Could not set up Gateway: {e}") from e


def destroy_gateway(gateway_id: str, region: str = DEFAULT_REGION) -> bool:
    """Delete a Gateway and its associated Cognito resources.

    Returns:
        True if cleanup succeeded, False otherwise.
    """
    try:
        from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

        client = GatewayClient(region_name=region)
        client.cleanup_gateway(gateway_id, None)
        logger.info("Gateway destroyed: %s", gateway_id)
        return True
    except ImportError:
        logger.warning("bedrock-agentcore-starter-toolkit not installed — cannot destroy gateway.")
        return False
    except AWS_ERRORS as e:
        logger.warning("Could not destroy Gateway %s: %s", gateway_id, e)
        return False
