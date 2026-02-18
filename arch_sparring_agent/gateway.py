"""Gateway setup and management for policy enforcement."""

import json
import logging
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .config import DEFAULT_REGION, IAM_PROPAGATION_TIMEOUT
from .exceptions import PolicySetupError

logger = logging.getLogger(__name__)


def list_gateways(region: str = DEFAULT_REGION) -> list:
    """List available Gateways using boto3 directly."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        response = client.list_gateways()
        return response.get("items", [])
    except (ClientError, BotoCoreError) as e:
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

        # ListGateways (GatewaySummary) doesn't include ARN or URL — construct the ARN
        sts = boto3.client("sts", region_name=region)
        account_id = sts.get_caller_identity()["Account"]
        gateway_arn = f"arn:aws:bedrock-agentcore:{region}:{account_id}:gateway/{gw_id}"

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
        sts = boto3.client("sts", region_name=region)
        account_id = sts.get_caller_identity()["Account"]

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
    except (ClientError, BotoCoreError) as e:
        logger.warning("Could not associate Gateway with Policy Engine: %s", e)
        logger.warning("You may need to associate them manually via the AWS Console.")
        return False


def setup_gateway(
    region: str = DEFAULT_REGION,
    gateway_name: str = "ArchReviewGateway",
) -> tuple[str | None, str | None]:
    """Create or retrieve a Gateway for policy enforcement.

    Returns:
        Tuple of (gateway_arn, gateway_id), or (None, None) on failure.
    """
    try:
        from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

        gateway_arn, gateway_id, gateway_url = _find_gateway_by_name(gateway_name, region)
        if gateway_id:
            logger.info("Using existing Gateway: %s", gateway_name)
            if gateway_url:
                logger.debug("Gateway URL: %s", gateway_url)
            logger.debug("Gateway ID: %s", gateway_id)
            return gateway_arn, gateway_id

        logger.info("Creating Gateway: %s...", gateway_name)
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

            observability = gateway.get("observability", {})
            if observability.get("status") == "error":
                logger.warning(
                    "Gateway created but observability (X-Ray tracing) could not be enabled. "
                    "To fix, run: aws xray update-trace-segment-destination "
                    "--destination CloudWatchLogs --region %s",
                    region,
                )

        except (ClientError, BotoCoreError) as create_error:
            if "already exists" in str(create_error).lower():
                gateway_arn, gateway_id, gateway_url = _find_gateway_by_name(gateway_name, region)
                if gateway_id:
                    logger.info("Using existing Gateway: %s", gateway_name)
                    if gateway_url:
                        logger.debug("Gateway URL: %s", gateway_url)
                    logger.debug("Gateway ID: %s", gateway_id)
                    return gateway_arn, gateway_id
            raise

        if isinstance(gateway, str):
            gateway = {"gatewayId": gateway}

        logger.info("Gateway created")
        logger.info("Configuring IAM permissions...")
        client.fix_iam_permissions(gateway)

        # Wait for IAM propagation with exponential backoff
        agentcore_client = boto3.client("bedrock-agentcore-control", region_name=region)
        gw_id = gateway.get("gatewayId") or gateway.get("id")
        if gw_id:
            _wait_for_iam_propagation(agentcore_client, gw_id)

        gateway_id = gw_id
        gateway_url = gateway.get("gatewayUrl") or gateway.get("url")
        gateway_arn = gateway.get("gatewayArn")

        if not gateway_arn and gateway_id:
            sts = boto3.client("sts", region_name=region)
            account_id = sts.get_caller_identity()["Account"]
            gateway_arn = f"arn:aws:bedrock-agentcore:{region}:{account_id}:gateway/{gateway_id}"

        logger.info("Gateway setup complete: %s", gateway_name)
        if gateway_url:
            logger.debug("Gateway URL: %s", gateway_url)
        logger.debug("Gateway ID: %s", gateway_id)

        _save_gateway_config(gateway, cognito_response, region)
        return gateway_arn, gateway_id

    except ImportError:
        logger.warning("bedrock-agentcore-starter-toolkit not installed.")
        logger.warning("Run: pip install bedrock-agentcore-starter-toolkit")
        return None, None
    except (ClientError, BotoCoreError, PolicySetupError) as e:
        import traceback

        logger.warning("Could not set up Gateway: %s", e)
        logger.debug("Details: %s", traceback.format_exc())
        logger.warning("Continuing without Gateway. Policies will not be created.")
        return None, None


def _save_gateway_config(gateway: dict, cognito_response: dict, region: str) -> None:
    """Save gateway config to ~/.arch-review for cleanup."""
    config = {
        "gateway_url": gateway.get("gatewayUrl"),
        "gateway_id": gateway.get("gatewayId"),
        "region": region,
        "client_info": cognito_response.get("client_info"),
    }

    config_path = Path.home() / ".arch-review" / "gateway_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))


def cleanup_gateway(region: str = DEFAULT_REGION) -> None:
    """Clean up Gateway and Cognito resources."""
    config_path = Path.home() / ".arch-review" / "gateway_config.json"
    if not config_path.exists():
        logger.info("No gateway config found. Nothing to clean up.")
        return

    try:
        from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

        config = json.loads(config_path.read_text())
        client = GatewayClient(region_name=config.get("region", region))
        client.cleanup_gateway(config["gateway_id"], config.get("client_info"))
        logger.info("Gateway cleanup complete!")
        config_path.unlink()
    except (ClientError, BotoCoreError, json.JSONDecodeError, OSError, KeyError) as e:
        logger.warning("Could not clean up Gateway: %s", e)
