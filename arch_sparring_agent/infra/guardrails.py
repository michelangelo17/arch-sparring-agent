"""Bedrock Guardrail setup and teardown for contextual grounding checks."""

from __future__ import annotations

import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ..config import DEFAULT_REGION, GROUNDING_THRESHOLD, GUARDRAIL_NAME
from ..exceptions import GuardrailSetupError

logger = logging.getLogger(__name__)


def setup_guardrails(region: str = DEFAULT_REGION) -> tuple[str, str]:
    """Create (or find) a Bedrock Guardrail with a contextual grounding policy.

    Idempotent — reuses an existing guardrail with the same name.

    Returns:
        Tuple of (guardrail_id, guardrail_version).

    Raises:
        GuardrailSetupError: If creation fails.
    """
    client = boto3.client("bedrock", region_name=region)

    existing_id = _find_guardrail_by_name(client)
    if existing_id:
        logger.info("Found existing guardrail: %s", existing_id)
        return existing_id, "DRAFT"

    return _create_guardrail(client)


def _find_guardrail_by_name(client) -> str | None:  # type: ignore[type-arg]
    """Find an existing guardrail by name. Returns guardrail_id or None."""
    try:
        response = client.list_guardrails()
        for guardrail in response.get("guardrails", []):
            if guardrail.get("name") == GUARDRAIL_NAME:
                return guardrail["id"]
    except (ClientError, BotoCoreError) as e:
        logger.warning("Could not list guardrails: %s", e)
    return None


def _create_guardrail(client) -> tuple[str, str]:  # type: ignore[type-arg]
    """Create a Bedrock Guardrail with contextual grounding enabled.

    Returns:
        Tuple of (guardrail_id, guardrail_version).
    """
    try:
        response = client.create_guardrail(
            name=GUARDRAIL_NAME,
            description="Contextual grounding checks for architecture review condenser outputs",
            contextualGroundingPolicyConfig={
                "filtersConfig": [
                    {
                        "type": "GROUNDING",
                        "threshold": GROUNDING_THRESHOLD,
                        "action": "BLOCK",
                        "enabled": True,
                    },
                ]
            },
            blockedInputMessaging="Request blocked by guardrail.",
            blockedOutputsMessaging="Response blocked by guardrail.",
        )

        guardrail_id = response["guardrailId"]
        version = response["version"]

        logger.info("Created Bedrock Guardrail: %s (version: %s)", guardrail_id, version)
        return guardrail_id, version

    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ConflictException":
            logger.info("Guardrail creation conflict — searching for existing guardrail")
            existing_id = _find_guardrail_by_name(client)
            if existing_id:
                return existing_id, "DRAFT"
        raise GuardrailSetupError(f"Failed to create guardrail: {e}") from e
    except BotoCoreError as e:
        raise GuardrailSetupError(f"Failed to create guardrail: {e}") from e


def destroy_guardrails(guardrail_id: str | None, region: str = DEFAULT_REGION) -> bool:
    """Delete a Bedrock Guardrail. Returns True on success."""
    if not guardrail_id:
        return True

    try:
        client = boto3.client("bedrock", region_name=region)
        client.delete_guardrail(guardrailIdentifier=guardrail_id)
        logger.info("Deleted Bedrock Guardrail: %s", guardrail_id)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            logger.info("Guardrail already absent: %s", guardrail_id)
            return True
        logger.warning("Failed to delete guardrail %s: %s", guardrail_id, e)
        return False
    except BotoCoreError as e:
        logger.warning("Failed to delete guardrail %s: %s", guardrail_id, e)
        return False
