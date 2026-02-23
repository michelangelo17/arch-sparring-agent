"""Policy engine and Cedar policy management."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ..config import (
    AGENT_ARCHITECTURE,
    AGENT_QUESTION,
    AGENT_REQUIREMENTS,
    AGENT_REVIEW,
    AGENT_SPARRING,
    DEFAULT_REGION,
)
from ..exceptions import AWS_ERRORS, PolicySetupError
from .gateway import associate_gateway_with_policy_engine, setup_gateway

logger = logging.getLogger(__name__)


def destroy_policy_engine(policy_engine_id: str, region: str = DEFAULT_REGION) -> bool:
    """Delete all policies in an engine, then delete the engine itself.

    Returns:
        True if the engine was destroyed, False otherwise.
    """
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        logger.info("Listing policies for engine: %s", policy_engine_id)

        deleted_count = 0
        next_token = None
        while True:
            kwargs = {"policyEngineId": policy_engine_id, "maxResults": 100}
            if next_token:
                kwargs["nextToken"] = next_token
            response = client.list_policies(**kwargs)
            policies = response.get("policies", [])
            logger.info("list_policies returned %d policies", len(policies))

            for p in policies:
                pid = p["policyId"]
                try:
                    client.delete_policy(policyEngineId=policy_engine_id, policyId=pid)
                    deleted_count += 1
                    logger.info("Deleted policy: %s", p.get("name", pid))
                except AWS_ERRORS as e:
                    logger.warning("Could not delete policy %s: %s", pid, e)

            next_token = response.get("nextToken")
            if not next_token:
                break

        if deleted_count:
            logger.info(
                "Deleted %d policies, waiting for propagation...",
                deleted_count,
            )
            time.sleep(5)

        client.delete_policy_engine(policyEngineId=policy_engine_id)
        logger.info("Policy Engine destroyed: %s", policy_engine_id)
        return True
    except AWS_ERRORS as e:
        logger.warning("Could not destroy Policy Engine %s: %s", policy_engine_id, e)
        return False


def setup_policy_engine(
    region: str = DEFAULT_REGION, policy_engine_name: str = "ArchReviewPolicyEngine"
) -> str:
    """Create or retrieve a Policy Engine.

    Returns:
        Engine ID string.

    Raises:
        PolicySetupError: If the engine cannot be created or found.
    """
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        engines = client.list_policy_engines()
        engine = next(
            (
                e
                for e in engines.get("policyEngines", [])
                if _policy_name_matches(e, policy_engine_name)
            ),
            None,
        )

        if engine:
            engine_id = engine["policyEngineId"]
            logger.info("Using existing Policy Engine: %s (ID: %s)", policy_engine_name, engine_id)
        else:
            response = client.create_policy_engine(
                name=policy_engine_name, description="Policy engine for architecture review agents"
            )
            engine_id = response.get("policyEngineId")
            if not engine_id:
                raise PolicySetupError(
                    f"Policy Engine created but no ID in response: {list(response.keys())}"
                )
            logger.info("Created Policy Engine: %s (ID: %s)", policy_engine_name, engine_id)

        return engine_id

    except AWS_ERRORS as e:
        raise PolicySetupError(f"Could not set up Policy Engine: {e}") from e


def _wait_for_policy_active(
    client: Any, policy_engine_id: str, policy_id: str, policy_name: str
) -> bool:
    """Wait for policy to become ACTIVE.

    Args:
        client: boto3 bedrock-agentcore-control client.
        policy_engine_id: The policy engine ID.
        policy_id: The policy ID to check.
        policy_name: Human-readable policy name for log messages.

    Returns:
        True if policy became ACTIVE, False otherwise.
    """
    logger.info("Verifying policy '%s' status...", policy_name)
    for _ in range(120):  # 120 * 0.5s = 60s max
        try:
            response = client.get_policy(policyEngineId=policy_engine_id, policyId=policy_id)
            status = response.get("status", "").upper()

            if status == "ACTIVE":
                logger.info("Policy '%s' is ACTIVE", policy_name)
                return True
            if "FAIL" in status:
                reasons = response.get("statusReasons", [])
                logger.error(
                    "Policy '%s' failed. Status: %s, Reasons: %s", policy_name, status, reasons
                )
                return False

            logger.debug("Policy '%s' status: %s", policy_name, status)
            time.sleep(0.5)
        except AWS_ERRORS as e:
            logger.debug("Error checking policy '%s' status: %s", policy_name, e)
            time.sleep(0.5)

    logger.warning("Timeout waiting for policy '%s' to become ACTIVE", policy_name)
    return False


def create_policy(
    policy_engine_id: str,
    policy_name: str,
    cedar_statement: str,
    description: str = "",
    region: str = DEFAULT_REGION,
    validation_mode: str = "FAIL_ON_ANY_FINDINGS",
) -> str | None:
    """Create a Cedar policy in a Policy Engine, or update if it already exists.

    Returns:
        Policy ID string, or None on failure.
    """
    logger.info("Checking policy: '%s'...", policy_name)
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        response = client.create_policy(
            policyEngineId=policy_engine_id,
            name=policy_name,
            definition={"cedar": {"statement": cedar_statement}},
            description=description or f"Policy for {policy_name}",
            validationMode=validation_mode,
        )
        policy_id = response.get("policyId")
        logger.info("Created policy '%s' (ID: %s)", policy_name, policy_id)
        # Verify policy becomes ACTIVE before returning
        if not _wait_for_policy_active(client, policy_engine_id, policy_id, policy_name):
            logger.error("Policy '%s' failed to become ACTIVE", policy_name)
            return None
        return policy_id
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        # Handle "already exists" as success
        if error_code in ("ConflictException", "ResourceAlreadyExistsException"):
            return _update_existing_policy(
                client,
                policy_engine_id,
                policy_name,
                cedar_statement,
                description,
                validation_mode,
            )
        logger.error("Error creating policy '%s': %s", policy_name, e)
        return None
    except BotoCoreError as e:
        logger.error("Error creating policy '%s': %s", policy_name, e)
        return None


def _policy_name_matches(resource: dict, target_name: str) -> bool:
    """Check if a resource's name matches the target.

    Case-insensitive comparison with prefix matching to handle
    AWS appending UUID suffixes to names.
    """
    value = resource.get("name", "")
    if not value:
        return False
    value_lower = value.lower()
    target_lower = target_name.lower()
    return value_lower == target_lower or value_lower.startswith(target_lower)


def _find_policy_id_by_name(client: Any, policy_engine_id: str, policy_name: str) -> str | None:
    """Find a policy ID by name with pagination.

    Uses case-insensitive prefix matching to handle AWS name normalisation.
    """
    next_token = None

    while True:
        kwargs = {"policyEngineId": policy_engine_id}
        if next_token:
            kwargs["nextToken"] = next_token

        response = client.list_policies(**kwargs)

        for p in response.get("policies", []):
            if _policy_name_matches(p, policy_name):
                logger.debug(
                    "Matched policy '%s' -> ID: %s (raw name: %s)",
                    policy_name,
                    p["policyId"],
                    p.get("name"),
                )
                return p["policyId"]

        next_token = response.get("nextToken")
        if not next_token:
            break

    return None


def _update_existing_policy(
    client: Any,
    policy_engine_id: str,
    policy_name: str,
    cedar_statement: str,
    description: str,
    validation_mode: str,
) -> str | None:
    """Find and update an existing policy by name.

    Returns:
        Policy ID string, or None on failure.
    """
    logger.info("Policy '%s' already exists. Updating...", policy_name)
    try:
        policy_id = _find_policy_id_by_name(client, policy_engine_id, policy_name)

        if policy_id:
            logger.info("Updating policy '%s' (ID: %s)...", policy_name, policy_id)
            client.update_policy(
                policyEngineId=policy_engine_id,
                policyId=policy_id,
                definition={"cedar": {"statement": cedar_statement}},
                description=description or f"Policy for {policy_name}",
                validationMode=validation_mode,
            )
            logger.info("Updated policy '%s' (ID: %s)", policy_name, policy_id)
            if not _wait_for_policy_active(client, policy_engine_id, policy_id, policy_name):
                logger.error("Policy '%s' failed to become ACTIVE after update", policy_name)
                return None
            return policy_id
        else:
            logger.error("Could not find existing policy ID for '%s'", policy_name)
            return None
    except AWS_ERRORS as update_error:
        logger.error("Error updating policy '%s': %s", policy_name, update_error)
        return None


@dataclass
class _PolicySpec:
    """Definition of a Cedar policy to create."""

    name: str
    cedar: str
    description: str
    validation_mode: str = "FAIL_ON_ANY_FINDINGS"


def _build_policy_specs(gateway_arn: str) -> list[_PolicySpec]:
    """Build the Cedar policy definitions for architecture review agents."""
    return [
        _PolicySpec(
            name="RequirementsAgentToolRestrictions",
            description="Restricts Requirements Agent to document reading and user interaction",
            cedar=f"""permit(
    principal is AgentCore::OAuthUser,
    action,
    resource == AgentCore::Gateway::"{gateway_arn}"
) when {{
    context has agentName && context.agentName == "{AGENT_REQUIREMENTS}" &&
    context has toolName &&
    ["read_document", "list_available_documents", "ask_user_question"].contains(context.toolName)
}};""",
        ),
        _PolicySpec(
            name="ArchitectureAgentToolRestrictions",
            description="Restricts Architecture Agent to CFN/diagram reading and user tools",
            cedar=f"""permit(
    principal is AgentCore::OAuthUser,
    action,
    resource == AgentCore::Gateway::"{gateway_arn}"
) when {{
    context has agentName && context.agentName == "{AGENT_ARCHITECTURE}" &&
    context has toolName &&
    [
        "read_cloudformation_template",
        "list_cloudformation_templates",
        "read_architecture_diagram",
        "list_architecture_diagrams",
        "list_source_files",
        "read_source_file",
        "search_source_code",
        "query_waf",
        "ask_user_question"
    ].contains(context.toolName)
}};""",
        ),
        _PolicySpec(
            name="ReviewAgentToolRestrictions",
            description="Allows Review Agent to use the WAF Knowledge Base query tool",
            cedar=f"""permit(
    principal is AgentCore::OAuthUser,
    action,
    resource == AgentCore::Gateway::"{gateway_arn}"
) when {{
    context has agentName && context.agentName == "{AGENT_REVIEW}" &&
    context has toolName &&
    ["query_waf"].contains(context.toolName)
}};""",
        ),
        _PolicySpec(
            name="DefaultDenyUnknownAgents",
            description="Denies access for unknown agents - only registered agents are allowed",
            validation_mode="IGNORE_ALL_FINDINGS",
            cedar=f"""forbid(
    principal is AgentCore::OAuthUser,
    action,
    resource == AgentCore::Gateway::"{gateway_arn}"
) unless {{
    context has agentName &&
    (context.agentName == "{AGENT_REQUIREMENTS}" ||
     context.agentName == "{AGENT_ARCHITECTURE}" ||
     context.agentName == "{AGENT_QUESTION}" ||
     context.agentName == "{AGENT_SPARRING}" ||
     context.agentName == "{AGENT_REVIEW}")
}};""",
        ),
    ]


def setup_architecture_review_policies(
    region: str = DEFAULT_REGION,
    policy_engine_name: str = "ArchReviewPolicyEngine",
    gateway_arn: str | None = None,
    gateway_name: str = "ArchReviewGateway",
) -> str:
    """Set up Cedar policies for agent tool restrictions.

    Returns:
        Policy engine ID.

    Raises:
        PolicySetupError: If gateway, engine, or policy creation fails.
    """
    gateway_id = None
    if not gateway_arn:
        gateway_arn, gateway_id = setup_gateway(region=region, gateway_name=gateway_name)

    engine_id = setup_policy_engine(region=region, policy_engine_name=policy_engine_name)

    logger.info("Verifying policies...")
    policies_created: list[str] = []
    policies_failed: list[str] = []

    for spec in _build_policy_specs(gateway_arn):
        policy_id = create_policy(
            engine_id,
            spec.name,
            spec.cedar,
            spec.description,
            region=region,
            validation_mode=spec.validation_mode,
        )
        if policy_id:
            policies_created.append(spec.name)
        else:
            policies_failed.append(spec.name)

    if policies_failed:
        raise PolicySetupError(
            f"Policy setup failed. {len(policies_failed)} policies could not be activated: "
            f"{', '.join(policies_failed)}"
        )

    if not policies_created:
        raise PolicySetupError("No policies were created.")

    logger.info("Verified %d policies:", len(policies_created))
    for name in policies_created:
        logger.debug("  - %s", name)

    gw_id = gateway_id
    if not gw_id and gateway_arn and "/gateway/" in gateway_arn:
        gw_id = gateway_arn.split("/gateway/")[-1]

    if gw_id:
        logger.info("Associating Gateway with Policy Engine...")
        associate_gateway_with_policy_engine(
            gateway_id=gw_id,
            policy_engine_id=engine_id,
            enforcement_mode="ENFORCE",
            region=region,
        )

    return engine_id
