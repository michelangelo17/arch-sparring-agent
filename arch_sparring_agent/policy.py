"""Policy engine and Cedar policy management."""

import logging
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .config import DEFAULT_REGION
from .gateway import associate_gateway_with_policy_engine, setup_gateway

logger = logging.getLogger(__name__)


def destroy_policy_engine(
    policy_engine_id: str, region: str = DEFAULT_REGION
) -> bool:
    """Delete all policies in an engine, then delete the engine itself.

    Returns:
        True if the engine was destroyed, False otherwise.
    """
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)

        # Delete all policies first (engine can't be deleted while policies exist)
        next_token = None
        while True:
            kwargs = {"policyEngineId": policy_engine_id}
            if next_token:
                kwargs["nextToken"] = next_token
            response = client.list_policies(**kwargs)
            for p in response.get("policies", []):
                pid = p["policyId"]
                try:
                    client.delete_policy(policyEngineId=policy_engine_id, policyId=pid)
                    logger.info("Deleted policy: %s", p.get("name", pid))
                except (ClientError, BotoCoreError) as e:
                    logger.warning("Could not delete policy %s: %s", pid, e)
            next_token = response.get("nextToken")
            if not next_token:
                break

        client.delete_policy_engine(policyEngineId=policy_engine_id)
        logger.info("Policy Engine destroyed: %s", policy_engine_id)
        return True
    except (ClientError, BotoCoreError) as e:
        logger.warning("Could not destroy Policy Engine %s: %s", policy_engine_id, e)
        return False


def setup_policy_engine(
    region: str = DEFAULT_REGION, policy_engine_name: str = "ArchReviewPolicyEngine"
) -> str | None:
    """Create or retrieve a Policy Engine.

    Returns:
        Engine ID string, or None on failure.
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
            logger.info("Created Policy Engine: %s (ID: %s)", policy_engine_name, engine_id)

        return engine_id

    except (ClientError, BotoCoreError) as e:
        logger.warning("Could not set up Policy Engine: %s", e)
        logger.warning("Continuing without policy controls.")
        return None


def _wait_for_policy_active(
    client, policy_engine_id: str, policy_id: str, policy_name: str
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
        except (ClientError, BotoCoreError) as e:
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


def _find_policy_id_by_name(
    client, policy_engine_id: str, policy_name: str
) -> str | None:
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
    client,
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
    except (ClientError, BotoCoreError) as update_error:
        logger.error("Error updating policy '%s': %s", policy_name, update_error)
        return None


def setup_architecture_review_policies(
    region: str = DEFAULT_REGION,
    policy_engine_name: str = "ArchReviewPolicyEngine",
    gateway_arn: str | None = None,
    gateway_name: str = "ArchReviewGateway",
) -> str | None:
    """Set up Cedar policies for agent tool restrictions.

    Returns:
        Policy engine ID, or None on failure.
    """
    gateway_id = None
    if not gateway_arn:
        gateway_arn, gateway_id = setup_gateway(region=region, gateway_name=gateway_name)
        if not gateway_arn:
            logger.warning(
                "Could not set up Gateway. Policies cannot be created without a Gateway."
            )
            return None

    engine_id = setup_policy_engine(region=region, policy_engine_name=policy_engine_name)
    if not engine_id:
        return None

    logger.info("Verifying policies...")
    policies_created = []
    policies_failed = []

    # RequirementsAnalyst: document and user interaction tools only
    requirements_cedar = f"""permit(
    principal is AgentCore::OAuthUser,
    action,
    resource == AgentCore::Gateway::"{gateway_arn}"
) when {{
    context has agentName && context.agentName == "RequirementsAnalyst" &&
    context has toolName &&
    ["read_document", "list_available_documents", "ask_user_question"].contains(context.toolName)
}};"""

    policy_id = create_policy(
        engine_id,
        "RequirementsAgentToolRestrictions",
        requirements_cedar,
        "Restricts Requirements Agent to only use document reading and user interaction tools",
        region=region,
    )
    if policy_id:
        policies_created.append("RequirementsAgentToolRestrictions")
    else:
        policies_failed.append("RequirementsAgentToolRestrictions")

    # ArchitectureEvaluator: CFN and diagram tools only
    architecture_cedar = f"""permit(
    principal is AgentCore::OAuthUser,
    action,
    resource == AgentCore::Gateway::"{gateway_arn}"
) when {{
    context has agentName && context.agentName == "ArchitectureEvaluator" &&
    context has toolName &&
    [
        "read_cloudformation_template",
        "list_cloudformation_templates",
        "read_architecture_diagram",
        "list_architecture_diagrams",
        "ask_user_question"
    ].contains(context.toolName)
}};"""

    policy_id = create_policy(
        engine_id,
        "ArchitectureAgentToolRestrictions",
        architecture_cedar,
        "Restricts Architecture Agent to only use CFN/diagram reading and user tools",
        region=region,
    )
    if policy_id:
        policies_created.append("ArchitectureAgentToolRestrictions")
    else:
        policies_failed.append("ArchitectureAgentToolRestrictions")

    # ReviewModerator: agent-to-agent communication only
    moderator_cedar = f"""permit(
    principal is AgentCore::OAuthUser,
    action,
    resource == AgentCore::Gateway::"{gateway_arn}"
) when {{
    context has agentName && context.agentName == "ReviewModerator" &&
    context has toolName &&
    ["get_requirements_analysis", "get_architecture_analysis"].contains(context.toolName)
}};"""

    policy_id = create_policy(
        engine_id,
        "ModeratorAgentToolRestrictions",
        moderator_cedar,
        "Restricts Moderator Agent to only use agent-to-agent communication tools",
        region=region,
    )
    if policy_id:
        policies_created.append("ModeratorAgentToolRestrictions")
    else:
        policies_failed.append("ModeratorAgentToolRestrictions")

    # Default deny: only registered agents are allowed (CRITICAL - must succeed)
    default_deny_cedar = f"""forbid(
    principal is AgentCore::OAuthUser,
    action,
    resource == AgentCore::Gateway::"{gateway_arn}"
) unless {{
    context has agentName &&
    (context.agentName == "RequirementsAnalyst" ||
     context.agentName == "ArchitectureEvaluator" ||
     context.agentName == "ReviewModerator" ||
     context.agentName == "QuestionAgent" ||
     context.agentName == "SparringAgent" ||
     context.agentName == "ReviewAgent")
}};"""

    policy_id = create_policy(
        engine_id,
        "DefaultDenyUnknownAgents",
        default_deny_cedar,
        "Denies access for unknown agents - only registered agents are allowed",
        region=region,
        validation_mode="IGNORE_ALL_FINDINGS",  # Intentionally restrictive policy
    )
    if policy_id:
        policies_created.append("DefaultDenyUnknownAgents")
    else:
        policies_failed.append("DefaultDenyUnknownAgents")

    # Fail if any policy failed - all policies are required for security
    if policies_failed:
        logger.error(
            "Policy setup failed. %d policies could not be activated: %s",
            len(policies_failed),
            ", ".join(policies_failed),
        )
        return None

    if policies_created:
        logger.info("Verified %d policies:", len(policies_created))
        for name in policies_created:
            logger.debug("  - %s", name)

        # Associate gateway with policy engine
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
    else:
        logger.warning("No policies were created.")
        return None
