import logging
import os
from datetime import datetime
from typing import Any

import boto3
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)

logger = logging.getLogger(__name__)

MODEL_ID = "amazon.nova-2-lite-v1:0"
DEFAULT_REGION = "eu-central-1"

# --- Tuning constants (override via environment variables) ---

# Context condenser: skip extraction for content shorter than this (chars)
CONDENSER_PASSTHROUGH_THRESHOLD = int(os.getenv("ARCH_REVIEW_PASSTHROUGH_THRESHOLD", "2000"))

# Context condenser: chunk size for fallback chunked extraction (chars)
CONDENSER_CHUNK_SIZE = int(os.getenv("ARCH_REVIEW_CHUNK_SIZE", "8000"))

# Context condenser: max chunks to process in fallback mode
CONDENSER_MAX_CHUNKS = int(os.getenv("ARCH_REVIEW_MAX_CHUNKS", "5"))

# Requirements agent: summarize documents longer than this (chars, ~6k tokens)
DOC_SUMMARY_THRESHOLD = int(os.getenv("ARCH_REVIEW_DOC_SUMMARY_THRESHOLD", "25000"))

# Requirements agent: use chunked summarization for documents longer than this (chars)
DOC_CHUNK_SUMMARY_THRESHOLD = int(os.getenv("ARCH_REVIEW_DOC_CHUNK_THRESHOLD", "100000"))

# Source analyzer: truncate source files longer than this (chars)
SOURCE_FILE_MAX_CHARS = int(os.getenv("ARCH_REVIEW_SOURCE_MAX_CHARS", "50000"))

# Diagram analyzer: max tokens for diagram analysis response
DIAGRAM_MAX_TOKENS = int(os.getenv("ARCH_REVIEW_DIAGRAM_MAX_TOKENS", "4000"))


def get_bedrock_client(region: str | None = None) -> Any:
    region = region or os.getenv("AWS_REGION", DEFAULT_REGION)
    return boto3.client("bedrock-runtime", region_name=region)


def check_model_access(model_id: str = MODEL_ID, region: str = DEFAULT_REGION) -> bool:
    """Verify access to the model."""
    try:
        bedrock = boto3.client("bedrock", region_name=region)
        response = bedrock.list_foundation_models()
        available_models = [m["modelId"] for m in response["modelSummaries"]]
        has_access = model_id in available_models
        if has_access:
            logger.info("Model %s is accessible", model_id)
        else:
            logger.error(
                "Model %s is not accessible. Available models: %d", model_id, len(available_models)
            )
        return has_access
    except Exception as e:
        logger.error("Error checking model access: %s", e)
        return False


def get_inference_profile_arn(model_id: str = MODEL_ID, region: str = DEFAULT_REGION) -> str | None:
    """Get inference profile ARN for a model."""
    try:
        sts = boto3.client("sts", region_name=region)
        account_id = sts.get_caller_identity()["Account"]
        profile_arn = f"arn:aws:bedrock:{region}:{account_id}:inference-profile/global.{model_id}"
        logger.info("Using inference profile: %s", profile_arn)
        return profile_arn
    except Exception as e:
        logger.warning("Could not get inference profile ARN: %s", e)
        return None


def _extract_memory_id(memory: dict) -> str | None:
    """Extract memory ID from response dict with various key formats."""
    for key in ["id", "memoryId", "memory_id", "Id"]:
        if key in memory:
            return memory[key]
    return None


def _find_memory_by_name(memories: list, memory_name: str) -> tuple[str | None, str | None]:
    """Find memory ID and status by name from list of memories."""
    for m in memories:
        m_name = m.get("name") or m.get("memoryName") or m.get("Name") or ""
        m_status = m.get("status") or m.get("Status") or ""
        if m_name == memory_name:
            return _extract_memory_id(m), m_status
        if memory_name in str(m.values()):
            return _extract_memory_id(m), m_status
    return None, None


def setup_agentcore_memory(
    region: str = DEFAULT_REGION,
    memory_name: str = "ArchReviewMemory",
    actor_id: str | None = None,
    session_id: str | None = None,
):
    """Setup AgentCore memory for agents."""
    import time

    client = MemoryClient(region_name=region)

    try:
        memories = client.list_memories()
        memory_id, memory_status = _find_memory_by_name(memories, memory_name)

        if memory_id:
            # Check if memory is active
            if memory_status and memory_status.upper() != "ACTIVE":
                logger.debug("Memory '%s' exists but status is %s", memory_name, memory_status)
                # Wait for it to become active (up to 60s)
                for _ in range(12):
                    time.sleep(5)
                    memories = client.list_memories()
                    memory_id, memory_status = _find_memory_by_name(memories, memory_name)
                    if memory_status and memory_status.upper() == "ACTIVE":
                        break
                    logger.debug("Waiting for memory to become active... (%s)", memory_status)

                if memory_status and memory_status.upper() != "ACTIVE":
                    logger.debug("Memory not active after waiting. Continuing without memory.")
                    return None, None

            logger.info("Using existing memory: %s", memory_name)
        else:
            memory = client.create_memory(
                name=memory_name, description="Memory for arch review agents"
            )
            memory_id = _extract_memory_id(memory)
            logger.info("Created memory: %s", memory_name)

            # Wait for memory to become active (up to 3 minutes)
            logger.info("Waiting for memory to initialize...")
            memory_status = "CREATING"
            for _ in range(36):  # 36 * 5s = 3 minutes max
                time.sleep(5)
                memories = client.list_memories()
                _, memory_status = _find_memory_by_name(memories, memory_name)
                status_upper = (memory_status or "").upper()

                if status_upper == "ACTIVE":
                    logger.info("Memory initialization done.")
                    break
                if status_upper == "FAILED":
                    logger.error("Memory creation failed.")
                    return None, None

                logger.debug("Memory status: %s", memory_status)
            else:
                logger.warning("Timeout waiting for memory. Status: %s", memory_status)
                return None, None

        if not memory_id:
            logger.warning("Could not resolve memory ID for '%s'. Skipping.", memory_name)
            return None, None

        actor_id = actor_id or f"actor_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        memory_config = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=session_id,
            actor_id=actor_id,
        )
        return memory_config, memory_id

    except Exception as e:
        logger.warning("Could not set up AgentCore Memory: %s", e)
        logger.warning("Continuing without memory.")
        return None, None


def create_session_manager(memory_config, actor_id: str | None = None):
    """Create a session manager for agent memory."""
    if not memory_config:
        return None

    if not actor_id:
        actor_id = getattr(
            memory_config, "actor_id", f"actor_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

    return AgentCoreMemorySessionManager(agentcore_memory_config=memory_config, actor_id=actor_id)


def setup_policy_engine(
    region: str = DEFAULT_REGION, policy_engine_name: str = "ArchReviewPolicyEngine"
):
    """Create or retrieve a Policy Engine."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        engines = client.list_policy_engines()
        engine = next(
            (e for e in engines.get("policyEngines", []) if e.get("name") == policy_engine_name),
            None,
        )

        if engine:
            engine_id = engine.get("policyEngineId")
            logger.info("Using existing Policy Engine: %s (ID: %s)", policy_engine_name, engine_id)
        else:
            response = client.create_policy_engine(
                name=policy_engine_name, description="Policy engine for architecture review agents"
            )
            engine_id = response.get("policyEngineId")
            logger.info("Created Policy Engine: %s (ID: %s)", policy_engine_name, engine_id)

        return engine_id

    except Exception as e:
        logger.warning("Could not set up Policy Engine: %s", e)
        logger.warning("Continuing without policy controls.")
        return None


def _wait_for_policy_active(client, policy_engine_id: str, policy_id: str) -> bool:
    """Wait for policy to become ACTIVE."""
    import time

    print("  Verifying status...", end="", flush=True)
    for _ in range(120):  # 120 * 0.5s = 60s max
        try:
            response = client.get_policy(policyEngineId=policy_engine_id, policyId=policy_id)
            # Check possible keys for status since GetPolicy response structure varies
            policy_data = response.get("policy") or response
            status = (policy_data.get("status") or policy_data.get("policyStatus") or "").upper()

            if status == "ACTIVE":
                print(" OK!")
                return True
            if "FAIL" in status:
                print(f"\n  ❌ Failed. Status: {status}")
                if "failureReason" in policy_data:
                    print(f"  Reason: {policy_data['failureReason']}")
                return False

            # Print status only if not just updating/creating to avoid noise
            if status not in ["UPDATING", "CREATING"]:
                print(f"[{status}]", end="", flush=True)
            else:
                print(".", end="", flush=True)

            time.sleep(0.5)
        except Exception:
            print("!", end="", flush=True)  # Show errors
            pass

    print(" (Timeout checking status)")
    return False  # Fail if we time out and didn't get ACTIVE


def _normalize_cedar(text: str) -> str:
    """Normalize cedar policy text for comparison."""
    return " ".join(text.split())


def create_policy(
    policy_engine_id: str,
    policy_name: str,
    cedar_statement: str,
    description: str = "",
    region: str = DEFAULT_REGION,
):
    """Create a Cedar policy in a Policy Engine, or use existing one."""
    logger.info("Checking Policy: %s...", policy_name)
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        response = client.create_policy(
            policyEngineId=policy_engine_id,
            name=policy_name,
            definition={"cedar": {"statement": cedar_statement}},
            description=description or f"Policy for {policy_name}",
            validationMode="FAIL_ON_ANY_FINDINGS",
        )
        policy_id = response.get("policyId")
        logger.info("Created policy %s (ID: %s)", policy_name, policy_id)
        # Verify policy becomes ACTIVE before returning
        if not _wait_for_policy_active(client, policy_engine_id, policy_id):
            logger.error("Policy %s failed to become ACTIVE", policy_name)
            return None
        return policy_id
    except Exception as e:
        error_msg = str(e).lower()
        # Handle "already exists" as success
        if "already exists" in error_msg or "conflictexception" in error_msg:
            logger.info("Policy '%s' exists. Updating...", policy_name)
            try:
                # Find policy ID with pagination
                policy_id = None
                next_token = None

                while True:
                    kwargs = {"policyEngineId": policy_engine_id}
                    if next_token:
                        kwargs["nextToken"] = next_token

                    response = client.list_policies(**kwargs)
                    # Try common keys for list response
                    policies = (
                        response.get("items")
                        or response.get("policies")
                        or response.get("policySummaries")
                        or []
                    )

                    existing_policy = next(
                        (p for p in policies if p.get("name") == policy_name), None
                    )

                    if existing_policy:
                        policy_id = existing_policy.get("policyId") or existing_policy.get("id")
                        break

                    next_token = response.get("nextToken")
                    if not next_token:
                        break

                if policy_id:
                    print("Updating...", end=" ", flush=True)
                    # Update policy
                    client.update_policy(
                        policyEngineId=policy_engine_id,
                        policyId=policy_id,
                        definition={"cedar": {"statement": cedar_statement}},
                        description=description or f"Policy for {policy_name}",
                        validationMode="FAIL_ON_ANY_FINDINGS",
                    )
                    logger.info("Updated policy %s (ID: %s)", policy_name, policy_id)
                    # Verify policy becomes ACTIVE before returning
                    if not _wait_for_policy_active(client, policy_engine_id, policy_id):
                        logger.error("Policy %s failed to become ACTIVE after update", policy_name)
                        return None
                    return policy_id
                else:
                    logger.error("Could not find existing policy ID for '%s'", policy_name)
                    return None
            except Exception as update_error:
                logger.error("Error updating policy: %s", update_error)
                return None

        logger.error("Error creating policy %s: %s", policy_name, e)
        return None


def setup_online_evaluation(
    region: str = DEFAULT_REGION, evaluation_name: str = "ArchReviewEvaluation"
):
    """Create or retrieve an Online Evaluation configuration."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        evaluations = client.list_online_evaluation_configs()
        evaluation = next(
            (
                e
                for e in evaluations.get("onlineEvaluationConfigs", [])
                if e.get("name") == evaluation_name
            ),
            None,
        )

        if evaluation:
            evaluation_id = evaluation.get("onlineEvaluationConfigId")
            logger.info(
                "Using existing Online Evaluation: %s (ID: %s)", evaluation_name, evaluation_id
            )
        else:
            response = client.create_online_evaluation_config(
                name=evaluation_name,
                description="Quality evaluation for architecture review agents",
            )
            evaluation_id = response.get("onlineEvaluationConfigId")
            logger.info("Created Online Evaluation: %s (ID: %s)", evaluation_name, evaluation_id)

        return evaluation_id

    except Exception as e:
        logger.warning("Could not set up Online Evaluation: %s", e)
        logger.warning("Continuing without quality evaluations.")
        return None


def setup_architecture_review_policies(
    region: str = DEFAULT_REGION,
    policy_engine_name: str = "ArchReviewPolicyEngine",
    gateway_arn: str | None = None,
    gateway_name: str = "ArchReviewGateway",
):
    """Set up Cedar policies for agent tool restrictions."""
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

    # Default deny: only registered agents are allowed
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
    )
    if policy_id:
        policies_created.append("DefaultDenyUnknownAgents")

    if policies_created:
        logger.info("Verified %d policies:", len(policies_created))
        for policy_name in policies_created:
            logger.debug("  - %s", policy_name)

        if gateway_id:
            logger.info("Associating Gateway with Policy Engine...")
            associate_gateway_with_policy_engine(
                gateway_id=gateway_id,
                policy_engine_id=engine_id,
                enforcement_mode="ENFORCE",
                region=region,
            )
        else:
            if gateway_arn and "/gateway/" in gateway_arn:
                extracted_id = gateway_arn.split("/gateway/")[-1]
                logger.info("Associating Gateway with Policy Engine...")
                associate_gateway_with_policy_engine(
                    gateway_id=extracted_id,
                    policy_engine_id=engine_id,
                    enforcement_mode="ENFORCE",
                    region=region,
                )

        return engine_id
    else:
        logger.warning("No policies were created.")
        return None


def associate_gateway_with_policy_engine(
    gateway_id: str,
    policy_engine_id: str,
    enforcement_mode: str = "ENFORCE",
    region: str = DEFAULT_REGION,
):
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
    except Exception as e:
        logger.warning("Could not associate Gateway with Policy Engine: %s", e)
        logger.warning("You may need to associate them manually via the AWS Console.")
        return False


def list_gateways(region: str = DEFAULT_REGION):
    """List available Gateways using boto3 directly."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        response = client.list_gateways()
        return response.get("items", [])
    except Exception as e:
        logger.warning("Could not list Gateways: %s", e)
        return []


def _find_gateway_by_name(gateway_name: str, region: str = DEFAULT_REGION):
    """Find an existing gateway by name (case-insensitive)."""
    gateways = list_gateways(region=region)
    gateway_name_lower = gateway_name.lower()

    for gw in gateways:
        if isinstance(gw, str):
            continue

        gw_name = gw.get("name") or gw.get("gatewayName") or gw.get("Name") or ""
        gw_id = gw.get("gatewayId") or gw.get("id") or ""

        # Match by name or ID prefix (API lowercases names in IDs)
        if gw_name.lower() == gateway_name_lower or gw_id.lower().startswith(
            gateway_name_lower.replace(" ", "")
        ):
            gateway_arn = gw.get("gatewayArn") or gw.get("arn")
            gateway_url = gw.get("gatewayUrl") or gw.get("url")

            if not gateway_arn and gw_id:
                sts = boto3.client("sts", region_name=region)
                account_id = sts.get_caller_identity()["Account"]
                gateway_arn = f"arn:aws:bedrock-agentcore:{region}:{account_id}:gateway/{gw_id}"

            return gateway_arn, gw_id, gateway_url

    return None, None, None


def setup_gateway(
    region: str = DEFAULT_REGION,
    gateway_name: str = "ArchReviewGateway",
):
    """Create or retrieve a Gateway for policy enforcement."""
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
                raise ValueError(f"Unexpected cognito response: {cognito_response[:100]}")

            authorizer_config = cognito_response.get("authorizer_config") or cognito_response.get(
                "authorizerConfig"
            )
            if not authorizer_config:
                raise ValueError(
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

        except Exception as create_error:
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

        import time

        logger.debug("Waiting for IAM propagation...")
        for _ in range(30):
            time.sleep(1)
        logger.debug("IAM propagation wait complete.")

        gateway_id = gateway.get("gatewayId") or gateway.get("id")
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
    except Exception as e:
        import traceback

        logger.warning("Could not set up Gateway: %s", e)
        logger.debug("Details: %s", traceback.format_exc())
        logger.warning("Continuing without Gateway. Policies will not be created.")
        return None, None


def _save_gateway_config(gateway: dict, cognito_response: dict, region: str):
    """Save gateway config to ~/.arch-review for cleanup."""
    import json
    from pathlib import Path

    config = {
        "gateway_url": gateway.get("gatewayUrl"),
        "gateway_id": gateway.get("gatewayId"),
        "region": region,
        "client_info": cognito_response.get("client_info"),
    }

    config_path = Path.home() / ".arch-review" / "gateway_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))


def cleanup_gateway(region: str = DEFAULT_REGION):
    """Clean up Gateway and Cognito resources."""
    import json
    from pathlib import Path

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
    except Exception as e:
        logger.warning("Could not clean up Gateway: %s", e)
