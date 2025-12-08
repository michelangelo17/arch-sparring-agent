import os
from datetime import datetime
from typing import Any

import boto3
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)

MODEL_ID = "amazon.nova-2-lite-v1:0"
DEFAULT_REGION = "eu-central-1"


def get_bedrock_client(region: str | None = None) -> Any:
    region = region or os.getenv("AWS_REGION", DEFAULT_REGION)
    return boto3.client("bedrock-runtime", region_name=region)


def check_model_access(model_id: str = MODEL_ID) -> bool:
    """Verify access to the model."""
    try:
        bedrock = boto3.client("bedrock", region_name=DEFAULT_REGION)
        response = bedrock.list_foundation_models()
        available_models = [m["modelId"] for m in response["modelSummaries"]]
        has_access = model_id in available_models
        if has_access:
            print(f"✓ Model {model_id} is accessible")
        else:
            print(
                f"✗ Model {model_id} is not accessible. Available models: {len(available_models)}"
            )
        return has_access
    except Exception as e:
        print(f"Error checking model access: {e}")
        return False


def get_inference_profile_arn(model_id: str = MODEL_ID, region: str = DEFAULT_REGION) -> str | None:
    """Get inference profile ARN for a model."""
    try:
        sts = boto3.client("sts", region_name=region)
        account_id = sts.get_caller_identity()["Account"]
        profile_arn = f"arn:aws:bedrock:{region}:{account_id}:inference-profile/global.{model_id}"
        print(f"✓ Using inference profile: {profile_arn}")
        return profile_arn
    except Exception as e:
        print(f"Warning: Could not get inference profile ARN: {e}")
        return None


def _extract_memory_id(memory: dict) -> str | None:
    """Extract memory ID from response dict with various key formats."""
    for key in ["id", "memoryId", "memory_id", "Id"]:
        if key in memory:
            return memory[key]
    return None


def _find_memory_by_name(memories: list, memory_name: str) -> str | None:
    """Find memory ID by name from list of memories."""
    for m in memories:
        m_name = m.get("name") or m.get("memoryName") or m.get("Name") or ""
        if m_name == memory_name:
            return _extract_memory_id(m)
        # Fallback: check if name appears in values
        if memory_name in str(m.values()):
            return _extract_memory_id(m)
    return None


def setup_agentcore_memory(
    region: str = DEFAULT_REGION,
    memory_name: str = "ArchReviewMemory",
    actor_id: str | None = None,
    session_id: str | None = None,
):
    """Setup AgentCore memory for agents."""
    client = MemoryClient(region_name=region)

    try:
        memories = client.list_memories()
        memory_id = _find_memory_by_name(memories, memory_name)

        if memory_id:
            print(f"✓ Using existing memory: {memory_name}")
        else:
            memory = client.create_memory(
                name=memory_name, description="Memory for arch review agents"
            )
            memory_id = _extract_memory_id(memory)
            print(f"✓ Created memory: {memory_name}")

        if not memory_id:
            print(f"Could not resolve memory ID for '{memory_name}'. Skipping.")
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
        print(f"Warning: Could not set up AgentCore Memory: {e}")
        print("Continuing without memory.")
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
            print(f"✓ Using existing Policy Engine: {policy_engine_name} (ID: {engine_id})")
        else:
            response = client.create_policy_engine(
                name=policy_engine_name, description="Policy engine for architecture review agents"
            )
            engine_id = response.get("policyEngineId")
            print(f"✓ Created Policy Engine: {policy_engine_name} (ID: {engine_id})")

        return engine_id

    except Exception as e:
        print(f"Warning: Could not set up Policy Engine: {e}")
        print("Continuing without policy controls.")
        return None


def create_policy(
    policy_engine_id: str,
    policy_name: str,
    cedar_statement: str,
    description: str = "",
    region: str = DEFAULT_REGION,
):
    """Create a Cedar policy in a Policy Engine, or use existing one."""
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
        print(f"✓ Created Policy: {policy_name} (ID: {policy_id})")
        return policy_id
    except Exception as e:
        # Handle "already exists" as success
        if "already exists" in str(e).lower():
            print(f"✓ Using existing Policy: {policy_name}")
            return policy_name  # Return name as ID placeholder
        print(f"Error creating policy {policy_name}: {e}")
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
            print(f"✓ Using existing Online Evaluation: {evaluation_name} (ID: {evaluation_id})")
        else:
            response = client.create_online_evaluation_config(
                name=evaluation_name,
                description="Quality evaluation for architecture review agents",
            )
            evaluation_id = response.get("onlineEvaluationConfigId")
            print(f"✓ Created Online Evaluation: {evaluation_name} (ID: {evaluation_id})")

        return evaluation_id

    except Exception as e:
        print(f"Warning: Could not set up Online Evaluation: {e}")
        print("Continuing without quality evaluations.")
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
            print("⚠️  Could not set up Gateway. Policies cannot be created without a Gateway.")
            return None

    engine_id = setup_policy_engine(region=region, policy_engine_name=policy_engine_name)
    if not engine_id:
        return None

    policies_created = []

    # RequirementsAnalyst: document and user interaction tools only
    requirements_cedar = f"""permit(
    principal is AgentCore::OAuthUser,
    action in [
        AgentCore::Action::"read_document",
        AgentCore::Action::"list_available_documents",
        AgentCore::Action::"ask_user_question"
    ],
    resource == AgentCore::Gateway::"{gateway_arn}"
) when {{
    context.input has agentName && context.input.agentName == "RequirementsAnalyst"
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
    action in [
        AgentCore::Action::"read_cloudformation_template",
        AgentCore::Action::"list_cloudformation_templates",
        AgentCore::Action::"read_architecture_diagram",
        AgentCore::Action::"list_architecture_diagrams",
        AgentCore::Action::"ask_user_question"
    ],
    resource == AgentCore::Gateway::"{gateway_arn}"
) when {{
    context.input has agentName && context.input.agentName == "ArchitectureEvaluator"
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
    action in [
        AgentCore::Action::"get_requirements_analysis",
        AgentCore::Action::"get_architecture_analysis"
    ],
    resource == AgentCore::Gateway::"{gateway_arn}"
) when {{
    context.input has agentName && context.input.agentName == "ReviewModerator"
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
    context.input has agentName &&
    (context.input.agentName == "RequirementsAnalyst" ||
     context.input.agentName == "ArchitectureEvaluator" ||
     context.input.agentName == "ReviewModerator" ||
     context.input.agentName == "QuestionAgent" ||
     context.input.agentName == "SparringAgent" ||
     context.input.agentName == "ReviewAgent")
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
        print(f"\n✓ Created {len(policies_created)} policies:")
        for policy_name in policies_created:
            print(f"  - {policy_name}")

        if gateway_id:
            print("\nAssociating Gateway with Policy Engine...")
            associate_gateway_with_policy_engine(
                gateway_id=gateway_id,
                policy_engine_id=engine_id,
                enforcement_mode="ENFORCE",
                region=region,
            )
        else:
            if gateway_arn and "/gateway/" in gateway_arn:
                extracted_id = gateway_arn.split("/gateway/")[-1]
                print("\nAssociating Gateway with Policy Engine...")
                associate_gateway_with_policy_engine(
                    gateway_id=extracted_id,
                    policy_engine_id=engine_id,
                    enforcement_mode="ENFORCE",
                    region=region,
                )

        return engine_id
    else:
        print("Warning: No policies were created.")
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
        print("✓ Associated Policy Engine with Gateway")
        print(f"  Gateway ID: {gateway_id}")
        print(f"  Policy Engine ARN: {policy_engine_arn}")
        print(f"  Enforcement mode: {enforcement_mode}")
        return True
    except Exception as e:
        print(f"Warning: Could not associate Gateway with Policy Engine: {e}")
        print("You may need to associate them manually via the AWS Console.")
        return False


def list_gateways(region: str = DEFAULT_REGION):
    """List available Gateways using boto3 directly."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        response = client.list_gateways()
        # API returns 'items' not 'gateways'
        return response.get("items", [])
    except Exception as e:
        print(f"Warning: Could not list Gateways: {e}")
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
            print(f"✓ Using existing Gateway: {gateway_name}")
            if gateway_url:
                print(f"  Gateway URL: {gateway_url}")
            print(f"  Gateway ID: {gateway_id}")
            return gateway_arn, gateway_id

        print(f"Creating Gateway: {gateway_name}...")
        client = GatewayClient(region_name=region)

        try:
            print("  Creating OAuth authorization server...")
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
            print("  ✓ Authorization server created")

            print("  Creating MCP Gateway...")
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
                    print(f"✓ Using existing Gateway: {gateway_name}")
                    if gateway_url:
                        print(f"  Gateway URL: {gateway_url}")
                    print(f"  Gateway ID: {gateway_id}")
                    return gateway_arn, gateway_id
            raise

        if isinstance(gateway, str):
            gateway = {"gatewayId": gateway}

        print("  ✓ Gateway created")
        print("  Configuring IAM permissions...")
        client.fix_iam_permissions(gateway)

        import time

        print("  Waiting for IAM propagation (30s)...", end="", flush=True)
        for i in range(30):
            time.sleep(1)
            if i % 5 == 0:
                print(".", end="", flush=True)
        print(" Done!")

        gateway_id = gateway.get("gatewayId") or gateway.get("id")
        gateway_url = gateway.get("gatewayUrl") or gateway.get("url")
        gateway_arn = gateway.get("gatewayArn")

        if not gateway_arn and gateway_id:
            sts = boto3.client("sts", region_name=region)
            account_id = sts.get_caller_identity()["Account"]
            gateway_arn = f"arn:aws:bedrock-agentcore:{region}:{account_id}:gateway/{gateway_id}"

        print(f"✓ Gateway setup complete: {gateway_name}")
        if gateway_url:
            print(f"  Gateway URL: {gateway_url}")
        print(f"  Gateway ID: {gateway_id}")

        _save_gateway_config(gateway, cognito_response, region)
        return gateway_arn, gateway_id

    except ImportError:
        print("Warning: bedrock-agentcore-starter-toolkit not installed.")
        print("Run: pip install bedrock-agentcore-starter-toolkit")
        return None, None
    except Exception as e:
        import traceback

        print(f"Warning: Could not set up Gateway: {e}")
        print(f"Details: {traceback.format_exc()}")
        print("Continuing without Gateway. Policies will not be created.")
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
        print("No gateway config found. Nothing to clean up.")
        return

    try:
        from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

        config = json.loads(config_path.read_text())
        client = GatewayClient(region_name=config.get("region", region))
        client.cleanup_gateway(config["gateway_id"], config.get("client_info"))
        print("✓ Gateway cleanup complete!")
        config_path.unlink()
    except Exception as e:
        print(f"Warning: Could not clean up Gateway: {e}")
