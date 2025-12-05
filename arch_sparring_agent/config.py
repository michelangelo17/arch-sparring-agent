import os
from datetime import datetime
from typing import Any

import boto3
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
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
    """Get inference profile ARN for Nova 2 Lite."""
    try:
        # Get AWS account ID
        sts = boto3.client("sts", region_name=region)
        account_id = sts.get_caller_identity()["Account"]

        # Construct inference profile ARN
        # Format: arn:aws:bedrock:REGION:ACCOUNT_ID:inference-profile/global.MODEL_ID
        profile_arn = f"arn:aws:bedrock:{region}:{account_id}:inference-profile/global.{model_id}"
        print(f"✓ Using inference profile: {profile_arn}")
        return profile_arn
    except Exception as e:
        print(f"Warning: Could not get inference profile ARN: {e}")
        return None


def setup_agentcore_memory(
    region: str = DEFAULT_REGION,
    memory_name: str = "ArchReviewMemory",
    actor_id: str | None = None,
    session_id: str | None = None,
):
    """Setup AgentCore memory for agents."""

    client = MemoryClient(region_name=region)
    memory_id = None

    try:
        # List memories and find by name
        memories = client.list_memories()

        # Try different possible field names for the memory name
        for m in memories:
            name = m.get("name") or m.get("memoryName") or m.get("Name")
            if name == memory_name:
                memory_id = m.get("id") or m.get("memoryId") or m.get("memory_id")
                break

        if not memory_id:
            # Create new memory if it doesn't exist
            try:
                memory = client.create_memory(
                    name=memory_name, description="Memory for arch review agents"
                )
                memory_id = memory.get("id") or memory.get("memoryId") or memory.get("memory_id")
            except Exception as create_error:
                if "already exists" in str(create_error).lower():
                    # Memory exists but we couldn't find it - skip memory for now
                    print(f"Memory '{memory_name}' exists but couldn't be listed. Skipping.")
                    return None, None
                raise

        if not memory_id:
            print("Could not get memory ID. Skipping memory.")
            return None, None

        # Generate IDs if not provided
        if not actor_id:
            actor_id = f"actor_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Create memory config with required fields
        retrieval_config = RetrievalConfig()
        retrieval_config_dict = retrieval_config.model_dump()

        memory_config = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=session_id,
            actor_id=actor_id,
            retrieval_config=retrieval_config_dict,
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

    # Use actor_id from memory_config if not provided
    if not actor_id:
        actor_id = (
            memory_config.actor_id
            if hasattr(memory_config, "actor_id")
            else f"actor_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

    session_manager = AgentCoreMemorySessionManager(memory_config=memory_config, actor_id=actor_id)

    return session_manager


def setup_policy_engine(
    region: str = DEFAULT_REGION, policy_engine_name: str = "ArchReviewPolicyEngine"
):
    """Create or retrieve a Policy Engine."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)

        # Check if policy engine already exists
        engines = client.list_policy_engines()
        engine = next(
            (e for e in engines.get("policyEngines", []) if e.get("name") == policy_engine_name),
            None,
        )

        if not engine:
            # Create new policy engine
            response = client.create_policy_engine(
                name=policy_engine_name, description="Policy engine for architecture review agents"
            )
            engine_id = response.get("policyEngineId")
            print(f"✓ Created Policy Engine: {policy_engine_name} (ID: {engine_id})")
            return engine_id
        else:
            engine_id = engine.get("policyEngineId")
            print(f"✓ Using existing Policy Engine: {policy_engine_name} (ID: {engine_id})")
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
    """Create a Cedar policy in a Policy Engine."""
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
        print(f"Error creating policy {policy_name}: {e}")
        return None


def setup_online_evaluation(
    region: str = DEFAULT_REGION, evaluation_name: str = "ArchReviewEvaluation"
):
    """Create or retrieve an Online Evaluation configuration."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)

        # Check if evaluation already exists
        evaluations = client.list_online_evaluation_configs()
        evaluation = next(
            (
                e
                for e in evaluations.get("onlineEvaluationConfigs", [])
                if e.get("name") == evaluation_name
            ),
            None,
        )

        if not evaluation:
            # Create new evaluation config
            response = client.create_online_evaluation_config(
                name=evaluation_name,
                description="Quality evaluation for architecture review agents",
            )
            evaluation_id = response.get("onlineEvaluationConfigId")
            print(f"✓ Created Online Evaluation: {evaluation_name} (ID: {evaluation_id})")
            return evaluation_id
        else:
            evaluation_id = evaluation.get("onlineEvaluationConfigId")
            print(f"✓ Using existing Online Evaluation: {evaluation_name} (ID: {evaluation_id})")
            return evaluation_id

    except Exception as e:
        print(f"Warning: Could not set up Online Evaluation: {e}")
        print("Continuing without quality evaluations.")
        return None


def setup_architecture_review_policies(
    region: str = DEFAULT_REGION, policy_engine_name: str = "ArchReviewPolicyEngine"
):
    """Set up Cedar policies for agent tool restrictions."""
    engine_id = setup_policy_engine(region=region, policy_engine_name=policy_engine_name)
    if not engine_id:
        return None

    policies_created = []

    # Requirements Agent: document and user tools only
    requirements_cedar = """permit(
    principal == Agent::"RequirementsAnalyst",
    action == Action::"ToolInvoke",
    resource
) when {
    resource.toolName in ["read_document", "list_available_documents", "ask_user_question"]
};

forbid(
    principal == Agent::"RequirementsAnalyst",
    action == Action::"ToolInvoke",
    resource
) when {
    resource.toolName not in ["read_document", "list_available_documents", "ask_user_question"]
};"""

    policy_id = create_policy(
        engine_id,
        "RequirementsAgentToolRestrictions",
        requirements_cedar,
        "Restricts Requirements Agent to only use document reading and user interaction tools",
        region=region,
    )
    if policy_id:
        policies_created.append("RequirementsAgentToolRestrictions")

    # Architecture Agent: CFN/diagram tools only
    architecture_cedar = """permit(
    principal == Agent::"ArchitectureEvaluator",
    action == Action::"ToolInvoke",
    resource
) when {
    resource.toolName in [
        "read_cloudformation_template",
        "list_cloudformation_templates",
        "read_architecture_diagram",
        "list_architecture_diagrams",
        "ask_user_question"
    ]
};

forbid(
    principal == Agent::"ArchitectureEvaluator",
    action == Action::"ToolInvoke",
    resource
) when {
    resource.toolName not in [
        "read_cloudformation_template",
        "list_cloudformation_templates",
        "read_architecture_diagram",
        "list_architecture_diagrams",
        "ask_user_question"
    ]
};"""

    policy_id = create_policy(
        engine_id,
        "ArchitectureAgentToolRestrictions",
        architecture_cedar,
        "Restricts Architecture Agent to only use CFN/diagram reading and user interaction tools",
        region=region,
    )
    if policy_id:
        policies_created.append("ArchitectureAgentToolRestrictions")

    # Moderator Agent: agent communication tools only
    moderator_cedar = """permit(
    principal == Agent::"ReviewModerator",
    action == Action::"ToolInvoke",
    resource
) when {
    resource.toolName in ["get_requirements_analysis", "get_architecture_analysis"]
};

forbid(
    principal == Agent::"ReviewModerator",
    action == Action::"ToolInvoke",
    resource
) when {
    resource.toolName not in ["get_requirements_analysis", "get_architecture_analysis"]
};"""

    policy_id = create_policy(
        engine_id,
        "ModeratorAgentToolRestrictions",
        moderator_cedar,
        "Restricts Moderator Agent to only use agent-to-agent communication tools",
        region=region,
    )
    if policy_id:
        policies_created.append("ModeratorAgentToolRestrictions")

    # Rate limiting
    # Note: Rate limiting may need to be implemented differently depending on Gateway capabilities
    rate_limit_cedar = """forbid(
    principal,
    action == Action::"BedrockInvoke",
    resource
) when {
    // Rate limiting: max 100 requests per minute
    // Note: This requires Gateway to track request counts - adjust based on actual Gateway schema
    context.requestCount > 100 &&
    context.timeWindow == 60
};"""

    policy_id = create_policy(
        engine_id,
        "RateLimiting",
        rate_limit_cedar,
        "Limits API calls to prevent excessive costs (max 100 requests per minute)",
        region=region,
    )
    if policy_id:
        policies_created.append("RateLimiting")

    if policies_created:
        print(f"\n✓ Created {len(policies_created)} policies:")
        for policy_name in policies_created:
            print(f"  - {policy_name}")
        print("\n⚠️  Note: Cedar policies are validated against the Gateway's tool schema.")
        print("   If policy creation fails, check that tool names match the Gateway configuration.")
        print("   Policy statements may need adjustment based on the actual Gateway schema.")
        return engine_id
    else:
        print("Warning: No policies were created.")
        return None
