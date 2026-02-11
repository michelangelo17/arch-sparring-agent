"""Configuration constants and basic AWS client setup."""

import logging
import os
import re
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from strands.models import BedrockModel

logger = logging.getLogger(__name__)

MODEL_ID = "amazon.nova-2-lite-v1:0"
DEFAULT_REGION = "eu-central-1"


REASONING_LEVELS = ("off", "low", "medium", "high")
DEFAULT_REASONING_LEVEL = "low"

# Claude budgetTokens mapping for reasoning levels
_CLAUDE_BUDGET_TOKENS = {"low": 1024, "medium": 4096, "high": 16384}


def _detect_model_family(model_id: str) -> str:
    """Detect model family from a model ID, inference profile, or ARN.

    Handles formats:
        "amazon.nova-2-lite-v1:0"                              -> "nova"
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"         -> "claude"
        "eu.mistral.pixtral-large-2502-v1:0"                   -> "mistral"
        "arn:aws:bedrock:...:inference-profile/global.amazon.nova-2-lite-v1:0"
                                                               -> "nova"
    """
    name = model_id.lower()

    # Strip full ARN prefix: everything up to and including "inference-profile/"
    arn_match = re.search(r"inference-profile/(.+)", name)
    if arn_match:
        name = arn_match.group(1)

    # Strip region prefix (eu., us., ap-*, global.)
    name = re.sub(r"^(eu|us|ap-\w+|global)\.", "", name)

    if name.startswith("amazon.nova"):
        return "nova"
    if name.startswith("anthropic.claude"):
        return "claude"
    if name.startswith("mistral."):
        return "mistral"
    return "unknown"


def create_model(
    model_id: str = MODEL_ID,
    reasoning: bool = False,
    reasoning_level: str = DEFAULT_REASONING_LEVEL,
) -> BedrockModel:
    """Create a BedrockModel, optionally with extended thinking enabled.

    Args:
        model_id: Bedrock model ID or inference profile ARN.
        reasoning: Enable extended thinking (reasoningConfig) for complex analysis.
        reasoning_level: Reasoning effort level: "off", "low", "medium", or "high".
    """
    kwargs: dict[str, Any] = {"model_id": model_id}

    if reasoning and reasoning_level != "off":
        family = _detect_model_family(model_id)
        level = (
            reasoning_level
            if reasoning_level in ("low", "medium", "high")
            else DEFAULT_REASONING_LEVEL
        )

        if family == "nova":
            kwargs["additional_request_fields"] = {
                "reasoningConfig": {
                    "type": "enabled",
                    "maxReasoningEffort": level,
                }
            }
        elif family == "claude":
            kwargs["additional_request_fields"] = {
                "reasoningConfig": {
                    "type": "enabled",
                    "budgetTokens": _CLAUDE_BUDGET_TOKENS[level],
                }
            }
        else:
            logger.warning(
                "Model family '%s' does not support extended thinking. "
                "Running %s without reasoning.",
                family,
                model_id,
            )

    return BedrockModel(**kwargs)


def _int_env(var: str, default: int) -> int:
    """Read an integer from an environment variable with validation."""
    raw = os.getenv(var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s='%s', using default %d", var, raw, default)
        return default


# --- Tuning constants (override via environment variables) ---

# Context condenser: skip extraction for content shorter than this (chars)
CONDENSER_PASSTHROUGH_THRESHOLD = _int_env("ARCH_REVIEW_PASSTHROUGH_THRESHOLD", 2000)

# Context condenser: chunk size for fallback chunked extraction (chars)
CONDENSER_CHUNK_SIZE = _int_env("ARCH_REVIEW_CHUNK_SIZE", 8000)

# Context condenser: max chunks to process in fallback mode
CONDENSER_MAX_CHUNKS = _int_env("ARCH_REVIEW_MAX_CHUNKS", 5)

# Requirements agent: summarize documents longer than this (chars, ~6k tokens)
DOC_SUMMARY_THRESHOLD = _int_env("ARCH_REVIEW_DOC_SUMMARY_THRESHOLD", 25000)

# Requirements agent: use chunked summarization for documents longer than this (chars)
DOC_CHUNK_SUMMARY_THRESHOLD = _int_env("ARCH_REVIEW_DOC_CHUNK_THRESHOLD", 100000)

# Source analyzer: max source file size (chars)
SOURCE_FILE_MAX_CHARS = _int_env("ARCH_REVIEW_SOURCE_MAX_CHARS", 50000)

# Diagram analyzer: max tokens for diagram analysis response
DIAGRAM_MAX_TOKENS = _int_env("ARCH_REVIEW_DIAGRAM_MAX_TOKENS", 4000)

# Gateway: max seconds to wait for IAM propagation after gateway creation
IAM_PROPAGATION_TIMEOUT = _int_env("ARCH_REVIEW_IAM_WAIT_TIMEOUT", 60)

# File size limits for tools
CFN_MAX_CHARS = _int_env("ARCH_REVIEW_CFN_MAX_CHARS", 500_000)
DOC_MAX_CHARS = _int_env("ARCH_REVIEW_DOC_MAX_CHARS", 500_000)
DIAGRAM_MAX_BYTES = _int_env("ARCH_REVIEW_DIAGRAM_MAX_BYTES", 10_000_000)


def get_bedrock_client(region: str | None = None) -> Any:
    """Create a Bedrock Runtime client."""
    region = region or os.getenv("AWS_REGION", DEFAULT_REGION)
    return boto3.client("bedrock-runtime", region_name=region)


def _strip_profile_prefix(model_id: str) -> str:
    """Strip region/inference profile prefix from a model ID.

    "eu.anthropic.claude-sonnet-4-5-v1:0" -> "anthropic.claude-sonnet-4-5-v1:0"
    "global.amazon.nova-2-lite-v1:0"      -> "amazon.nova-2-lite-v1:0"
    "amazon.nova-2-lite-v1:0"             -> "amazon.nova-2-lite-v1:0"
    """
    return re.sub(r"^(eu|us|ap-\w+|global)\.", "", model_id)


def _is_inference_profile(model_id: str) -> bool:
    """Check if a model ID is already a region-prefixed inference profile."""
    return bool(re.match(r"^(eu|us|ap-\w+|global)\.", model_id))


def check_model_access(model_id: str = MODEL_ID, region: str = DEFAULT_REGION) -> bool:
    """Verify access to the specified Bedrock model.

    Handles both base model IDs and region-prefixed inference profiles
    by stripping prefixes before comparing.
    """
    base_id = _strip_profile_prefix(model_id)
    try:
        bedrock = boto3.client("bedrock", region_name=region)
        response = bedrock.list_foundation_models()
        available_models = [m["modelId"] for m in response["modelSummaries"]]

        # Exact match first, then prefix match (handles version suffix differences)
        has_access = base_id in available_models or any(
            base_id.split(":")[0] in m for m in available_models
        )
        if has_access:
            logger.info("Model %s is accessible", model_id)
        else:
            logger.error(
                "Model %s is not accessible. Available models: %d",
                model_id,
                len(available_models),
            )
        return has_access
    except (ClientError, BotoCoreError) as e:
        logger.error("Error checking model access: %s", e)
        return False


def get_inference_profile_arn(model_id: str = MODEL_ID, region: str = DEFAULT_REGION) -> str | None:
    """Get inference profile ARN for a model.

    If the model ID is already a region-prefixed inference profile
    (e.g. "eu.anthropic.claude-*"), returns it as-is.
    For base model IDs, constructs a global inference profile ARN.
    """
    # Region-prefixed IDs are already inference profiles
    if _is_inference_profile(model_id):
        logger.info("Using inference profile: %s", model_id)
        return model_id

    try:
        sts = boto3.client("sts", region_name=region)
        account_id = sts.get_caller_identity()["Account"]
        profile_arn = f"arn:aws:bedrock:{region}:{account_id}:inference-profile/global.{model_id}"
        logger.info("Using inference profile: %s", profile_arn)
        return profile_arn
    except (ClientError, BotoCoreError) as e:
        logger.warning("Could not get inference profile ARN: %s", e)
        return None
