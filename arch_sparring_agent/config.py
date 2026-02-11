"""Configuration constants and basic AWS client setup."""

import logging
import os
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from strands.models import BedrockModel

logger = logging.getLogger(__name__)

MODEL_ID = "amazon.nova-2-lite-v1:0"
DEFAULT_REGION = "eu-central-1"


def create_model(model_id: str = MODEL_ID, reasoning: bool = False) -> BedrockModel:
    """Create a BedrockModel, optionally with extended thinking enabled.

    Args:
        model_id: Bedrock model ID or inference profile ARN.
        reasoning: Enable extended thinking (reasoningConfig) for complex analysis.
    """
    kwargs: dict[str, Any] = {"model_id": model_id}
    if reasoning:
        kwargs["additional_request_fields"] = {
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "high",
            }
        }
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


def check_model_access(model_id: str = MODEL_ID, region: str = DEFAULT_REGION) -> bool:
    """Verify access to the specified Bedrock model.

    Raises:
        ConfigurationError: If model is not accessible.
    """
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
    except (ClientError, BotoCoreError) as e:
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
    except (ClientError, BotoCoreError) as e:
        logger.warning("Could not get inference profile ARN: %s", e)
        return None
