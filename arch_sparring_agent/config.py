"""Configuration constants and basic AWS client setup."""

import logging
import os
from typing import Any

import boto3

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

# Gateway: max seconds to wait for IAM propagation after gateway creation
IAM_PROPAGATION_TIMEOUT = int(os.getenv("ARCH_REVIEW_IAM_WAIT_TIMEOUT", "60"))


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
