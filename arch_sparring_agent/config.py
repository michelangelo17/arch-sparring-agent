"""Configuration constants and model registry."""

import logging
import os
from typing import Any

from strands.models import BedrockModel

logger = logging.getLogger(__name__)

DEFAULT_REGION = "eu-central-1"

# --- Supported models (curated, verified EU inference profiles) ---

SUPPORTED_MODELS: dict[str, dict[str, str]] = {
    "nova-2-lite": {
        "model_id": "eu.amazon.nova-2-lite-v1:0",
        "description": "Amazon Nova 2 Lite (1M context)",
        "reasoning_type": "nova",
    },
    "opus-4.6": {
        "model_id": "eu.anthropic.claude-opus-4-6-v1",
        "description": "Claude Opus 4.6 (1M context)",
        "reasoning_type": "adaptive",
    },
}
DEFAULT_MODEL = "nova-2-lite"

# --- Reasoning levels ---

REASONING_LEVELS = ("off", "low", "medium", "high")
DEFAULT_REASONING_LEVEL = "low"


def _build_reasoning_fields(reasoning_type: str, level: str) -> dict[str, Any]:
    """Build additional_request_fields for reasoning based on model type.

    Nova uses reasoningConfig with maxReasoningEffort.
    Claude Opus 4.6 uses adaptive thinking with effort parameter.
    """
    if reasoning_type == "nova":
        return {
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": level,
            }
        }
    if reasoning_type == "adaptive":
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": level},
        }
    msg = f"Unknown reasoning_type: {reasoning_type}"
    raise ValueError(msg)


def create_model(
    model_name: str = DEFAULT_MODEL,
    reasoning: bool = False,
    reasoning_level: str = DEFAULT_REASONING_LEVEL,
) -> BedrockModel:
    """Create a BedrockModel from a supported model name.

    Args:
        model_name: Short name from SUPPORTED_MODELS (e.g. "nova-2-lite", "opus-4.6").
        reasoning: Enable extended thinking / reasoning for complex analysis.
        reasoning_level: Reasoning effort level: "off", "low", "medium", or "high".

    Raises:
        ValueError: If model_name is not in SUPPORTED_MODELS.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    model_config = SUPPORTED_MODELS[model_name]
    kwargs: dict[str, Any] = {"model_id": model_config["model_id"]}

    if reasoning and reasoning_level != "off":
        level = (
            reasoning_level
            if reasoning_level in ("low", "medium", "high")
            else DEFAULT_REASONING_LEVEL
        )
        kwargs["additional_request_fields"] = _build_reasoning_fields(
            model_config["reasoning_type"], level
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
