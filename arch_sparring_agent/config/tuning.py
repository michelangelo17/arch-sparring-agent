"""Tuning constants — all overridable via environment variables."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


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


def _float_env(var: str, default: float) -> float:
    """Read a float from an environment variable with validation."""
    raw = os.getenv(var)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s='%s', using default %.2f", var, raw, default)
        return default


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

# Source analyzer: max source file size (bytes)
SOURCE_MAX_BYTES = _int_env("ARCH_REVIEW_SOURCE_MAX_BYTES", 50000)

# Diagram analyzer: max tokens for diagram analysis response
DIAGRAM_MAX_TOKENS = _int_env("ARCH_REVIEW_DIAGRAM_MAX_TOKENS", 4000)

# Gateway: max seconds to wait for IAM propagation after gateway creation
IAM_PROPAGATION_TIMEOUT = _int_env("ARCH_REVIEW_IAM_WAIT_TIMEOUT", 60)

# File size limits for tools (bytes)
CFN_MAX_BYTES = _int_env("ARCH_REVIEW_CFN_MAX_BYTES", 500_000)
DOC_MAX_BYTES = _int_env("ARCH_REVIEW_DOC_MAX_BYTES", 500_000)
DIAGRAM_MAX_BYTES = _int_env("ARCH_REVIEW_DIAGRAM_MAX_BYTES", 10_000_000)

# Knowledge Base: seconds to wait for IAM role propagation during KB setup
IAM_KB_PROPAGATION_WAIT = _int_env("ARCH_REVIEW_KB_IAM_WAIT", 15)

# Policy Engine: max poll iterations waiting for policy to become ACTIVE (0.5s each)
POLICY_ACTIVE_MAX_POLLS = _int_env("ARCH_REVIEW_POLICY_ACTIVE_MAX_POLLS", 120)

# Guardrails: contextual grounding check
GROUNDING_THRESHOLD = _float_env("ARCH_REVIEW_GROUNDING_THRESHOLD", 0.7)
GROUNDING_CONTENT_CHUNK_SIZE = _int_env("ARCH_REVIEW_GROUNDING_CHUNK_SIZE", 4500)
GROUNDING_SOURCE_MAX_CHARS = _int_env("ARCH_REVIEW_GROUNDING_SOURCE_MAX_CHARS", 100_000)
GUARDRAIL_NAME = "ArchReviewGroundingGuardrail"
