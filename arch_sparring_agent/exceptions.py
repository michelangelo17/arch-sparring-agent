"""Custom exception hierarchy for arch-sparring-agent."""

from __future__ import annotations


class ArchReviewError(Exception):
    """Base exception for all arch-review errors."""


class ConfigurationError(ArchReviewError):
    """AWS config, model access, or credential issues."""


class PolicySetupError(ArchReviewError):
    """Policy engine, gateway, or Cedar policy failures."""


class ToolError(ArchReviewError):
    """Tool execution failures (file reading, diagram analysis, etc)."""


class GuardrailSetupError(ArchReviewError):
    """Bedrock Guardrail creation, update, or deletion failures."""
