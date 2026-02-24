"""Model registry, agent names, and model creation."""

from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

from strands.models import BedrockModel

logger = logging.getLogger(__name__)

DEFAULT_REGION = os.environ.get("AWS_REGION", "eu-central-1")

# --- Agent names (single source of truth for agents, Cedar policies, etc.) ---

AGENT_REQUIREMENTS = "RequirementsAnalyst"
AGENT_ARCHITECTURE = "ArchitectureEvaluator"
AGENT_QUESTION = "QuestionAgent"
AGENT_SPARRING = "SparringAgent"
AGENT_REVIEW = "ReviewAgent"

# --- Supported models (curated, verified EU inference profiles) ---


class ModelConfig(TypedDict):
    model_id: str
    description: str
    reasoning_type: str
    max_output_tokens: int


SUPPORTED_MODELS: dict[str, ModelConfig] = {
    "nova-2-lite": {
        "model_id": "eu.amazon.nova-2-lite-v1:0",
        "description": "Amazon Nova 2 Lite (1M context)",
        "reasoning_type": "nova",
        "max_output_tokens": 5000,
    },
    "opus-4.6": {
        "model_id": "eu.anthropic.claude-opus-4-6-v1",
        "description": "Claude Opus 4.6 (1M context)",
        "reasoning_type": "adaptive",
        "max_output_tokens": 16384,
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

    env_override = os.getenv("ARCH_REVIEW_MAX_OUTPUT_TOKENS")
    if env_override is not None:
        try:
            kwargs["max_tokens"] = int(env_override)
        except ValueError:
            logger.warning(
                "Invalid ARCH_REVIEW_MAX_OUTPUT_TOKENS='%s', using model default", env_override
            )
            kwargs["max_tokens"] = model_config["max_output_tokens"]
    else:
        kwargs["max_tokens"] = model_config["max_output_tokens"]

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
