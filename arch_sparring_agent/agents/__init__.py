"""Agent implementations for each review phase."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from strands.types.exceptions import MaxTokensReachedException

if TYPE_CHECKING:
    from strands import Agent

logger = logging.getLogger(__name__)

_SDK_RECOVERY_MARKER = "tool use was incomplete due to maximum token limits"


def collect_output(agent: Agent) -> str:
    """Extract all assistant text from an agent's conversation history.

    When a phase agent hits ``MaxTokensReachedException``, the SDK's recovery
    mechanism replaces truncated tool calls with error text and appends the
    recovered message to ``agent.messages``.  Rather than losing all work,
    this function walks the full history and collects every assistant text
    block, filtering out the SDK's recovery noise.

    The result is fed to the context condenser just like normal raw output,
    preserving the "extract, never truncate" contract.
    """
    parts: list[str] = []
    for msg in agent.messages:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content") or []:
            text = block.get("text")
            if text and _SDK_RECOVERY_MARKER not in text:
                parts.append(text)

    collected = "\n\n".join(parts)
    if collected:
        logger.info("Collected %d chars of partial output from agent history", len(collected))
    return collected


def safe_invoke(agent: Agent, prompt: str) -> str:
    """Invoke an agent, recovering partial output on max-tokens truncation.

    Mirrors the condenser's pattern: try the happy path, fall back to
    extracting whatever the agent produced rather than losing work.
    Re-raises ``MaxTokensReachedException`` only if there is genuinely
    no recoverable output.
    """
    try:
        return str(agent(prompt))
    except MaxTokensReachedException:
        logger.warning(
            "Agent %s hit max_tokens — collecting partial output from history",
            getattr(agent, "name", "unknown"),
        )
        output = collect_output(agent)
        if not output:
            raise
        return output


__all__ = [
    "architecture_agent",
    "collect_output",
    "kb_tool",
    "question_agent",
    "requirements_agent",
    "review_agent",
    "safe_invoke",
    "sparring_agent",
]
