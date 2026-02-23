"""Remediation agent for discussing and resolving review findings."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from botocore.exceptions import ClientError
from strands import Agent
from strands.models import BedrockModel
from strands.types.exceptions import ContextWindowOverflowException, MaxTokensReachedException

from ..config import DEFAULT_REGION
from ..infra.memory import create_session_manager, setup_agentcore_memory
from ..state import ReviewState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """You help resolve architecture review findings.

PROJECT: {project_name}
VERDICT: {verdict}

TECH STACK & REQUIREMENTS:
{requirements_summary}

ARCHITECTURE:
{architecture_summary}

GAPS:
{gaps_text}

RISKS:
{risks_text}

RECOMMENDATIONS:
{recs_text}

RULES:
- Keep responses SHORT (under 200 words)
- Only provide code when explicitly asked
- One issue at a time
- Use tech stack info for language-specific advice"""


def _format_list(items: list[dict], severity_key: str) -> str:
    """Format gaps or risks list for display."""
    if not items:
        return "  None identified"
    lines = [
        f"  {i}. [{item[severity_key].upper()}] {item['description']}"
        for i, item in enumerate(items, 1)
    ]
    return "\n".join(lines)


def _format_recommendations(items: list[str]) -> str:
    """Format recommendations list for display."""
    if not items:
        return "  None provided"
    return "\n".join(f"  {i}. {r}" for i, r in enumerate(items, 1))


def _format_header_list(items: list[dict], severity_key: str) -> str:
    """Format truncated list for header display."""
    lines = [
        f"  {i}. [{item[severity_key].upper()}] {item['description'][:60]}..."
        for i, item in enumerate(items, 1)
    ]
    return "\n".join(lines)


def _number_to_prompt(num: int, state: ReviewState) -> str | None:
    """Convert numeric input to discussion prompt."""
    if num <= len(state.gaps):
        return f"Discuss gap #{num}: {state.gaps[num - 1]['description']}"
    risk_num = num - len(state.gaps)
    if risk_num <= len(state.risks):
        return f"Discuss risk #{risk_num}: {state.risks[risk_num - 1]['description']}"
    return None


def create_remediation_agent(
    state: ReviewState,
    model: str | BedrockModel,
    region: str = DEFAULT_REGION,
) -> Agent:
    """Create agent for remediation discussions with session memory."""
    session_manager = None
    if state.project_name:
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", state.project_name)[:40]
        memory_name = f"Remediation_{safe_name}"
        session_id = f"session_{safe_name}"
        actor_id = f"user_{safe_name}"
        memory_config, _ = setup_agentcore_memory(
            region=region,
            memory_name=memory_name,
            session_id=session_id,
            actor_id=actor_id,
        )
        if memory_config:
            session_manager = create_session_manager(memory_config)
            logger.info("Session memory active (project: %s)", state.project_name)
        else:
            logger.warning(
                "Session memory could not be set up. "
                "Cross-session context will NOT be preserved. "
                "This session's discussions will be lost when you exit."
            )
    else:
        logger.warning(
            "No project name in review state. "
            "Session memory requires a project name for persistence."
        )

    gaps_text = _format_list(state.gaps, "severity")
    risks_text = _format_list(state.risks, "impact")
    recs_text = _format_recommendations(state.recommendations)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        project_name=state.project_name,
        verdict=state.verdict,
        requirements_summary=state.requirements_summary or "No requirements available.",
        architecture_summary=state.architecture_summary or "No architecture details.",
        gaps_text=gaps_text,
        risks_text=risks_text,
        recs_text=recs_text,
    )

    return Agent(
        name="RemediationAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[],
        session_manager=session_manager,
    )


def run_remediation(
    agent: Agent,
    state: ReviewState,
    output_fn: Callable[[str], None] | None = None,
) -> str:
    """Run interactive remediation session.

    Args:
        agent: The remediation agent.
        state: Review state with gaps/risks.
        output_fn: Optional callback for user-facing output (e.g. click.echo).
    """
    notes: list[str] = []

    def _emit(text: str) -> None:
        """Emit output via callback if provided."""
        if output_fn:
            output_fn(text)

    divider = "=" * 60
    parts = [
        f"\n{divider}",
        "REMEDIATION MODE",
        divider,
        f"\nReview ({state.timestamp[:10]}) found:",
        f"  - {len(state.gaps)} gaps",
        f"  - {len(state.risks)} risks",
        f"  - Verdict: {state.verdict}",
    ]

    if state.gaps:
        parts.append("\nGaps:")
        parts.append(_format_header_list(state.gaps, "severity"))

    if state.risks:
        parts.append("\nRisks:")
        parts.append(_format_header_list(state.risks, "impact"))

    parts.append("\nEnter a number to discuss, ask a question, or 'exit' to end.\n")
    header = "\n".join(parts)

    _emit(header)
    notes.append(header)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            _emit("\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "done", "q"):
            break

        notes.append(f"User: {user_input}")

        if user_input.isdigit():
            prompt = _number_to_prompt(int(user_input), state)
            if prompt:
                user_input = prompt

        response = str(agent(user_input))
        _emit(f"\n{response}")
        notes.append(f"Agent: {response}")

    try:
        prompt = "Summarize: issues discussed, decisions made, remaining. Under 150 words."
        summary = str(agent(prompt))
        notes.append(f"\n## Session Summary\n{summary}")
        _emit(f"\n{'=' * 60}\nSummary:\n{summary}\n{'=' * 60}")
    except (ContextWindowOverflowException, MaxTokensReachedException, ClientError) as e:
        logger.warning("Could not generate session summary: %s", e)

    return "\n\n".join(notes)
