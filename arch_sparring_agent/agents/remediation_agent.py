"""Remediation agent for discussing and resolving review findings."""

import logging
import re

from strands import Agent

from ..config import create_session_manager, setup_agentcore_memory
from ..state import ReviewState

logger = logging.getLogger(__name__)


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
    model_id: str,
    region: str = "eu-central-1",
) -> Agent:
    """Create agent for remediation discussions with session memory."""

    # Set up memory for session continuity (stable IDs for cross-session persistence)
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

    # Build context
    gaps_text = _format_list(state.gaps, "severity")
    risks_text = _format_list(state.risks, "impact")
    recs_text = _format_recommendations(state.recommendations)

    system_prompt = f"""You help resolve architecture review findings.

PROJECT: {state.project_name}
VERDICT: {state.verdict}

TECH STACK & REQUIREMENTS:
{state.requirements_summary or "No requirements available."}

ARCHITECTURE:
{state.architecture_summary or "No architecture details."}

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

    return Agent(
        name="RemediationAgent",
        model=model_id,
        system_prompt=system_prompt,
        tools=[],
        session_manager=session_manager,
    )


def run_remediation(agent: Agent, state: ReviewState, capture_fn=None) -> str:
    """Run interactive remediation session."""
    notes = []

    divider = "=" * 60
    parts = [
        f"\n{divider}",
        "REMEDIATION MODE",
        divider,
        f"\nReview ({state.timestamp[:10]}) found:",
        f"  • {len(state.gaps)} gaps",
        f"  • {len(state.risks)} risks",
        f"  • Verdict: {state.verdict}",
    ]

    if state.gaps:
        parts.append("\nGaps:")
        parts.append(_format_header_list(state.gaps, "severity"))

    if state.risks:
        parts.append("\nRisks:")
        parts.append(_format_header_list(state.risks, "impact"))

    parts.append("\nEnter a number to discuss, ask a question, or 'exit' to end.\n")
    header = "\n".join(parts)

    print(header)
    if capture_fn:
        capture_fn(header)
    notes.append(header)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
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
        print(f"\n{response}")
        if capture_fn:
            capture_fn(response)
        notes.append(f"Agent: {response}")

    try:
        prompt = "Summarize: issues discussed, decisions made, remaining. Under 150 words."
        summary = str(agent(prompt))
        notes.append(f"\n## Session Summary\n{summary}")
        print(f"\n{'=' * 60}\nSummary:\n{summary}\n{'=' * 60}")
    except Exception:
        pass

    return "\n\n".join(notes)
