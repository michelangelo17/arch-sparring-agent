"""Per-gap sparring orchestration for Phase 4.

Parses gaps from architecture findings, triages by severity, then spars
on each gap individually with bounded follow-up rounds.  Produces a
markdown summary consumed by the context condenser and final review agent.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from strands import Agent
from strands.models import BedrockModel

from ..agents import safe_invoke
from ..agents.sparring_agent import ClassificationOutcome, GapResult, create_sparring_agent
from ..exceptions import MODEL_ERRORS
from ..profiles import get_setting
from .extraction import parse_gaps_from_findings

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

_TRIAGE_PROMPT = """\
You are a severity classifier for architecture review gaps.

For each gap listed below, assign a severity: HIGH, MEDIUM, or LOW.

HIGH = security vulnerability, data loss risk, or compliance failure
MEDIUM = missing best practice or operational concern
LOW = nice-to-have or cosmetic issue

Respond with ONLY a numbered list in this exact format:
1. HIGH: <gap description>
2. MEDIUM: <gap description>
...

GAPS:
"""


@dataclass(frozen=True, slots=True)
class SparringGap:
    """A gap being sparred on, with severity from triage."""

    description: str
    severity: str


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------


def run_sparring(
    model: BedrockModel,
    arch_findings: str,
    qa_findings: str,
    profile: dict[str, Any] | None = None,
    *,
    output_fn: Callable[[str], None] | None = None,
) -> str:
    """Execute the per-gap sparring phase.

    Parses gaps, triages them, spars on each individually, and returns
    an assembled markdown summary.
    """
    raw_gaps = parse_gaps_from_findings(arch_findings, qa_findings)
    if not raw_gaps:
        summary = "No gaps identified — sparring phase skipped."
        if output_fn:
            output_fn(summary)
        return summary

    gaps = _triage_gaps(raw_gaps, model)
    results: list[GapResult] = []
    challenge_offset = 0

    for gap in gaps:
        result, count = _spar_single_gap(
            gap,
            model,
            profile,
            challenge_offset,
            _build_running_context(results),
            output_fn,
        )
        results.append(result)
        challenge_offset += count

    summary = _assemble_summary(results)
    if output_fn:
        output_fn(summary)
    return summary


# ------------------------------------------------------------------
# Triage
# ------------------------------------------------------------------


def _triage_gaps(raw_gaps: list[str], model: BedrockModel) -> list[SparringGap]:
    """Classify gaps as HIGH/MEDIUM/LOW via a single LLM call.

    Falls back to all-MEDIUM on model errors.  Any gap not matched in
    the LLM output also defaults to MEDIUM.
    """
    gap_list = "\n".join(f"{i + 1}. {g}" for i, g in enumerate(raw_gaps))
    severity_map: dict[str, str] = dict.fromkeys(raw_gaps, "MEDIUM")

    try:
        triage_agent = Agent(
            name="GapTriager",
            model=model,
            callback_handler=None,
            system_prompt=_TRIAGE_PROMPT,
            tools=[],
        )
        response = str(triage_agent(gap_list))
        severity_map = _parse_triage_response(response, raw_gaps)
    except MODEL_ERRORS:
        logger.warning("Gap triage failed — defaulting all gaps to MEDIUM")

    gaps = [
        SparringGap(
            description=g,
            severity=severity_map.get(g, "MEDIUM"),
        )
        for g in raw_gaps
    ]
    gaps.sort(key=lambda g: _SEVERITY_ORDER.get(g.severity, 1))
    return gaps


def _parse_triage_response(response: str, raw_gaps: list[str]) -> dict[str, str]:
    """Parse numbered severity lines, matching by index position."""
    severity_map: dict[str, str] = dict.fromkeys(raw_gaps, "MEDIUM")
    pattern = re.compile(r"^\s*(\d+)\.\s*(HIGH|MEDIUM|LOW)", re.IGNORECASE)
    for line in response.split("\n"):
        match = pattern.match(line)
        if not match:
            continue
        idx = int(match.group(1)) - 1
        severity = match.group(2).upper()
        if 0 <= idx < len(raw_gaps):
            severity_map[raw_gaps[idx]] = severity
    return severity_map


# ------------------------------------------------------------------
# Per-gap sparring
# ------------------------------------------------------------------


def _spar_single_gap(
    gap: SparringGap,
    model: BedrockModel,
    profile: dict[str, Any] | None,
    challenge_offset: int,
    running_context: str,
    output_fn: Callable[[str], None] | None,
) -> tuple[GapResult, int]:
    """Spar on a single gap with bounded rounds."""
    if output_fn:
        output_fn(f"\n### [{gap.severity}] {gap.description}\n")

    sa = create_sparring_agent(
        model,
        profile=profile,
        challenge_offset=challenge_offset,
    )
    max_rounds = _max_rounds_for(gap.severity, profile)

    def _to_gap_result(outcome: ClassificationOutcome) -> GapResult:
        return GapResult(gap.description, outcome.classification, outcome.reasoning)

    try:
        for round_num in range(max_rounds):
            prompt = _build_round_prompt(
                gap,
                running_context,
                round_num,
            )
            safe_invoke(sa.agent, prompt)
            outcome = sa.get_result()
            if outcome:
                return _to_gap_result(outcome), sa.challenge_count()

        safe_invoke(
            sa.agent,
            "Classify the gap based on the conversation so far. "
            "Call classify_gap with your conclusion.",
        )
        outcome = sa.get_result()
        if outcome:
            return _to_gap_result(outcome), sa.challenge_count()
    except MODEL_ERRORS:
        logger.warning(
            "Sparring failed for gap '%s' — marking as CONFIRMED_GAP",
            gap.description[:60],
        )

    return (
        GapResult(gap.description, "CONFIRMED_GAP", "Unresolved after max rounds"),
        sa.challenge_count(),
    )


def _max_rounds_for(severity: str, profile: dict[str, Any] | None) -> int:
    """Return the max sparring rounds for a given severity and profile."""
    value = get_setting(profile, "sparring", "max_rounds", severity.lower(), default=1)
    return value if isinstance(value, int) else 1


def _build_round_prompt(gap: SparringGap, running_context: str, round_num: int) -> str:
    """Build the prompt for a single sparring round."""
    parts = [f"GAP TO CHALLENGE: {gap.description}"]
    if running_context:
        parts.append(f"\nPRIOR GAP RESULTS:\n{running_context}")
    if round_num == 0:
        parts.append("\nChallenge this gap. Call challenge_user with your question.")
    else:
        parts.append(
            "\nThe user's answer was not convincing enough. "
            "Push back or call classify_gap if you've reached a conclusion."
        )
    return "\n".join(parts)


# ------------------------------------------------------------------
# Context and summary assembly
# ------------------------------------------------------------------


def _build_running_context(results: list[GapResult]) -> str:
    """Format prior gap results as plain text for cross-gap awareness."""
    if not results:
        return ""
    lines = []
    for r in results:
        lines.append(f"- {r.description}: {r.classification} ({r.reasoning})")
    return "\n".join(lines)


_CLASSIFICATION_HEADINGS = [
    ("Confirmed Gaps", "CONFIRMED_GAP"),
    ("Accepted Risks", "ACCEPTED_RISK"),
    ("Resolved", "RESOLVED"),
]


def _assemble_summary(results: list[GapResult]) -> str:
    """Group results by classification into a markdown summary."""
    sections: list[str] = []
    for heading, classification in _CLASSIFICATION_HEADINGS:
        group = [r for r in results if r.classification == classification]
        if group:
            items = "\n".join(f"- {r.description}: {r.reasoning}" for r in group)
            sections.append(f"### {heading}\n{items}")

    if not sections:
        return "No gaps to report."

    return "\n\n".join(sections)
