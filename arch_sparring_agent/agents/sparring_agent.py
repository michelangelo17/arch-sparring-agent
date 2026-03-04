"""Sparring agent for Phase 4 — challenging architectural decisions.

Exports ``create_sparring_agent`` (factory), ``ClassificationOutcome``,
``GapResult``, and ``SparringAgent`` (dataclasses consumed by the
review/sparring orchestrator).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from strands import Agent, tool
from strands.models import BedrockModel

from ..config import AGENT_SPARRING
from ..profiles import get_directive

VALID_CLASSIFICATIONS = frozenset({"CONFIRMED_GAP", "ACCEPTED_RISK", "RESOLVED"})

_SYSTEM_PROMPT = """\
You are an architecture sparring partner challenging ONE specific gap.

WORKFLOW:
1. Call challenge_user with a pointed question (2-3 sentences).
2. Evaluate the user's response:
   - WEAK (vague, hand-wavy): Follow up on the SAME gap — push for specifics.
   - STRONG (concrete evidence the gap doesn't exist): call classify_gap
     with RESOLVED.
   - ACKNOWLEDGED (user confirms gap, gives reasoning): call classify_gap
     with ACCEPTED_RISK.
3. If the answer is weak, challenge again before classifying.

RULES:
- Keep challenges SHORT (2-3 sentences max).
- Do NOT provide code examples or detailed solutions.
- Call challenge_user exactly ONCE per turn.
- Call classify_gap once you've reached a conclusion.

CLASSIFICATION RULES:
- "yes it's missing" or "it's intentional" = ACCEPTED_RISK
- "it's enabled by default" or "it's in file X" = RESOLVED
- Vague/weak defense after pushback = CONFIRMED_GAP"""


@dataclass(frozen=True, slots=True)
class ClassificationOutcome:
    """Result of the classify_gap tool (no description — added by orchestrator)."""

    classification: str
    reasoning: str


@dataclass(frozen=True, slots=True)
class GapResult:
    """Outcome of sparring on a single gap."""

    description: str
    classification: str
    reasoning: str


@dataclass(frozen=True, slots=True)
class SparringAgent:
    """Return type of ``create_sparring_agent``."""

    agent: Agent
    get_result: Callable[[], ClassificationOutcome | None]
    challenge_count: Callable[[], int]


def create_sparring_agent(
    model: BedrockModel,
    profile: dict[str, Any] | None = None,
    challenge_offset: int = 0,
) -> SparringAgent:
    """Create an agent for challenging a single architectural gap."""
    challenges_made: list[str] = []
    _result: list[ClassificationOutcome] = []

    @tool
    def challenge_user(challenge: str) -> str:
        """Challenge an architectural decision or gap."""
        challenges_made.append(challenge)
        idx = challenge_offset + len(challenges_made)
        print(f"\n\u2694\ufe0f  [{idx}] {challenge}")
        return input("Your response: ")

    @tool
    def classify_gap(classification: str, reasoning: str) -> str:
        """Classify the gap after sparring.

        classification must be CONFIRMED_GAP, ACCEPTED_RISK, or RESOLVED.
        """
        if classification not in VALID_CLASSIFICATIONS:
            return (
                f"Invalid classification '{classification}'. "
                f"Must be one of: {', '.join(sorted(VALID_CLASSIFICATIONS))}"
            )
        _result.clear()
        _result.append(ClassificationOutcome(classification, reasoning))
        return f"Gap classified as {classification}."

    system_prompt = _SYSTEM_PROMPT
    directive = get_directive(profile, "sparring")
    if directive:
        system_prompt += f"\n\n{directive}"

    agent = Agent(
        name=AGENT_SPARRING,
        model=model,
        callback_handler=None,
        system_prompt=system_prompt,
        tools=[challenge_user, classify_gap],
    )

    return SparringAgent(
        agent=agent,
        get_result=lambda: _result[0] if _result else None,
        challenge_count=lambda: len(challenges_made),
    )
