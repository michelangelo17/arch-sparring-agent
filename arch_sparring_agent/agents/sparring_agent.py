"""Sparring agent for Phase 4 - challenging architectural decisions."""

from __future__ import annotations

from typing import Any

from strands import Agent, tool
from strands.models import BedrockModel

from ..config import AGENT_SPARRING
from ..profiles import get_directive

_SYSTEM_PROMPT = """Challenge CONFIRMED gaps only. Be CONCISE.

CRITICAL: Call challenge_user exactly ONCE per turn. Wait for the user's
response before issuing the next challenge. NEVER batch multiple
challenge_user calls in a single response.

RULES:
- Only challenge items from "Features Not Found"
- Do NOT challenge verified features
- Keep challenges SHORT (2-3 sentences max)
- Do NOT provide code examples or detailed solutions
- Do NOT write long analyses - just ask pointed questions

After done_challenging, output a final summary that clearly separates:
- CONFIRMED GAPS: [gaps the user could not defend]
- RESOLVED: [gaps the user successfully defended with reason]
If all gaps were resolved, say "No confirmed gaps remain."

Call done_challenging when key issues are addressed."""


def create_sparring_agent(model: BedrockModel, profile: dict[str, Any] | None = None) -> Agent:
    """Create agent for challenging architectural decisions."""

    challenges_made = []

    @tool
    def challenge_user(challenge: str) -> str:
        """Challenge an architectural decision or gap."""
        challenges_made.append(challenge)
        print(f"\n⚔️  [{len(challenges_made)}] {challenge}")
        return input("Your response: ")

    @tool
    def done_challenging() -> str:
        """Signal completion of sparring phase."""
        return "Proceeding to final review."

    system_prompt = _SYSTEM_PROMPT

    directive = get_directive(profile, "sparring")
    if directive:
        system_prompt += f"\n\n{directive}"

    return Agent(
        name=AGENT_SPARRING,
        model=model,
        callback_handler=None,
        system_prompt=system_prompt,
        tools=[challenge_user, done_challenging],
    )


def run_sparring(agent: Agent, arch_findings: str, qa_findings: str) -> str:
    """Execute sparring phase with extracted findings."""
    result = agent(
        f"""Review the architecture findings and challenge any gaps. Be BRIEF — no code examples.

ARCHITECTURE FINDINGS:
{arch_findings}

CLARIFYING QUESTIONS & ANSWERS:
{qa_findings}

Challenge items from "Features Not Found" and any weaknesses identified
in the Q&A. Keep each challenge to 2-3 sentences.
Call done_challenging when done."""
    )
    return str(result)
