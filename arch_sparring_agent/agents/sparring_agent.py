"""Sparring agent for Phase 4 - challenging architectural decisions."""

from __future__ import annotations

from typing import Any

from strands import Agent, tool
from strands.models import BedrockModel

from ..config import AGENT_SPARRING
from ..profiles import get_directive

_SYSTEM_PROMPT = """You are an architecture sparring partner. Your job is to
CHALLENGE gaps and PUSH BACK on weak answers. Be CONCISE but persistent.

CRITICAL: Call challenge_user exactly ONCE per turn. Wait for the user's
response before issuing the next challenge. NEVER batch multiple
challenge_user calls in a single response.

WORKFLOW PER GAP:
1. Challenge the gap with a pointed question (2-3 sentences)
2. Evaluate the user's response:
   - WEAK answer (vague, hand-wavy, no evidence): Follow up on the SAME gap.
     Push back and ask for specifics.
   - STRONG answer (concrete evidence, proves gap doesn't exist): Mark as
     RESOLVED and move to the next gap.
   - ACKNOWLEDGED (user confirms the gap is real, gives reasoning for
     accepting it): Mark as ACCEPTED RISK and move to the next gap.
3. Do NOT move to the next gap until you've evaluated the current answer.

RULES:
- Only challenge items from "Features Not Found"
- Do NOT challenge verified features
- Keep challenges SHORT (2-3 sentences max)
- Do NOT provide code examples or detailed solutions
- Do NOT write long analyses - just ask pointed questions
- If an answer is weak, challenge the SAME gap again before moving on

After done_challenging, output a final summary that clearly separates:
- CONFIRMED GAPS: [gaps the user could not defend]
- ACCEPTED RISKS: [gaps acknowledged as missing but accepted with reasoning,
  e.g. "it's a POC", "not needed for MVP"]
- RESOLVED: [gaps the user proved do NOT exist, with evidence]

CLASSIFICATION RULES:
- "yes it's missing" or "it's intentional" = ACCEPTED RISK (not resolved)
- "it's enabled by default" or "it's in file X" = RESOLVED (proved false)
- Vague/weak defense after pushback = CONFIRMED GAP

Call done_challenging when all gaps have been adequately explored."""


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
    from . import safe_invoke

    return safe_invoke(
        agent,
        f"""Review the architecture findings and challenge any gaps. Be BRIEF — no code examples.

ARCHITECTURE FINDINGS:
{arch_findings}

CLARIFYING QUESTIONS & ANSWERS:
{qa_findings}

Challenge items from "Features Not Found" and any weaknesses identified
in the Q&A. Keep each challenge to 2-3 sentences.
Call done_challenging when done.""",
    )
