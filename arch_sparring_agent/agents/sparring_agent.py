"""Sparring agent for Phase 4 - challenging architectural decisions."""

from strands import Agent, tool
from strands.models import BedrockModel

from ..profiles import get_directive


def create_sparring_agent(model_id: str | BedrockModel) -> Agent:
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

    system_prompt = """Challenge CONFIRMED gaps only. Be CONCISE.

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

    directive = get_directive("sparring")
    if directive:
        system_prompt += f"\n\n{directive}"

    return Agent(
        name="SparringAgent",
        model=model_id,
        callback_handler=None,
        system_prompt=system_prompt,
        tools=[challenge_user, done_challenging],
    )


def run_sparring(agent: Agent, req_findings: str, arch_findings: str, qa_findings: str) -> str:
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
