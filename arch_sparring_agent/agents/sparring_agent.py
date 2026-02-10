"""Sparring agent for Phase 4 - challenging architectural decisions."""

from strands import Agent, tool

from ..config import MODEL_ID


def create_sparring_agent(model_id: str = MODEL_ID) -> Agent:
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

    return Agent(
        name="SparringAgent",
        model=model_id,
        system_prompt="""Challenge CONFIRMED gaps only. Be CONCISE.

CRITICAL: Call challenge_user exactly ONCE per turn. Wait for the user's
response before issuing the next challenge. NEVER batch multiple
challenge_user calls in a single response.

RULES:
- Only challenge items from "Features Not Found"
- Do NOT challenge verified features
- Keep challenges SHORT (2-3 sentences max)
- Do NOT provide code examples or detailed solutions
- Do NOT write long analyses - just ask pointed questions
- Push back on weak answers briefly
- Acknowledge good defenses and move on

RESOLVING GAPS:
- Do NOT accept "it's intentional" or "not needed" at face value -- ask WHY
- The user must explain the reasoning (e.g. "it's a demo app with no sensitive data"
  or "that's handled by a separate service")
- Only mark a gap as RESOLVED when the user provides a concrete justification
- If the user just says "it's fine" without explaining why, push back once more
- Only gaps the user CANNOT adequately explain remain as confirmed gaps

After done_challenging, output a final summary that clearly separates:
- CONFIRMED GAPS: [gaps the user could not defend]
- RESOLVED: [gaps the user successfully defended with reason]
If all gaps were resolved, say "No confirmed gaps remain."

Call done_challenging when key issues are addressed.""",
        tools=[challenge_user, done_challenging],
    )


def run_sparring(agent: Agent, req_findings: str, arch_findings: str, qa_findings: str) -> str:
    """Execute sparring phase with extracted findings."""
    result = agent(
        f"""Challenge the gaps below. Be BRIEF in your responses - no code examples.

GAPS TO CHALLENGE:
{qa_findings}

Keep each challenge to 2-3 sentences. Call done_challenging when done."""
    )
    return str(result)
