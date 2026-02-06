"""Sparring agent for Phase 4 - challenging architectural decisions."""

from strands import Agent, tool


def create_sparring_agent(model_id: str = "amazon.nova-2-lite-v1:0") -> Agent:
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

RULES:
- Only challenge items from "Features Not Found"
- Do NOT challenge verified features
- Keep challenges SHORT (2-3 sentences max)
- Do NOT provide code examples or detailed solutions
- Do NOT write long analyses - just ask pointed questions
- Push back on weak answers briefly
- Acknowledge good defenses and move on

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
