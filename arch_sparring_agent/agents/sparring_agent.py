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
        system_prompt="""Challenge architectural gaps and risks.

Role: Sparring partner - push back on decisions.

Guidelines:
- Be direct and constructive
- Focus on top 3-5 significant issues
- Push back on weak answers
- Acknowledge good defenses
- Challenge real risks, not preferences

Call done_challenging when key issues are addressed.""",
        tools=[challenge_user, done_challenging],
    )


def run_sparring(agent: Agent, req_summary: str, arch_summary: str, qa_context: str) -> str:
    """Execute sparring phase."""
    result = agent(
        f"""Challenge architectural gaps based on context:

REQUIREMENTS:
{req_summary}

ARCHITECTURE:
{arch_summary}

Q&A CONTEXT:
{qa_context}

Challenge decisions. Call done_challenging when ready."""
    )
    return str(result)
