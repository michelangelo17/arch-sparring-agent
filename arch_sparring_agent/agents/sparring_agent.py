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
        system_prompt="""Challenge CONFIRMED gaps only.

RULES:
- Only challenge items from "Features Not Found"
- Do NOT challenge features in "Features Verified" - those exist
- Focus on security, compliance, and architectural risks
- Push back on weak answers
- Acknowledge good defenses

Call done_challenging when key issues are addressed.""",
        tools=[challenge_user, done_challenging],
    )


def run_sparring(agent: Agent, req_summary: str, arch_summary: str, qa_context: str) -> str:
    """Execute sparring phase."""
    result = agent(
        f"""Challenge ONLY the gaps identified, not verified features.

Q&A CONTEXT (confirmed gaps):
{qa_context}

Challenge these gaps. Do NOT challenge features that exist.
Call done_challenging when ready."""
    )
    return str(result)
