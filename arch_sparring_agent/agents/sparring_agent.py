from strands import Agent, tool


def create_sparring_agent(model_id: str = "amazon.nova-2-lite-v1:0"):
    """Create the Sparring Agent for challenging architectural decisions."""

    challenges_made = []

    @tool
    def challenge_user(challenge: str) -> str:
        """Challenge the user on an architectural decision or gap."""
        challenges_made.append(challenge)
        print(f"\n⚔️  [{len(challenges_made)}] {challenge}")
        response = input("Your response: ")
        return response

    @tool
    def done_challenging() -> str:
        """Call when you've addressed the key architectural issues."""
        return "Proceeding to final review."

    agent = Agent(
        name="SparringAgent",
        model=model_id,
        system_prompt="""Challenge the user on architectural gaps and risks.

Your role is to be a sparring partner - push back on decisions.

Guidelines:
- Be direct and constructive
- Focus on the top 3-5 most significant issues
- Push back if answers seem weak or dismissive
- If they defend well, acknowledge and move on
- Challenge real risks, not minor preferences

Example challenges:
- "Why no auto-scaling when traffic is unpredictable?"
- "How will you handle data consistency during failures?"
- "This creates a single point of failure. What's the mitigation?"

Call done_challenging when you've addressed the key issues.""",
        tools=[challenge_user, done_challenging],
    )

    return agent


def run_sparring(
    agent: Agent, req_summary: str, arch_summary: str, qa_context: str
) -> str:
    """Run the sparring phase and return the discussion context."""
    result = agent(
        f"""Based on this context, challenge the user on key architectural gaps:

REQUIREMENTS:
{req_summary}

ARCHITECTURE:
{arch_summary}

Q&A CONTEXT:
{qa_context}

Challenge architectural decisions. Call done_challenging when ready."""
    )
    return str(result)

