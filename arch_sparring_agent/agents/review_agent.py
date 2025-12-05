from strands import Agent


def create_review_agent(model_id: str = "amazon.nova-2-lite-v1:0"):
    """Create the Review Agent for generating the final review."""

    agent = Agent(
        name="ReviewAgent",
        model=model_id,
        system_prompt="""Write a concise, actionable architecture review.

Format:
## Executive Summary
2-3 sentences summarizing the overall assessment.

## Key Gaps
- Bullet points of significant gaps identified
- Note which gaps were acknowledged by the user

## Top 3 Risks
1. Risk with severity and impact
2. Risk with severity and impact
3. Risk with severity and impact

## Top 3 Recommendations
1. Specific, actionable recommendation
2. Specific, actionable recommendation
3. Specific, actionable recommendation

Be specific. Reference actual components and decisions discussed.""",
        tools=[],
    )

    return agent


def generate_review(
    agent: Agent,
    req_summary: str,
    arch_summary: str,
    qa_context: str = "",
    sparring_context: str = "",
) -> str:
    """Generate the final architecture review."""
    prompt = f"""Write the final architecture review.

REQUIREMENTS SUMMARY:
{req_summary}

ARCHITECTURE SUMMARY:
{arch_summary}
"""
    if qa_context:
        prompt += f"\nQ&A DISCUSSION:\n{qa_context}"
    if sparring_context:
        prompt += f"\nSPARRING DISCUSSION:\n{sparring_context}"

    result = agent(prompt)
    return str(result)

