"""Review agent for Phase 5 - final review generation."""

from strands import Agent


def create_review_agent(model_id: str = "amazon.nova-2-lite-v1:0") -> Agent:
    """Create agent for generating final review."""

    return Agent(
        name="ReviewAgent",
        model=model_id,
        system_prompt="""Write architecture review.

Format:
## Executive Summary
2-3 sentences on overall assessment.

## Key Gaps
- Significant gaps identified
- Note acknowledged gaps

## Top 3 Risks
1. Risk with severity and impact
2. Risk with severity and impact
3. Risk with severity and impact

## Top 3 Recommendations
1. Specific, actionable recommendation
2. Specific, actionable recommendation
3. Specific, actionable recommendation

Be specific. Reference components and decisions discussed.""",
        tools=[],
    )


def generate_review(
    agent: Agent,
    req_summary: str,
    arch_summary: str,
    qa_context: str = "",
    sparring_context: str = "",
) -> str:
    """Generate final architecture review."""
    prompt = f"""Write the architecture review.

REQUIREMENTS:
{req_summary}

ARCHITECTURE:
{arch_summary}
"""
    if qa_context:
        prompt += f"\nQ&A DISCUSSION:\n{qa_context}"
    if sparring_context:
        prompt += f"\nSPARRING DISCUSSION:\n{sparring_context}"

    return str(agent(prompt))
