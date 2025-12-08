"""Review agent for Phase 5 - final review generation."""

from strands import Agent


def create_review_agent(model_id: str = "amazon.nova-2-lite-v1:0") -> Agent:
    """Create agent for generating final review."""

    return Agent(
        name="ReviewAgent",
        model=model_id,
        system_prompt="""Write architecture review based on CONFIRMED gaps only.

Format:
## Executive Summary
2-3 sentences on overall assessment.

## Confirmed Gaps
- Only gaps from "Features Not Found" or discussed in sparring
- Do NOT add gaps for verified features

## Top 3 Risks
1. Risk with severity and impact (from confirmed gaps only)
2. Risk with severity and impact
3. Risk with severity and impact

## Top 3 Recommendations
1. Specific, actionable recommendation
2. Specific, actionable recommendation
3. Specific, actionable recommendation

RULES:
- Only report gaps that were confirmed missing
- Features in "Features Verified" are NOT gaps
- Be specific. Reference components discussed.""",
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
    prompt = f"""Write review based on CONFIRMED gaps only.

ARCHITECTURE (note "Features Verified" vs "Features Not Found"):
{arch_summary}
"""
    if qa_context:
        prompt += f"\nCONFIRMED GAPS:\n{qa_context}"
    if sparring_context:
        prompt += f"\nSPARRING DISCUSSION:\n{sparring_context}"

    prompt += "\n\nOnly report gaps from 'Features Not Found'. Verified features are NOT gaps."

    return str(agent(prompt))
