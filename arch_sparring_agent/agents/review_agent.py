"""Review agent for Phase 5 - final review generation."""

from strands import Agent
from strands.models import BedrockModel


def create_review_agent(model_id: str | BedrockModel) -> Agent:
    """Create agent for generating final review."""

    return Agent(
        name="ReviewAgent",
        model=model_id,
        callback_handler=None,
        system_prompt="""Write architecture review based on CONFIRMED gaps only.

Format:
## Executive Summary
2-3 sentences on overall assessment.

## Confirmed Gaps
- Only gaps from "Features Not Found" or discussed in sparring
- If there are no genuine gaps, write "No confirmed gaps."
- Do NOT list items here that you then explain away as non-issues

## Risks
- Only risks arising from confirmed gaps (up to 3)
- If there are no confirmed gaps, write "No significant risks identified."
- Do NOT pad this section with hypothetical or minimal risks

## Recommendations
- Only actionable recommendations for confirmed gaps (up to 3)
- If there are no confirmed gaps, write "No recommendations." or offer
  optional improvements clearly labeled as "Nice to have" (not required)
- Do NOT invent requirements that weren't stated

## Verdict
- PASS: No gaps, or all requirements met (including via service defaults)
- PASS WITH CONCERNS: Has genuine gaps that warrant attention but no active vulnerabilities
- FAIL: Only for actively exploitable security vulnerabilities or violations of
  STATED requirements. Do NOT fail for missing best practices alone.

RULES:
- Only report gaps that are genuinely unmet requirements
- Features in "Features Verified" are NOT gaps
- Features covered by AWS service defaults (e.g. DynamoDB default encryption) are NOT gaps
- Do NOT assume compliance or regulatory requirements that aren't stated
- Items marked as "RESOLVED" in sparring/question phases are NOT confirmed gaps
- If a user defended or dismissed a gap (e.g. "intentional", "not needed"), it is resolved
- Do NOT pad sections with non-issues to fill space
- It is perfectly fine to have empty sections -- a clean architecture is a good result
- If all gaps were resolved, the verdict should be PASS, not PASS WITH CONCERNS
- Be specific. Reference components discussed.""",
        tools=[],
    )


def generate_review(
    agent: Agent,
    req_findings: str,
    arch_findings: str,
    qa_findings: str = "",
    sparring_findings: str = "",
) -> str:
    """Generate final architecture review from extracted findings."""
    prompt = "Write review based on CONFIRMED gaps only.\n"

    if qa_findings:
        prompt += f"\nCONFIRMED GAPS:\n{qa_findings}"
    if sparring_findings:
        prompt += f"\nRISK ASSESSMENT & DECISIONS:\n{sparring_findings}"

    # Include architecture findings as reference for component names
    if arch_findings:
        prompt += f"\nARCHITECTURE REFERENCE:\n{arch_findings}"

    prompt += (
        "\n\nOnly report gaps from 'Confirmed Gaps' or 'Features Not Found'. "
        "Verified features are NOT gaps."
    )

    return str(agent(prompt))
