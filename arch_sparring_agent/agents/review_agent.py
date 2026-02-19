"""Review agent for Phase 5 - final review generation."""

from strands import Agent
from strands.models import BedrockModel

from ..profiles import get_directive


def create_review_agent(model_id: str | BedrockModel) -> Agent:
    """Create agent for generating final review."""

    system_prompt = """Write architecture review based on CONFIRMED gaps only.

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

## Recommendations
- Only actionable recommendations for confirmed gaps (up to 3)
- If there are no confirmed gaps, write "No recommendations." or offer
  optional improvements clearly labeled as "Nice to have" (not required)

## Verdict
- PASS: No gaps, or all requirements met (including via service defaults)
- PASS WITH CONCERNS: Has genuine gaps that warrant attention but no active vulnerabilities
- FAIL: Only for actively exploitable security vulnerabilities or violations of
  STATED requirements.

Be specific. Reference components discussed."""

    directive = get_directive("review")
    if directive:
        system_prompt += f"\n\n{directive}"

    return Agent(
        name="ReviewAgent",
        model=model_id,
        callback_handler=None,
        system_prompt=system_prompt,
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
