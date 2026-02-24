"""Review agent for Phase 5 - final review generation."""

from __future__ import annotations

from typing import Any

from strands import Agent
from strands.models import BedrockModel

from ..config import AGENT_REVIEW
from ..profiles import get_directive
from .kb_tool import create_kb_tool

_SYSTEM_PROMPT = """Write architecture review based on CONFIRMED gaps only.

Format:
## Executive Summary
2-3 sentences on overall assessment.

## Confirmed Gaps
- Gaps from "Features Not Found" that were NOT resolved in sparring
- If there are no genuine gaps, write "No confirmed gaps."
- Do NOT list items here that you then explain away as non-issues

## Accepted Risks
- Items the user acknowledged as missing but accepted with reasoning
  (e.g. "it's a POC", "not needed for MVP", "intentional for now")
- These ARE real gaps but with documented risk acceptance
- If none, write "No accepted risks."

## Risks
- Risks arising from confirmed gaps AND accepted risks (up to 5)
- If there are no confirmed gaps or accepted risks, write "No significant risks identified."

## Recommendations
- Actionable recommendations for confirmed gaps (up to 3)
- Optional improvements for accepted risks clearly labeled as "Production readiness" (up to 3)
- If there are no confirmed gaps, write "No recommendations." or offer
  optional improvements clearly labeled as "Nice to have" (not required)

## Verdict
- PASS: No confirmed gaps AND no accepted risks, or all met via service defaults
- PASS WITH CONCERNS: No confirmed gaps but has accepted risks for production
- FAIL: Only for actively exploitable security vulnerabilities or violations of
  STATED requirements.

Be specific. Reference components discussed."""

_KB_ADDENDUM = """

WAF KNOWLEDGE BASE:
You have access to the AWS Well-Architected Framework via the query_waf tool.
Use it to reference specific WAF best practices when writing recommendations.
Cite WAF question IDs (e.g. SEC01-BP01) where relevant."""


def create_review_agent(
    model: BedrockModel,
    knowledge_base_id: str | None = None,
    region: str | None = None,
    profile: dict[str, Any] | None = None,
) -> Agent:
    """Create agent for generating final review."""

    tools: list[Any] = []

    if knowledge_base_id and region:
        tools.append(create_kb_tool(knowledge_base_id, region))

    system_prompt = _SYSTEM_PROMPT

    if knowledge_base_id and region:
        system_prompt += _KB_ADDENDUM

    directive = get_directive(profile, "review")
    if directive:
        system_prompt += f"\n\n{directive}"

    return Agent(
        name=AGENT_REVIEW,
        model=model,
        callback_handler=None,
        system_prompt=system_prompt,
        tools=tools,
    )


def run_review(
    agent: Agent,
    req_findings: str,
    arch_findings: str,
    qa_findings: str = "",
    sparring_findings: str = "",
) -> str:
    """Run final review phase from extracted findings."""
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
        "Items under 'Accepted Risks' go in the Accepted Risks section. "
        "Verified/Resolved features are NOT gaps."
    )

    from . import safe_invoke

    return safe_invoke(agent, prompt)
