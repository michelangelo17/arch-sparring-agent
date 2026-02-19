"""Non-interactive agents for CI/CD pipeline execution."""

from strands import Agent
from strands.models import BedrockModel

from ..profiles import get_directive


def create_ci_question_agent(model_id: str | BedrockModel) -> Agent:
    """Create agent that identifies gaps without user interaction."""
    return Agent(
        name="QuestionAgent",
        model=model_id,
        callback_handler=None,
        system_prompt="""You are a strict extractor. Your ONLY job is to copy items from
"Features Not Found" verbatim.

RULES:
1. Copy ONLY items listed under "Features Not Found"
2. Do NOT add ANY items not explicitly in that section
3. Do NOT analyze or interpret - just extract
4. If a feature is in "Features Verified", it is NOT a gap
5. If no "Features Not Found" section exists, output "No gaps identified"

Output exactly what's in "Features Not Found", nothing more.""",
        tools=[],
    )


def run_ci_questions(agent: Agent, arch_findings: str) -> str:
    """Execute CI question phase - identifies gaps without user interaction."""
    result = agent(
        f"""Copy the items from "Features Not Found" below. Do not add anything else.

ARCHITECTURE FINDINGS:
{arch_findings}

Copy items from "Features Not Found" section only:"""
    )
    return str(result)


def create_ci_sparring_agent(model_id: str | BedrockModel) -> Agent:
    """Create agent that challenges architecture without user interaction."""
    system_prompt = """Assess risks for the CONFIRMED GAPS provided.

RULES:
1. Only assess risks for gaps explicitly listed in the input
2. If no gaps provided, say "No significant risks identified"

Format per gap:
- Risk: [gap name] - Impact: [High/Medium/Low] - Mitigation: [action]"""

    directive = get_directive("ci")
    if directive:
        system_prompt += f"\n\n{directive}"

    return Agent(
        name="SparringAgent",
        model=model_id,
        callback_handler=None,
        system_prompt=system_prompt,
        tools=[],
    )


def run_ci_sparring(agent: Agent, qa_findings: str) -> str:
    """Execute CI sparring phase - challenges architecture without user interaction."""
    result = agent(
        f"""Assess risks ONLY for these confirmed gaps:

CONFIRMED GAPS:
{qa_findings}

For each gap above, provide: Risk, Impact, Mitigation.
Do NOT add risks for features not in the gaps list."""
    )
    return str(result)


def create_ci_review_agent(model_id: str | BedrockModel) -> Agent:
    """Create concise review agent for CI mode."""
    system_prompt = """Write a brief architecture review based ONLY on provided gaps/risks.

## Summary
2 sentences on overall assessment.

## Confirmed Gaps
List only genuine gaps from input. If input says "No gaps identified" or gaps are
empty/trivial, write "No confirmed gaps."
Do NOT list items here that you then dismiss as non-issues.

## Risks & Mitigations
List only risks from input. If no genuine gaps exist, write "No significant risks identified."

## Verdict
- PASS: No gaps, or all requirements met (including via service defaults)
- PASS WITH CONCERNS: Has genuine gaps that warrant attention but no active vulnerabilities
- FAIL: Only for actively exploitable security vulnerabilities or violations of
  STATED requirements.

RULES:
1. Do NOT invent gaps or risks not in the input
2. If "Features Verified" shows a feature exists, it is NOT a gap
3. Items marked "RESOLVED" are NOT confirmed gaps -- exclude them
4. Base verdict on CONFIRMED gaps only"""

    directive = get_directive("ci")
    if directive:
        system_prompt += f"\n\n{directive}"

    return Agent(
        name="ReviewAgent",
        model=model_id,
        callback_handler=None,
        system_prompt=system_prompt,
        tools=[],
    )


def generate_ci_review(
    agent: Agent,
    qa_findings: str = "",
    sparring_findings: str = "",
) -> str:
    """Generate concise CI review from extracted findings."""
    return str(
        agent(
            f"""Write review based ONLY on these inputs. Do not add new gaps.

CONFIRMED GAPS:
{qa_findings if qa_findings.strip() else "None identified"}

ASSESSED RISKS:
{sparring_findings if sparring_findings.strip() else "None identified"}

Base your verdict on the gaps and risks above. If none, verdict is PASS."""
        )
    )
