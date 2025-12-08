"""Non-interactive agents for CI/CD pipeline execution."""

from strands import Agent


def create_ci_question_agent(model_id: str = "amazon.nova-2-lite-v1:0") -> Agent:
    """Create agent that identifies gaps without user interaction."""
    return Agent(
        name="QuestionAgent",
        model=model_id,
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


def run_ci_questions(agent: Agent, req_summary: str, arch_summary: str) -> str:
    """Execute CI question phase - identifies gaps without user interaction."""
    result = agent(
        f"""Copy the items from "Features Not Found" below. Do not add anything else.

ARCHITECTURE ANALYSIS:
{arch_summary}

Copy items from "Features Not Found" section only:"""
    )
    return str(result)


def create_ci_sparring_agent(model_id: str = "amazon.nova-2-lite-v1:0") -> Agent:
    """Create agent that challenges architecture without user interaction."""
    return Agent(
        name="SparringAgent",
        model=model_id,
        system_prompt="""Assess risks for the CONFIRMED GAPS provided.

RULES:
1. Only assess risks for gaps explicitly listed in the input
2. Do NOT invent new gaps or risks
3. If a gap is about a missing feature (not security), impact is Medium
4. Security/compliance gaps are High impact
5. If no gaps provided, say "No significant risks identified"

Format per gap:
- Risk: [gap name] - Impact: [High/Medium/Low] - Mitigation: [action]""",
        tools=[],
    )


def run_ci_sparring(agent: Agent, req_summary: str, arch_summary: str, gaps_context: str) -> str:
    """Execute CI sparring phase - challenges architecture without user interaction."""
    result = agent(
        f"""Assess risks ONLY for these confirmed gaps:

CONFIRMED GAPS:
{gaps_context}

For each gap above, provide: Risk, Impact, Mitigation.
Do NOT add risks for features not in the gaps list."""
    )
    return str(result)


def create_ci_review_agent(model_id: str = "amazon.nova-2-lite-v1:0") -> Agent:
    """Create concise review agent for CI mode."""
    return Agent(
        name="ReviewAgent",
        model=model_id,
        system_prompt="""Write a brief architecture review based ONLY on provided gaps/risks.

## Summary
2 sentences on overall assessment.

## Confirmed Gaps
List only the gaps provided in input.

## Risks & Mitigations
List only risks provided in input.

## Verdict
- PASS: No gaps, or only minor gaps
- PASS WITH CONCERNS: Has gaps but no security/compliance blockers
- FAIL: Security vulnerability or compliance violation

RULES:
1. Do NOT invent gaps or risks not in the input
2. If "Features Verified" shows a feature exists, it is NOT a gap
3. Base verdict on CONFIRMED gaps only""",
        tools=[],
    )


def generate_ci_review(
    agent: Agent,
    req_summary: str,
    arch_summary: str,
    gaps_context: str = "",
    risks_context: str = "",
) -> str:
    """Generate concise CI review."""
    return str(
        agent(
            f"""Write review based ONLY on these inputs. Do not add new gaps.

CONFIRMED GAPS:
{gaps_context if gaps_context.strip() else "None identified"}

ASSESSED RISKS:
{risks_context if risks_context.strip() else "None identified"}

Base your verdict on the gaps and risks above. If none, verdict is PASS."""
        )
    )
