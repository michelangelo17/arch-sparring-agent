"""Question agent for Phase 3 - clarifying questions."""

from __future__ import annotations

import logging

from strands import Agent, tool
from strands.models import BedrockModel

from ..config import AGENT_QUESTION
from ..tools import search_content
from ..tools.cfn_analyzer import CloudFormationAnalyzer
from ..tools.source_analyzer import SourceAnalyzer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You verify gaps and identify concerns before asking users. Your workflow:

1. FIRST use list_templates to discover available template files
2. For each item in "Features Not Found":
   - Use search_templates with relevant CloudFormation keywords (e.g., for "encryption"
     search: SSESpecification, ServerSideEncryptionConfiguration, KMSKeyId, encrypt)
   - Use read_template with actual filenames from list_templates (do NOT guess filenames)
   - If source tools available, also use search_source
   - If found: it's NOT a gap, skip it
   - If NOT found after searching: use ask_user to confirm with the user

3. If "Features Not Found" is empty or says "None", proactively check for common
   architectural concerns using search tools:
   - Security: encryption at rest (SSE, KMS), encryption in transit (TLS/SSL)
   - Error handling: DLQ, retries, circuit breakers
   - Monitoring: alarms, dashboards, alerting
   - Access control: least-privilege IAM, resource policies
   For any concern NOT found in templates/source, use ask_user to ask the user about it.

4. CRITICAL: You MUST use the ask_user tool for ALL communication with the user.
   NEVER output analysis text directly -- always go through ask_user.
5. CRITICAL: Call ask_user exactly ONCE per turn. Wait for the user's response
   before issuing the next question. NEVER batch multiple ask_user calls in a
   single response.
6. If user says "no" or "none", move on
7. Call done_asking when done

IMPORTANT: Do NOT ask about things you can find in templates/source code.
IMPORTANT: Always use list_templates first -- never guess template filenames.
IMPORTANT: ALL questions and findings MUST be communicated via the ask_user tool.

HANDLING USER RESPONSES:
- Record the user's response as-is and pass it along
- If the user says something is intentional or not needed, note their reasoning
- Do NOT challenge or debate -- that is the sparring agent's job
- Include the user's responses alongside the gaps so the sparring agent has context

After done_asking, summarize ALL gaps with the user's responses so the sparring
agent has full context. Format:
- Gap: [description] -- User response: [what the user said]
- Gap: [description] -- Not addressed by user
Items verified in templates/source (not gaps) can be omitted."""


def create_question_agent(
    model: BedrockModel,
    templates_dir: str | None = None,
    source_dir: str | None = None,
) -> Agent:
    """Create agent for asking clarifying questions with verification tools."""

    questions_asked = []
    cfn_analyzer = CloudFormationAnalyzer(templates_dir) if templates_dir else None
    source_analyzer = SourceAnalyzer(source_dir) if source_dir else None

    @tool
    def ask_user(question: str) -> str:
        """Ask a clarifying question to the user."""
        questions_asked.append(question)
        print(f"\n❓ [{len(questions_asked)}] {question}")
        return input("Your answer: ")

    @tool
    def done_asking() -> str:
        """Signal completion of question phase."""
        return "Proceeding to sparring phase."

    tools = [ask_user, done_asking]

    if cfn_analyzer:

        @tool
        def list_templates() -> list[str]:
            """List available CloudFormation template files."""
            templates = cfn_analyzer.list_templates()
            logger.info("[%s] list_templates → %d files", AGENT_QUESTION, len(templates))
            return templates

        @tool
        def search_templates(pattern: str) -> str:
            """Search CloudFormation templates for a pattern (e.g., 'encryption')."""
            logger.info("[%s] search_templates(%r)", AGENT_QUESTION, pattern)
            results: list[str] = []
            for template_name in cfn_analyzer.list_templates():
                content = cfn_analyzer.read_template(template_name)
                match_block = search_content(content, pattern, template_name)
                if match_block:
                    results.append(match_block)
            if not results:
                logger.info("[%s] search_templates(%r) → no matches", AGENT_QUESTION, pattern)
                return f"No matches for '{pattern}' in CloudFormation templates."
            logger.info(
                "[%s] search_templates(%r) → %d matches", AGENT_QUESTION, pattern, len(results)
            )
            return "".join(results[:10])

        @tool
        def read_template(filename: str) -> str:
            """Read a specific CloudFormation template."""
            logger.info("[%s] read_template(%s)", AGENT_QUESTION, filename)
            return cfn_analyzer.read_template(filename)

        tools.extend([list_templates, search_templates, read_template])
        logger.info("[%s] CFN tools enabled (dir: %s)", AGENT_QUESTION, templates_dir)

    if source_analyzer:

        @tool
        def search_source(pattern: str) -> str:
            """Search source code for a pattern."""
            logger.info("[%s] search_source(%r)", AGENT_QUESTION, pattern)
            result = source_analyzer.search_source(pattern)
            logger.info("[%s] search_source(%r) → %d chars", AGENT_QUESTION, pattern, len(result))
            return result

        @tool
        def read_source(filename: str) -> str:
            """Read a specific source file."""
            logger.info("[%s] read_source(%s)", AGENT_QUESTION, filename)
            content = source_analyzer.read_source_file(filename)
            logger.info("[%s] read_source(%s) → %d chars", AGENT_QUESTION, filename, len(content))
            return content

        tools.extend([search_source, read_source])
        logger.info("[%s] Source tools enabled (dir: %s)", AGENT_QUESTION, source_dir)

    return Agent(
        name=AGENT_QUESTION,
        model=model,
        callback_handler=None,
        system_prompt=_SYSTEM_PROMPT,
        tools=tools,
    )


def run_questions(agent: Agent, arch_findings: str) -> str:
    """Execute question phase with extracted findings."""
    from . import safe_invoke

    return safe_invoke(
        agent,
        f"""Review the "Features Not Found" section below.

BEFORE asking the user:
1. Use search_templates to look for each "missing" feature
2. Use search_source if source code tools are available
3. Only ask about items you truly cannot find

ARCHITECTURE FINDINGS:
{arch_findings}

Verify gaps via search first, then ask about remaining unknowns. Call done_asking when ready.""",
    )
