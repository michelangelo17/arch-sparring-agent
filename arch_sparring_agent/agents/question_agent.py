"""Question agent for Phase 3 - clarifying questions."""

from strands import Agent, tool

from ..config import MODEL_ID
from ..tools.cfn_analyzer import CloudFormationAnalyzer
from ..tools.source_analyzer import SourceAnalyzer


def create_question_agent(
    model_id: str = MODEL_ID,
    templates_dir: str | None = None,
    source_dir: str | None = None,
) -> Agent:
    """Create agent for asking clarifying questions with verification tools."""

    questions_asked = []

    # Setup analyzers for verification
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

    # Add verification tools if directories provided
    if cfn_analyzer:

        @tool
        def list_templates() -> list[str]:
            """List available CloudFormation template files."""
            return cfn_analyzer.list_templates()

        @tool
        def search_templates(pattern: str) -> str:
            """Search CloudFormation templates for a pattern (e.g., 'encryption')."""
            results = []
            for template_name in cfn_analyzer.list_templates():
                content = cfn_analyzer.read_template(template_name)
                if pattern.lower() in content.lower():
                    lines = content.split("\n")
                    matches = []
                    for i, line in enumerate(lines, 1):
                        if pattern.lower() in line.lower():
                            matches.append(f"  L{i}: {line.strip()}")
                    if matches:
                        results.append(f"\n{template_name}:\n" + "\n".join(matches[:5]))
            if not results:
                return f"No matches for '{pattern}' in CloudFormation templates."
            return "".join(results[:10])

        @tool
        def read_template(filename: str) -> str:
            """Read a specific CloudFormation template."""
            return cfn_analyzer.read_template(filename)

        tools.extend([list_templates, search_templates, read_template])

    if source_analyzer:

        @tool
        def search_source(pattern: str) -> str:
            """Search source code for a pattern."""
            return source_analyzer.search_source(pattern)

        @tool
        def read_source(filename: str) -> str:
            """Read a specific source file."""
            return source_analyzer.read_source_file(filename)

        tools.extend([search_source, read_source])

    system_prompt = """You verify gaps and identify concerns before asking users. Your workflow:

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
5. One question at a time via ask_user
6. If user says "no" or "none", move on
7. Call done_asking when done

IMPORTANT: Do NOT ask about things you can find in templates/source code.
IMPORTANT: Always use list_templates first -- never guess template filenames.
IMPORTANT: ALL questions and findings MUST be communicated via the ask_user tool.
After done_asking, summarize confirmed gaps in 2-3 bullet points max."""

    return Agent(
        name="QuestionAgent",
        model=model_id,
        system_prompt=system_prompt,
        tools=tools,
    )


def run_questions(agent: Agent, req_findings: str, arch_findings: str) -> str:
    """Execute question phase with extracted findings."""
    result = agent(
        f"""Review the "Features Not Found" section below.

BEFORE asking the user:
1. Use search_templates to look for each "missing" feature
2. Use search_source if source code tools are available
3. Only ask about items you truly cannot find

ARCHITECTURE FINDINGS:
{arch_findings}

Verify gaps via search first, then ask about remaining unknowns. Call done_asking when ready."""
    )
    return str(result)
