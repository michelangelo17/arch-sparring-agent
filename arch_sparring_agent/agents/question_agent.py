"""Question agent for Phase 3 - clarifying questions."""

from strands import Agent, tool

from ..tools.cfn_analyzer import CloudFormationAnalyzer
from ..tools.source_analyzer import SourceAnalyzer


def create_question_agent(
    model_id: str = "amazon.nova-2-lite-v1:0",
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

        tools.extend([search_templates, read_template])

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

    system_prompt = """You verify gaps before asking users. Your workflow:

1. For each item in "Features Not Found":
   - FIRST use search_templates or search_source to look for evidence
   - Search for relevant keywords (e.g., for "encryption" search: SSESpecification,
     ServerSideEncryptionConfiguration, KMSKeyId, encrypt)
   - If found: it's NOT a gap, skip it
   - If NOT found after searching: ask the user

2. Only call ask_user for gaps you could NOT verify via search
3. One question at a time
4. If user says "no" or "none", move on
5. Call done_asking when done

IMPORTANT: Do NOT ask about things you can find in templates/source code.
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
