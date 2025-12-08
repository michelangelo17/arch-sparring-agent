"""Architecture analysis agent for Phase 2."""

from strands import Agent, tool

from ..config import create_session_manager
from ..tools.cfn_analyzer import CloudFormationAnalyzer
from ..tools.diagram_analyzer import DiagramAnalyzer
from ..tools.source_analyzer import SourceAnalyzer


def create_architecture_agent(
    templates_dir: str,
    diagrams_dir: str,
    model_id: str = "amazon.nova-2-lite-v1:0",
    memory_config=None,
    source_dir: str | None = None,
) -> Agent:
    """Create agent for analyzing CloudFormation templates, diagrams, and source code."""

    cfn_analyzer = CloudFormationAnalyzer(templates_dir)
    diagram_analyzer = DiagramAnalyzer(diagrams_dir)
    source_analyzer = SourceAnalyzer(source_dir) if source_dir else None

    @tool
    def read_cloudformation_template(filename: str) -> str:
        """Read a CloudFormation template."""
        return cfn_analyzer.read_template(filename)

    @tool
    def list_cloudformation_templates() -> list[str]:
        """List available CloudFormation templates."""
        return cfn_analyzer.list_templates()

    @tool
    def read_architecture_diagram(filename: str) -> str:
        """Analyze an architecture diagram image."""
        return diagram_analyzer.read_diagram(filename)

    @tool
    def list_architecture_diagrams() -> list[str]:
        """List available architecture diagrams."""
        return diagram_analyzer.list_diagrams()

    tools = [
        read_cloudformation_template,
        list_cloudformation_templates,
        read_architecture_diagram,
        list_architecture_diagrams,
    ]

    if source_analyzer:

        @tool
        def list_source_files() -> list[str]:
            """List Lambda handler and application source files."""
            return source_analyzer.list_source_files()

        @tool
        def read_source_file(filename: str) -> str:
            """Read a source code file to understand business logic."""
            return source_analyzer.read_source_file(filename)

        @tool
        def search_source_code(pattern: str) -> str:
            """Search for a pattern in source code (e.g., 'language', 'aspect', 'cache')."""
            return source_analyzer.search_source(pattern)

        tools.extend([list_source_files, read_source_file, search_source_code])

    session_manager = None
    if memory_config:
        session_manager = create_session_manager(memory_config)

    base_prompt = """Analyze infrastructure and verify feature implementations.

Tasks:
1. Read ALL CloudFormation templates
2. Analyze architecture diagrams"""

    if source_analyzer:
        base_prompt += """
3. For EACH requirement, search source code to verify it exists:
   - search_source_code("language") for language handling
   - search_source_code("aspect") for aspect extraction
   - search_source_code("cache") for caching logic
   - search_source_code("pii") for PII scanning
4. Read files to confirm implementation details

CRITICAL: You MUST search source code before marking features as "Not Found"."""

    base_prompt += """

Output format:
### Components
List infrastructure components

### Features Verified
- Feature: [file and line where found]

### Features Not Found
- Feature: [only if searched and not found]"""

    return Agent(
        name="ArchitectureEvaluator",
        model=model_id,
        system_prompt=base_prompt,
        tools=tools,
        session_manager=session_manager,
    )
