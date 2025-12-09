"""Architecture analysis agent for Phase 2."""

from strands import Agent, tool

from ..tools.cfn_analyzer import CloudFormationAnalyzer
from ..tools.diagram_analyzer import DiagramAnalyzer
from ..tools.source_analyzer import SourceAnalyzer


def create_architecture_agent(
    templates_dir: str,
    diagrams_dir: str,
    model_id: str = "amazon.nova-2-lite-v1:0",
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
            """Search for a pattern in source code."""
            return source_analyzer.search_source(pattern)

        tools.extend([list_source_files, read_source_file, search_source_code])

    base_prompt = """Analyze infrastructure and verify feature implementations.

UNDERSTANDING THE SOURCES:
- CloudFormation: DEPLOYED infrastructure (IAM policies, resource configs, what EXISTS)
- Source code: RUNTIME behavior (SDK calls, business logic, how resources are USED)

Both are important:
- CloudFormation shows what permissions/resources ARE configured
- Source code shows what permissions/resources are NEEDED (via SDK calls)
- If source code makes SDK calls not covered by CloudFormation IAM, that's a gap

Tasks:
1. Read ALL CloudFormation templates for deployed infrastructure
2. Analyze architecture diagrams"""

    if source_analyzer:
        base_prompt += """
3. Search source code to understand:
   - Business logic implementation
   - SDK calls (dynamodb, s3, sns, etc.) - what services are used at runtime
   - Environment variables - what resources are referenced
4. If source includes IaC definitions (CDK/Terraform), note them but use CloudFormation for
   actual deployed config since that's what's synthesized and deployed"""

    base_prompt += """

Output format:
### Components
List from CloudFormation + SDK calls observed in source code

### Features Verified
- Feature: [evidence from CFN or source code]

### Features Not Found
- Feature: [only if searched and not found]"""

    return Agent(
        name="ArchitectureEvaluator",
        model=model_id,
        system_prompt=base_prompt,
        tools=tools,
    )
