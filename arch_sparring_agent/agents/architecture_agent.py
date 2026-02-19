"""Architecture analysis agent for Phase 2."""

from strands import Agent, tool
from strands.models import BedrockModel

from ..profiles import get_directive
from ..tools.cfn_analyzer import CloudFormationAnalyzer
from ..tools.diagram_analyzer import DiagramAnalyzer
from ..tools.source_analyzer import SourceAnalyzer


def create_architecture_agent(
    templates_dir: str,
    diagrams_dir: str,
    model_id: str | BedrockModel,
    source_dir: str | None = None,
    knowledge_base_id: str | None = None,
    region: str | None = None,
) -> Agent:
    """Create agent for analyzing CloudFormation templates, diagrams, and source code."""

    # DiagramAnalyzer uses raw Bedrock converse API and needs a string model_id
    diagram_model_id = (
        model_id.config["model_id"] if isinstance(model_id, BedrockModel) else model_id
    )

    cfn_analyzer = CloudFormationAnalyzer(templates_dir)
    diagram_analyzer = DiagramAnalyzer(diagrams_dir, model_id=diagram_model_id)
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

    if knowledge_base_id and region:
        from ..tools.kb_client import KnowledgeBaseClient

        _kb_client = KnowledgeBaseClient(knowledge_base_id, region)

        @tool
        def query_waf(query: str) -> str:
            """Query the AWS Well-Architected Framework knowledge base for best practices."""
            return _kb_client.query(query)

        tools.append(query_waf)

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
- Include features satisfied by AWS service defaults (note "via AWS default" as evidence)

### Features Not Found
- Feature: [only if searched AND not covered by service defaults]"""

    if knowledge_base_id:
        base_prompt += """

WAF KNOWLEDGE BASE:
You have access to the AWS Well-Architected Framework via the query_waf tool.
Use it to look up best practices when evaluating security controls, reliability
patterns, performance strategies, and cost optimization. Cite specific WAF
recommendations in your analysis."""

    directive = get_directive("architecture")
    if directive:
        base_prompt += f"\n\n{directive}"

    return Agent(
        name="ArchitectureEvaluator",
        model=model_id,
        callback_handler=None,
        system_prompt=base_prompt,
        tools=tools,
    )
