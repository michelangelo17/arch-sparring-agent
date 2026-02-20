"""Architecture analysis agent for Phase 2."""

import logging

from strands import Agent, tool
from strands.models import BedrockModel

from ..config import AGENT_ARCHITECTURE
from ..profiles import get_directive
from ..tools.cfn_analyzer import CloudFormationAnalyzer
from ..tools.diagram_analyzer import DiagramAnalyzer
from ..tools.source_analyzer import SourceAnalyzer
from . import create_kb_tool

logger = logging.getLogger(__name__)

_BASE_PROMPT = """Analyze infrastructure and verify feature implementations.

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

_SOURCE_TASK_TEMPLATE = """
{task_num}. Search source code to understand:
   - Business logic implementation
   - SDK calls (dynamodb, s3, sns, etc.) - what services are used at runtime
   - Environment variables - what resources are referenced
{next_task}. If source includes IaC definitions (CDK/Terraform), note them but use
   CloudFormation for actual deployed config since that's what's deployed"""

_OUTPUT_FORMAT = """

Output format:
### Components
List from CloudFormation + SDK calls observed in source code

### Features Verified
- Feature: [evidence from CFN or source code]
- Include features satisfied by AWS service defaults (note "via AWS default" as evidence)

### Features Not Found
- Feature: [only if searched AND not covered by service defaults]"""

_WAF_QUERY_PROMPT = """Based on the architecture you just analyzed, use the query_waf tool to
assess this architecture against the AWS Well-Architected Framework.

1. Identify which WAF lenses apply based on the services and patterns you found
   (e.g. Lambda/API Gateway → serverless-applications-lens,
   multi-tenant → saas-lens, Bedrock/SageMaker → generative-ai-lens).

2. Call query_waf for EACH relevant pillar:
   - Security: encryption, IAM least privilege, network controls
   - Reliability: failure handling, recovery, scaling limits
   - Performance: right-sizing, caching, timeout alignment
   - Cost Optimization: provisioning mode, unused resources
   - Operational Excellence: monitoring, alerting, observability

Use the sources parameter to scope queries:
- "well-architected-framework" = core WAF (always include)
- Technology lenses: "serverless-applications-lens", "saas-lens",
  "analytics-lens", "machine-learning-lens", "generative-ai-lens",
  "iot-lens", "container-build-lens", "devops-guidance",
  "high-performance-computing-lens", "streaming-media-lens",
  "amazon-opensearch-service-lens", "migration-lens"
- Industry lenses: "games-industry-lens",
  "financial-services-industry-lens", "healthcare-industry-lens",
  "government-lens", "sap-lens", "supply-chain-lens",
  "connected-mobility-lens",
  "modern-industrial-data-technology-lens",
  "mergers-and-acquisitions-lens"

Pass sources=["well-architected-framework", "<lens>"] to get both
core and lens-specific results.

CRITICAL: Only include recommendations that apply to the ACTUAL services and
compute model in use. If the architecture is serverless (Lambda, Step Functions,
DynamoDB), do NOT recommend EC2-specific practices like Spot Instances, Reserved
Instances, or instance right-sizing. Contextualize every recommendation against
what you actually found in the templates and source code. Flag real gaps only.

MANDATORY: You MUST call query_waf for each pillar. Do NOT write WAF findings
from your own knowledge.

Output format:
### WAF Assessment
- Pillar — Finding: [WAF recommendation vs actual implementation]
  Include the [source: ...] tag from the query_waf results so the reader knows
  which WAF document or lens the recommendation came from.
  Example: "Security — No encryption at rest for DynamoDB [source: well-architected-framework]"
  Example: "Reliability — No async retry strategy [source: serverless-applications-lens]"
  If the query_waf result has no source tag, use the s3 URI instead."""


def create_architecture_agent(
    templates_dir: str,
    diagrams_dir: str,
    model: str | BedrockModel,
    source_dir: str | None = None,
    knowledge_base_id: str | None = None,
    region: str | None = None,
    profile: dict | None = None,
) -> Agent:
    """Create agent for analyzing CloudFormation templates, diagrams, and source code."""

    # DiagramAnalyzer uses raw Bedrock converse API and needs a string model_id
    diagram_model_id = model.config["model_id"] if isinstance(model, BedrockModel) else model

    cfn_analyzer = CloudFormationAnalyzer(templates_dir)
    diagram_analyzer = DiagramAnalyzer(diagrams_dir, model_id=diagram_model_id)
    source_analyzer = SourceAnalyzer(source_dir) if source_dir else None

    @tool
    def read_cloudformation_template(filename: str) -> str:
        """Read a CloudFormation template."""
        logger.info("[%s] read_cloudformation_template(%s)", AGENT_ARCHITECTURE, filename)
        return cfn_analyzer.read_template(filename)

    @tool
    def list_cloudformation_templates() -> list[str]:
        """List available CloudFormation templates."""
        templates = cfn_analyzer.list_templates()
        logger.info("[%s] list_cfn_templates → %d files", AGENT_ARCHITECTURE, len(templates))
        return templates

    @tool
    def read_architecture_diagram(filename: str) -> str:
        """Analyze an architecture diagram image."""
        logger.info("[%s] read_architecture_diagram(%s)", AGENT_ARCHITECTURE, filename)
        return diagram_analyzer.read_diagram(filename)

    @tool
    def list_architecture_diagrams() -> list[str]:
        """List available architecture diagrams."""
        diagrams = diagram_analyzer.list_diagrams()
        logger.info("[%s] list_architecture_diagrams → %d files", AGENT_ARCHITECTURE, len(diagrams))
        return diagrams

    tools = [
        read_cloudformation_template,
        list_cloudformation_templates,
        read_architecture_diagram,
        list_architecture_diagrams,
    ]
    logger.info("[%s] CFN tools enabled (dir: %s)", AGENT_ARCHITECTURE, templates_dir)
    logger.info("[%s] Diagram tools enabled (dir: %s)", AGENT_ARCHITECTURE, diagrams_dir)

    if source_analyzer:

        @tool
        def list_source_files() -> list[str]:
            """List Lambda handler and application source files."""
            files = source_analyzer.list_source_files()
            logger.info("[%s] list_source_files → %d files", AGENT_ARCHITECTURE, len(files))
            return files

        @tool
        def read_source_file(filename: str) -> str:
            """Read a source code file to understand business logic."""
            logger.info("[%s] read_source(%s)", AGENT_ARCHITECTURE, filename)
            content = source_analyzer.read_source_file(filename)
            logger.info(
                "[%s] read_source(%s) → %d chars", AGENT_ARCHITECTURE, filename, len(content)
            )
            return content

        @tool
        def search_source_code(pattern: str) -> str:
            """Search for a pattern in source code."""
            logger.info("[%s] search_source(%r)", AGENT_ARCHITECTURE, pattern)
            result = source_analyzer.search_source(pattern)
            logger.info(
                "[%s] search_source(%r) → %d chars", AGENT_ARCHITECTURE, pattern, len(result)
            )
            return result

        tools.extend([list_source_files, read_source_file, search_source_code])
        logger.info("[%s] Source tools enabled (dir: %s)", AGENT_ARCHITECTURE, source_dir)

    if knowledge_base_id and region:
        tools.append(create_kb_tool(knowledge_base_id, region, with_sources_filter=True))
        logger.info(
            "[%s] KB tool enabled (kb: %s, region: %s)",
            AGENT_ARCHITECTURE,
            knowledge_base_id,
            region,
        )

    base_prompt = _BASE_PROMPT

    task_num = 3
    if source_analyzer:
        base_prompt += _SOURCE_TASK_TEMPLATE.format(task_num=task_num, next_task=task_num + 1)
        task_num += 2

    base_prompt += _OUTPUT_FORMAT

    directive = get_directive(profile, "architecture")
    if directive:
        base_prompt += f"\n\n{directive}"

    return Agent(
        name=AGENT_ARCHITECTURE,
        model=model,
        callback_handler=None,
        system_prompt=base_prompt,
        tools=tools,
    )


def run_architecture(agent: Agent, requirements: str, *, has_kb: bool = False) -> str:
    """Execute architecture analysis, optionally followed by WAF KB assessment.

    Args:
        agent: The architecture agent (created via ``create_architecture_agent``).
        requirements: Extracted requirements from Phase 1.
        has_kb: If True, runs a second pass querying the WAF Knowledge Base.

    The agent is stateful -- conversation history from the first pass is
    available during the second pass, so the KB queries can reference
    specific components and patterns found in the infrastructure analysis.
    """
    result = agent(
        f"""Analyze all templates, diagrams, and source code.

REQUIREMENTS:
{requirements}

Summarize architecture, patterns, and verify which requirements have implementations."""
    )
    arch_output = str(result)

    if has_kb:
        waf_result = agent(_WAF_QUERY_PROMPT)
        arch_output += "\n" + str(waf_result)

    return arch_output
