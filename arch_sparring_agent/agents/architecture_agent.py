"""Architecture analysis agent for Phase 2."""

from strands import Agent, tool

from ..config import create_session_manager
from ..tools.cfn_analyzer import CloudFormationAnalyzer
from ..tools.diagram_analyzer import DiagramAnalyzer


def create_architecture_agent(
    templates_dir: str,
    diagrams_dir: str,
    model_id: str = "amazon.nova-2-lite-v1:0",
    memory_config=None,
) -> Agent:
    """Create agent for analyzing CloudFormation templates and diagrams."""

    cfn_analyzer = CloudFormationAnalyzer(templates_dir)
    diagram_analyzer = DiagramAnalyzer(diagrams_dir)

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

    session_manager = None
    if memory_config:
        session_manager = create_session_manager(memory_config)

    return Agent(
        name="ArchitectureEvaluator",
        model=model_id,
        system_prompt="""Analyze CloudFormation templates and architecture diagrams.

Tasks:
1. Read templates using read_cloudformation_template
2. Analyze diagrams using read_architecture_diagram
3. Identify patterns, components, and integration points

Focus on: security, reliability, performance, cost, operational excellence.""",
        tools=[
            read_cloudformation_template,
            list_cloudformation_templates,
            read_architecture_diagram,
            list_architecture_diagrams,
        ],
        session_manager=session_manager,
    )
