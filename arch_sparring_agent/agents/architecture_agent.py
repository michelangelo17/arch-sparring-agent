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
    """
    Create the Architecture Analysis Agent.

    templates_dir can point to regular CloudFormation templates or CDK cdk.out/ directory.
    """

    cfn_analyzer = CloudFormationAnalyzer(templates_dir)
    diagram_analyzer = DiagramAnalyzer(diagrams_dir)

    @tool
    def read_cloudformation_template(filename: str) -> str:
        """Read a CloudFormation template file."""
        return cfn_analyzer.read_template(filename)

    @tool
    def list_cloudformation_templates() -> list[str]:
        """List all CloudFormation templates."""
        return cfn_analyzer.list_templates()

    @tool
    def read_architecture_diagram(filename: str) -> str:
        """Analyze an architecture diagram image."""
        return diagram_analyzer.read_diagram(filename)

    @tool
    def list_architecture_diagrams() -> list[str]:
        """List all architecture diagram images."""
        return diagram_analyzer.list_diagrams()

    @tool
    def ask_user_question(question: str) -> str:
        """Ask the user a clarifying question and get their response."""
        print(f"\n❓ {question}")
        response = input("Your answer: ")
        return response

    # Set up session manager if memory config provided
    session_manager = None
    if memory_config:
        session_manager = create_session_manager(memory_config)

    agent = Agent(
        name="ArchitectureEvaluator",
        model=model_id,
        system_prompt="""AWS architecture evaluator. Be concise.

Tasks:
1. Read CloudFormation templates using read_cloudformation_template
2. Analyze diagrams using read_architecture_diagram
3. Identify patterns, risks, and issues
4. When unclear, USE ask_user_question tool to ask the user directly

IMPORTANT: When you need clarification, call ask_user_question - don't just write questions.

Focus on: security, reliability, performance, cost, operational excellence.""",
        tools=[
            read_cloudformation_template,
            list_cloudformation_templates,
            read_architecture_diagram,
            list_architecture_diagrams,
            ask_user_question,
        ],
        session_manager=session_manager,
    )

    return agent
