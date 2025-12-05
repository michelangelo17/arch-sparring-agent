from strands import Agent, tool

from ..config import create_session_manager


def create_requirements_agent(
    documents_dir: str,
    model_id: str = "amazon.nova-2-lite-v1:0",
    memory_config=None,
) -> Agent:
    """Create the Requirements Analysis Agent."""

    from ..tools.document_parser import DocumentParser

    parser = DocumentParser(documents_dir)

    @tool
    def read_document(filename: str) -> str:
        """Read a markdown document."""
        doc = parser.read_markdown_file(filename)
        return f"Content from {filename}:\n\n{doc['content']}"

    @tool
    def list_available_documents() -> list[str]:
        """List all available markdown documents."""
        return parser.list_documents()

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
        name="RequirementsAnalyst",
        model=model_id,
        system_prompt="""Requirements analyst for architecture reviews. Be concise.

Tasks:
1. Read markdown documents using read_document tool
2. Extract requirements, constraints, NFRs
3. When information is missing, USE ask_user_question tool to ask the user directly

IMPORTANT: When you need clarification, call ask_user_question - don't just write questions.

Return a brief summary of requirements found.""",
        tools=[read_document, list_available_documents, ask_user_question],
        session_manager=session_manager,
    )

    return agent
