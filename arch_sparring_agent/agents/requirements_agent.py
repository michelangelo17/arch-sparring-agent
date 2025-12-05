"""Requirements analysis agent for Phase 1."""

from strands import Agent, tool

from ..config import create_session_manager


def create_requirements_agent(
    documents_dir: str,
    model_id: str = "amazon.nova-2-lite-v1:0",
    memory_config=None,
) -> Agent:
    """Create agent for analyzing requirements documents."""

    from ..tools.document_parser import DocumentParser

    parser = DocumentParser(documents_dir)

    @tool
    def read_document(filename: str) -> str:
        """Read a markdown document."""
        doc = parser.read_markdown_file(filename)
        return f"Content from {filename}:\n\n{doc['content']}"

    @tool
    def list_available_documents() -> list[str]:
        """List available markdown documents."""
        return parser.list_documents()

    session_manager = None
    if memory_config:
        session_manager = create_session_manager(memory_config)

    return Agent(
        name="RequirementsAnalyst",
        model=model_id,
        system_prompt="""Analyze requirements documents.

Tasks:
1. Read documents using read_document
2. Extract requirements, constraints, NFRs
3. Return concise summary

Focus on functional requirements, constraints, and non-functional requirements.""",
        tools=[read_document, list_available_documents],
        session_manager=session_manager,
    )
