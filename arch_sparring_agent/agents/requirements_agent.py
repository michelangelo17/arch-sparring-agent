"""Requirements analysis agent for Phase 1."""

from strands import Agent, tool


def create_requirements_agent(
    documents_dir: str,
    model_id: str = "amazon.nova-2-lite-v1:0",
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

    return Agent(
        name="RequirementsAnalyst",
        model=model_id,
        system_prompt="""Analyze requirements documents.

Tasks:
1. List all documents using list_available_documents
2. Read EVERY document using read_document (do not skip any)
3. Extract requirements, constraints, NFRs from ALL documents
4. Return concise summary

IMPORTANT: Read ALL markdown files including README.md. Do not skip any documents.
Look for deployment instructions, troubleshooting guides, and prerequisites.""",
        tools=[read_document, list_available_documents],
    )
