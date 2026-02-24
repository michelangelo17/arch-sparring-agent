"""Requirements analysis agent for Phase 1."""

from __future__ import annotations

from strands import Agent, tool
from strands.models import BedrockModel

from ..config import AGENT_REQUIREMENTS, DOC_CHUNK_SUMMARY_THRESHOLD, DOC_SUMMARY_THRESHOLD
from ..exceptions import MODEL_ERRORS
from ..review.context_condenser import chunked_extract

_SYSTEM_PROMPT = """Analyze requirements documents.

Tasks:
1. List documents
2. Read EVERY document
3. Extract requirements/constraints/NFRs
4. Return a structured summary

Format:
### Functional Requirements
- Item 1
- Item 2

### Non-Functional Requirements
- Item 1

### Constraints
- Item 1

Do NOT copy text verbatim. Summarize."""

_SUMMARIZE_PROMPT = (
    "Summarize this document part. Preserve ALL requirements, constraints, "
    "and technical details. Be concise but comprehensive."
)


def create_requirements_agent(
    documents_dir: str,
    model: BedrockModel,
) -> Agent:
    """Create agent for analyzing requirements documents."""

    from ..tools.document_parser import DocumentParser

    parser = DocumentParser(documents_dir)

    @tool
    def read_document(filename: str) -> str:
        """Read a markdown document. Summarizes if too long."""
        doc = parser.read_markdown_file(filename)
        content = str(doc["content"])

        if len(content) <= DOC_SUMMARY_THRESHOLD:
            return f"Content from {filename}:\n\n{content}"

        try:
            if len(content) > DOC_CHUNK_SUMMARY_THRESHOLD:
                summary = chunked_extract(content, _SUMMARIZE_PROMPT, model)
            else:
                summarizer = Agent(
                    name="DocSummarizer",
                    model=model,
                    callback_handler=None,
                    system_prompt=_SUMMARIZE_PROMPT,
                    tools=[],
                )
                summary = str(summarizer(f"Summarize this content:\n\n{content}"))

            return f"Content from {filename} (Summarized):\n\n{summary}"
        except MODEL_ERRORS as e:
            try:
                summary = chunked_extract(content, _SUMMARIZE_PROMPT, model)
                return f"Content from {filename} (Chunk Summarized after error):\n\n{summary}"
            except MODEL_ERRORS as chunk_err:
                return (
                    f"Error reading {filename}: Could not summarize ({e}) "
                    f"or chunk-summarize ({chunk_err})"
                )

    @tool
    def list_available_documents() -> list[str]:
        """List available markdown documents."""
        return parser.list_documents()

    return Agent(
        name=AGENT_REQUIREMENTS,
        model=model,
        callback_handler=None,
        system_prompt=_SYSTEM_PROMPT,
        tools=[read_document, list_available_documents],
    )
