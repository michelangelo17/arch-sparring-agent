"""Requirements analysis agent for Phase 1."""

from strands import Agent, tool

from ..config import DOC_CHUNK_SUMMARY_THRESHOLD, DOC_SUMMARY_THRESHOLD
from ..context_condenser import _chunked_extract


def create_requirements_agent(
    documents_dir: str,
    model_id: str = "amazon.nova-2-lite-v1:0",
) -> Agent:
    """Create agent for analyzing requirements documents."""

    from ..tools.document_parser import DocumentParser

    parser = DocumentParser(documents_dir)

    summarize_prompt = (
        "Summarize this document part. Preserve ALL requirements, constraints, "
        "and technical details. Be concise but comprehensive."
    )

    @tool
    def read_document(filename: str) -> str:
        """Read a markdown document. Summarizes if too long."""
        doc = parser.read_markdown_file(filename)
        content = str(doc["content"])

        # If content > threshold (~6k tokens), summarize to avoid context overflow
        if len(content) > DOC_SUMMARY_THRESHOLD:
            summarizer = Agent(
                name="DocSummarizer",
                model=model_id,
                system_prompt=summarize_prompt,
                tools=[],
            )
            try:
                # Use chunked extraction for very large files
                if len(content) > DOC_CHUNK_SUMMARY_THRESHOLD:
                    summary = _chunked_extract(content, summarize_prompt, model_id)
                else:
                    summary = str(summarizer(f"Summarize this content:\n\n{content}"))

                return f"Content from {filename} (Summarized):\n\n{summary}"
            except Exception as e:
                # Fallback: try chunking with smaller chunks if single pass failed
                try:
                    summary = _chunked_extract(content, summarize_prompt, model_id)
                    return f"Content from {filename} (Chunk Summarized after error):\n\n{summary}"
                except Exception as chunk_err:
                    return (
                        f"Error reading {filename}: Could not summarize ({e}) "
                        f"or chunk-summarize ({chunk_err})"
                    )

        return f"Content from {filename}:\n\n{content}"

    @tool
    def list_available_documents() -> list[str]:
        """List available markdown documents."""
        return parser.list_documents()

    return Agent(
        name="RequirementsAnalyst",
        model=model_id,
        system_prompt="""Analyze requirements documents.

Tasks:
1. List documents
2. Read EVERY document
3. Extract requirements/constraints/NFRs
4. Return CONCISE summary (max 400 words)

Format:
### Functional Requirements
- Item 1
- Item 2

### Non-Functional Requirements
- Item 1

### Constraints
- Item 1

Do NOT copy text verbatim. Summarize.""",
        tools=[read_document, list_available_documents],
    )
