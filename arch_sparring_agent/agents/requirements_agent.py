"""Requirements analysis agent for Phase 1."""

from strands import Agent, tool


def _chunk_summarize(content: str, agent: Agent, chunk_size: int = 50000) -> str:
    """Summarize large content in chunks to avoid truncation."""
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    summaries = []

    for i, chunk in enumerate(chunks, 1):
        try:
            prompt = (
                f"Summarize Part {i}/{len(chunks)}. "
                f"Preserve all requirements and constraints:\n\n{chunk}"
            )
            result = str(agent(prompt))
            summaries.append(f"--- Part {i} Summary ---\n{result}")
        except Exception as e:
            summaries.append(f"--- Part {i} Error ---\nCould not summarize: {e}")

    return "\n\n".join(summaries)


def create_requirements_agent(
    documents_dir: str,
    model_id: str = "amazon.nova-2-lite-v1:0",
) -> Agent:
    """Create agent for analyzing requirements documents."""

    from ..tools.document_parser import DocumentParser

    parser = DocumentParser(documents_dir)

    @tool
    def read_document(filename: str) -> str:
        """Read a markdown document. Summarizes if too long."""
        doc = parser.read_markdown_file(filename)
        content = str(doc["content"])

        # If content > 25k chars (~6k tokens), summarize to avoid context overflow
        if len(content) > 25000:
            summarizer = Agent(
                name="DocSummarizer",
                model=model_id,
                system_prompt=(
                    "Summarize this document part. Preserve ALL requirements, constraints, "
                    "and technical details. Be concise but comprehensive."
                ),
                tools=[],
            )
            try:
                # Use chunked summarization for very large files (>100k chars)
                if len(content) > 100000:
                    summary = _chunk_summarize(content, summarizer)
                else:
                    summary = str(summarizer(f"Summarize this content:\n\n{content}"))

                return f"Content from {filename} (Summarized):\n\n{summary}"
            except Exception as e:
                # Fallback: try chunking if single pass failed
                try:
                    summary = _chunk_summarize(content, summarizer, chunk_size=30000)
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
