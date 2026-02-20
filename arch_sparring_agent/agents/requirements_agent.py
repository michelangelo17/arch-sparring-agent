"""Requirements analysis agent for Phase 1."""

from botocore.exceptions import ClientError
from strands import Agent, tool
from strands.models import BedrockModel
from strands.types.exceptions import ContextWindowOverflowException, MaxTokensReachedException

from ..config import AGENT_REQUIREMENTS, DOC_CHUNK_SUMMARY_THRESHOLD, DOC_SUMMARY_THRESHOLD
from ..context_condenser import chunked_extract

_SYSTEM_PROMPT = """Analyze requirements documents.

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

Do NOT copy text verbatim. Summarize."""

_SUMMARIZE_PROMPT = (
    "Summarize this document part. Preserve ALL requirements, constraints, "
    "and technical details. Be concise but comprehensive."
)


def create_requirements_agent(
    documents_dir: str,
    model_id: str | BedrockModel,
) -> Agent:
    """Create agent for analyzing requirements documents."""

    from ..tools.document_parser import DocumentParser

    parser = DocumentParser(documents_dir)

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
                callback_handler=None,
                system_prompt=_SUMMARIZE_PROMPT,
                tools=[],
            )
            try:
                if len(content) > DOC_CHUNK_SUMMARY_THRESHOLD:
                    summary = chunked_extract(content, _SUMMARIZE_PROMPT, model_id)
                else:
                    summary = str(summarizer(f"Summarize this content:\n\n{content}"))

                return f"Content from {filename} (Summarized):\n\n{summary}"
            except (ContextWindowOverflowException, MaxTokensReachedException, ClientError) as e:
                try:
                    summary = chunked_extract(content, _SUMMARIZE_PROMPT, model_id)
                    return f"Content from {filename} (Chunk Summarized after error):\n\n{summary}"
                except (
                    ContextWindowOverflowException,
                    MaxTokensReachedException,
                    ClientError,
                ) as chunk_err:
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
        name=AGENT_REQUIREMENTS,
        model=model_id,
        callback_handler=None,
        system_prompt=_SYSTEM_PROMPT,
        tools=[read_document, list_available_documents],
    )
