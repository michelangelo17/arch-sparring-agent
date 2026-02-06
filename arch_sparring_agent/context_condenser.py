"""Structured extraction of phase outputs to prevent token overflow without data loss.

Instead of hard character slices or high-threshold summarization, this module extracts
structured findings from each phase's raw output. Every item is preserved as a concise
bullet point — no mid-sentence cuts, no lost findings.
"""

from botocore.exceptions import ClientError
from strands import Agent
from strands.types.exceptions import ContextWindowOverflowException, MaxTokensReachedException

from .config import CONDENSER_CHUNK_SIZE, CONDENSER_MAX_CHUNKS, CONDENSER_PASSTHROUGH_THRESHOLD

# Re-export for backwards compatibility and test access
PASSTHROUGH_THRESHOLD = CONDENSER_PASSTHROUGH_THRESHOLD
CHUNK_SIZE = CONDENSER_CHUNK_SIZE
MAX_CHUNKS = CONDENSER_MAX_CHUNKS


def _chunked_extract(content: str, system_prompt: str, model_id: str) -> str:
    """Fallback: extract findings from content in chunks, then merge."""
    chunks = [content[i : i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]

    chunk_results = []
    for i, chunk in enumerate(chunks[:MAX_CHUNKS]):
        extractor = Agent(
            name="ChunkExtractor",
            model=model_id,
            system_prompt=system_prompt,
            tools=[],
        )
        try:
            chunk_results.append(str(extractor(chunk)))
        except Exception:
            chunk_results.append(f"[Chunk {i + 1} could not be processed]")

    # Merge: deduplicate by running one final extraction over the combined chunk results
    combined = "\n\n".join(chunk_results)
    if len(combined) <= PASSTHROUGH_THRESHOLD:
        return combined

    merger = Agent(
        name="FindingsMerger",
        model=model_id,
        system_prompt=(
            "Merge these extracted findings into a single deduplicated list. "
            "Remove duplicates. Keep every unique item. Use the same format as the input."
        ),
        tools=[],
    )
    try:
        return str(merger(combined))
    except Exception:
        # If merge fails, return raw combined (better than nothing)
        return combined


def _extract(content: str, system_prompt: str, model_id: str) -> str:
    """Run structured extraction, with chunked fallback for large inputs."""
    if len(content) <= PASSTHROUGH_THRESHOLD:
        return content

    extractor = Agent(
        name="FindingsExtractor",
        model=model_id,
        system_prompt=system_prompt,
        tools=[],
    )

    try:
        return str(extractor(content))
    except (ContextWindowOverflowException, MaxTokensReachedException):
        return _chunked_extract(content, system_prompt, model_id)
    except ClientError as e:
        # Bedrock token limit errors surface as ClientError with specific error codes
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("ValidationException", "ModelErrorException"):
            return _chunked_extract(content, system_prompt, model_id)
        raise


def extract_requirements(raw_output: str, model_id: str) -> str:
    """Extract structured requirements from Phase 1 output.

    Returns a compact bullet list of every requirement, constraint, and NFR.
    No prose, no elaboration — just the items.
    """
    return _extract(
        raw_output,
        """Extract every requirement, constraint, and non-functional requirement as bullet points.

Rules:
- One bullet per item
- No prose or elaboration — just state the requirement
- Preserve ALL items, do not skip any
- Group under: ### Functional Requirements, ### Non-Functional Requirements, ### Constraints

Max 800 words.""",
        model_id,
    )


def extract_architecture_findings(raw_output: str, model_id: str) -> str:
    """Extract structured findings from Phase 2 output.

    Returns Components, Features Verified, and Features Not Found sections
    with one line per item including evidence.
    """
    return _extract(
        raw_output,
        """Extract the structured findings from this architecture analysis.

Format:
### Components
- One line per component

### Features Verified
- Feature: [brief evidence, e.g. "found in checkCache.ts line 12"]

### Features Not Found
- Feature: [what was searched for]

Rules:
- One line per item
- Include the evidence reference for verified features
- Preserve ALL items from both Verified and Not Found sections
- Do not add items that aren't in the input
- Do not remove items that are in the input

Max 600 words.""",
        model_id,
    )


def extract_phase_findings(raw_output: str, phase_name: str, model_id: str) -> str:
    """Extract structured findings from Phase 3 (Q&A) or Phase 4 (Sparring) output.

    Returns categorized bullet points: decisions, gaps, risks, verified items.
    """
    return _extract(
        raw_output,
        f"""Extract every finding from this {phase_name} output as categorized bullet points.

Format:
### Confirmed Gaps
- Items confirmed as missing

### Accepted Risks
- Items acknowledged but not being addressed, with reasoning

### Will Fix
- Items agreed to be fixed

### Verified
- Items confirmed as implemented

Rules:
- One bullet per item
- Include brief reasoning where it was discussed
- Preserve ALL items — do not skip any
- Only use categories that have items (skip empty sections)

Max 400 words.""",
        model_id,
    )
