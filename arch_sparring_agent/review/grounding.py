"""Contextual grounding checks via Bedrock Guardrails ApplyGuardrail API.

Validates that context condenser outputs are faithfully grounded in the raw
phase output they were extracted from. Uses post-hoc validation (ApplyGuardrail)
rather than inline model wrapping because the content has already been generated.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import boto3

from ..config import (
    DEFAULT_REGION,
    GROUNDING_CONTENT_CHUNK_SIZE,
    GROUNDING_SOURCE_MAX_CHARS,
)
from ..exceptions import AWS_ERRORS

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of a contextual grounding check."""

    passed: bool
    grounding_score: float
    action: str
    details: str


@dataclass
class GuardrailsChecker:
    """Runs contextual grounding checks against a deployed Bedrock Guardrail."""

    guardrail_id: str
    guardrail_version: str
    region: str

    def __post_init__(self) -> None:
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
        )

    def check_grounding(self, source: str, query: str, content: str) -> GroundingResult:
        """Check whether *content* is grounded in *source* for the given *query*.

        Chunks *content* at GROUNDING_CONTENT_CHUNK_SIZE to respect the 5K char
        API limit.  Returns the worst (lowest) grounding score across all chunks
        so that a single ungrounded chunk fails the overall check.

        Truncates *source* at GROUNDING_SOURCE_MAX_CHARS and *query* at 1000
        chars (API limits).  On API errors, logs a warning and returns a passing
        result — guardrail unavailability must not block reviews.
        """
        source = self._truncate(source, GROUNDING_SOURCE_MAX_CHARS, "grounding source")
        query = self._truncate(query, 1000, "query")

        if not content.strip():
            return GroundingResult(
                passed=True, grounding_score=1.0, action="NONE", details="Empty content — skipped"
            )

        chunks = self._chunk_content(content)
        worst_score = 1.0
        worst_action = "NONE"
        all_details: list[str] = []

        for i, chunk in enumerate(chunks):
            result = self._apply_guardrail(source, query, chunk)
            if result is None:
                continue

            score = self._extract_grounding_score(result)
            action = result.get("action", "NONE")

            if score < worst_score:
                worst_score = score
                worst_action = action

            if action == "GUARDRAIL_INTERVENED":
                all_details.append(f"Chunk {i + 1}/{len(chunks)}: score={score:.2f} (intervened)")
            else:
                all_details.append(f"Chunk {i + 1}/{len(chunks)}: score={score:.2f}")

        passed = worst_action != "GUARDRAIL_INTERVENED"
        details = "; ".join(all_details) if all_details else "No grounding data returned"

        return GroundingResult(
            passed=passed,
            grounding_score=worst_score,
            action=worst_action,
            details=details,
        )

    def _apply_guardrail(self, source: str, query: str, content: str) -> dict | None:
        """Call ApplyGuardrail for a single content chunk. Returns None on error."""
        try:
            return self._client.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source="OUTPUT",
                content=[
                    {
                        "text": {
                            "text": source,
                            "qualifiers": ["grounding_source"],
                        }
                    },
                    {
                        "text": {
                            "text": query,
                            "qualifiers": ["query"],
                        }
                    },
                    {
                        "text": {
                            "text": content,
                        }
                    },
                ],
            )
        except AWS_ERRORS as e:
            logger.warning("Guardrail API call failed, skipping grounding check: %s", e)
            return None

    @staticmethod
    def _extract_grounding_score(response: dict) -> float:
        """Extract the grounding score from the ApplyGuardrail response.

        Looks through all assessments for contextualGroundingPolicy filters
        with type "GROUNDING" and returns the lowest score found.
        """
        lowest = 1.0
        for assessment in response.get("assessments", []):
            grounding_policy = assessment.get("contextualGroundingPolicy", {})
            for filt in grounding_policy.get("filters", []):
                if filt.get("type") == "GROUNDING":
                    lowest = min(lowest, filt.get("score", 1.0))
        return lowest

    @staticmethod
    def _truncate(text: str, max_chars: int, label: str) -> str:
        if len(text) <= max_chars:
            return text
        logger.warning(
            "Truncating %s from %d to %d chars for grounding check",
            label,
            len(text),
            max_chars,
        )
        return text[:max_chars]

    @staticmethod
    def _chunk_content(content: str) -> list[str]:
        """Split content into chunks up to GROUNDING_CONTENT_CHUNK_SIZE.

        Splits on paragraph boundaries (``\\n\\n``) first, then merges
        paragraphs into chunks that fit the size limit.  Falls back to a
        hard slice when a single paragraph exceeds the limit.
        """
        limit = GROUNDING_CONTENT_CHUNK_SIZE
        paragraphs = content.split("\n\n")
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            candidate = f"{current}\n\n{para}" if current else para
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(para) <= limit:
                    current = para
                else:
                    for i in range(0, len(para), limit):
                        chunks.append(para[i : i + limit])
                    current = ""

        if current:
            chunks.append(current)
        return chunks or [content]


def create_guardrails_checker(
    guardrail_id: str | None,
    guardrail_version: str | None,
    region: str | None = None,
) -> GuardrailsChecker | None:
    """Create a GuardrailsChecker if guardrail config is present, else None."""
    if not guardrail_id or not guardrail_version:
        return None
    resolved_region = region or os.environ.get("AWS_REGION", DEFAULT_REGION)
    return GuardrailsChecker(
        guardrail_id=guardrail_id,
        guardrail_version=guardrail_version,
        region=resolved_region,
    )
