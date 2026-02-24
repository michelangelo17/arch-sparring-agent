"""Markdown parsing and structured extraction of review outputs."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..state import Gap, ReviewState, Risk

if TYPE_CHECKING:
    from .orchestrator import ReviewResult


# ---------------------------------------------------------------------------
# Markdown parsing helpers
# ---------------------------------------------------------------------------


def infer_severity(text: str) -> str:
    """Infer severity from keywords in text."""
    text_lower = text.lower()
    if "critical" in text_lower or "high" in text_lower:
        return "high"
    if "low" in text_lower or "minor" in text_lower:
        return "low"
    return "medium"


def has_duplicate_description(text: str, items: list[Gap] | list[Risk]) -> bool:
    """Check if *text* is a prefix-duplicate of any item's description."""
    prefix = text[:50]
    return any(prefix in item.description for item in items)


def has_duplicate_string(text: str, items: list[str]) -> bool:
    """Check if *text* is a prefix-duplicate of any string in *items*."""
    prefix = text[:50]
    return any(prefix in item for item in items)


def strip_markdown(text: str) -> str:
    """Remove common markdown inline formatting."""
    text = text.strip()
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def _header_level(line: str) -> int:
    """Return the markdown header level (1-6), or 0 if not a header."""
    match = re.match(r"^(#{1,6})\s+", line.strip())
    return len(match.group(1)) if match else 0


def _is_section_header(line: str, keyword: str) -> bool:
    """Check if *line* is a markdown/bold section header containing *keyword*."""
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if keyword.lower() not in lower:
        return False
    if _header_level(stripped) > 0:
        return True
    if stripped.startswith("**"):
        return True
    return False


def _is_gap_section_header(line: str) -> bool:
    """Check if *line* is a gap-list section header (not a numbered gap reference).

    Matches: "## Confirmed Gaps", "### Key Gaps", "## Gaps"
    Rejects: "##### Gap 1: Feature - ..." (a sub-header referencing a specific gap)
    """
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if "gap" not in lower:
        return False
    if re.search(r"gap\s*\d+\s*:", lower):
        return False
    if _header_level(stripped) > 0:
        return True
    if stripped.startswith("**"):
        return True
    return False


def _is_list_item(line: str) -> bool:
    """Check if a line is a bullet or numbered list item."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped[0] in "-*":
        return True
    if re.match(r"^\d+[.)]\s", stripped):
        return True
    return False


def _extract_list_text(line: str) -> str:
    """Strip the bullet/number prefix from a list item."""
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped[0] in "-*":
        return stripped[1:].strip()
    match = re.match(r"^\d+[.)]\s*(.*)", stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _is_next_section_at_level(line: str, max_level: int) -> bool:
    """True if *line* is a header at *max_level* or higher (lower number)."""
    level = _header_level(line)
    return level > 0 and level <= max_level


# ---------------------------------------------------------------------------
# Generic section extractor
# ---------------------------------------------------------------------------


def _extract_section_items(
    text: str,
    header_matcher: Callable[[str], bool],
    *,
    item_filter: Callable[[str], str | None] | None = None,
) -> list[str]:
    """Extract list-item texts from sections whose headers match *header_matcher*.

    Walks through *text* line by line. When a matching header is found, all
    subsequent list items are collected (stripped of markdown formatting) until
    a same-level-or-higher header ends the section.

    Args:
        text: The markdown text to parse.
        header_matcher: Callable that returns True for section-header lines.
        item_filter: Optional transform applied to each raw item text.
            Return None to skip the item; return a string to include it.
    """
    items: list[str] = []
    in_section = False
    section_level = 0

    for line in text.split("\n"):
        if header_matcher(line):
            in_section = True
            section_level = _header_level(line) or 2
            continue
        if in_section and _is_next_section_at_level(line, section_level):
            in_section = False
            continue
        if in_section and _is_list_item(line):
            item_text = strip_markdown(_extract_list_text(line))
            if not item_text:
                continue
            if item_filter:
                result = item_filter(item_text)
                if result is not None:
                    items.append(result)
            else:
                items.append(item_text)

    return items


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


def extract_state_from_review(review_result: ReviewResult) -> ReviewState:
    """Extract structured state from review result."""
    project_name = Path.cwd().name

    gaps = extract_gaps(review_result.review, review_result.qa_context)
    risks = extract_risks(review_result.review, review_result.sparring_context)
    recommendations = extract_recommendations(review_result.review)
    verdict = extract_verdict(review_result.review)

    return ReviewState(
        timestamp=datetime.now().isoformat(),
        project_name=project_name,
        gaps=gaps,
        risks=risks,
        recommendations=recommendations,
        verdict=verdict,
        requirements_summary=review_result.requirements_summary,
        architecture_summary=review_result.architecture_summary,
    )


def extract_gaps(review_text: str, gaps_context: str) -> list[Gap]:
    """Extract gaps from review text and context."""
    gaps: list[Gap] = []
    gap_id = 1

    def _collect_gap(text: str, severity_fn: Callable[[str], str] = infer_severity) -> None:
        nonlocal gap_id
        if len(text) > 5 and not has_duplicate_description(text, gaps):
            gaps.append(Gap(id=f"gap-{gap_id}", description=text[:200], severity=severity_fn(text)))
            gap_id += 1

    for source in [review_text, gaps_context]:
        for item in _extract_section_items(source, _is_gap_section_header):
            _collect_gap(item)

    if "not found" in review_text.lower():
        for item in _extract_section_items(
            review_text, lambda line: _is_section_header(line, "not found")
        ):
            _collect_gap(item, severity_fn=lambda _: "medium")

    return gaps[:20]


def extract_risks(review_text: str, risks_context: str) -> list[Risk]:
    """Extract risks from review text and context.

    Handles flat lists ("## Risks" with bullet points) and nested formats
    ("#### Risks & Mitigations" with "##### Gap N:" sub-headers).
    """
    risks: list[Risk] = []
    risk_id = 1

    def _risk_filter(text: str) -> str | None:
        lower = text.lower()
        if lower.startswith("impact:") or lower.startswith("mitigation:"):
            return None
        if lower.startswith("risk:"):
            text = text[5:].strip()
        return text if len(text) > 10 else None

    for source in [review_text, risks_context]:
        for item in _extract_section_items(
            source, lambda line: _is_section_header(line, "risk"), item_filter=_risk_filter
        ):
            if not has_duplicate_description(item, risks):
                risks.append(
                    Risk(id=f"risk-{risk_id}", description=item[:200], impact=infer_severity(item))
                )
                risk_id += 1

    return risks[:10]


def extract_recommendations(review_text: str) -> list[str]:
    """Extract recommendations, falling back to mitigation entries from risk sections."""
    recommendations: list[str] = []

    for item in _extract_section_items(
        review_text, lambda line: _is_section_header(line, "recommendation")
    ):
        if len(item) > 10:
            recommendations.append(item[:300])

    if not recommendations:
        recommendations = extract_mitigations_as_recommendations(review_text)

    return recommendations[:10]


def extract_mitigations_as_recommendations(review_text: str) -> list[str]:
    """Extract "Mitigation:" entries from risk/mitigation sections."""
    recommendations: list[str] = []

    def _mitigation_filter(text: str) -> str | None:
        if text.lower().startswith("mitigation:"):
            rec = text[11:].strip()
            if len(rec) > 10 and not has_duplicate_string(rec, recommendations):
                return rec[:300]
        return None

    for item in _extract_section_items(
        review_text,
        lambda line: _is_section_header(line, "risk") or _is_section_header(line, "mitigation"),
        item_filter=_mitigation_filter,
    ):
        recommendations.append(item)

    return recommendations


def extract_verdict(review_text: str) -> str:
    """Extract verdict string from review text.

    Returns one of: "PASS", "PASS WITH CONCERNS", "FAIL".
    """
    clean_text = review_text.replace("**", "")
    review_lower = clean_text.lower()

    if "verdict" in review_lower:
        after_verdict = review_lower.split("verdict")[-1][:100]
        if "fail" in after_verdict:
            return "FAIL"
        elif "pass with concerns" in after_verdict:
            return "PASS WITH CONCERNS"
        elif "pass" in after_verdict:
            return "PASS"

    critical_terms = ["critical", "severe", "major vulnerability", "security vulnerability"]
    if any(term in review_lower for term in critical_terms):
        return "FAIL"

    if "impact: high" in review_lower or "impact high" in review_lower:
        return "PASS WITH CONCERNS"

    return "PASS"
