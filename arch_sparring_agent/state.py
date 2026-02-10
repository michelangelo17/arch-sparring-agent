"""Review state management for remediation mode."""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ReviewState:
    """Structured state from a completed review."""

    timestamp: str
    version: str = "1.0"
    project_name: str = ""
    gaps: list[dict] = field(default_factory=list)
    risks: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    verdict: str = ""
    requirements_summary: str = ""
    architecture_summary: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ReviewState":
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in review state: {e}") from e
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> "ReviewState":
        file_path = Path(path)
        try:
            return cls.from_json(file_path.read_text())
        except OSError as e:
            raise ValueError(f"Could not read state file '{file_path}': {e}") from e

    def save(self, path: str | Path):
        Path(path).write_text(self.to_json())


def _infer_severity(text: str) -> str:
    """Infer severity."""
    text_lower = text.lower()
    if "critical" in text_lower or "high" in text_lower:
        return "high"
    if "low" in text_lower or "minor" in text_lower:
        return "low"
    return "medium"


def _is_duplicate(text: str, items: list[dict], key: str = "description") -> bool:
    """Check for duplicates."""
    prefix = text[:50]
    return any(prefix in item[key] for item in items)


def _is_duplicate_str(text: str, items: list[str]) -> bool:
    """Check for duplicates in a string list."""
    prefix = text[:50]
    return any(prefix in item for item in items)


def _strip_markdown(text: str) -> str:
    """Remove common markdown inline formatting."""
    text = text.strip()
    # Remove bold/italic markers: **text**, *text*, __text__, _text_
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)
    # Remove inline code backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def _header_level(line: str) -> int:
    """Return the markdown header level (1-6), or 0 if not a header."""
    match = re.match(r"^(#{1,6})\s+", line.strip())
    return len(match.group(1)) if match else 0


def _is_section_header(line: str, keyword: str) -> bool:
    """Check if a line is a section header containing the keyword (case-insensitive).

    Matches markdown headers (## ...) and bold headers (**...**) containing the keyword.
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Markdown header: any level of # containing the keyword
    if _header_level(stripped) > 0 and keyword.lower() in stripped.lower():
        return True
    # Bold header or keyword at start: **Gaps**, Key Gaps, etc.
    if keyword.lower() in stripped.lower() and stripped.startswith("**"):
        return True
    return False


def _is_gap_section_header(line: str) -> bool:
    """Check if a line is a gap list section header (not a numbered gap reference).

    Matches: "## Confirmed Gaps", "### Key Gaps", "## Gaps"
    Rejects: "##### Gap 1: Feature - ..." (a sub-header referencing a specific gap)
    """
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    # Must contain "gap" in a section-title sense
    if "gap" not in lower:
        return False
    # Reject numbered gap references like "Gap 1:", "Gap 2:"
    if re.search(r"gap\s*\d+\s*:", lower):
        return False
    # Accept markdown headers or bold headers
    if _header_level(stripped) > 0:
        return True
    if stripped.startswith("**"):
        return True
    return False


def _is_list_item(line: str) -> bool:
    """Check if a line is a list item (bullet or numbered)."""
    stripped = line.strip()
    if not stripped:
        return False
    # Bullet: - item or * item
    if stripped.startswith("-") or stripped.startswith("*"):
        return True
    # Numbered: 1. item, 1) item, etc.
    if re.match(r"^\d+[.)]\s", stripped):
        return True
    return False


def _extract_list_text(line: str) -> str:
    """Extract the text content from a list item, stripping bullet/number prefix."""
    stripped = line.strip()
    # Remove bullet prefix
    if stripped.startswith("-") or stripped.startswith("*"):
        return stripped[1:].strip()
    # Remove numbered prefix: "1. " or "1) "
    match = re.match(r"^\d+[.)]\s*(.*)", stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _is_next_section_at_level(line: str, max_level: int) -> bool:
    """Check if a line starts a section at the given level or higher (lower number).

    For example, if we're in a #### section, only ## or ### would end it,
    not ##### sub-headers within it.
    """
    level = _header_level(line)
    return level > 0 and level <= max_level


def extract_state_from_review(review_result: dict) -> ReviewState:
    """Extract structured state from review result."""
    review_text = review_result.get("review", "")
    project_name = Path.cwd().name

    gaps = _extract_gaps(review_text, review_result.get("gaps", ""))
    risks = _extract_risks(review_text, review_result.get("risks", ""))
    recommendations = _extract_recommendations(review_text)
    verdict = extract_verdict(review_text)

    return ReviewState(
        timestamp=datetime.now().isoformat(),
        project_name=project_name,
        gaps=gaps,
        risks=risks,
        recommendations=recommendations,
        verdict=verdict,
        requirements_summary=review_result.get("requirements_summary", ""),
        architecture_summary=review_result.get("architecture_summary", ""),
    )


def _extract_gaps(review_text: str, gaps_context: str) -> list[dict]:
    """Extract gaps from review and context."""
    gaps = []
    gap_id = 1

    for source in [review_text, gaps_context]:
        lines = source.split("\n")
        in_gaps_section = False
        section_level = 0

        for line in lines:
            if _is_gap_section_header(line):
                in_gaps_section = True
                section_level = _header_level(line) or 2
                continue
            # Only exit on a header at the same level or higher (not sub-headers)
            if in_gaps_section and _is_next_section_at_level(line, section_level):
                in_gaps_section = False
                continue

            if in_gaps_section and _is_list_item(line):
                gap_text = _strip_markdown(_extract_list_text(line))
                if gap_text and len(gap_text) > 5 and not _is_duplicate(gap_text, gaps):
                    gaps.append(
                        {
                            "id": f"gap-{gap_id}",
                            "description": gap_text[:200],
                            "severity": _infer_severity(gap_text),
                        }
                    )
                    gap_id += 1

    # Check "Features Not Found" section
    if "not found" in review_text.lower():
        lines = review_text.split("\n")
        in_not_found = False
        section_level = 0
        for line in lines:
            if _is_section_header(line, "not found"):
                in_not_found = True
                section_level = _header_level(line) or 2
                continue
            if in_not_found and _is_next_section_at_level(line, section_level):
                in_not_found = False
                continue
            if in_not_found and _is_list_item(line):
                gap_text = _strip_markdown(_extract_list_text(line))
                if gap_text and len(gap_text) > 5 and not _is_duplicate(gap_text, gaps):
                    gaps.append(
                        {
                            "id": f"gap-{gap_id}",
                            "description": gap_text[:200],
                            "severity": "medium",
                        }
                    )
                    gap_id += 1

    return gaps[:20]


def _extract_risks(review_text: str, risks_context: str) -> list[dict]:
    """Extract risks from review.

    Handles two common formats:
    1. Flat list: "## Risks" followed by bullet points
    2. Nested: "#### Risks & Mitigations" with "##### Gap N:" sub-headers
       containing "- **Risk**: ..." bullets
    """
    risks = []
    risk_id = 1

    for source in [review_text, risks_context]:
        lines = source.split("\n")
        in_risks_section = False
        section_level = 0

        for line in lines:
            if _is_section_header(line, "risk"):
                in_risks_section = True
                section_level = _header_level(line) or 2
                continue
            # Only exit on headers at the same level or higher -- allow sub-headers
            if in_risks_section and _is_next_section_at_level(line, section_level):
                in_risks_section = False
                continue

            if in_risks_section and _is_list_item(line):
                risk_text = _strip_markdown(_extract_list_text(line))
                # Skip "Impact:" and "Mitigation:" lines -- only extract "Risk:" lines
                # or general risk descriptions
                risk_lower = risk_text.lower()
                if risk_lower.startswith("impact:") or risk_lower.startswith("mitigation:"):
                    continue
                # Strip leading "Risk:" prefix if present
                if risk_lower.startswith("risk:"):
                    risk_text = risk_text[5:].strip()
                if risk_text and len(risk_text) > 10 and not _is_duplicate(risk_text, risks):
                    risks.append(
                        {
                            "id": f"risk-{risk_id}",
                            "description": risk_text[:200],
                            "impact": _infer_severity(risk_text),
                        }
                    )
                    risk_id += 1

    return risks[:10]


def _extract_recommendations(review_text: str) -> list[str]:
    """Extract recommendations from review.

    Looks for a "Recommendations" section first. If not found, falls back to
    extracting "Mitigation:" entries from "Risks" sections.
    """
    recommendations = []
    lines = review_text.split("\n")
    in_recommendations = False
    section_level = 0

    for line in lines:
        if _is_section_header(line, "recommendation"):
            in_recommendations = True
            section_level = _header_level(line) or 2
            continue
        if in_recommendations and _is_next_section_at_level(line, section_level):
            in_recommendations = False
            continue

        if in_recommendations and _is_list_item(line):
            rec_text = _strip_markdown(_extract_list_text(line))
            if rec_text and len(rec_text) > 10:
                recommendations.append(rec_text[:300])

    # Fallback: extract mitigations from risks section if no recommendations found
    if not recommendations:
        recommendations = _extract_mitigations_as_recommendations(review_text)

    return recommendations[:10]


def _extract_mitigations_as_recommendations(review_text: str) -> list[str]:
    """Extract mitigation entries from risk sections as recommendations."""
    recommendations = []
    lines = review_text.split("\n")
    in_risks_section = False
    section_level = 0

    for line in lines:
        if _is_section_header(line, "risk") or _is_section_header(line, "mitigation"):
            in_risks_section = True
            section_level = _header_level(line) or 2
            continue
        if in_risks_section and _is_next_section_at_level(line, section_level):
            in_risks_section = False
            continue

        if in_risks_section and _is_list_item(line):
            item_text = _strip_markdown(_extract_list_text(line))
            item_lower = item_text.lower()
            if item_lower.startswith("mitigation:"):
                rec_text = item_text[11:].strip()
                if (
                    rec_text
                    and len(rec_text) > 10
                    and not _is_duplicate_str(rec_text, recommendations)
                ):
                    recommendations.append(rec_text[:300])

    return recommendations


def extract_verdict(review_text: str) -> str:
    """Extract verdict string from review text.

    Returns one of: "PASS", "PASS WITH CONCERNS", "FAIL".
    """
    # Strip markdown bold markers for matching
    clean_text = review_text.replace("**", "")
    review_lower = clean_text.lower()

    # Check explicit verdict
    if "verdict" in review_lower:
        after_verdict = review_lower.split("verdict")[-1][:100]
        if "fail" in after_verdict:
            return "FAIL"
        elif "pass with concerns" in after_verdict:
            return "PASS WITH CONCERNS"
        elif "pass" in after_verdict:
            return "PASS"

    # Fallback: infer from content
    critical_terms = ["critical", "severe", "major vulnerability", "security vulnerability"]
    if any(term in review_lower for term in critical_terms):
        return "FAIL"

    has_high_impact = "impact: high" in review_lower or "impact high" in review_lower
    if has_high_impact:
        return "PASS WITH CONCERNS"

    # Default to PASS if no concerning signals
    return "PASS"
