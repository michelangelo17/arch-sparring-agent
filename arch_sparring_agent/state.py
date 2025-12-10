"""Review state management for remediation mode."""

import json
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
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> "ReviewState":
        return cls.from_json(Path(path).read_text())

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


def extract_state_from_review(review_result: dict) -> ReviewState:
    """Extract structured state from review result."""
    review_text = review_result.get("review", "")
    project_name = Path.cwd().name

    gaps = _extract_gaps(review_text, review_result.get("gaps", ""))
    risks = _extract_risks(review_text, review_result.get("risks", ""))
    recommendations = _extract_recommendations(review_text)
    verdict = _extract_verdict(review_text)

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

        for line in lines:
            line_lower = line.lower().strip()

            if "gap" in line_lower and ("##" in line or "key" in line_lower):
                in_gaps_section = True
                continue
            if in_gaps_section and line.strip().startswith("##"):
                in_gaps_section = False
                continue

            if in_gaps_section and line.strip().startswith("-"):
                gap_text = line.strip().lstrip("-").strip()
                if gap_text and len(gap_text) > 5:
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
        for line in lines:
            if "not found" in line.lower() and "#" in line:
                in_not_found = True
                continue
            if in_not_found and line.strip().startswith("#"):
                in_not_found = False
                continue
            if in_not_found and line.strip().startswith("-"):
                gap_text = line.strip().lstrip("-").strip()
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
    """Extract risks from review."""
    risks = []
    risk_id = 1

    for source in [review_text, risks_context]:
        lines = source.split("\n")
        in_risks_section = False

        for line in lines:
            line_lower = line.lower().strip()

            if "risk" in line_lower and ("##" in line or "top" in line_lower):
                in_risks_section = True
                continue
            if in_risks_section and line.strip().startswith("##"):
                in_risks_section = False
                continue

            if in_risks_section:
                stripped = line.strip()
                if stripped and (stripped[0].isdigit() or stripped.startswith("-")):
                    risk_text = stripped.lstrip("0123456789.-) ").strip()
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
    """Extract recommendations from review."""
    recommendations = []
    lines = review_text.split("\n")
    in_recommendations = False

    for line in lines:
        line_lower = line.lower().strip()

        if "recommendation" in line_lower and "#" in line:
            in_recommendations = True
            continue
        if in_recommendations and line.strip().startswith("##"):
            in_recommendations = False
            continue

        if in_recommendations:
            stripped = line.strip()
            if stripped and (stripped[0].isdigit() or stripped.startswith("-")):
                rec_text = stripped.lstrip("0123456789.-) ").strip()
                if rec_text and len(rec_text) > 10:
                    recommendations.append(rec_text[:300])

    return recommendations[:10]


def _extract_verdict(review_text: str) -> str:
    """Extract verdict from review."""
    review_lower = review_text.lower()

    if "verdict" in review_lower:
        after_verdict = review_lower.split("verdict")[-1][:100]
        if "fail" in after_verdict:
            return "FAIL"
        elif "pass with concerns" in after_verdict:
            return "PASS WITH CONCERNS"
        elif "pass" in after_verdict:
            return "PASS"

    return "UNKNOWN"
