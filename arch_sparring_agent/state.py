"""Review state management for remediation mode."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Gap:
    """A single architecture gap identified during review."""

    id: str
    description: str
    severity: str


@dataclass
class Risk:
    """A single risk identified during review."""

    id: str
    description: str
    impact: str


@dataclass
class ReviewState:
    """Structured state from a completed review."""

    timestamp: str
    version: str = "1.0"
    project_name: str = ""
    gaps: list[Gap] = field(default_factory=list)
    risks: list[Risk] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    verdict: str = ""
    requirements_summary: str = ""
    architecture_summary: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> ReviewState:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in review state: {e}") from e
        data["gaps"] = [Gap(**g) for g in data.get("gaps", [])]
        data["risks"] = [Risk(**r) for r in data.get("risks", [])]
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> ReviewState:
        file_path = Path(path)
        try:
            return cls.from_json(file_path.read_text())
        except OSError as e:
            raise ValueError(f"Could not read state file '{file_path}': {e}") from e

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())
