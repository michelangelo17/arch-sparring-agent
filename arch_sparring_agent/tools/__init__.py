"""Analysis tools for CloudFormation, diagrams, documents, and source code."""

from __future__ import annotations

from pathlib import Path

from ..exceptions import ToolError


def validate_path(base_dir: Path, filename: str) -> Path:
    """Resolve path and verify it stays within the base directory."""
    file_path = (base_dir / filename).resolve()
    if not file_path.is_relative_to(base_dir.resolve()):
        raise ToolError(f"Path traversal detected: {filename}")
    return file_path


def validate_file_size(path: Path, max_bytes: int, env_var: str) -> None:
    """Raise ToolError if *path* exceeds *max_bytes* bytes.

    The error message references *env_var* so the user knows how to override.
    """
    file_size = path.stat().st_size
    if file_size <= max_bytes:
        return

    size_label = f"{file_size / 1_000_000:.1f}MB"
    limit_label = f"{max_bytes / 1_000_000:.1f}MB"

    raise ToolError(
        f"'{path.name}' is {size_label} which exceeds the {limit_label} limit. "
        f"Reduce the file size or increase the limit with {env_var}."
    )


def search_content(content: str, pattern: str, filename: str, max_matches: int = 5) -> str | None:
    """Case-insensitive line search within *content*.

    Returns a formatted match block for *filename*, or None if no matches.
    At most *max_matches* matching lines are included.
    """
    pattern_lower = pattern.lower()
    if pattern_lower not in content.lower():
        return None

    matches: list[str] = []
    for i, line in enumerate(content.split("\n"), 1):
        if pattern_lower in line.lower():
            matches.append(f"  L{i}: {line.strip()}")

    if not matches:
        return None

    result = f"\n{filename}:\n" + "\n".join(matches[:max_matches])
    if len(matches) > max_matches:
        result += f"\n  ... and {len(matches) - max_matches} more matches"
    return result
