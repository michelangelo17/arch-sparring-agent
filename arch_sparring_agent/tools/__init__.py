"""Analysis tools for CloudFormation, diagrams, documents, and source code."""

from pathlib import Path

from ..exceptions import ToolError


def validate_path(base_dir: Path, filename: str) -> Path:
    """Resolve path and verify it stays within the base directory."""
    file_path = (base_dir / filename).resolve()
    if not file_path.is_relative_to(base_dir.resolve()):
        raise ToolError(f"Path traversal detected: {filename}")
    return file_path


def validate_file_size(path: Path, max_size: int, env_var: str) -> None:
    """Raise ToolError if *path* exceeds *max_size* bytes.

    The error message references *env_var* so the user knows how to override.
    """
    file_size = path.stat().st_size
    if file_size <= max_size:
        return

    is_bytes = "BYTES" in env_var.upper()
    if is_bytes:
        size_label = f"{file_size / 1_000_000:.1f}MB"
        limit_label = f"{max_size / 1_000_000:.1f}MB"
    else:
        size_label = f"{file_size / 1_000:.0f}KB"
        limit_label = f"{max_size / 1_000:.0f}KB"

    raise ToolError(
        f"'{path.name}' is {size_label} which exceeds the {limit_label} limit. "
        f"Reduce the file size or increase the limit with {env_var}."
    )
