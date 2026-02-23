"""Source code analyzer for Lambda handlers and application logic."""

from __future__ import annotations

from pathlib import Path

from ..config import SOURCE_MAX_BYTES
from ..exceptions import ToolError
from . import search_content, validate_file_size, validate_path


class SourceAnalyzer:
    """Reads source code files to understand business logic."""

    SUPPORTED_EXTENSIONS = {".ts", ".js", ".py", ".java", ".go", ".cs"}

    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)

    def list_source_files(self) -> list[str]:
        """List source files recursively, excluding node_modules/venv/etc."""
        if not self.source_dir.exists():
            return []

        exclude_dirs = {"node_modules", ".venv", "venv", "__pycache__", ".git", "dist", "build"}
        files = []

        for ext in self.SUPPORTED_EXTENSIONS:
            for path in self.source_dir.rglob(f"*{ext}"):
                if not any(excluded in path.parts for excluded in exclude_dirs):
                    files.append(str(path.relative_to(self.source_dir)))

        return sorted(files)

    def read_source_file(self, filename: str) -> str:
        """Read a source file's contents.

        Raises:
            ToolError: If file not found, not a file, or unsupported type.
        """
        path = validate_path(self.source_dir, filename)
        if not path.exists():
            raise ToolError(f"File not found: {filename}")
        if not path.is_file():
            raise ToolError(f"Not a file: {filename}")
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ToolError(f"Unsupported file type: {path.suffix}")

        validate_file_size(path, SOURCE_MAX_BYTES, "ARCH_REVIEW_SOURCE_MAX_BYTES")
        return path.read_text(encoding="utf-8")

    def search_source(self, pattern: str) -> str:
        """Search for a pattern across all source files."""
        results: list[str] = []

        for filepath in self.list_source_files():
            content = self.read_source_file(filepath)
            match_block = search_content(content, pattern, filepath)
            if match_block:
                results.append(match_block)

        if not results:
            return f"No matches found for: {pattern}"
        return "".join(results[:10])
