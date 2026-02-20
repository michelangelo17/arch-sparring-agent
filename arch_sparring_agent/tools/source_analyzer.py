"""Source code analyzer for Lambda handlers and application logic."""

from pathlib import Path

from ..config import SOURCE_FILE_MAX_CHARS
from ..exceptions import ToolError
from ..utils import validate_path


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

        file_size = path.stat().st_size
        if file_size > SOURCE_FILE_MAX_CHARS:
            size_kb = file_size / 1_000
            limit_kb = SOURCE_FILE_MAX_CHARS / 1_000
            raise ToolError(
                f"Source file '{filename}' is {size_kb:.0f}KB which exceeds the "
                f"{limit_kb:.0f}KB limit. Split large files or increase the "
                f"limit with ARCH_REVIEW_SOURCE_MAX_CHARS."
            )

        return path.read_text(encoding="utf-8")

    def search_source(self, pattern: str) -> str:
        """Search for a pattern across all source files."""
        results = []
        pattern_lower = pattern.lower()

        for filepath in self.list_source_files():
            content = self.read_source_file(filepath)
            if pattern_lower in content.lower():
                lines = content.split("\n")
                matches = []
                for i, line in enumerate(lines, 1):
                    if pattern_lower in line.lower():
                        matches.append(f"  L{i}: {line.strip()}")
                if matches:
                    results.append(f"\n{filepath}:\n" + "\n".join(matches[:5]))
                    if len(matches) > 5:
                        results.append(f"  ... and {len(matches) - 5} more matches")

        if not results:
            return f"No matches found for: {pattern}"
        return "".join(results[:10])
