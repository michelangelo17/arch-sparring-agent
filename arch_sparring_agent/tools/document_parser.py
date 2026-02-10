"""Markdown document parser for requirements files."""

from pathlib import Path

import frontmatter

from ..config import DOC_MAX_CHARS
from ..exceptions import ToolError


def _validate_path(base_dir: Path, filename: str) -> Path:
    """Resolve path and verify it stays within the base directory."""
    file_path = (base_dir / filename).resolve()
    if not file_path.is_relative_to(base_dir.resolve()):
        raise ToolError(f"Path traversal detected: {filename}")
    return file_path


class DocumentParser:
    """Reads markdown documents with frontmatter support."""

    def __init__(self, document_dir: str):
        self.document_dir = Path(document_dir)

    def read_markdown_file(self, filename: str) -> dict:
        """Read markdown file, returning content and metadata."""
        file_path = _validate_path(self.document_dir, filename)
        if not file_path.exists():
            raise ToolError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        if file_size > DOC_MAX_CHARS:
            size_kb = file_size / 1_000
            limit_kb = DOC_MAX_CHARS / 1_000
            raise ToolError(
                f"Document '{filename}' is {size_kb:.0f}KB which exceeds the "
                f"{limit_kb:.0f}KB limit. Split large documents or increase the "
                f"limit with ARCH_REVIEW_DOC_MAX_CHARS."
            )

        doc = frontmatter.loads(file_path.read_text(encoding="utf-8"))
        return {"filename": filename, "content": doc.content, "metadata": doc.metadata}

    def list_documents(self) -> list[str]:
        """List markdown files in directory."""
        return [f.name for f in self.document_dir.glob("*.md")]
