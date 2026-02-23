"""Markdown document parser for requirements files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import frontmatter

from ..config import DOC_MAX_BYTES
from ..exceptions import ToolError
from . import validate_file_size, validate_path


class ParsedDocument(TypedDict):
    """Structured return type for ``read_markdown_file``."""

    filename: str
    content: str
    metadata: dict[str, Any]


class DocumentParser:
    """Reads markdown documents with frontmatter support."""

    def __init__(self, document_dir: str):
        self.document_dir = Path(document_dir)

    def read_markdown_file(self, filename: str) -> ParsedDocument:
        """Read markdown file, returning content and metadata."""
        file_path = validate_path(self.document_dir, filename)
        if not file_path.exists():
            raise ToolError(f"File not found: {file_path}")

        validate_file_size(file_path, DOC_MAX_BYTES, "ARCH_REVIEW_DOC_MAX_BYTES")

        doc = frontmatter.loads(file_path.read_text(encoding="utf-8"))
        return {"filename": filename, "content": doc.content, "metadata": doc.metadata}

    def list_documents(self) -> list[str]:
        """List markdown files in directory."""
        return [f.name for f in self.document_dir.glob("*.md")]
