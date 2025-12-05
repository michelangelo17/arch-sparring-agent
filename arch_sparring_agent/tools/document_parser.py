from pathlib import Path

import frontmatter


class DocumentParser:
    """Reads markdown documents. The agent extracts relevant information from the content."""

    def __init__(self, document_dir: str):
        self.document_dir = Path(document_dir)

    def read_markdown_file(self, filename: str) -> dict:
        """Read a markdown file and parse frontmatter + content."""
        file_path = self.document_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            doc = frontmatter.load(f)

        return {
            "filename": filename,
            "content": doc.content,
            "metadata": doc.metadata,
        }

    def list_documents(self) -> list[str]:
        """List all markdown files in the document directory."""
        return [f.name for f in self.document_dir.glob("*.md")]
