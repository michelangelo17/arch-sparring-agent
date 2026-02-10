"""CloudFormation template reader."""

from pathlib import Path

from ..config import CFN_MAX_CHARS
from ..exceptions import ToolError


def _validate_path(base_dir: Path, filename: str) -> Path:
    """Resolve path and verify it stays within the base directory."""
    file_path = (base_dir / filename).resolve()
    if not file_path.is_relative_to(base_dir.resolve()):
        raise ToolError(f"Path traversal detected: {filename}")
    return file_path


class CloudFormationAnalyzer:
    """Reads CloudFormation templates (YAML, JSON, CDK output)."""

    def __init__(self, templates_dir: str):
        self.templates_dir = Path(templates_dir)

    def list_templates(self) -> list[str]:
        """List CloudFormation template files recursively (supports nested CDK output)."""
        templates = []
        for ext in ["*.yaml", "*.yml", "*.json", "*.template.json"]:
            templates.extend(
                str(f.relative_to(self.templates_dir)) for f in self.templates_dir.rglob(ext)
            )
        return sorted(templates)

    def read_template(self, filename: str) -> str:
        """Read template file as string."""
        file_path = _validate_path(self.templates_dir, filename)
        if not file_path.exists():
            raise ToolError(f"Template not found: {filename}")

        file_size = file_path.stat().st_size
        if file_size > CFN_MAX_CHARS:
            size_kb = file_size / 1_000
            limit_kb = CFN_MAX_CHARS / 1_000
            raise ToolError(
                f"Template '{filename}' is {size_kb:.0f}KB which exceeds the "
                f"{limit_kb:.0f}KB limit. Split large templates or increase the "
                f"limit with ARCH_REVIEW_CFN_MAX_CHARS."
            )

        return file_path.read_text(encoding="utf-8")
