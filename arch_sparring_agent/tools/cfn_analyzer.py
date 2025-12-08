"""CloudFormation template reader."""

from pathlib import Path


class CloudFormationAnalyzer:
    """Reads CloudFormation templates (YAML, JSON, CDK output)."""

    def __init__(self, templates_dir: str):
        self.templates_dir = Path(templates_dir)

    def list_templates(self) -> list[str]:
        """List CloudFormation template files."""
        templates = []
        for ext in ["*.yaml", "*.yml", "*.json", "*.template.json"]:
            templates.extend(f.name for f in self.templates_dir.glob(ext))
        return templates

    def read_template(self, filename: str) -> str:
        """Read template file as string."""
        file_path = self.templates_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Template not found: {filename}")
        return file_path.read_text(encoding="utf-8")
