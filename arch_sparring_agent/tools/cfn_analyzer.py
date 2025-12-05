from pathlib import Path


class CloudFormationAnalyzer:
    """
    Reads CloudFormation templates. The agent analyzes the content to understand
    resources, patterns, dependencies, and architecture.

    Works for both regular CloudFormation templates and CDK synthesized output.
    """

    def __init__(self, templates_dir: str):
        self.templates_dir = Path(templates_dir)

    def list_templates(self) -> list[str]:
        """List all CloudFormation templates in the directory."""
        templates = []
        for ext in ["*.yaml", "*.yml", "*.json", "*.template.json"]:
            templates.extend(f.name for f in self.templates_dir.glob(ext))
        return templates

    def read_template(self, filename: str) -> str:
        """Read a CloudFormation template file and return its content as a string."""
        file_path = self.templates_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Template not found: {filename}")

        with open(file_path, encoding="utf-8") as f:
            return f.read()
