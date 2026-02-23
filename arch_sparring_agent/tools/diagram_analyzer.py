"""Architecture diagram analyzer using Bedrock multimodal."""

from __future__ import annotations

import io
import os
from pathlib import Path

import boto3
from PIL import Image, UnidentifiedImageError

from ..config import DEFAULT_REGION, DIAGRAM_MAX_BYTES, DIAGRAM_MAX_TOKENS
from ..exceptions import AWS_ERRORS, ToolError
from . import validate_file_size, validate_path


class DiagramAnalyzer:
    """Analyzes architecture diagrams via Bedrock Converse API."""

    def __init__(self, diagrams_dir: str, model_id: str, region: str | None = None):
        self.diagrams_dir = Path(diagrams_dir)
        self.model_id = model_id
        resolved_region = region or os.getenv("AWS_REGION", DEFAULT_REGION)
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=resolved_region)

    def read_diagram(self, filename: str) -> str:
        """Analyze diagram and return text description."""
        image_path = validate_path(self.diagrams_dir, filename)
        if not image_path.exists():
            raise ToolError(f"Diagram not found: {filename}")

        validate_file_size(image_path, DIAGRAM_MAX_BYTES, "ARCH_REVIEW_DIAGRAM_MAX_BYTES")

        image_bytes = image_path.read_bytes()

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()
        except (OSError, UnidentifiedImageError) as e:
            raise ToolError(f"Invalid image: {e}") from e

        suffix = Path(filename).suffix.lower()
        if suffix == ".png":
            img_format = "png"
        elif suffix in (".jpg", ".jpeg"):
            img_format = "jpeg"
        else:
            raise ToolError(f"Unsupported format: {filename}. Use PNG or JPEG.")

        try:
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": img_format,
                                    "source": {"bytes": image_bytes},
                                }
                            },
                            {
                                "text": (
                                    "Describe this architecture diagram: components, "
                                    "relationships, data flows, and patterns."
                                )
                            },
                        ],
                    }
                ],
                inferenceConfig={"maxTokens": DIAGRAM_MAX_TOKENS, "temperature": 0.1},
            )

            if "output" in response and "message" in response["output"]:
                content = response["output"]["message"].get("content", [])
                return "\n".join(item.get("text", "") for item in content if "text" in item)
            return str(response)

        except AWS_ERRORS as e:
            raise ToolError(f"Bedrock API error: {e}") from e

    def list_diagrams(self) -> list[str]:
        """List diagram files (PNG, JPEG), returning paths relative to diagrams_dir."""
        extensions = {".png", ".jpg", ".jpeg"}
        return sorted(
            str(f.relative_to(self.diagrams_dir))
            for f in self.diagrams_dir.rglob("*")
            if f.suffix.lower() in extensions
        )
