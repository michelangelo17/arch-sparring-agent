"""Architecture diagram analyzer using Bedrock multimodal."""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image, UnidentifiedImageError

from ..config import DEFAULT_REGION, DIAGRAM_MAX_BYTES, DIAGRAM_MAX_TOKENS
from ..exceptions import ToolError
from . import validate_file_size, validate_path


class DiagramAnalyzer:
    """Analyzes architecture diagrams via Bedrock Converse API."""

    def __init__(self, diagrams_dir: str, model_id: str):
        self.diagrams_dir = Path(diagrams_dir)
        self.model_id = model_id
        region = os.getenv("AWS_REGION", DEFAULT_REGION)
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)

    def _read_image_bytes(self, image_path: Path) -> bytes:
        """Read raw image bytes from file."""
        with open(image_path, "rb") as f:
            return f.read()

    def read_diagram(self, filename: str) -> str:
        """Analyze diagram and return text description."""
        image_path = validate_path(self.diagrams_dir, filename)
        if not image_path.exists():
            raise ToolError(f"Diagram not found: {filename}")

        validate_file_size(image_path, DIAGRAM_MAX_BYTES, "ARCH_REVIEW_DIAGRAM_MAX_BYTES")

        try:
            with Image.open(image_path) as img:
                img.verify()
        except (OSError, UnidentifiedImageError) as e:
            raise ToolError(f"Invalid image: {e}") from e

        image_bytes = self._read_image_bytes(image_path)

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

        except (ClientError, BotoCoreError) as e:
            raise ToolError(f"Bedrock API error: {e}") from e

    def list_diagrams(self) -> list[str]:
        """List diagram files (PNG, JPEG)."""
        extensions = [".png", ".jpg", ".jpeg"]
        return [f.name for f in self.diagrams_dir.iterdir() if f.suffix.lower() in extensions]
