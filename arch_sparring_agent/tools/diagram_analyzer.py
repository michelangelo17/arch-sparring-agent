"""Architecture diagram analyzer using Bedrock multimodal."""

from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image, UnidentifiedImageError

from ..config import (
    DIAGRAM_MAX_BYTES,
    DIAGRAM_MAX_TOKENS,
    MODEL_ID,
    get_bedrock_client,
    get_inference_profile_arn,
)
from ..exceptions import ToolError


def _validate_path(base_dir: Path, filename: str) -> Path:
    """Resolve path and verify it stays within the base directory."""
    file_path = (base_dir / filename).resolve()
    if not file_path.is_relative_to(base_dir.resolve()):
        raise ToolError(f"Path traversal detected: {filename}")
    return file_path


class DiagramAnalyzer:
    """Analyzes architecture diagrams via Bedrock Converse API."""

    def __init__(self, diagrams_dir: str, model_id: str = MODEL_ID):
        self.diagrams_dir = Path(diagrams_dir)
        self.model_id = model_id
        self.bedrock_client = get_bedrock_client()

    def _read_image_bytes(self, image_path: Path) -> bytes:
        """Read raw image bytes from file."""
        with open(image_path, "rb") as f:
            return f.read()

    def read_diagram(self, filename: str) -> str:
        """Analyze diagram and return text description."""
        image_path = _validate_path(self.diagrams_dir, filename)
        if not image_path.exists():
            raise ToolError(f"Diagram not found: {filename}")

        # Check file size before reading
        file_size = image_path.stat().st_size
        if file_size > DIAGRAM_MAX_BYTES:
            size_mb = file_size / 1_000_000
            limit_mb = DIAGRAM_MAX_BYTES / 1_000_000
            raise ToolError(
                f"Diagram '{filename}' is {size_mb:.1f}MB which exceeds the "
                f"{limit_mb:.1f}MB limit. Reduce the image size or increase the "
                f"limit with ARCH_REVIEW_DIAGRAM_MAX_BYTES."
            )

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

        inference_profile_arn = get_inference_profile_arn(self.model_id)

        try:
            # Use inference profile if available, otherwise fall back to direct model ID
            model_param = (
                {"inferenceProfileArn": inference_profile_arn}
                if inference_profile_arn
                else {"modelId": self.model_id}
            )
            response = self.bedrock_client.converse(
                **model_param,
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
