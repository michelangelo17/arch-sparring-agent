"""Architecture diagram analyzer using Bedrock multimodal."""

import base64
from pathlib import Path

from PIL import Image

from ..config import MODEL_ID, get_bedrock_client, get_inference_profile_arn


class DiagramAnalyzer:
    """Analyzes architecture diagrams via Bedrock Converse API."""

    def __init__(self, diagrams_dir: str):
        self.diagrams_dir = Path(diagrams_dir)
        self.bedrock_client = get_bedrock_client()

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def read_diagram(self, filename: str) -> str:
        """Analyze diagram and return text description."""
        image_path = self.diagrams_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Diagram not found: {filename}")

        try:
            Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Invalid image: {e}") from e

        image_base64 = self.encode_image(image_path)

        suffix = Path(filename).suffix.lower()
        if suffix == ".png":
            img_format = "png"
        elif suffix in (".jpg", ".jpeg"):
            img_format = "jpeg"
        else:
            raise ValueError(f"Unsupported format: {filename}. Use PNG or JPEG.")

        inference_profile_arn = get_inference_profile_arn(MODEL_ID)
        if not inference_profile_arn:
            raise RuntimeError("Could not get inference profile ARN.")

        try:
            response = self.bedrock_client.converse(
                inferenceProfileArn=inference_profile_arn,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": img_format,
                                    "source": {"bytes": base64.b64decode(image_base64)},
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
                inferenceConfig={"maxTokens": 4000, "temperature": 0.1},
            )

            if "output" in response and "message" in response["output"]:
                content = response["output"]["message"].get("content", [])
                return "\n".join(item.get("text", "") for item in content if "text" in item)
            return str(response)

        except Exception as e:
            raise RuntimeError(f"Bedrock API error: {e}") from e

    def list_diagrams(self) -> list[str]:
        """List diagram files (PNG, JPEG)."""
        extensions = [".png", ".jpg", ".jpeg"]
        return [f.name for f in self.diagrams_dir.iterdir() if f.suffix.lower() in extensions]
