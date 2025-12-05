import base64
from pathlib import Path

from PIL import Image

from ..config import MODEL_ID, get_bedrock_client, get_inference_profile_arn


class DiagramAnalyzer:
    """
    Analyzes architecture diagrams using multimodal capabilities.
    """

    def __init__(self, diagrams_dir: str):
        self.diagrams_dir = Path(diagrams_dir)
        self.bedrock_client = get_bedrock_client()

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for Bedrock API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def read_diagram(self, filename: str) -> str:
        """
        Read and analyze an architecture diagram using multimodal capabilities.
        Returns a text description of the diagram for the agent to understand.
        """
        image_path = self.diagrams_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Diagram not found: {filename}")

        # Verify it's an image
        try:
            Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Not a valid image file: {e}") from e

        # Encode image
        image_base64 = self.encode_image(image_path)

        # Determine image format (Bedrock supports PNG and JPEG)
        filename_lower = filename.lower()
        if filename_lower.endswith(".png"):
            img_format = "png"
        elif filename_lower.endswith((".jpg", ".jpeg")):
            img_format = "jpeg"
        else:
            raise ValueError(f"Unsupported image format: {filename}. Supported formats: PNG, JPEG")

        # Call Bedrock using Converse API
        # Note: Nova 2 Lite requires inference profile ARN for direct Converse API calls
        # Format: arn:aws:bedrock:REGION:ACCOUNT_ID:inference-profile/global.MODEL_ID
        # Strands SDK Agent handles this automatically
        try:
            # Get inference profile ARN (required for Nova 2 Lite)
            inference_profile_arn = get_inference_profile_arn(MODEL_ID)
            if not inference_profile_arn:
                raise RuntimeError(
                    "Could not get inference profile ARN. "
                    "Nova 2 Lite requires inference profile for direct Converse API calls."
                )

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
                                    "Describe this architecture diagram in detail, including all "
                                    "components, relationships, data flows, and architectural "
                                    "patterns."
                                )
                            },
                        ],
                    }
                ],
                inferenceConfig={"maxTokens": 4000, "temperature": 0.1},
            )

            # Extract response from Converse API
            if "output" in response and "message" in response["output"]:
                content = response["output"]["message"].get("content", [])
                text_parts = [item.get("text", "") for item in content if "text" in item]
                return "\n".join(text_parts)
            else:
                return str(response)
        except Exception as e:
            raise RuntimeError(f"Bedrock API error analyzing diagram: {e}") from e

    def list_diagrams(self) -> list[str]:
        """List all diagram image files (PNG and JPEG only)."""
        extensions = [".png", ".jpg", ".jpeg"]
        return [f.name for f in self.diagrams_dir.iterdir() if f.suffix.lower() in extensions]
