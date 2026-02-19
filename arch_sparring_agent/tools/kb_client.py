"""Client for querying the Bedrock Knowledge Base (WAF best practices)."""

from __future__ import annotations

import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class KnowledgeBaseClient:
    """Wraps the ``bedrock-agent-runtime`` Retrieve API for WAF queries."""

    def __init__(self, knowledge_base_id: str, region: str):
        self.client = boto3.client("bedrock-agent-runtime", region_name=region)
        self.kb_id = knowledge_base_id

    def query(self, query: str, num_results: int = 5) -> str:
        """Retrieve relevant WAF best-practice passages for *query*.

        Returns a formatted string with source attributions, or an error
        message if the query fails.
        """
        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": num_results}
                },
            )
        except (ClientError, BotoCoreError) as e:
            logger.warning("KB query failed: %s", e)
            return f"[Knowledge Base query failed: {e}]"

        results = response.get("retrievalResults", [])
        if not results:
            return "[No relevant WAF best practices found for this query.]"

        parts: list[str] = []
        for i, result in enumerate(results, 1):
            content = result.get("content", {}).get("text", "")
            location = result.get("location", {})
            source = ""
            if location.get("type") == "S3":
                uri = location.get("s3Location", {}).get("uri", "")
                source = f" (source: {uri})" if uri else ""

            score = result.get("score")
            score_str = f" [relevance: {score:.2f}]" if score is not None else ""

            parts.append(f"### Result {i}{score_str}{source}\n{content}")

        return "\n\n".join(parts)
