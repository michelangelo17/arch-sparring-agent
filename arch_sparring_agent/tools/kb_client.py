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

    def query(
        self,
        query: str,
        num_results: int = 5,
        sources: list[str] | None = None,
    ) -> str:
        """Retrieve relevant WAF best-practice passages for *query*.

        Args:
            query: Natural-language question.
            num_results: Maximum results to return.
            sources: Optional list of source identifiers to restrict results
                to (e.g. ``["well-architected-framework",
                "serverless-applications-lens"]``).  When omitted, all
                sources are searched.

        Returns a formatted string with source attributions, or an error
        message if the query fails.
        """
        vector_cfg: dict = {"numberOfResults": num_results}

        if sources:
            if len(sources) == 1:
                vector_cfg["filter"] = {"equals": {"key": "source", "value": sources[0]}}
            else:
                vector_cfg["filter"] = {
                    "orAll": [{"equals": {"key": "source", "value": s}} for s in sources]
                }

        logger.debug("KB query: %r | filter: %s", query, sources or "none (all sources)")

        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={"vectorSearchConfiguration": vector_cfg},
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

            metadata = result.get("metadata", {})
            source_tag = metadata.get("source", "")

            location = result.get("location", {})
            s3_uri = ""
            if location.get("type") == "S3":
                s3_uri = location.get("s3Location", {}).get("uri", "")

            attribution = ""
            if source_tag:
                attribution = f" [source: {source_tag}]"
            elif s3_uri:
                attribution = f" (s3: {s3_uri})"

            score = result.get("score")
            score_str = f" [relevance: {score:.2f}]" if score is not None else ""

            parts.append(f"### Result {i}{score_str}{attribution}\n{content}")

        return "\n\n".join(parts)
