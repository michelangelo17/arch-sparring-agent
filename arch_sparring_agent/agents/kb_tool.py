"""Factory for the query_waf tool backed by a Knowledge Base."""

from __future__ import annotations

import logging
from collections.abc import Callable

from strands import tool

from ..tools.kb_client import KnowledgeBaseClient

logger = logging.getLogger(__name__)


def create_kb_tool(knowledge_base_id: str, region: str) -> Callable[..., str]:
    """Create a ``query_waf`` tool backed by a KnowledgeBaseClient.

    Args:
        knowledge_base_id: Bedrock Knowledge Base ID.
        region: AWS region for the KB.
    """
    kb_client = KnowledgeBaseClient(knowledge_base_id, region)

    @tool
    def query_waf(query: str, sources: list[str] | None = None) -> str:
        """Query the AWS Well-Architected Framework knowledge base.

        Args:
            query: Natural-language question about best practices.
            sources: Optional filter -- restrict results to these sources.
                Use "well-architected-framework" for core WAF content.
                Use a lens slug for lens-specific content, e.g.
                "serverless-applications-lens", "saas-lens".
                Omit to search all sources.
        """
        logger.info("query_waf(%r, sources=%s)", query[:80], sources or "all")
        result = kb_client.query(query, sources=sources)
        logger.info("query_waf → %d chars", len(result))
        return result

    return query_waf
