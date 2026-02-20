"""Agent implementations for each review phase."""

import logging

from strands import tool

from ..tools.kb_client import KnowledgeBaseClient

logger = logging.getLogger(__name__)


def create_kb_tool(knowledge_base_id: str, region: str, *, with_sources_filter: bool = False):
    """Create a ``query_waf`` tool backed by a KnowledgeBaseClient.

    Args:
        knowledge_base_id: Bedrock Knowledge Base ID.
        region: AWS region for the KB.
        with_sources_filter: If True, the tool accepts an optional ``sources``
            parameter for filtering results by WAF lens.
    """
    kb_client = KnowledgeBaseClient(knowledge_base_id, region)

    if with_sources_filter:

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

    else:

        @tool
        def query_waf(query: str) -> str:
            """Query the AWS Well-Architected Framework knowledge base for best practices."""
            logger.info("query_waf(%r)", query[:80])
            result = kb_client.query(query)
            logger.info("query_waf → %d chars", len(result))
            return result

    return query_waf
