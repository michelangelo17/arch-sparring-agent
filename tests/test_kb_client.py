"""Tests for the Knowledge Base query client."""

import logging
from unittest.mock import MagicMock, patch

from arch_sparring_agent.tools.kb_client import KnowledgeBaseClient


@patch("arch_sparring_agent.tools.kb_client.boto3")
def test_query_returns_formatted_results(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": "Enable encryption at rest for all data stores."},
                "location": {
                    "type": "S3",
                    "s3Location": {"uri": "s3://bucket/security/sec01-bp01.md"},
                },
                "metadata": {"source": "well-architected-framework", "pillar": "security"},
                "score": 0.85,
            },
            {
                "content": {"text": "Use TLS for all data in transit."},
                "location": {
                    "type": "S3",
                    "s3Location": {"uri": "s3://bucket/security/sec02.md"},
                },
                "metadata": {"source": "serverless-applications-lens"},
                "score": 0.72,
            },
        ]
    }

    client = KnowledgeBaseClient("kb-123", "eu-central-1")
    result = client.query("encryption best practices")

    assert "encryption at rest" in result
    assert "TLS" in result
    assert "[source: well-architected-framework]" in result
    assert "[source: serverless-applications-lens]" in result
    assert "[relevance: 0.85]" in result


@patch("arch_sparring_agent.tools.kb_client.boto3")
def test_query_returns_message_on_empty_results(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.retrieve.return_value = {"retrievalResults": []}

    client = KnowledgeBaseClient("kb-123", "eu-central-1")
    result = client.query("something obscure")

    assert "No relevant WAF best practices" in result


@patch("arch_sparring_agent.tools.kb_client.boto3")
def test_query_handles_api_error(mock_boto3):
    from arch_sparring_agent.tools.kb_client import ClientError as ModuleClientError

    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.retrieve.side_effect = ModuleClientError("ResourceNotFoundException")

    client = KnowledgeBaseClient("kb-123", "eu-central-1")
    result = client.query("test query")

    assert "Knowledge Base query failed" in result


@patch("arch_sparring_agent.tools.kb_client.boto3")
def test_query_with_custom_num_results(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.retrieve.return_value = {"retrievalResults": []}

    client = KnowledgeBaseClient("kb-123", "eu-central-1")
    client.query("test", num_results=10)

    call_kwargs = mock_client.retrieve.call_args[1]
    config = call_kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]
    assert config["numberOfResults"] == 10


@patch("arch_sparring_agent.tools.kb_client.boto3")
def test_query_handles_missing_score(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": "Best practice content."},
                "location": {"type": "S3", "s3Location": {"uri": "s3://b/f.md"}},
            }
        ]
    }

    client = KnowledgeBaseClient("kb-123", "eu-central-1")
    result = client.query("test")

    assert "Best practice content" in result
    assert "relevance" not in result
    # No metadata → falls back to S3 URI
    assert "(s3: s3://b/f.md)" in result


@patch("arch_sparring_agent.tools.kb_client.boto3")
def test_query_logs_filter(mock_boto3, caplog):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.retrieve.return_value = {"retrievalResults": []}

    client = KnowledgeBaseClient("kb-123", "eu-central-1")
    with caplog.at_level(logging.DEBUG, logger="arch_sparring_agent.tools.kb_client"):
        client.query("test", sources=["serverless-applications-lens"])

    assert "serverless-applications-lens" in caplog.text


@patch("arch_sparring_agent.tools.kb_client.boto3")
def test_query_logs_no_filter(mock_boto3, caplog):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.retrieve.return_value = {"retrievalResults": []}

    client = KnowledgeBaseClient("kb-123", "eu-central-1")
    with caplog.at_level(logging.DEBUG, logger="arch_sparring_agent.tools.kb_client"):
        client.query("test")

    assert "none (all sources)" in caplog.text


@patch("arch_sparring_agent.tools.kb_client.boto3")
def test_query_source_metadata_preferred_over_s3_uri(mock_boto3):
    """When metadata.source exists, it is shown instead of the S3 URI."""
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": "Some content."},
                "location": {"type": "S3", "s3Location": {"uri": "s3://b/file.md"}},
                "metadata": {"source": "data-analytics-lens"},
                "score": 0.9,
            }
        ]
    }

    client = KnowledgeBaseClient("kb-123", "eu-central-1")
    result = client.query("analytics")

    assert "[source: data-analytics-lens]" in result
    assert "s3://b/file.md" not in result
