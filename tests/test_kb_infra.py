"""Tests for arch_sparring_agent.kb.infra."""

from unittest.mock import MagicMock, patch

import pytest

from arch_sparring_agent.kb.infra import (
    _create_bedrock_kb,
    _create_data_bucket,
    _create_data_source,
    _create_vector_bucket,
    destroy_knowledge_base,
    setup_knowledge_base,
)
from tests.conftest import FakeClientError


@patch("arch_sparring_agent.kb.infra.get_account_id", return_value="123456789012")
@patch("arch_sparring_agent.kb.infra.time")
@patch("arch_sparring_agent.kb.infra.boto3")
def test_setup_knowledge_base_creates_all_resources(mock_boto3, mock_time, _mock_account):
    mock_s3 = MagicMock()
    mock_s3_vectors = MagicMock()
    mock_iam = MagicMock()
    mock_bedrock = MagicMock()

    def make_client(service, **kwargs):
        return {
            "s3": mock_s3,
            "s3vectors": mock_s3_vectors,
            "iam": mock_iam,
            "bedrock-agent": mock_bedrock,
        }.get(service, MagicMock())

    mock_boto3.client.side_effect = make_client
    mock_s3_vectors.create_vector_bucket.return_value = {
        "vectorBucketArn": "arn:aws:s3vectors:us-east-1:123:bucket/test"
    }
    mock_s3_vectors.create_index.return_value = {
        "indexArn": "arn:aws:s3vectors:us-east-1:123:bucket/test/index/test"
    }
    mock_iam.create_role.return_value = {"Role": {"Arn": "arn:aws:iam::123:role/test"}}
    mock_iam.get_role.return_value = {"Role": {"Arn": "arn:aws:iam::123:role/test"}}
    mock_bedrock.create_knowledge_base.return_value = {
        "knowledgeBase": {"knowledgeBaseId": "kb-123"}
    }
    mock_bedrock.create_data_source.return_value = {"dataSource": {"dataSourceId": "ds-456"}}

    kb_id, bucket_name = setup_knowledge_base(region="us-east-1")

    assert kb_id == "kb-123"
    assert "123456789012" in bucket_name


@patch("arch_sparring_agent.kb.infra.boto3")
def test_create_data_bucket_handles_already_owned(mock_boto3):
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    mock_s3.create_bucket.side_effect = FakeClientError("BucketAlreadyOwnedByYou")

    _create_data_bucket("test-bucket", "us-east-1")

    mock_s3.create_bucket.assert_called_once()


@patch("arch_sparring_agent.kb.infra.boto3")
def test_create_vector_bucket_handles_conflict(mock_boto3):
    mock_s3v = MagicMock()
    mock_boto3.client.return_value = mock_s3v
    mock_s3v.create_vector_bucket.side_effect = FakeClientError("ConflictException")
    mock_s3v.list_vector_buckets.return_value = {
        "vectorBuckets": [
            {
                "vectorBucketName": "test-vectors",
                "vectorBucketArn": "arn:aws:s3vectors:us-east-1:123:bucket/test",
            }
        ]
    }

    result = _create_vector_bucket("test-vectors", "us-east-1")

    assert result == "arn:aws:s3vectors:us-east-1:123:bucket/test"


@patch("arch_sparring_agent.infra.polling.time")
@patch("arch_sparring_agent.kb.infra.boto3")
def test_create_bedrock_kb_retries_on_iam_not_propagated(mock_boto3, mock_poll_time):
    mock_bedrock = MagicMock()
    mock_boto3.client.return_value = mock_bedrock
    mock_poll_time.monotonic.side_effect = [0.0, 1.0, 2.0, 3.0]
    mock_bedrock.create_knowledge_base.side_effect = [
        FakeClientError("not authorized"),
        {"knowledgeBase": {"knowledgeBaseId": "kb-retry"}},
    ]

    result = _create_bedrock_kb(
        "arn:aws:iam::123:role/test",
        "arn:aws:s3vectors:us-east-1:123:bucket/test",
        "arn:aws:s3vectors:us-east-1:123:bucket/test/index/test",
        "us-east-1",
    )

    assert result == "kb-retry"
    assert mock_bedrock.create_knowledge_base.call_count == 2


@patch("arch_sparring_agent.infra.polling.time")
@patch("arch_sparring_agent.kb.infra.boto3")
def test_create_bedrock_kb_raises_after_max_retries(mock_boto3, mock_poll_time):
    mock_bedrock = MagicMock()
    mock_boto3.client.return_value = mock_bedrock

    call_count = [0]

    def advancing_monotonic():
        call_count[0] += 1
        return float(call_count[0] * 100)

    mock_poll_time.monotonic.side_effect = advancing_monotonic

    error_msg = "not authorized to perform this action"

    class NotAuthorizedError(FakeClientError):
        def __str__(self):
            return error_msg

        response = {"Error": {"Code": "ValidationException"}}

    mock_bedrock.create_knowledge_base.side_effect = NotAuthorizedError()

    from arch_sparring_agent.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        _create_bedrock_kb(
            "arn:aws:iam::123:role/test",
            "arn:aws:s3vectors:us-east-1:123:bucket/test",
            "arn:aws:s3vectors:us-east-1:123:bucket/test/index/test",
            "us-east-1",
        )


@patch("arch_sparring_agent.kb.infra.boto3")
def test_create_data_source_handles_conflict(mock_boto3):
    mock_bedrock = MagicMock()
    mock_boto3.client.return_value = mock_bedrock
    mock_bedrock.create_data_source.side_effect = FakeClientError("ConflictException")
    mock_bedrock.list_data_sources.return_value = {
        "dataSourceSummaries": [
            {"dataSourceId": "ds-existing", "name": "arch-review-waf-data-source"}
        ]
    }

    result = _create_data_source("kb-123", "test-bucket", "us-east-1")

    assert result == "ds-existing"


@patch("arch_sparring_agent.kb.infra.get_account_id", return_value="123456789012")
@patch("arch_sparring_agent.kb.infra._delete_kb_role")
@patch("arch_sparring_agent.kb.infra._delete_vector_bucket")
@patch("arch_sparring_agent.kb.infra._delete_data_bucket")
@patch("arch_sparring_agent.kb.infra._delete_bedrock_kb")
def test_destroy_knowledge_base_teardown(
    mock_delete_kb, mock_delete_data, mock_delete_vector, mock_delete_role, mock_account
):
    destroy_knowledge_base("kb-123", "test-bucket", region="us-east-1")

    mock_delete_kb.assert_called_once_with("kb-123", "us-east-1")
    mock_delete_data.assert_called_once_with("test-bucket", "us-east-1")
    mock_delete_vector.assert_called_once()
    mock_delete_role.assert_called_once()
