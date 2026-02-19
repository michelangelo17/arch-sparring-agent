"""Knowledge Base infrastructure — S3 data bucket, S3 Vectors, IAM role, Bedrock KB."""

import json
import logging
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ..config import DEFAULT_REGION

logger = logging.getLogger(__name__)

KB_BUCKET_PREFIX = "arch-review-kb"
VECTOR_BUCKET_PREFIX = "arch-review-vectors"
KB_NAME = "arch-review-waf-kb"
VECTOR_INDEX_NAME = "waf-index"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSION = 1024


def _get_account_id(region: str) -> str:
    sts = boto3.client("sts", region_name=region)
    return sts.get_caller_identity()["Account"]


def _data_bucket_name(account_id: str, region: str) -> str:
    return f"{KB_BUCKET_PREFIX}-{account_id}-{region}"


def _vector_bucket_name(account_id: str, region: str) -> str:
    return f"{VECTOR_BUCKET_PREFIX}-{account_id}-{region}"


# ---------------------------------------------------------------------------
# S3 data bucket (stores the scraped WAF markdown files)
# ---------------------------------------------------------------------------


def _create_data_bucket(bucket_name: str, region: str) -> None:
    s3 = boto3.client("s3", region_name=region)
    try:
        create_kwargs: dict = {"Bucket": bucket_name}
        if region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**create_kwargs)
        logger.info("Created S3 data bucket: %s", bucket_name)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            logger.info("S3 data bucket already exists: %s", bucket_name)
        else:
            raise


def _delete_data_bucket(bucket_name: str, region: str) -> None:
    s3 = boto3.resource("s3", region_name=region)
    bucket = s3.Bucket(bucket_name)
    try:
        bucket.objects.all().delete()
        bucket.delete()
        logger.info("Deleted S3 data bucket: %s", bucket_name)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "NoSuchBucket":
            logger.info("S3 data bucket already absent: %s", bucket_name)
        else:
            logger.warning("Could not delete S3 data bucket %s: %s", bucket_name, e)


# ---------------------------------------------------------------------------
# S3 Vectors (vector bucket + index)
# ---------------------------------------------------------------------------


def _create_vector_bucket(vector_bucket_name: str, region: str) -> str:
    """Create an S3 vector bucket. Returns the vector bucket ARN."""
    s3v = boto3.client("s3vectors", region_name=region)
    try:
        resp = s3v.create_vector_bucket(vectorBucketName=vector_bucket_name)
        arn = resp["vectorBucketArn"]
        logger.info("Created S3 vector bucket: %s", vector_bucket_name)
        return arn
    except ClientError as e:
        if "ConflictException" in str(e) or "already exists" in str(e).lower():
            arn = _find_vector_bucket_arn(s3v, vector_bucket_name)
            if arn:
                logger.info("Using existing S3 vector bucket: %s", vector_bucket_name)
                return arn
        raise


def _find_vector_bucket_arn(s3v, name: str) -> str | None:
    resp = s3v.list_vector_buckets()
    for vb in resp.get("vectorBuckets", []):
        if vb.get("vectorBucketName") == name:
            return vb.get("vectorBucketArn")
    return None


def _create_vector_index(vector_bucket_name: str, index_name: str, region: str) -> str:
    """Create a vector index inside the vector bucket. Returns the index ARN."""
    s3v = boto3.client("s3vectors", region_name=region)
    try:
        resp = s3v.create_index(
            vectorBucketName=vector_bucket_name,
            indexName=index_name,
            dataType="float32",
            dimension=EMBEDDING_DIMENSION,
            distanceMetric="cosine",
        )
        arn = resp["indexArn"]
        logger.info("Created vector index: %s", index_name)
        return arn
    except ClientError as e:
        if "ConflictException" in str(e) or "already exists" in str(e).lower():
            arn = _find_vector_index_arn(s3v, vector_bucket_name, index_name)
            if arn:
                logger.info("Using existing vector index: %s", index_name)
                return arn
        raise


def _find_vector_index_arn(s3v, vector_bucket_name: str, index_name: str) -> str | None:
    resp = s3v.list_indexes(vectorBucketName=vector_bucket_name)
    for idx in resp.get("indexes", []):
        if idx.get("indexName") == index_name:
            return idx.get("indexArn")
    return None


def _delete_vector_bucket(vector_bucket_name: str, region: str) -> None:
    s3v = boto3.client("s3vectors", region_name=region)

    try:
        indexes = s3v.list_indexes(vectorBucketName=vector_bucket_name)
        for idx in indexes.get("indexes", []):
            try:
                s3v.delete_index(
                    vectorBucketName=vector_bucket_name,
                    indexName=idx["indexName"],
                )
                logger.info("Deleted vector index: %s", idx["indexName"])
            except (ClientError, BotoCoreError) as e:
                logger.warning("Could not delete index %s: %s", idx["indexName"], e)
    except (ClientError, BotoCoreError):
        pass

    try:
        s3v.delete_vector_bucket(vectorBucketName=vector_bucket_name)
        logger.info("Deleted S3 vector bucket: %s", vector_bucket_name)
    except ClientError as e:
        if "NotFoundException" in str(e):
            logger.info("S3 vector bucket already absent: %s", vector_bucket_name)
        else:
            logger.warning("Could not delete vector bucket %s: %s", vector_bucket_name, e)
    except BotoCoreError as e:
        logger.warning("Could not delete vector bucket %s: %s", vector_bucket_name, e)


# ---------------------------------------------------------------------------
# IAM role for Bedrock KB
# ---------------------------------------------------------------------------


def _create_kb_role(
    account_id: str, data_bucket_name: str, vector_bucket_name: str, region: str
) -> str:
    """Create IAM role that lets Bedrock KB read from S3 and use S3 Vectors."""
    iam = boto3.client("iam", region_name=region)
    role_name = "arch-review-kb-role"

    trust_policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": account_id},
                    },
                }
            ],
        }
    )

    try:
        resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=trust_policy,
            Description="Allows Bedrock Knowledge Base to access S3 and S3 Vectors",
        )
        role_arn = resp["Role"]["Arn"]
        logger.info("Created IAM role: %s", role_name)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "EntityAlreadyExists":
            role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
            logger.info("Using existing IAM role: %s", role_name)
        else:
            raise

    embedding_arn = f"arn:aws:bedrock:{region}::foundation-model/{EMBEDDING_MODEL}"
    inline_policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{data_bucket_name}",
                        f"arn:aws:s3:::{data_bucket_name}/*",
                    ],
                },
                {
                    "Effect": "Allow",
                    "Action": "bedrock:InvokeModel",
                    "Resource": embedding_arn,
                },
                {
                    "Effect": "Allow",
                    "Action": "s3vectors:*",
                    "Resource": (
                        f"arn:aws:s3vectors:{region}:{account_id}:"
                        f"vector-bucket/{vector_bucket_name}/*"
                    ),
                },
            ],
        }
    )

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="arch-review-kb-access",
        PolicyDocument=inline_policy,
    )

    time.sleep(10)
    return role_arn


def _delete_kb_role(region: str) -> None:
    iam = boto3.client("iam", region_name=region)
    role_name = "arch-review-kb-role"
    try:
        iam.delete_role_policy(RoleName=role_name, PolicyName="arch-review-kb-access")
    except (ClientError, BotoCoreError):
        pass
    try:
        iam.delete_role(RoleName=role_name)
        logger.info("Deleted IAM role: %s", role_name)
    except (ClientError, BotoCoreError) as e:
        logger.warning("Could not delete IAM role %s: %s", role_name, e)


# ---------------------------------------------------------------------------
# Bedrock Knowledge Base
# ---------------------------------------------------------------------------


def _create_bedrock_kb(
    role_arn: str,
    vector_bucket_arn: str,
    vector_index_arn: str,
    region: str,
) -> str:
    """Create the Bedrock Knowledge Base backed by S3 Vectors.

    Returns the KB ID.
    """
    bedrock = boto3.client("bedrock-agent", region_name=region)
    embedding_arn = f"arn:aws:bedrock:{region}::foundation-model/{EMBEDDING_MODEL}"

    try:
        resp = bedrock.create_knowledge_base(
            name=KB_NAME,
            description=("AWS Well-Architected Framework best practices for architecture review"),
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": embedding_arn,
                },
            },
            storageConfiguration={
                "type": "S3_VECTORS",
                "s3VectorsConfiguration": {
                    "vectorBucketArn": vector_bucket_arn,
                    "indexArn": vector_index_arn,
                    "indexName": VECTOR_INDEX_NAME,
                },
            },
        )
        kb_id = resp["knowledgeBase"]["knowledgeBaseId"]
        logger.info("Created Bedrock Knowledge Base: %s (ID: %s)", KB_NAME, kb_id)
        return kb_id
    except ClientError as e:
        if "ConflictException" in str(e) or "already exists" in str(e).lower():
            kb_id = _find_kb_by_name(bedrock)
            if kb_id:
                logger.info("Using existing Bedrock Knowledge Base: %s", KB_NAME)
                return kb_id
        raise


def _find_kb_by_name(bedrock) -> str | None:
    paginator = bedrock.get_paginator("list_knowledge_bases")
    for page in paginator.paginate():
        for kb in page.get("knowledgeBaseSummaries", []):
            if kb.get("name") == KB_NAME:
                return kb["knowledgeBaseId"]
    return None


def _create_data_source(kb_id: str, bucket_name: str, region: str) -> str:
    """Create an S3 data source for the KB. Returns the data source ID."""
    bedrock = boto3.client("bedrock-agent", region_name=region)

    try:
        resp = bedrock.create_data_source(
            knowledgeBaseId=kb_id,
            name="waf-content",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{bucket_name}",
                },
            },
            vectorIngestionConfiguration={
                "chunkingConfiguration": {"chunkingStrategy": "NONE"},
            },
        )
        ds_id = resp["dataSource"]["dataSourceId"]
        logger.info("Created KB data source: %s", ds_id)
        return ds_id
    except ClientError as e:
        if "ConflictException" in str(e) or "already exists" in str(e).lower():
            ds_id = _find_data_source(bedrock, kb_id)
            if ds_id:
                logger.info("Using existing KB data source")
                return ds_id
        raise


def _find_data_source(bedrock, kb_id: str) -> str | None:
    resp = bedrock.list_data_sources(knowledgeBaseId=kb_id)
    sources = resp.get("dataSourceSummaries", [])
    return sources[0]["dataSourceId"] if sources else None


def _delete_bedrock_kb(kb_id: str, region: str) -> None:
    bedrock = boto3.client("bedrock-agent", region_name=region)
    try:
        sources = bedrock.list_data_sources(knowledgeBaseId=kb_id)
        for ds in sources.get("dataSourceSummaries", []):
            try:
                bedrock.delete_data_source(
                    knowledgeBaseId=kb_id,
                    dataSourceId=ds["dataSourceId"],
                )
            except (ClientError, BotoCoreError):
                pass
        bedrock.delete_knowledge_base(knowledgeBaseId=kb_id)
        logger.info("Deleted Bedrock Knowledge Base: %s", kb_id)
    except (ClientError, BotoCoreError) as e:
        logger.warning("Could not delete KB %s: %s", kb_id, e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_knowledge_base(region: str = DEFAULT_REGION) -> tuple[str, str]:
    """Provision all KB infrastructure.

    Returns:
        Tuple of (knowledge_base_id, data_bucket_name).
    """
    account_id = _get_account_id(region)
    data_bucket = _data_bucket_name(account_id, region)
    vector_bucket = _vector_bucket_name(account_id, region)

    _create_data_bucket(data_bucket, region)

    vector_bucket_arn = _create_vector_bucket(vector_bucket, region)
    vector_index_arn = _create_vector_index(vector_bucket, VECTOR_INDEX_NAME, region)

    role_arn = _create_kb_role(account_id, data_bucket, vector_bucket, region)

    kb_id = _create_bedrock_kb(role_arn, vector_bucket_arn, vector_index_arn, region)
    _create_data_source(kb_id, data_bucket, region)

    logger.info(
        "Knowledge Base setup complete — KB ID: %s, Bucket: %s",
        kb_id,
        data_bucket,
    )
    return kb_id, data_bucket


def destroy_knowledge_base(
    kb_id: str | None,
    bucket_name: str | None,
    region: str = DEFAULT_REGION,
) -> None:
    """Tear down all KB infrastructure."""
    if kb_id:
        _delete_bedrock_kb(kb_id, region)

    account_id = _get_account_id(region)
    vector_bucket = _vector_bucket_name(account_id, region)
    _delete_vector_bucket(vector_bucket, region)

    _delete_kb_role(region)

    if bucket_name:
        _delete_data_bucket(bucket_name, region)

    logger.info("Knowledge Base teardown complete")
