"""Upload scraped WAF content to S3 and trigger Bedrock KB ingestion."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import boto3
import yaml

from ..config import DEFAULT_REGION
from ..exceptions import AWS_ERRORS
from ..infra.polling import poll_until

logger = logging.getLogger(__name__)

METADATA_KEYS = ("source", "pillar")


def _extract_frontmatter(md_path: Path) -> dict[str, str]:
    """Extract YAML frontmatter metadata from a markdown file."""
    text = md_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    try:
        fm = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return {}
    if not isinstance(fm, dict):
        return {}
    return {k: str(v) for k, v in fm.items() if k in METADATA_KEYS and v}


def _build_sidecar(metadata: dict[str, str]) -> str:
    """Build a Bedrock KB ``.metadata.json`` sidecar from extracted fields."""
    return json.dumps({"metadataAttributes": metadata}, separators=(",", ":"))


def upload_to_s3(
    content_dir: str | Path,
    bucket_name: str,
    region: str = DEFAULT_REGION,
) -> int:
    """Upload markdown files and metadata sidecars to *bucket_name*.

    For each ``.md`` file that contains YAML frontmatter with ``source``
    and/or ``pillar``, a companion ``.metadata.json`` file is uploaded so
    Bedrock KB can use those fields for filtered retrieval.

    Returns the number of content files uploaded.
    """
    content_path = Path(content_dir)
    s3 = boto3.client("s3", region_name=region)
    count = 0

    for md_file in content_path.rglob("*.md"):
        key = str(md_file.relative_to(content_path))
        s3.upload_file(str(md_file), bucket_name, key)
        count += 1

        fm = _extract_frontmatter(md_file)
        if fm:
            sidecar_key = f"{key}.metadata.json"
            s3.put_object(
                Bucket=bucket_name,
                Key=sidecar_key,
                Body=_build_sidecar(fm),
                ContentType="application/json",
            )

        if count % 50 == 0:
            logger.info("Uploaded %d files...", count)

    logger.info("Uploaded %d files to s3://%s", count, bucket_name)
    return count


def _find_data_source_id(kb_id: str, region: str) -> str | None:
    """Find the first data source associated with a KB."""
    bedrock = boto3.client("bedrock-agent", region_name=region)
    resp = bedrock.list_data_sources(knowledgeBaseId=kb_id)
    sources = resp.get("dataSourceSummaries", [])
    return sources[0]["dataSourceId"] if sources else None


def start_ingestion(kb_id: str, data_source_id: str, region: str = DEFAULT_REGION) -> str:
    """Start a KB ingestion job. Returns the ingestion job ID."""
    bedrock = boto3.client("bedrock-agent", region_name=region)
    resp = bedrock.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=data_source_id,
    )
    job_id = resp["ingestionJob"]["ingestionJobId"]
    logger.info("Started ingestion job: %s", job_id)
    return job_id


def wait_for_ingestion(
    kb_id: str,
    data_source_id: str,
    job_id: str,
    region: str = DEFAULT_REGION,
    timeout: int = 600,
) -> bool:
    """Poll until the ingestion job completes.

    Returns True on success, False on failure or timeout.
    """
    bedrock = boto3.client("bedrock-agent", region_name=region)

    def _check() -> bool | None:
        try:
            resp = bedrock.get_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=data_source_id,
                ingestionJobId=job_id,
            )
            status = resp["ingestionJob"]["status"]
            if status == "COMPLETE":
                stats = resp["ingestionJob"].get("statistics", {})
                logger.info(
                    "Ingestion complete — scanned: %s, indexed: %s, failed: %s",
                    stats.get("numberOfDocumentsScanned", "?"),
                    stats.get("numberOfNewDocumentsIndexed", "?"),
                    stats.get("numberOfDocumentsFailed", "?"),
                )
                return True
            if status == "FAILED":
                reasons = resp["ingestionJob"].get("failureReasons", [])
                logger.error("Ingestion failed: %s", reasons)
                raise StopIteration
            logger.debug("Ingestion status: %s", status)
            return None
        except AWS_ERRORS as e:
            logger.warning("Error polling ingestion status: %s", e)
            return None

    return poll_until(_check, interval=15.0, timeout=timeout, desc="KB ingestion") is not None


def sync_kb(
    content_dir: str | Path,
    kb_id: str,
    bucket_name: str,
    region: str = DEFAULT_REGION,
) -> bool:
    """Upload content to S3, trigger ingestion, and wait for completion.

    Returns True on success.
    """
    upload_to_s3(content_dir, bucket_name, region)

    ds_id = _find_data_source_id(kb_id, region)
    if not ds_id:
        logger.error("No data source found for KB %s", kb_id)
        return False

    job_id = start_ingestion(kb_id, ds_id, region)
    return wait_for_ingestion(kb_id, ds_id, job_id, region)
