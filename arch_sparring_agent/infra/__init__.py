"""AWS infrastructure lifecycle: Gateway, Policy Engine, Memory, and shared config."""

from __future__ import annotations

import functools

import boto3

from .shared_config import (
    SSM_PARAMETER_NAME,
    SharedConfig,
    delete_from_ssm,
    load_from_ssm,
    save_to_ssm,
)


@functools.lru_cache(maxsize=4)
def get_account_id(region: str) -> str:
    """Return the AWS account ID for the current caller (cached per region)."""
    sts = boto3.client("sts", region_name=region)
    return sts.get_caller_identity()["Account"]


def build_gateway_arn(region: str, account_id: str, gateway_id: str) -> str:
    """Construct the ARN for a Bedrock AgentCore Gateway."""
    return f"arn:aws:bedrock-agentcore:{region}:{account_id}:gateway/{gateway_id}"


__all__ = [
    "SSM_PARAMETER_NAME",
    "SharedConfig",
    "build_gateway_arn",
    "delete_from_ssm",
    "get_account_id",
    "load_from_ssm",
    "save_to_ssm",
]
