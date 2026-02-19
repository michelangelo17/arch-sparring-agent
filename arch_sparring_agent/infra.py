"""Shared infrastructure config — SSM-based discovery for deploy-once, use-many."""

import json
import logging
from dataclasses import asdict, dataclass

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .config import DEFAULT_REGION
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

SSM_PARAMETER_NAME = "/arch-review/config"


@dataclass
class SharedConfig:
    """Shared infrastructure identifiers written by `deploy`, read by `run`."""

    gateway_id: str
    gateway_arn: str
    policy_engine_id: str
    region: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "SharedConfig":
        return cls(**json.loads(raw))


def save_to_ssm(config: SharedConfig) -> None:
    """Write shared config to SSM Parameter Store."""
    client = boto3.client("ssm", region_name=config.region)
    client.put_parameter(
        Name=SSM_PARAMETER_NAME,
        Description="arch-review shared infrastructure config",
        Value=config.to_json(),
        Type="SecureString",
        Overwrite=True,
    )
    logger.info("Saved config to SSM: %s", SSM_PARAMETER_NAME)


def load_from_ssm(region: str = DEFAULT_REGION) -> SharedConfig:
    """Read shared config from SSM Parameter Store.

    Raises:
        ConfigurationError: If the parameter does not exist (deploy not run yet).
    """
    try:
        client = boto3.client("ssm", region_name=region)
        response = client.get_parameter(Name=SSM_PARAMETER_NAME, WithDecryption=True)
        raw = response["Parameter"]["Value"]
        config = SharedConfig.from_json(raw)
        logger.debug("Loaded config from SSM: %s", SSM_PARAMETER_NAME)
        return config
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ParameterNotFound":
            raise ConfigurationError(
                "No infrastructure found. Run `arch-review deploy` first."
            ) from e
        raise ConfigurationError(f"Could not read SSM parameter: {e}") from e
    except (BotoCoreError, json.JSONDecodeError, TypeError, KeyError) as e:
        raise ConfigurationError(f"Could not load shared config: {e}") from e


def delete_from_ssm(region: str = DEFAULT_REGION) -> None:
    """Delete the shared config SSM parameter."""
    try:
        client = boto3.client("ssm", region_name=region)
        client.delete_parameter(Name=SSM_PARAMETER_NAME)
        logger.info("Deleted SSM parameter: %s", SSM_PARAMETER_NAME)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ParameterNotFound":
            logger.info("SSM parameter already absent: %s", SSM_PARAMETER_NAME)
        else:
            raise
