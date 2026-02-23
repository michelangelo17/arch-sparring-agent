"""Tests for KB-related CLI commands and SharedConfig extensions."""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arch_sparring_agent.cli import cli
from arch_sparring_agent.infra import SharedConfig


def test_default_kb_fields_are_none():
    config = SharedConfig(
        gateway_id="gw-1",
        gateway_arn="arn:gw",
        policy_engine_id="pe-1",
        region="eu-central-1",
    )
    assert config.knowledge_base_id is None
    assert config.kb_bucket_name is None


def test_roundtrip_with_kb_fields():
    config = SharedConfig(
        gateway_id="gw-1",
        gateway_arn="arn:gw",
        policy_engine_id="pe-1",
        region="eu-central-1",
        knowledge_base_id="kb-abc",
        kb_bucket_name="arch-review-kb-123-eu-central-1",
    )
    raw = config.to_json()
    restored = SharedConfig.from_json(raw)
    assert restored.knowledge_base_id == "kb-abc"
    assert restored.kb_bucket_name == "arch-review-kb-123-eu-central-1"


def test_from_json_tolerates_missing_kb_fields():
    old_json = json.dumps(
        {
            "gateway_id": "gw-1",
            "gateway_arn": "arn:gw",
            "policy_engine_id": "pe-1",
            "region": "eu-central-1",
        }
    )
    config = SharedConfig.from_json(old_json)
    assert config.knowledge_base_id is None
    assert config.kb_bucket_name is None


def test_from_json_rejects_unknown_fields():
    future_json = json.dumps(
        {
            "gateway_id": "gw-1",
            "gateway_arn": "arn:gw",
            "policy_engine_id": "pe-1",
            "region": "eu-central-1",
            "knowledge_base_id": "kb-x",
            "kb_bucket_name": "bucket",
            "some_future_field": "value",
        }
    )
    with pytest.raises(TypeError):
        SharedConfig.from_json(future_json)


def test_deploy_help_shows_with_kb_option():
    runner = CliRunner()
    result = runner.invoke(cli, ["deploy", "--help"])
    assert result.exit_code == 0
    assert "--with-kb" in result.output


def test_kb_sync_help_shows_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["kb", "sync", "--help"])
    assert result.exit_code == 0
    assert "--content-dir" in result.output
    assert "--region" in result.output


def test_kb_group_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["kb", "--help"])
    assert result.exit_code == 0
    assert "sync" in result.output


@patch("arch_sparring_agent.cli.common.load_from_ssm")
def test_sync_fails_without_kb_in_config(mock_load):
    mock_load.return_value = SharedConfig(
        gateway_id="gw-1",
        gateway_arn="arn:gw",
        policy_engine_id="pe-1",
        region="eu-central-1",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["kb", "sync"])
    assert result.exit_code != 0
    assert "No Knowledge Base" in result.output


@patch("arch_sparring_agent.cli.deploy.save_to_ssm")
@patch("arch_sparring_agent.cli.deploy._deploy_infra")
def test_deploy_without_kb_leaves_fields_none(mock_deploy_infra, mock_save):
    mock_deploy_infra.return_value = ("arn:gw", "gw-1", "pe-1")
    runner = CliRunner()
    result = runner.invoke(cli, ["deploy", "--region", "eu-central-1"])
    assert result.exit_code == 0

    saved_config = mock_save.call_args[0][0]
    assert saved_config.knowledge_base_id is None
    assert saved_config.kb_bucket_name is None


@patch("arch_sparring_agent.cli.deploy.save_to_ssm")
@patch("arch_sparring_agent.cli.deploy._deploy_infra")
def test_deploy_with_kb_populates_fields(mock_deploy_infra, mock_save):
    mock_deploy_infra.return_value = ("arn:gw", "gw-1", "pe-1")

    with patch("arch_sparring_agent.kb.infra.setup_knowledge_base") as mock_kb_setup:
        mock_kb_setup.return_value = ("kb-abc", "my-bucket")
        runner = CliRunner()
        result = runner.invoke(cli, ["deploy", "--with-kb", "--region", "eu-central-1"])

    assert result.exit_code == 0
    saved_config = mock_save.call_args[0][0]
    assert saved_config.knowledge_base_id == "kb-abc"
    assert saved_config.kb_bucket_name == "my-bucket"


@patch("arch_sparring_agent.cli.deploy.delete_from_ssm")
@patch("arch_sparring_agent.cli.deploy.destroy_gateway")
@patch("arch_sparring_agent.cli.deploy.destroy_policy_engine")
@patch("arch_sparring_agent.cli.common.load_from_ssm")
def test_destroy_tears_down_kb_when_present(
    mock_load, mock_destroy_pe, mock_destroy_gw, mock_del_ssm
):
    mock_load.return_value = SharedConfig(
        gateway_id="gw-1",
        gateway_arn="arn:gw",
        policy_engine_id="pe-1",
        region="eu-central-1",
        knowledge_base_id="kb-abc",
        kb_bucket_name="my-bucket",
    )

    with patch("arch_sparring_agent.kb.infra.destroy_knowledge_base") as mock_kb_destroy:
        runner = CliRunner()
        result = runner.invoke(cli, ["destroy", "--confirm", "--region", "eu-central-1"])
        mock_kb_destroy.assert_called_once_with("kb-abc", "my-bucket", region="eu-central-1")

    assert result.exit_code == 0
