"""Tests for KB-related CLI commands and SharedConfig extensions."""

import json
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from arch_sparring_agent.cli import cli
from arch_sparring_agent.infra import SharedConfig


class TestSharedConfigKBFields(unittest.TestCase):
    def test_default_kb_fields_are_none(self):
        config = SharedConfig(
            gateway_id="gw-1",
            gateway_arn="arn:gw",
            policy_engine_id="pe-1",
            region="eu-central-1",
        )
        self.assertIsNone(config.knowledge_base_id)
        self.assertIsNone(config.kb_bucket_name)

    def test_roundtrip_with_kb_fields(self):
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
        self.assertEqual(restored.knowledge_base_id, "kb-abc")
        self.assertEqual(restored.kb_bucket_name, "arch-review-kb-123-eu-central-1")

    def test_from_json_tolerates_missing_kb_fields(self):
        """Configs saved before KB support should still deserialize."""
        old_json = json.dumps(
            {
                "gateway_id": "gw-1",
                "gateway_arn": "arn:gw",
                "policy_engine_id": "pe-1",
                "region": "eu-central-1",
            }
        )
        config = SharedConfig.from_json(old_json)
        self.assertIsNone(config.knowledge_base_id)
        self.assertIsNone(config.kb_bucket_name)

    def test_from_json_ignores_unknown_fields(self):
        """Future fields in SSM shouldn't break old clients."""
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
        config = SharedConfig.from_json(future_json)
        self.assertEqual(config.knowledge_base_id, "kb-x")


class TestDeployWithKBHelp(unittest.TestCase):
    def test_deploy_help_shows_with_kb_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["deploy", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--with-kb", result.output)


class TestKBSyncHelp(unittest.TestCase):
    def test_kb_sync_help_shows_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["kb", "sync", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--content-dir", result.output)
        self.assertIn("--region", result.output)

    def test_kb_group_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["kb", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("sync", result.output)


class TestKBSyncRequiresKB(unittest.TestCase):
    @patch("arch_sparring_agent.cli.load_from_ssm")  # imported in cli/__init__.py
    def test_sync_fails_without_kb_in_config(self, mock_load):
        mock_load.return_value = SharedConfig(
            gateway_id="gw-1",
            gateway_arn="arn:gw",
            policy_engine_id="pe-1",
            region="eu-central-1",
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["kb", "sync"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No Knowledge Base", result.output)


class TestDeployWithKB(unittest.TestCase):
    @patch("arch_sparring_agent.cli.deploy.save_to_ssm")
    @patch("arch_sparring_agent.cli.deploy._deploy_infra")
    def test_deploy_without_kb_leaves_fields_none(self, mock_deploy_infra, mock_save):
        mock_deploy_infra.return_value = ("arn:gw", "gw-1", "pe-1")
        runner = CliRunner()
        result = runner.invoke(cli, ["deploy", "--region", "eu-central-1"])
        self.assertEqual(result.exit_code, 0)

        saved_config = mock_save.call_args[0][0]
        self.assertIsNone(saved_config.knowledge_base_id)
        self.assertIsNone(saved_config.kb_bucket_name)

    @patch("arch_sparring_agent.cli.deploy.save_to_ssm")
    @patch("arch_sparring_agent.cli.deploy._deploy_infra")
    def test_deploy_with_kb_populates_fields(self, mock_deploy_infra, mock_save):
        mock_deploy_infra.return_value = ("arn:gw", "gw-1", "pe-1")

        with patch("arch_sparring_agent.kb.infra.setup_knowledge_base") as mock_kb_setup:
            mock_kb_setup.return_value = ("kb-abc", "my-bucket")
            runner = CliRunner()
            result = runner.invoke(cli, ["deploy", "--with-kb", "--region", "eu-central-1"])

        self.assertEqual(result.exit_code, 0)
        saved_config = mock_save.call_args[0][0]
        self.assertEqual(saved_config.knowledge_base_id, "kb-abc")
        self.assertEqual(saved_config.kb_bucket_name, "my-bucket")


class TestDestroyWithKB(unittest.TestCase):
    @patch("arch_sparring_agent.cli.deploy.delete_from_ssm")
    @patch("arch_sparring_agent.cli.deploy.destroy_gateway")
    @patch("arch_sparring_agent.cli.deploy.destroy_policy_engine")
    @patch("arch_sparring_agent.cli.load_from_ssm")
    def test_destroy_tears_down_kb_when_present(
        self, mock_load, mock_destroy_pe, mock_destroy_gw, mock_del_ssm
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

        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
