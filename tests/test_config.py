"""Unit tests for arch_sparring_agent.config."""

import os
import unittest
from unittest.mock import MagicMock, patch

from arch_sparring_agent import config
from tests.conftest import FakeClientError


class TestIntEnv(unittest.TestCase):
    def test_valid_int(self):
        with patch.dict(os.environ, {"TEST_INT_VAR": "42"}, clear=False):
            self.assertEqual(config._int_env("TEST_INT_VAR", 10), 42)

    def test_invalid_int_logs_warning_and_returns_default(self):
        with patch.dict(os.environ, {"TEST_INT_VAR": "not_a_number"}, clear=False):
            with self.assertLogs(config.__name__, level="WARNING") as cm:
                result = config._int_env("TEST_INT_VAR", 10)
            self.assertEqual(result, 10)
            self.assertTrue(any("Invalid integer" in m for m in cm.output))

    def test_missing_env_var_returns_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_MISSING_VAR", None)
            self.assertEqual(config._int_env("TEST_MISSING_VAR", 99), 99)


class TestCheckModelAccess(unittest.TestCase):
    def test_model_found(self):
        mock_bedrock = MagicMock()
        mock_bedrock.list_foundation_models.return_value = {
            "modelSummaries": [{"modelId": "amazon.nova-2-lite-v1:0"}],
        }
        with patch("arch_sparring_agent.config.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_bedrock
            self.assertTrue(config.check_model_access("amazon.nova-2-lite-v1:0", "eu-central-1"))

    def test_model_not_found(self):
        mock_bedrock = MagicMock()
        mock_bedrock.list_foundation_models.return_value = {
            "modelSummaries": [{"modelId": "other-model"}],
        }
        with patch("arch_sparring_agent.config.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_bedrock
            self.assertFalse(config.check_model_access("amazon.nova-2-lite-v1:0", "eu-central-1"))

    def test_client_error_returns_false(self):
        mock_bedrock = MagicMock()
        mock_bedrock.list_foundation_models.side_effect = FakeClientError()
        with patch("arch_sparring_agent.config.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_bedrock
            self.assertFalse(config.check_model_access("amazon.nova-2-lite-v1:0", "eu-central-1"))


class TestGetInferenceProfileArn(unittest.TestCase):
    def test_success(self):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        with patch("arch_sparring_agent.config.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_sts
            arn = config.get_inference_profile_arn("amazon.nova-2-lite-v1:0", "eu-central-1")
        expected = (
            "arn:aws:bedrock:eu-central-1:123456789012:inference-profile/"
            "global.amazon.nova-2-lite-v1:0"
        )
        self.assertEqual(arn, expected)

    def test_client_error_returns_none(self):
        with patch("arch_sparring_agent.config.boto3") as mock_boto3:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.side_effect = FakeClientError()
            mock_boto3.client.return_value = mock_sts
            result = config.get_inference_profile_arn("amazon.nova-2-lite-v1:0", "eu-central-1")
            self.assertIsNone(result)
