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


class TestDetectModelFamily(unittest.TestCase):
    """Tests for _detect_model_family()."""

    def test_nova_base_id(self):
        self.assertEqual(config._detect_model_family("amazon.nova-2-lite-v1:0"), "nova")

    def test_nova_region_prefixed(self):
        self.assertEqual(config._detect_model_family("eu.amazon.nova-2-lite-v1:0"), "nova")

    def test_nova_full_arn(self):
        arn = (
            "arn:aws:bedrock:eu-central-1:123456789012:"
            "inference-profile/global.amazon.nova-2-lite-v1:0"
        )
        self.assertEqual(config._detect_model_family(arn), "nova")

    def test_claude_region_prefixed(self):
        self.assertEqual(
            config._detect_model_family("eu.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "claude",
        )

    def test_claude_base_id(self):
        self.assertEqual(
            config._detect_model_family("anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "claude",
        )

    def test_claude_global_prefix(self):
        self.assertEqual(
            config._detect_model_family("global.anthropic.claude-opus-4-5-20251101-v1:0"),
            "claude",
        )

    def test_mistral_region_prefixed(self):
        self.assertEqual(
            config._detect_model_family("eu.mistral.pixtral-large-2502-v1:0"),
            "mistral",
        )

    def test_mistral_base_id(self):
        self.assertEqual(
            config._detect_model_family("mistral.pixtral-large-2502-v1:0"),
            "mistral",
        )

    def test_unknown_model(self):
        self.assertEqual(config._detect_model_family("somevendor.some-model-v1:0"), "unknown")

    def test_case_insensitive(self):
        self.assertEqual(config._detect_model_family("AMAZON.NOVA-2-lite-v1:0"), "nova")


class TestCreateModel(unittest.TestCase):
    """Tests for create_model() with family-aware reasoning config."""

    def test_nova_reasoning_low(self):
        model = config.create_model(
            "amazon.nova-2-lite-v1:0", reasoning=True, reasoning_level="low"
        )
        expected = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}}
        self.assertEqual(model.config.get("additional_request_fields"), expected)

    def test_nova_reasoning_high(self):
        model = config.create_model(
            "amazon.nova-2-lite-v1:0", reasoning=True, reasoning_level="high"
        )
        expected = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}}
        self.assertEqual(model.config.get("additional_request_fields"), expected)

    def test_claude_reasoning_low(self):
        model = config.create_model(
            "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
            reasoning=True,
            reasoning_level="low",
        )
        expected = {"reasoningConfig": {"type": "enabled", "budgetTokens": 1024}}
        self.assertEqual(model.config.get("additional_request_fields"), expected)

    def test_claude_reasoning_medium(self):
        model = config.create_model(
            "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
            reasoning=True,
            reasoning_level="medium",
        )
        expected = {"reasoningConfig": {"type": "enabled", "budgetTokens": 4096}}
        self.assertEqual(model.config.get("additional_request_fields"), expected)

    def test_claude_reasoning_high(self):
        model = config.create_model(
            "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
            reasoning=True,
            reasoning_level="high",
        )
        expected = {"reasoningConfig": {"type": "enabled", "budgetTokens": 16384}}
        self.assertEqual(model.config.get("additional_request_fields"), expected)

    def test_mistral_reasoning_logs_warning(self):
        with self.assertLogs(config.__name__, level="WARNING") as cm:
            model = config.create_model(
                "eu.mistral.pixtral-large-2502-v1:0",
                reasoning=True,
                reasoning_level="medium",
            )
        self.assertIsNone(model.config.get("additional_request_fields"))
        self.assertTrue(any("does not support extended thinking" in m for m in cm.output))

    def test_reasoning_off_skips_config(self):
        model = config.create_model(
            "amazon.nova-2-lite-v1:0", reasoning=True, reasoning_level="off"
        )
        self.assertIsNone(model.config.get("additional_request_fields"))

    def test_no_reasoning_flag_skips_config(self):
        model = config.create_model("amazon.nova-2-lite-v1:0", reasoning=False)
        self.assertIsNone(model.config.get("additional_request_fields"))

    def test_invalid_level_falls_back_to_default(self):
        model = config.create_model(
            "amazon.nova-2-lite-v1:0", reasoning=True, reasoning_level="invalid"
        )
        # Should fall back to DEFAULT_REASONING_LEVEL ("low")
        expected = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}}
        self.assertEqual(model.config.get("additional_request_fields"), expected)


class TestIsInferenceProfile(unittest.TestCase):
    """Tests for _is_inference_profile()."""

    def test_eu_prefix(self):
        self.assertTrue(config._is_inference_profile("eu.anthropic.claude-sonnet-4-5-v1:0"))

    def test_us_prefix(self):
        self.assertTrue(config._is_inference_profile("us.amazon.nova-2-lite-v1:0"))

    def test_global_prefix(self):
        self.assertTrue(config._is_inference_profile("global.anthropic.claude-opus-4-5-v1:0"))

    def test_ap_prefix(self):
        self.assertTrue(config._is_inference_profile("ap-northeast.amazon.nova-2-lite-v1:0"))

    def test_base_id(self):
        self.assertFalse(config._is_inference_profile("amazon.nova-2-lite-v1:0"))


class TestStripProfilePrefix(unittest.TestCase):
    """Tests for _strip_profile_prefix()."""

    def test_eu_prefix(self):
        self.assertEqual(
            config._strip_profile_prefix("eu.anthropic.claude-sonnet-4-5-v1:0"),
            "anthropic.claude-sonnet-4-5-v1:0",
        )

    def test_global_prefix(self):
        self.assertEqual(
            config._strip_profile_prefix("global.amazon.nova-2-lite-v1:0"),
            "amazon.nova-2-lite-v1:0",
        )

    def test_base_id_unchanged(self):
        self.assertEqual(
            config._strip_profile_prefix("amazon.nova-2-lite-v1:0"),
            "amazon.nova-2-lite-v1:0",
        )


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

    def test_region_prefixed_model_found(self):
        """Region-prefixed model should be matched after stripping prefix."""
        mock_bedrock = MagicMock()
        mock_bedrock.list_foundation_models.return_value = {
            "modelSummaries": [{"modelId": "anthropic.claude-sonnet-4-5-20250929-v1:0"}],
        }
        with patch("arch_sparring_agent.config.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_bedrock
            self.assertTrue(
                config.check_model_access(
                    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0", "eu-central-1"
                )
            )

    def test_prefix_match_without_version(self):
        """Prefix matching handles version suffix differences."""
        mock_bedrock = MagicMock()
        mock_bedrock.list_foundation_models.return_value = {
            "modelSummaries": [{"modelId": "anthropic.claude-sonnet-4-5-20250929-v1:0"}],
        }
        with patch("arch_sparring_agent.config.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_bedrock
            self.assertTrue(
                config.check_model_access(
                    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0", "eu-central-1"
                )
            )


class TestGetInferenceProfileArn(unittest.TestCase):
    def test_success_base_model(self):
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

    def test_region_prefixed_returns_as_is(self):
        """Region-prefixed inference profiles should be returned as-is."""
        result = config.get_inference_profile_arn(
            "eu.anthropic.claude-sonnet-4-5-20250929-v1:0", "eu-central-1"
        )
        self.assertEqual(result, "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

    def test_global_prefixed_returns_as_is(self):
        result = config.get_inference_profile_arn(
            "global.anthropic.claude-opus-4-5-20251101-v1:0", "eu-central-1"
        )
        self.assertEqual(result, "global.anthropic.claude-opus-4-5-20251101-v1:0")

    def test_client_error_returns_none(self):
        with patch("arch_sparring_agent.config.boto3") as mock_boto3:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.side_effect = FakeClientError()
            mock_boto3.client.return_value = mock_sts
            result = config.get_inference_profile_arn("amazon.nova-2-lite-v1:0", "eu-central-1")
            self.assertIsNone(result)
