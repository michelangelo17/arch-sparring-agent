"""Unit tests for arch_sparring_agent.config."""

import os
import unittest
from unittest.mock import patch

from arch_sparring_agent import config


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


class TestSupportedModels(unittest.TestCase):
    """Tests for SUPPORTED_MODELS registry."""

    def test_default_model_exists(self):
        self.assertIn(config.DEFAULT_MODEL, config.SUPPORTED_MODELS)

    def test_all_models_have_required_fields(self):
        for name, model in config.SUPPORTED_MODELS.items():
            self.assertIn("model_id", model, f"{name} missing model_id")
            self.assertIn("description", model, f"{name} missing description")
            self.assertIn("reasoning_field", model, f"{name} missing reasoning_field")

    def test_nova_model_id(self):
        self.assertEqual(
            config.SUPPORTED_MODELS["nova-2-lite"]["model_id"],
            "eu.amazon.nova-2-lite-v1:0",
        )

    def test_opus_model_id(self):
        self.assertEqual(
            config.SUPPORTED_MODELS["opus-4.6"]["model_id"],
            "eu.anthropic.claude-opus-4-6-v1",
        )


class TestCreateModel(unittest.TestCase):
    """Tests for create_model() with curated model registry."""

    def test_nova_no_reasoning(self):
        model = config.create_model("nova-2-lite", reasoning=False)
        self.assertEqual(model.config["model_id"], "eu.amazon.nova-2-lite-v1:0")
        self.assertIsNone(model.config.get("additional_request_fields"))

    def test_nova_reasoning_low(self):
        model = config.create_model("nova-2-lite", reasoning=True, reasoning_level="low")
        expected = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}}
        self.assertEqual(model.config.get("additional_request_fields"), expected)

    def test_nova_reasoning_high(self):
        model = config.create_model("nova-2-lite", reasoning=True, reasoning_level="high")
        expected = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}}
        self.assertEqual(model.config.get("additional_request_fields"), expected)

    def test_opus_no_reasoning(self):
        model = config.create_model("opus-4.6", reasoning=False)
        self.assertEqual(model.config["model_id"], "eu.anthropic.claude-opus-4-6-v1")
        self.assertIsNone(model.config.get("additional_request_fields"))

    def test_opus_reasoning_medium(self):
        model = config.create_model("opus-4.6", reasoning=True, reasoning_level="medium")
        expected = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "medium"}}
        self.assertEqual(model.config.get("additional_request_fields"), expected)

    def test_reasoning_off_skips_config(self):
        model = config.create_model("nova-2-lite", reasoning=True, reasoning_level="off")
        self.assertIsNone(model.config.get("additional_request_fields"))

    def test_no_reasoning_flag_skips_config(self):
        model = config.create_model("nova-2-lite", reasoning=False)
        self.assertIsNone(model.config.get("additional_request_fields"))

    def test_invalid_level_falls_back_to_default(self):
        model = config.create_model("nova-2-lite", reasoning=True, reasoning_level="invalid")
        # Should fall back to DEFAULT_REASONING_LEVEL ("low")
        expected = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}}
        self.assertEqual(model.config.get("additional_request_fields"), expected)

    def test_unknown_model_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            config.create_model("nonexistent-model")
        self.assertIn("nonexistent-model", str(ctx.exception))
        self.assertIn("Supported models", str(ctx.exception))

    def test_default_model_name(self):
        model = config.create_model()
        self.assertEqual(model.config["model_id"], "eu.amazon.nova-2-lite-v1:0")
