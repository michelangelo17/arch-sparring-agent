"""Unit tests for arch_sparring_agent.config."""

import os
from unittest.mock import patch

import pytest

from arch_sparring_agent import config


def test_valid_int():
    with patch.dict(os.environ, {"TEST_INT_VAR": "42"}, clear=False):
        assert config._int_env("TEST_INT_VAR", 10) == 42


def test_invalid_int_logs_warning_and_returns_default(caplog):
    with patch.dict(os.environ, {"TEST_INT_VAR": "not_a_number"}, clear=False):
        with caplog.at_level("WARNING", logger=config.__name__):
            result = config._int_env("TEST_INT_VAR", 10)
        assert result == 10
        assert any("Invalid integer" in m for m in caplog.messages)


def test_missing_env_var_returns_default():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TEST_MISSING_VAR", None)
        assert config._int_env("TEST_MISSING_VAR", 99) == 99


def test_default_model_exists():
    assert config.DEFAULT_MODEL in config.SUPPORTED_MODELS


def test_all_models_have_required_fields():
    for name, model in config.SUPPORTED_MODELS.items():
        assert "model_id" in model, f"{name} missing model_id"
        assert "description" in model, f"{name} missing description"
        assert "reasoning_type" in model, f"{name} missing reasoning_type"


def test_nova_model_id():
    assert (
        config.SUPPORTED_MODELS["nova-2-lite"]["model_id"]
        == "eu.amazon.nova-2-lite-v1:0"
    )


def test_opus_model_id():
    assert (
        config.SUPPORTED_MODELS["opus-4.6"]["model_id"]
        == "eu.anthropic.claude-opus-4-6-v1"
    )


def test_nova_no_reasoning():
    model = config.create_model("nova-2-lite", reasoning=False)
    assert model.config["model_id"] == "eu.amazon.nova-2-lite-v1:0"
    assert model.config.get("additional_request_fields") is None


def test_nova_reasoning_low():
    model = config.create_model("nova-2-lite", reasoning=True, reasoning_level="low")
    expected = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}}
    assert model.config.get("additional_request_fields") == expected


def test_nova_reasoning_high():
    model = config.create_model("nova-2-lite", reasoning=True, reasoning_level="high")
    expected = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}}
    assert model.config.get("additional_request_fields") == expected


def test_opus_no_reasoning():
    model = config.create_model("opus-4.6", reasoning=False)
    assert model.config["model_id"] == "eu.anthropic.claude-opus-4-6-v1"
    assert model.config.get("additional_request_fields") is None


def test_opus_reasoning_medium():
    model = config.create_model("opus-4.6", reasoning=True, reasoning_level="medium")
    expected = {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
    }
    assert model.config.get("additional_request_fields") == expected


def test_opus_reasoning_high():
    model = config.create_model("opus-4.6", reasoning=True, reasoning_level="high")
    expected = {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    assert model.config.get("additional_request_fields") == expected


def test_reasoning_off_skips_config():
    model = config.create_model("nova-2-lite", reasoning=True, reasoning_level="off")
    assert model.config.get("additional_request_fields") is None


def test_no_reasoning_flag_skips_config():
    model = config.create_model("nova-2-lite", reasoning=False)
    assert model.config.get("additional_request_fields") is None


def test_invalid_level_falls_back_to_default():
    model = config.create_model("nova-2-lite", reasoning=True, reasoning_level="invalid")
    expected = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}}
    assert model.config.get("additional_request_fields") == expected


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError) as ctx:
        config.create_model("nonexistent-model")
    assert "nonexistent-model" in str(ctx.value)
    assert "Supported models" in str(ctx.value)


def test_default_model_name():
    model = config.create_model()
    assert model.config["model_id"] == "eu.amazon.nova-2-lite-v1:0"
