"""Tests for agents.collect_output and agents.safe_invoke."""

from unittest.mock import MagicMock

import pytest

from arch_sparring_agent.agents import collect_output, safe_invoke
from tests.conftest import FakeMaxTokensReached


def _make_agent(messages=None):
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.messages = messages or []
    return agent


# --- collect_output ---


def test_collect_output_extracts_assistant_text():
    agent = _make_agent(
        [
            {"role": "user", "content": [{"text": "user prompt"}]},
            {"role": "assistant", "content": [{"text": "first analysis"}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "1"}}]},
            {"role": "assistant", "content": [{"text": "second analysis"}]},
        ]
    )
    result = collect_output(agent)
    assert "first analysis" in result
    assert "second analysis" in result
    assert "user prompt" not in result


def test_collect_output_filters_sdk_recovery_messages():
    agent = _make_agent(
        [
            {"role": "assistant", "content": [{"text": "real findings"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "text": "The selected tool read_cfn's tool use was incomplete "
                        "due to maximum token limits being reached."
                    }
                ],
            },
        ]
    )
    result = collect_output(agent)
    assert "real findings" in result
    assert "maximum token limits" not in result


def test_collect_output_skips_non_text_blocks():
    agent = _make_agent(
        [
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"name": "foo", "input": {}, "toolUseId": "1"}},
                    {"text": "kept text"},
                ],
            },
        ]
    )
    result = collect_output(agent)
    assert result == "kept text"


def test_collect_output_returns_empty_for_no_assistant_text():
    agent = _make_agent(
        [
            {"role": "user", "content": [{"text": "only user messages"}]},
        ]
    )
    assert collect_output(agent) == ""


def test_collect_output_handles_empty_history():
    agent = _make_agent([])
    assert collect_output(agent) == ""


# --- safe_invoke ---


def test_safe_invoke_returns_normal_result():
    agent = _make_agent()
    agent.return_value = "clean output"
    result = safe_invoke(agent, "test prompt")
    assert result == "clean output"
    agent.assert_called_once_with("test prompt")


def test_safe_invoke_recovers_on_max_tokens():
    agent = _make_agent(
        [
            {"role": "assistant", "content": [{"text": "partial findings"}]},
        ]
    )
    agent.side_effect = FakeMaxTokensReached("max tokens")
    result = safe_invoke(agent, "test prompt")
    assert result == "partial findings"


def test_safe_invoke_reraises_when_no_output():
    agent = _make_agent([])
    agent.side_effect = FakeMaxTokensReached("max tokens")
    with pytest.raises(FakeMaxTokensReached):
        safe_invoke(agent, "test prompt")
