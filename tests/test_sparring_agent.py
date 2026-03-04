"""Tests for the sparring agent factory (agents/sparring_agent.py)."""

from unittest.mock import MagicMock, patch

from arch_sparring_agent.agents.sparring_agent import (
    ClassificationOutcome,
    SparringAgent,
    create_sparring_agent,
)

_MOD = "arch_sparring_agent.agents.sparring_agent"


def _passthrough_tool(fn):
    """A fake @tool that preserves the original function."""
    fn.tool_name = fn.__name__
    return fn


def _get_tool(tools, name):
    """Find a tool by __name__ from the Agent constructor's tools list."""
    for t in tools:
        if getattr(t, "__name__", None) == name:
            return t
    return None


def test_create_sparring_agent_returns_sparring_agent():
    model = MagicMock(name="model")
    sa = create_sparring_agent(model)
    assert isinstance(sa, SparringAgent)
    assert sa.agent is not None
    assert callable(sa.get_result)
    assert callable(sa.challenge_count)
    assert sa.get_result() is None
    assert sa.challenge_count() == 0


def test_classify_gap_tool_sets_result():
    with patch(f"{_MOD}.tool", _passthrough_tool):
        with patch(f"{_MOD}.Agent") as mock_agent_cls:
            model = MagicMock(name="model")
            sa = create_sparring_agent(model)
            tools = mock_agent_cls.call_args[1]["tools"]

    classify_fn = _get_tool(tools, "classify_gap")
    assert classify_fn is not None

    result_msg = classify_fn(classification="RESOLVED", reasoning="proved it")
    assert "RESOLVED" in result_msg

    outcome = sa.get_result()
    assert isinstance(outcome, ClassificationOutcome)
    assert outcome.classification == "RESOLVED"
    assert outcome.reasoning == "proved it"


def test_classify_gap_rejects_invalid_classification():
    with patch(f"{_MOD}.tool", _passthrough_tool):
        with patch(f"{_MOD}.Agent") as mock_agent_cls:
            model = MagicMock(name="model")
            sa = create_sparring_agent(model)
            tools = mock_agent_cls.call_args[1]["tools"]

    classify_fn = _get_tool(tools, "classify_gap")
    assert classify_fn is not None

    result_msg = classify_fn(classification="INVALID", reasoning="whatever")
    assert "Invalid" in result_msg
    assert sa.get_result() is None


def test_profile_directive_appended():
    with patch(f"{_MOD}.Agent") as mock_agent_cls:
        model = MagicMock(name="model")
        profile = {"directives": {"sparring": "Be extra strict"}}
        create_sparring_agent(model, profile=profile)
        system_prompt = mock_agent_cls.call_args[1]["system_prompt"]
    assert "Be extra strict" in system_prompt


def test_challenge_counter_offset():
    with patch(f"{_MOD}.tool", _passthrough_tool):
        with patch(f"{_MOD}.Agent") as mock_agent_cls:
            model = MagicMock(name="model")
            sa = create_sparring_agent(model, challenge_offset=5)
            tools = mock_agent_cls.call_args[1]["tools"]

    challenge_fn = _get_tool(tools, "challenge_user")
    assert challenge_fn is not None

    with patch("builtins.input", return_value="response"):
        with patch("builtins.print"):
            challenge_fn(challenge="test challenge")

    assert sa.challenge_count() == 1
