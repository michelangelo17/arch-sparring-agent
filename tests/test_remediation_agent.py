"""Tests for arch_sparring_agent.agents.remediation_agent."""

from unittest.mock import MagicMock, patch

from arch_sparring_agent.agents.remediation_agent import (
    _format_list,
    _format_recommendations,
    _number_to_prompt,
    create_remediation_agent,
    run_remediation,
)
from arch_sparring_agent.state import ReviewState


def _make_state(**overrides):
    """Create a ReviewState with sensible defaults."""
    defaults = {
        "timestamp": "2024-01-15T10:00:00",
        "project_name": "TestProject",
        "gaps": [
            {"id": "gap-1", "description": "Missing auth", "severity": "high"},
            {"id": "gap-2", "description": "No logging", "severity": "medium"},
        ],
        "risks": [
            {"id": "risk-1", "description": "DDoS vulnerability", "impact": "high"},
        ],
        "recommendations": ["Add WAF", "Enable CloudTrail"],
        "verdict": "PASS WITH CONCERNS",
        "requirements_summary": "REST API on ECS",
        "architecture_summary": "ECS + ALB + RDS",
    }
    defaults.update(overrides)
    return ReviewState(**defaults)


def test_format_list_with_items():
    items = [
        {"id": "gap-1", "description": "Missing auth", "severity": "high"},
        {"id": "gap-2", "description": "No logging", "severity": "medium"},
    ]
    result = _format_list(items, "severity")

    assert "[HIGH]" in result
    assert "[MEDIUM]" in result
    assert "Missing auth" in result
    assert "No logging" in result


def test_format_list_empty():
    result = _format_list([], "severity")
    assert "None identified" in result


def test_format_recommendations_with_items():
    result = _format_recommendations(["Add WAF", "Enable CloudTrail"])
    assert "Add WAF" in result
    assert "Enable CloudTrail" in result


def test_format_recommendations_empty():
    result = _format_recommendations([])
    assert "None provided" in result


def test_number_to_prompt_gap():
    state = _make_state()
    prompt = _number_to_prompt(1, state)
    assert prompt is not None
    assert "gap #1" in prompt.lower()
    assert "Missing auth" in prompt


def test_number_to_prompt_risk():
    state = _make_state()
    prompt = _number_to_prompt(3, state)
    assert prompt is not None
    assert "risk #1" in prompt.lower()
    assert "DDoS vulnerability" in prompt


def test_number_to_prompt_out_of_range():
    state = _make_state()
    prompt = _number_to_prompt(99, state)
    assert prompt is None


@patch("arch_sparring_agent.agents.remediation_agent.setup_agentcore_memory")
@patch("arch_sparring_agent.agents.remediation_agent.Agent")
def test_create_remediation_agent_builds_agent(mock_agent_cls, mock_setup_mem):
    mock_setup_mem.return_value = (None, None)
    state = _make_state()

    create_remediation_agent(state=state, model="test-model", region="us-east-1")

    mock_agent_cls.assert_called_once()
    call_kwargs = mock_agent_cls.call_args[1]
    assert "RemediationAgent" == call_kwargs["name"]
    assert "Missing auth" in call_kwargs["system_prompt"]
    assert "DDoS vulnerability" in call_kwargs["system_prompt"]
    assert "Add WAF" in call_kwargs["system_prompt"]
    assert "TestProject" in call_kwargs["system_prompt"]


@patch("arch_sparring_agent.agents.remediation_agent.setup_agentcore_memory")
@patch("arch_sparring_agent.agents.remediation_agent.Agent")
def test_create_remediation_agent_with_memory(mock_agent_cls, mock_setup_mem):
    mock_config = MagicMock()
    mock_setup_mem.return_value = (mock_config, "mem-123")
    state = _make_state()

    create_remediation_agent(state=state, model="test-model", region="us-east-1")

    call_kwargs = mock_agent_cls.call_args[1]
    assert call_kwargs["session_manager"] is not None


@patch("arch_sparring_agent.agents.remediation_agent.setup_agentcore_memory")
@patch("arch_sparring_agent.agents.remediation_agent.Agent")
def test_run_remediation_displays_header(mock_agent_cls, mock_setup_mem):
    mock_setup_mem.return_value = (None, None)
    state = _make_state()

    mock_agent = MagicMock()
    mock_agent.return_value = "Agent response"
    mock_agent.__call__ = MagicMock(return_value="Session summary")
    mock_agent_cls.return_value = mock_agent

    output_lines = []

    with patch("builtins.input", side_effect=["exit"]):
        run_remediation(mock_agent, state, output_fn=output_lines.append)

    assert any("REMEDIATION MODE" in line for line in output_lines)
    assert any("2 gaps" in line for line in output_lines)
    assert any("1 risks" in line for line in output_lines)


@patch("arch_sparring_agent.agents.remediation_agent.setup_agentcore_memory")
@patch("arch_sparring_agent.agents.remediation_agent.Agent")
def test_run_remediation_processes_user_input(mock_agent_cls, mock_setup_mem):
    mock_setup_mem.return_value = (None, None)
    state = _make_state()

    mock_agent = MagicMock()
    mock_agent.__call__ = MagicMock(side_effect=["Good question!", "Summary of session"])

    output_lines = []

    with patch("builtins.input", side_effect=["How do I fix gap 1?", "exit"]):
        notes = run_remediation(mock_agent, state, output_fn=output_lines.append)

    assert "User: How do I fix gap 1?" in notes
