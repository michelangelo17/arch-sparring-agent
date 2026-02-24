from __future__ import annotations

from unittest.mock import MagicMock, patch

from arch_sparring_agent.agents.question_agent import _SYSTEM_PROMPT, run_questions


def test_system_prompt_includes_output_constraint():
    assert "OUTPUT CONSTRAINT" in _SYSTEM_PROMPT
    assert "ONLY the gap" in _SYSTEM_PROMPT


@patch("arch_sparring_agent.agents.safe_invoke")
def test_run_questions_prompt_includes_output_constraint(mock_safe_invoke):
    mock_safe_invoke.return_value = "gap summary"
    agent = MagicMock()

    run_questions(agent, "### Features Not Found\n- No encryption")

    prompt = mock_safe_invoke.call_args[0][1]
    assert "ONLY the gap summary" in prompt
