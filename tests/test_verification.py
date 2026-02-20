"""Tests for the service-defaults verification step."""

from unittest.mock import MagicMock, patch

import pytest

from arch_sparring_agent.orchestrator import ReviewOrchestrator


def _make_orchestrator():
    return ReviewOrchestrator(
        requirements_agent=MagicMock(),
        architecture_agent=MagicMock(),
        question_agent=MagicMock(),
        sparring_agent=MagicMock(),
        review_agent=MagicMock(),
        standard_model=MagicMock(name="standard_model"),
    )


def test_passthrough_when_no_features_not_found():
    orch = _make_orchestrator()
    findings = "### Components\n- DynamoDB\n\n### Features Verified\n- Encryption: via default"
    result = orch._verify_against_defaults(findings)
    assert result == findings


@patch("arch_sparring_agent.orchestrator.Agent")
def test_calls_verifier_when_features_not_found_present(mock_agent_cls):
    orch = _make_orchestrator()
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent
    mock_agent.return_value = "verified findings"

    findings = "### Features Verified\n- X\n\n### Features Not Found\n- Encryption at rest"
    result = orch._verify_against_defaults(findings)

    mock_agent_cls.assert_called_once()
    assert result == "verified findings"


@patch("arch_sparring_agent.orchestrator.Agent")
def test_returns_original_on_verifier_failure(mock_agent_cls):
    orch = _make_orchestrator()
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent
    mock_agent.side_effect = RuntimeError("model error")

    findings = "### Features Not Found\n- Something"
    result = orch._verify_against_defaults(findings)

    assert result == findings


@pytest.fixture
def verification_integration_mocks():
    patch("arch_sparring_agent.orchestrator.extract_requirements").start().side_effect = (
        lambda c, m: c
    )

    mock_extract_arch = patch(
        "arch_sparring_agent.orchestrator.extract_architecture_findings"
    ).start()
    mock_extract_arch.side_effect = lambda c, m: f"### Features Not Found\n- item from {c}"

    patch("arch_sparring_agent.orchestrator.extract_phase_findings").start().side_effect = (
        lambda c, p, m: c
    )

    patch("arch_sparring_agent.orchestrator.run_questions").start().return_value = "q"
    patch("arch_sparring_agent.orchestrator.run_sparring").start().return_value = "s"
    patch("arch_sparring_agent.orchestrator.run_review").start().return_value = "review"

    try:
        yield
    finally:
        patch.stopall()


@patch.object(ReviewOrchestrator, "_verify_against_defaults")
def test_verify_called_during_run_review(mock_verify, verification_integration_mocks):
    mock_verify.side_effect = lambda x: x.replace("Not Found", "Verified (via service default)")

    mock_req = MagicMock()
    mock_req.return_value = "req summary"
    mock_arch = MagicMock()
    mock_arch.return_value = "arch summary"

    orch = ReviewOrchestrator(
        requirements_agent=mock_req,
        architecture_agent=mock_arch,
        question_agent=MagicMock(),
        sparring_agent=MagicMock(),
        review_agent=MagicMock(),
        standard_model=MagicMock(name="standard_model"),
    )
    orch.run_review()

    mock_verify.assert_called_once()
    call_arg = mock_verify.call_args[0][0]
    assert "Features Not Found" in call_arg
