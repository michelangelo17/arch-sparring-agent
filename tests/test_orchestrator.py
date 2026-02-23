from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from arch_sparring_agent.infra import SharedConfig
from arch_sparring_agent.review.grounding import GroundingResult
from arch_sparring_agent.review.orchestrator import ReviewOrchestrator

_ORCH = "arch_sparring_agent.review.orchestrator"

FAKE_CONFIG = SharedConfig(
    gateway_id="gw-123",
    gateway_arn="arn:aws:bedrock-agentcore:eu-central-1:111111111111:gateway/gw-123",
    policy_engine_id="pe-456",
    region="eu-central-1",
)

FAKE_CONFIG_WITH_GUARDRAILS = SharedConfig(
    gateway_id="gw-123",
    gateway_arn="arn:aws:bedrock-agentcore:eu-central-1:111111111111:gateway/gw-123",
    policy_engine_id="pe-456",
    region="eu-central-1",
    guardrail_id="gr-test123",
    guardrail_version="DRAFT",
)


@pytest.fixture
def create_mocks():
    mock_reasoning_model = MagicMock(name="reasoning_model")
    mock_standard_model = MagicMock(name="standard_model")

    def fake_create_model(model_name, reasoning=False, reasoning_level="low"):
        return mock_reasoning_model if reasoning else mock_standard_model

    mock_create_model = patch(f"{_ORCH}.create_model", side_effect=fake_create_model).start()

    mock_create_req = patch(f"{_ORCH}.create_requirements_agent").start()
    mock_create_arch = patch(f"{_ORCH}.create_architecture_agent").start()
    mock_create_quest = patch(f"{_ORCH}.create_question_agent").start()
    mock_create_spar = patch(f"{_ORCH}.create_sparring_agent").start()
    mock_create_rev = patch(f"{_ORCH}.create_review_agent").start()
    mock_create_guardrails = patch(f"{_ORCH}.create_guardrails_checker").start()
    mock_create_guardrails.return_value = None

    try:
        yield SimpleNamespace(
            mock_create_model=mock_create_model,
            mock_create_req=mock_create_req,
            mock_create_arch=mock_create_arch,
            mock_create_quest=mock_create_quest,
            mock_create_spar=mock_create_spar,
            mock_create_rev=mock_create_rev,
            mock_standard_model=mock_standard_model,
            mock_create_guardrails=mock_create_guardrails,
        )
    finally:
        patch.stopall()


def test_create_builds_all_agents(create_mocks):
    orch = ReviewOrchestrator.create(
        documents_dir="docs",
        templates_dir="tmpl",
        diagrams_dir="diag",
        shared_config=FAKE_CONFIG,
    )

    create_mocks.mock_create_req.assert_called()
    create_mocks.mock_create_arch.assert_called()
    create_mocks.mock_create_quest.assert_called()
    create_mocks.mock_create_spar.assert_called()
    create_mocks.mock_create_rev.assert_called()

    assert orch.standard_model is not None
    assert orch._has_kb is False


def test_create_sets_has_kb_when_knowledge_base_configured(create_mocks):
    config_with_kb = SharedConfig(
        gateway_id="gw-123",
        gateway_arn="arn:aws:bedrock-agentcore:eu-central-1:111111111111:gateway/gw-123",
        policy_engine_id="pe-456",
        region="eu-central-1",
        knowledge_base_id="kb-789",
    )
    orch = ReviewOrchestrator.create(
        documents_dir="docs",
        templates_dir="tmpl",
        diagrams_dir="diag",
        shared_config=config_with_kb,
    )

    assert orch._has_kb is True


def test_reasoning_level_off_uses_standard_model_for_all(create_mocks):
    ReviewOrchestrator.create(
        documents_dir="docs",
        templates_dir="tmpl",
        diagrams_dir="diag",
        shared_config=FAKE_CONFIG,
        reasoning_level="off",
    )

    calls = create_mocks.mock_create_model.call_args_list
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert not kwargs.get("reasoning", False)


def test_model_name_passed_to_create_model(create_mocks):
    ReviewOrchestrator.create(
        documents_dir="docs",
        templates_dir="tmpl",
        diagrams_dir="diag",
        shared_config=FAKE_CONFIG,
        model_name="opus-4.6",
    )

    first_call = create_mocks.mock_create_model.call_args_list[0]
    assert first_call[0][0] == "opus-4.6"


@pytest.fixture
def run_review_mocks():
    mock_standard_model = MagicMock(name="standard_model")

    mock_req_agent = MagicMock()
    mock_req_agent.return_value = "Requirements Summary"

    mock_arch_agent = MagicMock()
    mock_quest_agent = MagicMock()
    mock_spar_agent = MagicMock()
    mock_rev_agent = MagicMock()

    mock_extract_req = patch(f"{_ORCH}.extract_requirements").start()
    mock_extract_req.side_effect = lambda content, model: f"[extracted] {content}"

    mock_extract_arch = patch(f"{_ORCH}.extract_architecture_findings").start()
    mock_extract_arch.side_effect = lambda content, model: f"[extracted] {content}"

    mock_extract_phase = patch(f"{_ORCH}.extract_phase_findings").start()
    mock_extract_phase.side_effect = lambda content, phase, model: f"[extracted:{phase}] {content}"

    mock_run_arch = patch(f"{_ORCH}.run_architecture").start()
    mock_run_arch.return_value = "Architecture Summary"

    mock_run_questions = patch(f"{_ORCH}.run_questions").start()
    mock_run_sparring = patch(f"{_ORCH}.run_sparring").start()
    mock_gen_review = patch(f"{_ORCH}.run_review").start()

    try:
        yield SimpleNamespace(
            mock_standard_model=mock_standard_model,
            mock_req_agent=mock_req_agent,
            mock_arch_agent=mock_arch_agent,
            mock_quest_agent=mock_quest_agent,
            mock_spar_agent=mock_spar_agent,
            mock_rev_agent=mock_rev_agent,
            mock_extract_req=mock_extract_req,
            mock_extract_arch=mock_extract_arch,
            mock_extract_phase=mock_extract_phase,
            mock_run_arch=mock_run_arch,
            mock_run_questions=mock_run_questions,
            mock_run_sparring=mock_run_sparring,
            mock_gen_review=mock_gen_review,
        )
    finally:
        patch.stopall()


def _make_orchestrator(run_review_mocks, **kwargs):
    defaults = {
        "requirements_agent": run_review_mocks.mock_req_agent,
        "architecture_agent": run_review_mocks.mock_arch_agent,
        "question_agent": run_review_mocks.mock_quest_agent,
        "sparring_agent": run_review_mocks.mock_spar_agent,
        "review_agent": run_review_mocks.mock_rev_agent,
        "standard_model": run_review_mocks.mock_standard_model,
    }
    defaults.update(kwargs)
    return ReviewOrchestrator(**defaults)


def test_run_review(run_review_mocks):
    orch = _make_orchestrator(run_review_mocks)

    run_review_mocks.mock_run_questions.return_value = "Questions Context"
    run_review_mocks.mock_run_sparring.return_value = "Sparring Context"
    run_review_mocks.mock_gen_review.return_value = "Final Review"

    result = orch.run_review()

    run_review_mocks.mock_req_agent.assert_called()
    run_review_mocks.mock_run_arch.assert_called_once_with(
        orch.architecture_agent,
        "[extracted] Requirements Summary",
        has_kb=False,
    )

    run_review_mocks.mock_extract_req.assert_called_once_with(
        "Requirements Summary", orch.standard_model
    )
    run_review_mocks.mock_extract_arch.assert_called_once_with(
        "Architecture Summary", orch.standard_model
    )

    run_review_mocks.mock_run_questions.assert_called_with(
        orch.question_agent,
        "[extracted] Architecture Summary",
    )

    run_review_mocks.mock_run_sparring.assert_called()
    sparring_call_args = run_review_mocks.mock_run_sparring.call_args
    assert sparring_call_args[0][1] == "[extracted] Architecture Summary"
    assert sparring_call_args[0][2] == "[extracted:Q&A] Questions Context"

    run_review_mocks.mock_gen_review.assert_called()

    assert result.review == "Final Review"
    assert result.requirements_summary == "Requirements Summary"
    assert result.requirements_findings == "[extracted] Requirements Summary"
    assert result.architecture_summary == "Architecture Summary"
    assert result.architecture_findings == "[extracted] Architecture Summary"
    assert result.qa_context == "Questions Context"
    assert result.qa_findings == "[extracted:Q&A] Questions Context"
    assert result.sparring_context == "Sparring Context"
    assert result.sparring_findings == "[extracted:Sparring] Sparring Context"


def test_run_review_passes_has_kb_flag(run_review_mocks):
    orch = _make_orchestrator(run_review_mocks, has_kb=True)

    run_review_mocks.mock_run_questions.return_value = "Q"
    run_review_mocks.mock_run_sparring.return_value = "S"
    run_review_mocks.mock_gen_review.return_value = "Review"

    orch.run_review()

    run_review_mocks.mock_run_arch.assert_called_once_with(
        orch.architecture_agent,
        "[extracted] Requirements Summary",
        has_kb=True,
    )


def test_output_fn_callback(run_review_mocks):
    captured = []
    orch = _make_orchestrator(run_review_mocks, output_fn=captured.append)

    run_review_mocks.mock_run_questions.return_value = "Q Context"
    run_review_mocks.mock_run_sparring.return_value = "S Context"
    run_review_mocks.mock_gen_review.return_value = "Final"

    orch.run_review()

    assert len(captured) > 0
    assert captured[0] == "=" * 60


def test_create_no_guardrails_by_default(create_mocks):
    orch = ReviewOrchestrator.create(
        documents_dir="docs",
        templates_dir="tmpl",
        diagrams_dir="diag",
        shared_config=FAKE_CONFIG,
    )

    create_mocks.mock_create_guardrails.assert_called_once_with(None, None, region="eu-central-1")
    assert orch._guardrails is None


def test_create_sets_guardrails_when_configured(create_mocks):
    mock_checker = MagicMock()
    mock_checker.guardrail_id = "gr-test123"
    create_mocks.mock_create_guardrails.return_value = mock_checker

    orch = ReviewOrchestrator.create(
        documents_dir="docs",
        templates_dir="tmpl",
        diagrams_dir="diag",
        shared_config=FAKE_CONFIG_WITH_GUARDRAILS,
    )

    create_mocks.mock_create_guardrails.assert_called_once_with(
        "gr-test123", "DRAFT", region="eu-central-1"
    )
    assert orch._guardrails is mock_checker


def test_run_review_calls_grounding_checks(run_review_mocks):
    mock_guardrails = MagicMock()
    mock_guardrails.check_grounding.return_value = GroundingResult(
        passed=True, grounding_score=0.9, action="NONE", details="ok"
    )
    orch = _make_orchestrator(run_review_mocks, guardrails=mock_guardrails)

    run_review_mocks.mock_run_questions.return_value = "Q Context"
    run_review_mocks.mock_run_sparring.return_value = "S Context"
    run_review_mocks.mock_gen_review.return_value = "Final Review"

    orch.run_review()

    assert mock_guardrails.check_grounding.call_count == 4

    call_args_list = mock_guardrails.check_grounding.call_args_list
    phases = [call[0][1] for call in call_args_list]
    assert any("Extract requirements" in q for q in phases)
    assert any("Extract architecture" in q for q in phases)
    assert any("Extract Q&A" in q for q in phases)
    assert any("Extract sparring" in q for q in phases)


def test_run_review_no_grounding_checks_without_guardrails(run_review_mocks):
    orch = _make_orchestrator(run_review_mocks)

    run_review_mocks.mock_run_questions.return_value = "Q"
    run_review_mocks.mock_run_sparring.return_value = "S"
    run_review_mocks.mock_gen_review.return_value = "R"

    result = orch.run_review()

    assert result.review == "R"
    assert orch._guardrails is None
