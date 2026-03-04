"""Tests for the per-gap sparring orchestration (review/sparring.py)."""

from unittest.mock import MagicMock, patch

from arch_sparring_agent.agents.sparring_agent import GapResult, SparringAgent
from arch_sparring_agent.review.sparring import (
    SparringGap,
    _assemble_summary,
    _build_running_context,
    _max_rounds_for,
    _parse_triage_response,
    _spar_single_gap,
    _triage_gaps,
    run_sparring,
)

_MOD = "arch_sparring_agent.review.sparring"


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------


def test_triage_gaps_assigns_severity():
    model = MagicMock(name="model")
    triage_agent = MagicMock()
    triage_agent.return_value = "1. HIGH: Encryption at rest\n2. LOW: Missing docs\n"

    with patch(f"{_MOD}.Agent", return_value=triage_agent):
        gaps = _triage_gaps(["Encryption at rest", "Missing docs"], model)

    assert len(gaps) == 2
    assert gaps[0].severity == "HIGH"
    assert gaps[0].description == "Encryption at rest"
    assert gaps[1].severity == "LOW"


def test_triage_gaps_fallback_on_error():
    model = MagicMock(name="model")

    with patch(f"{_MOD}.Agent", side_effect=Exception("boom")):
        with patch(f"{_MOD}.MODEL_ERRORS", (Exception,)):
            gaps = _triage_gaps(["Gap A", "Gap B"], model)

    assert all(g.severity == "MEDIUM" for g in gaps)


def test_parse_triage_response_fuzzy_match():
    raw_gaps = ["Encryption at rest for DynamoDB", "Missing CloudWatch alarms"]
    response = "1. HIGH: Encryption at rest for Dyna\n2. LOW: Missing CloudWatch ala"
    result = _parse_triage_response(response, raw_gaps)
    assert result["Encryption at rest for DynamoDB"] == "HIGH"
    assert result["Missing CloudWatch alarms"] == "LOW"


def test_parse_triage_response_unmatched_defaults_medium():
    raw_gaps = ["Some unique gap"]
    response = "1. HIGH: Totally different text"
    result = _parse_triage_response(response, raw_gaps)
    assert result["Some unique gap"] == "MEDIUM"


# ---------------------------------------------------------------------------
# Per-gap sparring
# ---------------------------------------------------------------------------


def _make_sparring_agent(classify_on_round=None, classification="RESOLVED"):
    """Helper to build a mock SparringAgent."""
    call_count = [0]
    _result = [None]

    def fake_get_result():
        return _result[0]

    def fake_invoke(prompt):
        call_count[0] += 1
        if classify_on_round is not None and call_count[0] >= classify_on_round:
            _result[0] = GapResult("", classification, "test reasoning")
        return "agent output"

    mock_agent = MagicMock()
    mock_agent.side_effect = fake_invoke

    return SparringAgent(
        agent=mock_agent,
        get_result=fake_get_result,
        challenge_count=lambda: call_count[0],
    )


def test_spar_single_gap_resolved():
    model = MagicMock(name="model")
    gap = SparringGap("Missing encryption", "findings", "HIGH")
    profile = {"settings": {"sparring": {"max_rounds": {"high": 3}}}}

    sa = _make_sparring_agent(classify_on_round=1, classification="RESOLVED")
    with patch(f"{_MOD}.create_sparring_agent", return_value=sa):
        with patch(f"{_MOD}.safe_invoke", side_effect=lambda agent, prompt: agent(prompt)):
            result, count = _spar_single_gap(gap, model, profile, 0, "", None)

    assert result.classification == "RESOLVED"
    assert count >= 1


def test_spar_single_gap_accepted_risk():
    model = MagicMock(name="model")
    gap = SparringGap("Missing logging", "findings", "MEDIUM")
    profile = {"settings": {"sparring": {"max_rounds": {"medium": 2}}}}

    sa = _make_sparring_agent(classify_on_round=1, classification="ACCEPTED_RISK")
    with patch(f"{_MOD}.create_sparring_agent", return_value=sa):
        with patch(f"{_MOD}.safe_invoke", side_effect=lambda agent, prompt: agent(prompt)):
            result, _ = _spar_single_gap(gap, model, profile, 0, "", None)

    assert result.classification == "ACCEPTED_RISK"


def test_spar_single_gap_max_rounds_confirmed():
    model = MagicMock(name="model")
    gap = SparringGap("Missing backup", "findings", "LOW")
    profile = {"settings": {"sparring": {"max_rounds": {"low": 1}}}}

    sa = _make_sparring_agent(classify_on_round=None)
    with patch(f"{_MOD}.create_sparring_agent", return_value=sa):
        with patch(f"{_MOD}.safe_invoke", side_effect=lambda agent, prompt: agent(prompt)):
            result, _ = _spar_single_gap(gap, model, profile, 0, "", None)

    assert result.classification == "CONFIRMED_GAP"


def test_final_classify_turn():
    """Agent doesn't classify during rounds but does on the final turn."""
    model = MagicMock(name="model")
    gap = SparringGap("Missing auth", "findings", "HIGH")
    profile = {"settings": {"sparring": {"max_rounds": {"high": 2}}}}

    sa = _make_sparring_agent(classify_on_round=3, classification="ACCEPTED_RISK")
    with patch(f"{_MOD}.create_sparring_agent", return_value=sa):
        with patch(f"{_MOD}.safe_invoke", side_effect=lambda agent, prompt: agent(prompt)):
            result, _ = _spar_single_gap(gap, model, profile, 0, "", None)

    assert result.classification == "ACCEPTED_RISK"


def test_single_gap_error_does_not_crash():
    model = MagicMock(name="model")
    gap = SparringGap("Broken gap", "findings", "HIGH")
    profile = {"settings": {"sparring": {"max_rounds": {"high": 2}}}}

    sa = _make_sparring_agent()
    with patch(f"{_MOD}.create_sparring_agent", return_value=sa):
        with patch(f"{_MOD}.safe_invoke", side_effect=Exception("model error")):
            with patch(f"{_MOD}.MODEL_ERRORS", (Exception,)):
                result, _ = _spar_single_gap(gap, model, profile, 0, "", None)

    assert result.classification == "CONFIRMED_GAP"


# ---------------------------------------------------------------------------
# Running context
# ---------------------------------------------------------------------------


def test_running_context_accumulates():
    results = [
        GapResult("Gap A", "RESOLVED", "proved it"),
        GapResult("Gap B", "ACCEPTED_RISK", "it's a POC"),
    ]
    ctx = _build_running_context(results)
    assert "Gap A" in ctx
    assert "RESOLVED" in ctx
    assert "Gap B" in ctx
    assert "ACCEPTED_RISK" in ctx


def test_running_context_empty():
    assert _build_running_context([]) == ""


# ---------------------------------------------------------------------------
# Summary assembly
# ---------------------------------------------------------------------------


def test_assemble_summary_groups_by_classification():
    results = [
        GapResult("Gap A", "CONFIRMED_GAP", "not defended"),
        GapResult("Gap B", "ACCEPTED_RISK", "POC"),
        GapResult("Gap C", "RESOLVED", "proved false"),
    ]
    summary = _assemble_summary(results)
    assert "### Confirmed Gaps" in summary
    assert "### Accepted Risks" in summary
    assert "### Resolved" in summary
    assert "Gap A" in summary
    assert "Gap B" in summary
    assert "Gap C" in summary


def test_assemble_summary_omits_empty_sections():
    results = [GapResult("Gap A", "RESOLVED", "proved it")]
    summary = _assemble_summary(results)
    assert "### Resolved" in summary
    assert "Confirmed" not in summary
    assert "Accepted" not in summary


def test_assemble_summary_no_results():
    assert _assemble_summary([]) == "No gaps to report."


# ---------------------------------------------------------------------------
# Full run_sparring
# ---------------------------------------------------------------------------


def test_run_sparring_empty_gaps():
    model = MagicMock(name="model")
    with patch(f"{_MOD}.parse_gaps_from_findings", return_value=[]):
        result = run_sparring(model, "no gaps", "no gaps")
    assert "skipped" in result.lower()


def test_run_sparring_assembles_summary():
    model = MagicMock(name="model")
    gaps = ["Gap A", "Gap B"]
    triaged = [
        SparringGap("Gap A", "findings", "HIGH"),
        SparringGap("Gap B", "findings", "MEDIUM"),
    ]

    with patch(f"{_MOD}.parse_gaps_from_findings", return_value=gaps):
        with patch(f"{_MOD}._triage_gaps", return_value=triaged):
            with patch(
                f"{_MOD}._spar_single_gap",
                side_effect=[
                    (GapResult("Gap A", "RESOLVED", "proved"), 1),
                    (GapResult("Gap B", "CONFIRMED_GAP", "weak answer"), 1),
                ],
            ):
                result = run_sparring(model, "arch", "qa")

    assert "### Resolved" in result
    assert "### Confirmed Gaps" in result
    assert "Gap A" in result
    assert "Gap B" in result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_max_rounds_for_uses_profile():
    profile = {"settings": {"sparring": {"max_rounds": {"high": 5}}}}
    assert _max_rounds_for("HIGH", profile) == 5


def test_max_rounds_for_defaults_to_one():
    assert _max_rounds_for("HIGH", None) == 1
