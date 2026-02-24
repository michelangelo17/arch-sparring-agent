from arch_sparring_agent.review.extraction import (
    extract_gaps,
    extract_recommendations,
    extract_risks,
    extract_state_from_review,
    extract_verdict,
    has_duplicate_description,
    has_duplicate_string,
    infer_severity,
    strip_markdown,
)
from arch_sparring_agent.review.orchestrator import ReviewResult
from arch_sparring_agent.state import Gap


def test_strips_bold():
    assert strip_markdown("**bold text**") == "bold text"


def test_strips_italic():
    assert strip_markdown("*italic*") == "italic"


def test_strips_inline_code():
    assert strip_markdown("`some_code`") == "some_code"


def test_strips_underscores():
    assert strip_markdown("__underline__") == "underline"


def test_strips_mixed():
    assert strip_markdown("**Risk**: `some_func` is *broken*") == "Risk: some_func is broken"


def test_plain_text_unchanged():
    assert strip_markdown("plain text") == "plain text"


def test_infer_severity():
    assert infer_severity("This is a critical issue") == "high"
    assert infer_severity("High latency") == "high"
    assert infer_severity("Low priority") == "low"
    assert infer_severity("Minor bug") == "low"
    assert infer_severity("Some other issue") == "medium"


def test_has_duplicate_description():
    items = [
        Gap(id="g-1", description="Database is unencrypted", severity="high"),
        Gap(id="g-2", description="Missing unit tests", severity="medium"),
    ]
    assert has_duplicate_description("Database", items)
    assert has_duplicate_description("Missing unit", items)
    assert not has_duplicate_description("Something completely new", items)


def test_has_duplicate_string():
    items = ["Database is unencrypted", "Missing unit tests"]
    assert has_duplicate_string("Database", items)
    assert not has_duplicate_string("Something completely new", items)


def test_extract_verdict():
    assert extract_verdict("Overall Verdict: PASS") == "PASS"
    assert extract_verdict("Verdict: FAIL because of security") == "FAIL"
    assert extract_verdict("Verdict: PASS WITH CONCERNS due to latency") == "PASS WITH CONCERNS"
    assert extract_verdict("No clear result") == "PASS"
    assert extract_verdict("Critical security flaw found") == "FAIL"
    assert extract_verdict("Risk Impact: High severity") == "PASS WITH CONCERNS"


def test_extract_gaps_flat_list():
    review_text = """
## Gaps
- Missing authentication
- No backup strategy
"""
    gaps = extract_gaps(review_text, "")
    assert len(gaps) == 2
    assert gaps[0].description == "Missing authentication"
    assert gaps[0].id == "gap-1"
    assert gaps[1].description == "No backup strategy"
    assert gaps[1].id == "gap-2"


def test_extract_gaps_confirmed_gaps_header():
    """Test '#### Confirmed Gaps' header style."""
    review_text = """
#### Confirmed Gaps
1. Feature: Cost optimization through time-based cooldown
2. Feature: Monitoring via CloudWatch Logs and X-Ray tracing
"""
    gaps = extract_gaps(review_text, "")
    assert len(gaps) == 2
    assert "Cost optimization" in gaps[0].description
    assert "Monitoring" in gaps[1].description


def test_extract_gaps_ignores_numbered_gap_subheaders():
    """Ensure '##### Gap 1: ...' sub-headers inside Risks don't create spurious gaps."""
    review_text = """
#### Confirmed Gaps
1. Cost optimization
2. Monitoring

#### Risks & Mitigations
##### Gap 1: Feature - Cost optimization
- **Risk**: Inefficient resource utilization
- **Impact**: **Medium**
- **Mitigation**: Implement cooldown periods

##### Gap 2: Feature - Monitoring
- **Risk**: Insufficient visibility
- **Impact**: **Medium**
- **Mitigation**: Integrate logging
"""
    gaps = extract_gaps(review_text, "")
    assert len(gaps) == 2
    descriptions = [g.description for g in gaps]
    for desc in descriptions:
        assert "Risk" not in desc
        assert "Impact" not in desc
        assert "Mitigation" not in desc


def test_extract_gaps_features_not_found():
    review_text = """
## Features Not Found
- Rate limiting
"""
    gaps = extract_gaps(review_text, "")
    assert len(gaps) == 1
    assert gaps[0].description == "Rate limiting"
    assert gaps[0].severity == "medium"


def test_extract_gaps_strips_markdown():
    review_text = """
## Gaps
- **Missing** `encryption` at rest
"""
    gaps = extract_gaps(review_text, "")
    assert len(gaps) == 1
    assert gaps[0].description == "Missing encryption at rest"


def test_extract_risks_flat_list():
    review_text = """
## Top Risks
1. Data loss potential
2. Security breach
"""
    risks = extract_risks(review_text, "")
    assert len(risks) == 2
    assert risks[0].description == "Data loss potential"
    assert risks[0].id == "risk-1"
    assert risks[1].description == "Security breach"
    assert risks[1].id == "risk-2"


def test_extract_risks_nested_with_gap_subheaders():
    """Test the nested format: #### Risks & Mitigations with ##### Gap N: sub-headers."""
    review_text = """
#### Risks & Mitigations
##### Gap 1: Feature - Cost optimization
- **Risk**: Inefficient resource utilization and increased costs
- **Impact**: **Medium**
- **Mitigation**: Implement cooldown periods

##### Gap 2: Feature - Monitoring
- **Risk**: Insufficient visibility into application performance
- **Impact**: **Medium**
- **Mitigation**: Integrate comprehensive logging
"""
    risks = extract_risks(review_text, "")
    assert len(risks) == 2
    assert "Inefficient resource utilization" in risks[0].description
    assert "Insufficient visibility" in risks[1].description


def test_extract_risks_skips_impact_and_mitigation():
    """Impact and Mitigation bullets should not appear as risks."""
    review_text = """
#### Risks & Mitigations
##### Gap 1: Something
- **Risk**: Actual risk description here
- **Impact**: **High**
- **Mitigation**: Do something about it
"""
    risks = extract_risks(review_text, "")
    assert len(risks) == 1
    assert risks[0].description == "Actual risk description here"


def test_extract_risks_strips_risk_prefix():
    """Risk: prefix should be removed from risk description."""
    review_text = """
#### Risks
- Risk: Something bad might happen
"""
    risks = extract_risks(review_text, "")
    assert len(risks) == 1
    assert risks[0].description == "Something bad might happen"


def test_extract_recommendations_direct_section():
    review_text = """
## Recommendations
1. Implement caching
- Use Multi-Factor Authentication
"""
    recs = extract_recommendations(review_text)
    assert len(recs) == 2
    assert recs[0] == "Implement caching"
    assert recs[1] == "Use Multi-Factor Authentication"


def test_extract_recommendations_falls_back_to_mitigations():
    """When there's no Recommendations section, extract mitigations from Risks."""
    review_text = """
#### Risks & Mitigations
##### Gap 1: Something
- **Risk**: Bad thing
- **Impact**: **Medium**
- **Mitigation**: Implement cooldown periods to prevent rapid allocations

##### Gap 2: Another thing
- **Risk**: Another bad thing
- **Impact**: **Medium**
- **Mitigation**: Integrate comprehensive logging using CloudWatch
"""
    recs = extract_recommendations(review_text)
    assert len(recs) == 2
    assert "cooldown periods" in recs[0]
    assert "comprehensive logging" in recs[1]


def test_extract_recommendations_prefers_direct_section():
    """If both Recommendations and Mitigations exist, use Recommendations."""
    review_text = """
#### Risks & Mitigations
##### Gap 1: Something
- **Risk**: Bad thing
- **Mitigation**: Do X

## Recommendations
- Do Y instead
"""
    recs = extract_recommendations(review_text)
    assert len(recs) == 1
    assert recs[0] == "Do Y instead"


def test_extract_state_from_review_flat_format():
    review_result = ReviewResult(
        review="""
## Gaps
- Missing Gap 1

## Top Risks
1. Major Risk 1

## Recommendations
- Recommendation 1

Verdict: PASS
""",
        full_session="",
        requirements_summary="Reqs",
        requirements_findings="",
        architecture_summary="Arch",
        architecture_findings="",
        qa_context="",
        qa_findings="",
        sparring_context="",
        sparring_findings="",
    )

    state = extract_state_from_review(review_result)

    assert state.verdict == "PASS"
    assert len(state.gaps) == 1
    assert len(state.risks) == 1
    assert len(state.recommendations) == 1
    assert state.requirements_summary == "Reqs"


def test_extract_state_from_review_ci_format():
    """Full integration test with the nested CI-style review format."""
    review_result = ReviewResult(
        review="""### Architecture Review

#### Summary
The architecture shows potential inefficiencies in resource utilization.

#### Confirmed Gaps
1. Feature: Cost optimization through time-based cooldown and review count detection
2. Feature: Monitoring via CloudWatch Logs and X-Ray tracing

#### Risks & Mitigations
##### Gap 1: Feature - Cost optimization through time-based cooldown
- **Risk**: Inefficient resource utilization and increased operational costs
- **Impact**: **Medium**
- **Mitigation**: Implement time-based cooldown periods to prevent rapid allocations

##### Gap 2: Feature - Monitoring via CloudWatch Logs and X-Ray tracing
- **Risk**: Insufficient visibility into application performance
- **Impact**: **Medium**
- **Mitigation**: Integrate comprehensive logging using CloudWatch Logs

#### Features Not Found
- Rate limiting

Verdict: PASS WITH CONCERNS
""",
        full_session="",
        requirements_summary="Reqs",
        requirements_findings="",
        architecture_summary="Arch",
        architecture_findings="",
        qa_context="",
        qa_findings="",
        sparring_context="",
        sparring_findings="",
    )

    state = extract_state_from_review(review_result)

    assert len(state.gaps) == 3
    gap_descs = [g.description for g in state.gaps]
    assert any("Cost optimization" in d for d in gap_descs)
    assert any("Monitoring" in d for d in gap_descs)
    assert any("Rate limiting" in d for d in gap_descs)
    for g in state.gaps:
        assert "Risk" not in g.description
        assert "Mitigation" not in g.description

    assert len(state.risks) == 2
    assert "Inefficient resource" in state.risks[0].description
    assert "Insufficient visibility" in state.risks[1].description

    assert len(state.recommendations) == 2
    assert any("cooldown" in r for r in state.recommendations)
    assert any("logging" in r for r in state.recommendations)

    assert state.verdict == "PASS WITH CONCERNS"
