import tempfile
import unittest
from pathlib import Path

from arch_sparring_agent.orchestrator import ReviewResult
from arch_sparring_agent.state import (
    ReviewState,
    _extract_gaps,
    _extract_recommendations,
    _extract_risks,
    _infer_severity,
    _is_duplicate,
    _strip_markdown,
    extract_state_from_review,
    extract_verdict,
)


class TestReviewState(unittest.TestCase):
    def test_review_state_serialization(self):
        """Test ReviewState to/from JSON."""
        state = ReviewState(
            timestamp="2023-10-27T10:00:00",
            project_name="TestProject",
            gaps=[{"id": "gap-1", "description": "Missing auth", "severity": "high"}],
            risks=[{"id": "risk-1", "description": "DDoS", "impact": "high"}],
            recommendations=["Add WAF"],
            verdict="PASS WITH CONCERNS",
            requirements_summary="Reqs",
            architecture_summary="Arch",
        )

        json_str = state.to_json()
        loaded_state = ReviewState.from_json(json_str)

        self.assertEqual(loaded_state, state)
        self.assertEqual(loaded_state.project_name, "TestProject")
        self.assertEqual(len(loaded_state.gaps), 1)
        self.assertEqual(loaded_state.gaps[0]["id"], "gap-1")

    def test_review_state_file_io(self):
        """Test ReviewState save and load from file."""
        state = ReviewState(timestamp="2023-10-27T10:00:00", project_name="FileTest")

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            path = Path(tmp.name)

        try:
            state.save(path)
            loaded_state = ReviewState.from_file(path)
            self.assertEqual(loaded_state, state)
        finally:
            path.unlink()


class TestStripMarkdown(unittest.TestCase):
    def test_strips_bold(self):
        self.assertEqual(_strip_markdown("**bold text**"), "bold text")

    def test_strips_italic(self):
        self.assertEqual(_strip_markdown("*italic*"), "italic")

    def test_strips_inline_code(self):
        self.assertEqual(_strip_markdown("`some_code`"), "some_code")

    def test_strips_underscores(self):
        self.assertEqual(_strip_markdown("__underline__"), "underline")

    def test_strips_mixed(self):
        self.assertEqual(
            _strip_markdown("**Risk**: `some_func` is *broken*"),
            "Risk: some_func is broken",
        )

    def test_plain_text_unchanged(self):
        self.assertEqual(_strip_markdown("plain text"), "plain text")


class TestExtractionLogic(unittest.TestCase):
    def test_infer_severity(self):
        self.assertEqual(_infer_severity("This is a critical issue"), "high")
        self.assertEqual(_infer_severity("High latency"), "high")
        self.assertEqual(_infer_severity("Low priority"), "low")
        self.assertEqual(_infer_severity("Minor bug"), "low")
        self.assertEqual(_infer_severity("Some other issue"), "medium")

    def test_is_duplicate(self):
        items = [{"description": "Database is unencrypted"}, {"description": "Missing unit tests"}]

        # Current logic checks if new text is a substring of existing item
        self.assertTrue(_is_duplicate("Database", items))
        self.assertTrue(_is_duplicate("Missing unit", items))
        self.assertFalse(_is_duplicate("Something completely new", items))

    def test_extract_verdict(self):
        self.assertEqual(extract_verdict("Overall Verdict: PASS"), "PASS")
        self.assertEqual(extract_verdict("Verdict: FAIL because of security"), "FAIL")
        self.assertEqual(
            extract_verdict("Verdict: PASS WITH CONCERNS due to latency"), "PASS WITH CONCERNS"
        )
        # Fallback: infer from content
        self.assertEqual(extract_verdict("No clear result"), "PASS")  # Default
        self.assertEqual(extract_verdict("Critical security flaw found"), "FAIL")
        self.assertEqual(extract_verdict("Risk Impact: High severity"), "PASS WITH CONCERNS")

    # ----------------------------------------------------------------
    # Gap extraction
    # ----------------------------------------------------------------
    def test_extract_gaps_flat_list(self):
        review_text = """
## Gaps
- Missing authentication
- No backup strategy
"""
        gaps = _extract_gaps(review_text, "")
        self.assertEqual(len(gaps), 2)
        self.assertEqual(gaps[0]["description"], "Missing authentication")
        self.assertEqual(gaps[0]["id"], "gap-1")
        self.assertEqual(gaps[1]["description"], "No backup strategy")
        self.assertEqual(gaps[1]["id"], "gap-2")

    def test_extract_gaps_confirmed_gaps_header(self):
        """Test '#### Confirmed Gaps' header style."""
        review_text = """
#### Confirmed Gaps
1. Feature: Cost optimization through time-based cooldown
2. Feature: Monitoring via CloudWatch Logs and X-Ray tracing
"""
        gaps = _extract_gaps(review_text, "")
        self.assertEqual(len(gaps), 2)
        self.assertIn("Cost optimization", gaps[0]["description"])
        self.assertIn("Monitoring", gaps[1]["description"])

    def test_extract_gaps_ignores_numbered_gap_subheaders(self):
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
        gaps = _extract_gaps(review_text, "")
        # Should only have the 2 from Confirmed Gaps, not Risk/Impact/Mitigation lines
        self.assertEqual(len(gaps), 2)
        descriptions = [g["description"] for g in gaps]
        for desc in descriptions:
            self.assertNotIn("Risk", desc)
            self.assertNotIn("Impact", desc)
            self.assertNotIn("Mitigation", desc)

    def test_extract_gaps_features_not_found(self):
        review_text = """
## Features Not Found
- Rate limiting
"""
        gaps = _extract_gaps(review_text, "")
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["description"], "Rate limiting")
        self.assertEqual(gaps[0]["severity"], "medium")

    def test_extract_gaps_strips_markdown(self):
        review_text = """
## Gaps
- **Missing** `encryption` at rest
"""
        gaps = _extract_gaps(review_text, "")
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["description"], "Missing encryption at rest")

    # ----------------------------------------------------------------
    # Risk extraction
    # ----------------------------------------------------------------
    def test_extract_risks_flat_list(self):
        review_text = """
## Top Risks
1. Data loss potential
2. Security breach
"""
        risks = _extract_risks(review_text, "")
        self.assertEqual(len(risks), 2)
        self.assertEqual(risks[0]["description"], "Data loss potential")
        self.assertEqual(risks[0]["id"], "risk-1")
        self.assertEqual(risks[1]["description"], "Security breach")
        self.assertEqual(risks[1]["id"], "risk-2")

    def test_extract_risks_nested_with_gap_subheaders(self):
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
        risks = _extract_risks(review_text, "")
        self.assertEqual(len(risks), 2)
        self.assertIn("Inefficient resource utilization", risks[0]["description"])
        self.assertIn("Insufficient visibility", risks[1]["description"])

    def test_extract_risks_skips_impact_and_mitigation(self):
        """Impact and Mitigation bullets should not appear as risks."""
        review_text = """
#### Risks & Mitigations
##### Gap 1: Something
- **Risk**: Actual risk description here
- **Impact**: **High**
- **Mitigation**: Do something about it
"""
        risks = _extract_risks(review_text, "")
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0]["description"], "Actual risk description here")

    def test_extract_risks_strips_risk_prefix(self):
        """Risk: prefix should be removed from risk description."""
        review_text = """
#### Risks
- Risk: Something bad might happen
"""
        risks = _extract_risks(review_text, "")
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0]["description"], "Something bad might happen")

    # ----------------------------------------------------------------
    # Recommendation extraction
    # ----------------------------------------------------------------
    def test_extract_recommendations_direct_section(self):
        review_text = """
## Recommendations
1. Implement caching
- Use Multi-Factor Authentication
"""
        recs = _extract_recommendations(review_text)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0], "Implement caching")
        self.assertEqual(recs[1], "Use Multi-Factor Authentication")

    def test_extract_recommendations_falls_back_to_mitigations(self):
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
        recs = _extract_recommendations(review_text)
        self.assertEqual(len(recs), 2)
        self.assertIn("cooldown periods", recs[0])
        self.assertIn("comprehensive logging", recs[1])

    def test_extract_recommendations_prefers_direct_section(self):
        """If both Recommendations and Mitigations exist, use Recommendations."""
        review_text = """
#### Risks & Mitigations
##### Gap 1: Something
- **Risk**: Bad thing
- **Mitigation**: Do X

## Recommendations
- Do Y instead
"""
        recs = _extract_recommendations(review_text)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0], "Do Y instead")

    # ----------------------------------------------------------------
    # Full integration: CI-style review
    # ----------------------------------------------------------------
    def test_extract_state_from_review_flat_format(self):
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
            gaps="",
            gaps_findings="",
            risks="",
            risks_findings="",
        )

        state = extract_state_from_review(review_result)

        self.assertEqual(state.verdict, "PASS")
        self.assertEqual(len(state.gaps), 1)
        self.assertEqual(len(state.risks), 1)
        self.assertEqual(len(state.recommendations), 1)
        self.assertEqual(state.requirements_summary, "Reqs")

    def test_extract_state_from_review_ci_format(self):
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
            gaps="",
            gaps_findings="",
            risks="",
            risks_findings="",
        )

        state = extract_state_from_review(review_result)

        # Should have 3 gaps: 2 from Confirmed Gaps + 1 from Features Not Found
        self.assertEqual(len(state.gaps), 3)
        gap_descs = [g["description"] for g in state.gaps]
        self.assertTrue(any("Cost optimization" in d for d in gap_descs))
        self.assertTrue(any("Monitoring" in d for d in gap_descs))
        self.assertTrue(any("Rate limiting" in d for d in gap_descs))
        # No risk/impact/mitigation text in gaps
        for g in state.gaps:
            self.assertNotIn("Risk", g["description"])
            self.assertNotIn("Mitigation", g["description"])

        # Should have 2 risks (one per Gap sub-header)
        self.assertEqual(len(state.risks), 2)
        self.assertIn("Inefficient resource", state.risks[0]["description"])
        self.assertIn("Insufficient visibility", state.risks[1]["description"])

        # Recommendations should be extracted from mitigations (no Recommendations section)
        self.assertEqual(len(state.recommendations), 2)
        self.assertTrue(any("cooldown" in r for r in state.recommendations))
        self.assertTrue(any("logging" in r for r in state.recommendations))

        self.assertEqual(state.verdict, "PASS WITH CONCERNS")


if __name__ == "__main__":
    unittest.main()
