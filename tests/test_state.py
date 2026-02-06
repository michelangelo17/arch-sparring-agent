import tempfile
import unittest
from pathlib import Path

from arch_sparring_agent.state import (
    ReviewState,
    _extract_gaps,
    _extract_recommendations,
    _extract_risks,
    _extract_verdict,
    _infer_severity,
    _is_duplicate,
    extract_state_from_review,
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
        self.assertEqual(_extract_verdict("Overall Verdict: PASS"), "PASS")
        self.assertEqual(_extract_verdict("Verdict: FAIL because of security"), "FAIL")
        self.assertEqual(
            _extract_verdict("Verdict: PASS WITH CONCERNS due to latency"), "PASS WITH CONCERNS"
        )
        self.assertEqual(_extract_verdict("No clear result"), "UNKNOWN")

    def test_extract_gaps(self):
        review_text = """
        ## Gaps
        - Missing authentication
        - No backup strategy
        """
        gaps_context = ""

        gaps = _extract_gaps(review_text, gaps_context)
        self.assertEqual(len(gaps), 2)
        self.assertEqual(gaps[0]["description"], "Missing authentication")
        self.assertEqual(gaps[0]["id"], "gap-1")
        self.assertEqual(gaps[1]["description"], "No backup strategy")
        self.assertEqual(gaps[1]["id"], "gap-2")

    def test_extract_gaps_features_not_found(self):
        review_text = """
        ## Features Not Found
        - Rate limiting
        """
        gaps_context = ""
        gaps = _extract_gaps(review_text, gaps_context)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["description"], "Rate limiting")
        self.assertEqual(gaps[0]["severity"], "medium")

    def test_extract_risks(self):
        review_text = """
        ## Top Risks
        1. Data loss potential
        2. Security breach
        """
        risks_context = ""

        risks = _extract_risks(review_text, risks_context)
        self.assertEqual(len(risks), 2)
        self.assertEqual(risks[0]["description"], "Data loss potential")
        self.assertEqual(risks[0]["id"], "risk-1")
        self.assertEqual(risks[1]["description"], "Security breach")
        self.assertEqual(risks[1]["id"], "risk-2")

    def test_extract_recommendations(self):
        review_text = """
        ## Recommendations
        1. Implement caching
        - Use Multi-Factor Authentication
        """

        recs = _extract_recommendations(review_text)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0], "Implement caching")
        self.assertEqual(recs[1], "Use Multi-Factor Authentication")

    def test_extract_state_from_review(self):
        review_result = {
            "review": """
            ## Gaps
            - Missing Gap 1

            ## Top Risks
            1. Major Risk 1

            ## Recommendations
            - Recommendation 1

            Verdict: PASS
            """,
            "gaps": "",
            "risks": "",
            "requirements_summary": "Reqs",
            "architecture_summary": "Arch",
        }

        state = extract_state_from_review(review_result)

        if len(state.gaps) != 1:
            print(f"Gaps found: {state.gaps}")

        self.assertEqual(state.verdict, "PASS")
        self.assertEqual(len(state.gaps), 1)
        self.assertEqual(len(state.risks), 1)
        self.assertEqual(len(state.recommendations), 1)
        self.assertEqual(state.requirements_summary, "Reqs")


if __name__ == "__main__":
    unittest.main()
