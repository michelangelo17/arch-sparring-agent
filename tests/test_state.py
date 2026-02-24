import tempfile
from pathlib import Path

from arch_sparring_agent.state import Gap, ReviewState, Risk


def test_review_state_serialization():
    """Test ReviewState to/from JSON."""
    state = ReviewState(
        timestamp="2023-10-27T10:00:00",
        project_name="TestProject",
        gaps=[Gap(id="gap-1", description="Missing auth", severity="high")],
        risks=[Risk(id="risk-1", description="DDoS", impact="high")],
        recommendations=["Add WAF"],
        verdict="PASS WITH CONCERNS",
        requirements_summary="Reqs",
        architecture_summary="Arch",
    )

    json_str = state.to_json()
    loaded_state = ReviewState.from_json(json_str)

    assert loaded_state == state
    assert loaded_state.project_name == "TestProject"
    assert len(loaded_state.gaps) == 1
    assert loaded_state.gaps[0].id == "gap-1"


def test_review_state_file_io():
    """Test ReviewState save and load from file."""
    state = ReviewState(timestamp="2023-10-27T10:00:00", project_name="FileTest")

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        path = Path(tmp.name)

    try:
        state.save(path)
        loaded_state = ReviewState.from_file(path)
        assert loaded_state == state
    finally:
        path.unlink()
