"""Unit tests for arch_sparring_agent.cli."""

import pytest
from click.testing import CliRunner

from arch_sparring_agent.cli import cli, get_verdict_and_exit_code, get_version


def test_get_version_returns_string():
    v = get_version()
    assert isinstance(v, str)
    assert len(v) > 0


def test_verdict_pass_returns_0():
    verdict, code = get_verdict_and_exit_code("Verdict: PASS")
    assert verdict == "PASS"
    assert code == 0


def test_verdict_fail_returns_1():
    verdict, code = get_verdict_and_exit_code("Verdict: FAIL")
    assert verdict == "FAIL"
    assert code == 1


def test_verdict_pass_with_concerns_returns_2():
    verdict, code = get_verdict_and_exit_code("Verdict: PASS WITH CONCERNS")
    assert verdict == "PASS WITH CONCERNS"
    assert code == 2


def test_pass_with_concerns_high_impact_still_returns_2():
    text = "Verdict: PASS WITH CONCERNS\n\nImpact: High"
    verdict, code = get_verdict_and_exit_code(text)
    assert verdict == "PASS WITH CONCERNS"
    assert code == 2


@pytest.fixture
def runner():
    return CliRunner()


def test_help_exits_0(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Architecture Review Sparring Partner" in result.output


def test_version_shows_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "arch-review" in result.output


def test_run_help_shows_options(runner):
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--documents-dir" in result.output


def test_deploy_help_shows_options(runner):
    result = runner.invoke(cli, ["deploy", "--help"])
    assert result.exit_code == 0
    assert "--region" in result.output


def test_destroy_help_shows_options(runner):
    result = runner.invoke(cli, ["destroy", "--help"])
    assert result.exit_code == 0
    assert "--confirm" in result.output


def test_remediate_help_shows_options(runner):
    result = runner.invoke(cli, ["remediate", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


def test_run_missing_required_dirs_shows_error(runner):
    result = runner.invoke(
        cli,
        [
            "run",
            "--documents-dir",
            "/nonexistent/documents",
            "--templates-dir",
            "/nonexistent/templates",
            "--diagrams-dir",
            "/nonexistent/diagrams",
        ],
    )
    assert result.exit_code != 0
    assert "documents-dir" in result.output
