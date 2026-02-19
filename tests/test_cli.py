"""Unit tests for arch_sparring_agent.cli."""

import unittest

from click.testing import CliRunner

from arch_sparring_agent.cli import _get_verdict_and_exit_code, cli, get_version


class TestGetVersion(unittest.TestCase):
    def test_returns_string(self):
        v = get_version()
        self.assertIsInstance(v, str)
        self.assertGreater(len(v), 0)


class TestGetVerdictAndExitCode(unittest.TestCase):
    def test_verdict_pass_returns_0(self):
        verdict, code = _get_verdict_and_exit_code("Verdict: PASS")
        self.assertEqual(verdict, "PASS")
        self.assertEqual(code, 0)

    def test_verdict_fail_returns_1(self):
        verdict, code = _get_verdict_and_exit_code("Verdict: FAIL")
        self.assertEqual(verdict, "FAIL")
        self.assertEqual(code, 1)

    def test_verdict_pass_with_concerns_returns_2(self):
        verdict, code = _get_verdict_and_exit_code("Verdict: PASS WITH CONCERNS")
        self.assertEqual(verdict, "PASS WITH CONCERNS")
        self.assertEqual(code, 2)

    def test_pass_with_concerns_high_impact_strict_returns_fail_1(self):
        text = "Verdict: PASS WITH CONCERNS\n\nImpact: High"
        verdict, code = _get_verdict_and_exit_code(text, strict=True)
        self.assertEqual(verdict, "FAIL")
        self.assertEqual(code, 1)


class TestCliGroup(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help_exits_0(self):
        result = self.runner.invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Architecture Review Sparring Partner", result.output)

    def test_version_shows_version(self):
        result = self.runner.invoke(cli, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("arch-review", result.output)

    def test_run_help_shows_options(self):
        result = self.runner.invoke(cli, ["run", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--documents-dir", result.output)

    def test_deploy_help_shows_options(self):
        result = self.runner.invoke(cli, ["deploy", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--region", result.output)

    def test_destroy_help_shows_options(self):
        result = self.runner.invoke(cli, ["destroy", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--confirm", result.output)

    def test_remediate_help_shows_options(self):
        result = self.runner.invoke(cli, ["remediate", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--model", result.output)

    def test_run_missing_required_dirs_shows_error(self):
        result = self.runner.invoke(
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
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("documents-dir", result.output)
