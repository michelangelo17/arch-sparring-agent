"""Unit tests for arch_sparring_agent.cli."""

import unittest

from click.testing import CliRunner

from arch_sparring_agent.cli import _get_verdict_and_exit_code, get_version, main


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


class TestMainCli(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help_exits_0(self):
        result = self.runner.invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Architecture Review Sparring Partner", result.output)
        self.assertIn("--help", result.output)

    def test_version_shows_version(self):
        result = self.runner.invoke(main, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("arch-review", result.output)

    def test_missing_required_dirs_shows_error(self):
        result = self.runner.invoke(
            main,
            [
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
