"""Tests for arch_sparring_agent.profiles and CLI profile commands."""

import shutil
import unittest
from pathlib import Path

import yaml
from click.testing import CliRunner

from arch_sparring_agent.cli import cli
from arch_sparring_agent.exceptions import ConfigurationError
from arch_sparring_agent.profiles import (
    BUILTIN_DIR,
    get_directive,
    get_profile_path,
    list_profiles,
    load_profile,
    reset,
)


class TestLoadProfile(unittest.TestCase):
    def setUp(self):
        reset()

    def tearDown(self):
        reset()

    def test_load_default(self):
        load_profile("default")
        directive = get_directive("sparring")
        self.assertIn("Push back", directive)

    def test_load_strict(self):
        load_profile("strict")
        directive = get_directive("sparring")
        self.assertIn("firmly", directive)

    def test_load_lightweight(self):
        load_profile("lightweight")
        directive = get_directive("sparring")
        self.assertIn("prototype", directive)

    def test_load_nonexistent_raises(self):
        with self.assertRaises(ConfigurationError) as ctx:
            load_profile("nonexistent_profile_xyz")
        self.assertIn("nonexistent_profile_xyz", str(ctx.exception))

    def test_get_directive_without_loading_returns_empty(self):
        self.assertEqual(get_directive("sparring"), "")

    def test_get_directive_unknown_agent_returns_empty(self):
        load_profile("default")
        self.assertEqual(get_directive("nonexistent_agent"), "")


class TestProfileResolutionOrder(unittest.TestCase):
    """Test that project-level profiles take precedence over builtin."""

    def setUp(self):
        reset()
        self.tmp_project_dir = Path.cwd() / ".arch-review" / "profiles"
        self.tmp_project_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        reset()
        if self.tmp_project_dir.exists():
            shutil.rmtree(Path.cwd() / ".arch-review" / "profiles")
            project_root = Path.cwd() / ".arch-review"
            if project_root.exists() and not any(project_root.iterdir()):
                project_root.rmdir()

    def test_project_profile_overrides_builtin(self):
        custom = {
            "name": "default",
            "description": "Project override",
            "directives": {"sparring": "Custom project sparring directive"},
        }
        with open(self.tmp_project_dir / "default.yaml", "w") as f:
            yaml.dump(custom, f)

        load_profile("default")
        self.assertEqual(get_directive("sparring"), "Custom project sparring directive")


class TestProfileResolutionUserDir(unittest.TestCase):
    """Test that user-level profiles take precedence over builtin."""

    def setUp(self):
        reset()
        self.tmp_user_dir = Path.home() / ".config" / "arch-review" / "profiles"
        self.created_file = self.tmp_user_dir / "_test_user_profile.yaml"
        self.tmp_user_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        reset()
        if self.created_file.exists():
            self.created_file.unlink()

    def test_user_profile_found(self):
        custom = {
            "name": "test",
            "description": "User test profile",
            "directives": {"sparring": "User sparring directive"},
        }
        with open(self.created_file, "w") as f:
            yaml.dump(custom, f)

        load_profile("_test_user_profile")
        self.assertEqual(get_directive("sparring"), "User sparring directive")


class TestListProfiles(unittest.TestCase):
    def test_builtin_profiles_listed(self):
        result = list_profiles()
        self.assertIn("default", result["builtin"])
        self.assertIn("strict", result["builtin"])
        self.assertIn("lightweight", result["builtin"])


class TestGetProfilePath(unittest.TestCase):
    def test_builtin_default_path(self):
        path = get_profile_path("default")
        self.assertIsNotNone(path)
        self.assertEqual(path.name, "default.yaml")
        self.assertTrue(path.is_file())

    def test_nonexistent_returns_none(self):
        self.assertIsNone(get_profile_path("nonexistent_xyz"))


class TestDirectiveContent(unittest.TestCase):
    """Verify that each built-in profile has directives for the expected agents."""

    def setUp(self):
        reset()

    def tearDown(self):
        reset()

    def test_default_has_all_agent_directives(self):
        load_profile("default")
        for agent in ("sparring", "review", "architecture", "ci"):
            with self.subTest(agent=agent):
                self.assertTrue(len(get_directive(agent)) > 0, f"Missing directive for {agent}")

    def test_strict_has_all_agent_directives(self):
        load_profile("strict")
        for agent in ("sparring", "review", "architecture", "ci"):
            with self.subTest(agent=agent):
                self.assertTrue(len(get_directive(agent)) > 0, f"Missing directive for {agent}")

    def test_lightweight_has_all_agent_directives(self):
        load_profile("lightweight")
        for agent in ("sparring", "review", "architecture", "ci"):
            with self.subTest(agent=agent):
                self.assertTrue(len(get_directive(agent)) > 0, f"Missing directive for {agent}")

    def test_profiles_differ(self):
        load_profile("default")
        default_sparring = get_directive("sparring")
        reset()

        load_profile("strict")
        strict_sparring = get_directive("sparring")
        reset()

        load_profile("lightweight")
        lightweight_sparring = get_directive("sparring")

        self.assertNotEqual(default_sparring, strict_sparring)
        self.assertNotEqual(default_sparring, lightweight_sparring)
        self.assertNotEqual(strict_sparring, lightweight_sparring)


class TestBuiltinYamlValidity(unittest.TestCase):
    """Ensure all YAML files in the profiles directory parse correctly."""

    def test_all_yamls_parse(self):
        for yaml_file in BUILTIN_DIR.glob("*.yaml"):
            with self.subTest(file=yaml_file.name):
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                self.assertIsInstance(data, dict)
                self.assertIn("name", data)
                self.assertIn("description", data)
                self.assertIn("directives", data)
                self.assertIsInstance(data["directives"], dict)


class TestCliProfilesCommands(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_profiles_list(self):
        result = self.runner.invoke(cli, ["profiles", "list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("default", result.output)
        self.assertIn("strict", result.output)
        self.assertIn("lightweight", result.output)

    def test_profiles_show_default(self):
        result = self.runner.invoke(cli, ["profiles", "show", "default"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("directives", result.output)
        self.assertIn("sparring", result.output)

    def test_profiles_show_nonexistent(self):
        result = self.runner.invoke(cli, ["profiles", "show", "nonexistent_xyz"])
        self.assertNotEqual(result.exit_code, 0)

    def test_profiles_help(self):
        result = self.runner.invoke(cli, ["profiles", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("list", result.output)
        self.assertIn("show", result.output)
        self.assertIn("create", result.output)

    def test_run_help_shows_profile_option(self):
        result = self.runner.invoke(cli, ["run", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--profile", result.output)

    def test_remediate_help_shows_profile_option(self):
        result = self.runner.invoke(cli, ["remediate", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--profile", result.output)


class TestCliProfilesCreate(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.user_dir = Path.home() / ".config" / "arch-review" / "profiles"
        self.created_file = self.user_dir / "_test_create_profile.yaml"

    def tearDown(self):
        if self.created_file.exists():
            self.created_file.unlink()

    def test_create_from_default(self):
        result = self.runner.invoke(
            cli, ["profiles", "create", "_test_create_profile", "--from", "default"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(self.created_file.exists())
        with open(self.created_file) as f:
            data = yaml.safe_load(f)
        self.assertIn("directives", data)
        self.assertIn("sparring", data["directives"])

    def test_create_already_exists_fails(self):
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.created_file.write_text("existing")
        result = self.runner.invoke(
            cli, ["profiles", "create", "_test_create_profile", "--from", "default"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("already exists", result.output)
