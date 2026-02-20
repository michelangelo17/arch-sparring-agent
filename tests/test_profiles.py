"""Tests for arch_sparring_agent.profiles and CLI profile commands."""

import shutil
from pathlib import Path

import pytest
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
)


def test_load_default():
    profile = load_profile("default")
    directive = get_directive(profile, "sparring")
    assert "Push back" in directive


def test_load_strict():
    profile = load_profile("strict")
    directive = get_directive(profile, "sparring")
    assert "firmly" in directive


def test_load_lightweight():
    profile = load_profile("lightweight")
    directive = get_directive(profile, "sparring")
    assert "prototype" in directive


def test_load_nonexistent_raises():
    with pytest.raises(ConfigurationError) as exc_info:
        load_profile("nonexistent_profile_xyz")
    assert "nonexistent_profile_xyz" in str(exc_info.value)


def test_get_directive_without_profile_returns_empty():
    assert get_directive(None, "sparring") == ""


def test_get_directive_unknown_agent_returns_empty():
    profile = load_profile("default")
    assert get_directive(profile, "nonexistent_agent") == ""


@pytest.fixture
def tmp_project_profiles_dir():
    tmp_project_dir = Path.cwd() / ".arch-review" / "profiles"
    tmp_project_dir.mkdir(parents=True, exist_ok=True)
    yield tmp_project_dir
    if tmp_project_dir.exists():
        shutil.rmtree(Path.cwd() / ".arch-review" / "profiles")
        project_root = Path.cwd() / ".arch-review"
        if project_root.exists() and not any(project_root.iterdir()):
            project_root.rmdir()


def test_project_profile_overrides_builtin(tmp_project_profiles_dir):
    custom = {
        "name": "default",
        "description": "Project override",
        "directives": {"sparring": "Custom project sparring directive"},
    }
    with open(tmp_project_profiles_dir / "default.yaml", "w") as f:
        yaml.dump(custom, f)

    profile = load_profile("default")
    assert get_directive(profile, "sparring") == "Custom project sparring directive"


@pytest.fixture
def tmp_user_profile_file():
    tmp_user_dir = Path.home() / ".config" / "arch-review" / "profiles"
    created_file = tmp_user_dir / "_test_user_profile.yaml"
    tmp_user_dir.mkdir(parents=True, exist_ok=True)
    yield created_file
    if created_file.exists():
        created_file.unlink()


def test_user_profile_found(tmp_user_profile_file):
    custom = {
        "name": "test",
        "description": "User test profile",
        "directives": {"sparring": "User sparring directive"},
    }
    with open(tmp_user_profile_file, "w") as f:
        yaml.dump(custom, f)

    profile = load_profile("_test_user_profile")
    assert get_directive(profile, "sparring") == "User sparring directive"


def test_builtin_profiles_listed():
    result = list_profiles()
    assert "default" in result["builtin"]
    assert "strict" in result["builtin"]
    assert "lightweight" in result["builtin"]


def test_builtin_default_path():
    path = get_profile_path("default")
    assert path is not None
    assert path.name == "default.yaml"
    assert path.is_file()


def test_nonexistent_returns_none():
    assert get_profile_path("nonexistent_xyz") is None


def test_default_has_all_agent_directives():
    profile = load_profile("default")
    for agent in ("sparring", "review", "architecture"):
        assert len(get_directive(profile, agent)) > 0


def test_strict_has_all_agent_directives():
    profile = load_profile("strict")
    for agent in ("sparring", "review", "architecture"):
        assert len(get_directive(profile, agent)) > 0


def test_lightweight_has_all_agent_directives():
    profile = load_profile("lightweight")
    for agent in ("sparring", "review", "architecture"):
        assert len(get_directive(profile, agent)) > 0


def test_profiles_differ():
    default_sparring = get_directive(load_profile("default"), "sparring")
    strict_sparring = get_directive(load_profile("strict"), "sparring")
    lightweight_sparring = get_directive(load_profile("lightweight"), "sparring")

    assert default_sparring != strict_sparring
    assert default_sparring != lightweight_sparring
    assert strict_sparring != lightweight_sparring


def test_all_yamls_parse():
    for yaml_file in BUILTIN_DIR.glob("*.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "name" in data
        assert "description" in data
        assert "directives" in data
        assert isinstance(data["directives"], dict)


def test_profiles_list():
    runner = CliRunner()
    result = runner.invoke(cli, ["profiles", "list"])
    assert result.exit_code == 0
    assert "default" in result.output
    assert "strict" in result.output
    assert "lightweight" in result.output


def test_profiles_show_default():
    runner = CliRunner()
    result = runner.invoke(cli, ["profiles", "show", "default"])
    assert result.exit_code == 0
    assert "directives" in result.output
    assert "sparring" in result.output


def test_profiles_show_nonexistent():
    runner = CliRunner()
    result = runner.invoke(cli, ["profiles", "show", "nonexistent_xyz"])
    assert result.exit_code != 0


def test_profiles_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["profiles", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "show" in result.output
    assert "create" in result.output


def test_run_help_shows_profile_option():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--profile" in result.output


def test_remediate_help_shows_profile_option():
    runner = CliRunner()
    result = runner.invoke(cli, ["remediate", "--help"])
    assert result.exit_code == 0
    assert "--profile" in result.output


@pytest.fixture
def profile_create_teardown():
    user_dir = Path.home() / ".config" / "arch-review" / "profiles"
    created_file = user_dir / "_test_create_profile.yaml"
    yield created_file
    if created_file.exists():
        created_file.unlink()


def test_create_from_default(profile_create_teardown):
    runner = CliRunner()
    created_file = profile_create_teardown
    result = runner.invoke(cli, ["profiles", "create", "_test_create_profile", "--from", "default"])
    assert result.exit_code == 0
    assert created_file.exists()
    with open(created_file) as f:
        data = yaml.safe_load(f)
    assert "directives" in data
    assert "sparring" in data["directives"]


def test_create_already_exists_fails(profile_create_teardown):
    user_dir = Path.home() / ".config" / "arch-review" / "profiles"
    created_file = profile_create_teardown
    user_dir.mkdir(parents=True, exist_ok=True)
    created_file.write_text("existing")
    runner = CliRunner()
    result = runner.invoke(cli, ["profiles", "create", "_test_create_profile", "--from", "default"])
    assert result.exit_code != 0
    assert "already exists" in result.output
