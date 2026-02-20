"""Profile loading and directive resolution for customizable review behavior."""

from pathlib import Path

import yaml

from .exceptions import ConfigurationError

BUILTIN_DIR = Path(__file__).parent / "profiles"
USER_DIR = Path.home() / ".config" / "arch-review" / "profiles"


def _project_dir() -> Path:
    """Return the project-level profiles directory (evaluated at call time)."""
    return Path.cwd() / ".arch-review" / "profiles"


def _search_order() -> list[Path]:
    """Return profile search directories: project -> user -> built-in."""
    return [_project_dir(), USER_DIR, BUILTIN_DIR]


def load_profile(name: str = "default") -> dict:
    """Load a profile by name from the first matching directory.

    Resolution order: project (.arch-review/profiles/) -> user (~/.config/arch-review/profiles/)
    -> built-in (package).

    Returns the parsed profile dict.
    """
    for directory in _search_order():
        path = directory / f"{name}.yaml"
        if path.is_file():
            with open(path) as f:
                return yaml.safe_load(f) or {}

    available = [p.stem for p in BUILTIN_DIR.glob("*.yaml")]
    raise ConfigurationError(
        f"Profile '{name}' not found. Available built-in profiles: {', '.join(available)}"
    )


def get_directive(profile: dict | None, agent_name: str) -> str:
    """Return the directive for an agent from a loaded profile.

    Returns empty string if profile is None or the profile has no
    directive for the given agent.
    """
    if profile is None:
        return ""
    directives = profile.get("directives", {})
    return directives.get(agent_name, "")


def list_profiles() -> dict[str, list[str]]:
    """List available profiles grouped by source."""
    result: dict[str, list[str]] = {"builtin": [], "user": [], "project": []}
    for label, directory in [
        ("builtin", BUILTIN_DIR),
        ("user", USER_DIR),
        ("project", _project_dir()),
    ]:
        if directory.is_dir():
            result[label] = sorted(p.stem for p in directory.glob("*.yaml"))
    return result


def get_profile_path(name: str) -> Path | None:
    """Return the path to a profile file, or None if not found."""
    for directory in _search_order():
        path = directory / f"{name}.yaml"
        if path.is_file():
            return path
    return None
