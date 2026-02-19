"""Profile loading and directive resolution for customizable review behavior."""

from pathlib import Path

import yaml

from .exceptions import ConfigurationError

BUILTIN_DIR = Path(__file__).parent / "profiles"
USER_DIR = Path.home() / ".config" / "arch-review" / "profiles"
PROJECT_DIR = Path.cwd() / ".arch-review" / "profiles"

_SEARCH_ORDER = [PROJECT_DIR, USER_DIR, BUILTIN_DIR]

_loaded: dict | None = None


def load_profile(name: str = "default") -> None:
    """Load a profile by name from the first matching directory.

    Resolution order: project (.arch-review/profiles/) -> user (~/.config/arch-review/profiles/)
    -> built-in (package).
    """
    global _loaded

    for directory in _SEARCH_ORDER:
        path = directory / f"{name}.yaml"
        if path.is_file():
            with open(path) as f:
                _loaded = yaml.safe_load(f) or {}
            return

    available = [p.stem for p in BUILTIN_DIR.glob("*.yaml")]
    raise ConfigurationError(
        f"Profile '{name}' not found. Available built-in profiles: {', '.join(available)}"
    )


def get_directive(agent_name: str) -> str:
    """Return the directive for an agent from the currently loaded profile.

    Returns empty string if no profile is loaded or the profile has no
    directive for the given agent.
    """
    if _loaded is None:
        return ""
    directives = _loaded.get("directives", {})
    return directives.get(agent_name, "")


def list_profiles() -> dict[str, list[str]]:
    """List available profiles grouped by source."""
    result: dict[str, list[str]] = {"builtin": [], "user": [], "project": []}
    for label, directory in [
        ("builtin", BUILTIN_DIR),
        ("user", USER_DIR),
        ("project", PROJECT_DIR),
    ]:
        if directory.is_dir():
            result[label] = sorted(p.stem for p in directory.glob("*.yaml"))
    return result


def get_profile_path(name: str) -> Path | None:
    """Return the path to a profile file, or None if not found."""
    for directory in _SEARCH_ORDER:
        path = directory / f"{name}.yaml"
        if path.is_file():
            return path
    return None


def reset() -> None:
    """Reset loaded state (for testing)."""
    global _loaded
    _loaded = None
