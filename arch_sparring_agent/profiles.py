"""Profile loading and directive resolution for customizable review behavior."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar, overload

import yaml

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

BUILTIN_DIR = Path(__file__).parent / "profiles"
USER_DIR = Path.home() / ".config" / "arch-review" / "profiles"

_EXPECTED_KEYS = {"name", "description", "directives", "settings"}

T = TypeVar("T")


def project_dir() -> Path:
    """Return the project-level profiles directory (evaluated at call time)."""
    return Path.cwd() / ".arch-review" / "profiles"


def _search_order() -> list[Path]:
    """Return profile search directories: project -> user -> built-in."""
    return [project_dir(), USER_DIR, BUILTIN_DIR]


def _validate_profile(data: dict[str, Any], path: Path) -> None:
    """Warn about unexpected keys in a loaded profile."""
    unknown = set(data.keys()) - _EXPECTED_KEYS
    if unknown:
        logger.warning(
            "Profile %s contains unknown keys: %s (expected: %s)",
            path.name,
            ", ".join(sorted(unknown)),
            ", ".join(sorted(_EXPECTED_KEYS)),
        )


def load_profile(name: str = "default") -> dict[str, Any]:
    """Load a profile by name from the first matching directory.

    Resolution order: project (.arch-review/profiles/) -> user (~/.config/arch-review/profiles/)
    -> built-in (package).

    Returns the parsed profile dict.
    """
    for directory in _search_order():
        path = directory / f"{name}.yaml"
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            _validate_profile(data, path)
            return data

    available = [p.stem for p in BUILTIN_DIR.glob("*.yaml")]
    raise ConfigurationError(
        f"Profile '{name}' not found. Available built-in profiles: {', '.join(available)}"
    )


def get_directive(profile: dict[str, Any] | None, agent_name: str) -> str:
    """Return the directive for an agent from a loaded profile.

    Returns empty string if profile is None or the profile has no
    directive for the given agent.
    """
    if profile is None:
        return ""
    directives = profile.get("directives", {})
    return directives.get(agent_name, "")


@overload
def get_setting(profile: dict[str, Any] | None, *keys: str, default: T) -> T: ...


@overload
def get_setting(profile: dict[str, Any] | None, *keys: str) -> Any: ...


def get_setting(profile: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """Retrieve a nested setting value from a loaded profile.

    Walks the ``settings`` sub-dict using *keys* as successive lookups.
    Returns *default* when the profile is ``None`` or the key path is missing.
    """
    if profile is None:
        return default
    current: Any = profile.get("settings", {})
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def list_profiles() -> dict[str, list[str]]:
    """List available profiles grouped by source."""
    result: dict[str, list[str]] = {"builtin": [], "user": [], "project": []}
    for label, directory in [
        ("builtin", BUILTIN_DIR),
        ("user", USER_DIR),
        ("project", project_dir()),
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
