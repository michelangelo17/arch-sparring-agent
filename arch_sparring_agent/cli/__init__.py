"""CLI entry point for architecture review tool."""

import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click

from ..exceptions import ConfigurationError
from ..infra import SharedConfig, load_from_ssm
from ..state import extract_verdict

# Exit codes
EXIT_SUCCESS = 0
EXIT_HIGH_RISK = 1
EXIT_MEDIUM_RISK = 2
EXIT_ERROR = 3

# Defaults
DEFAULT_OUTPUT_DIR = ".arch-review"
DEFAULT_REVIEW_FILE = "review.md"
DEFAULT_STATE_FILE = "state.json"
DEFAULT_REMEDIATION_FILE = "remediation-notes.md"


def get_env_or_default(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


def get_version() -> str:
    """Get package version from metadata."""
    try:
        return version("arch-sparring-agent")
    except PackageNotFoundError:
        return "unknown"


def _configure_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )
    logging.getLogger("arch_sparring_agent").setLevel(level)


def _get_output_dir(output_dir: str | None) -> Path:
    """Get output directory, creating if needed."""
    path = Path(output_dir or DEFAULT_OUTPUT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_verdict_and_exit_code(review_text: str) -> tuple[str, int]:
    """Map verdict string to exit code."""
    verdict = extract_verdict(review_text)

    if verdict == "FAIL":
        return "FAIL", EXIT_HIGH_RISK
    if verdict == "PASS WITH CONCERNS":
        return "PASS WITH CONCERNS", EXIT_MEDIUM_RISK
    return "PASS", EXIT_SUCCESS


def _load_shared_config(region: str) -> SharedConfig:
    """Load shared config from SSM, exiting with a helpful message on failure."""
    try:
        return load_from_ssm(region)
    except ConfigurationError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


@click.group()
@click.version_option(version=get_version(), prog_name="arch-review")
def cli():
    """Architecture Review Sparring Partner

    \b
    Workflow:
      1. arch-review deploy          Deploy shared infra (once per account)
      2. arch-review run             Run architecture review
      3. arch-review remediate       Discuss previous findings
      4. arch-review destroy         Tear down shared infra
    """


# Import submodules to trigger command registration
from . import deploy as _deploy  # noqa: E402, F401
from . import kb as _kb  # noqa: E402, F401
from . import profiles_cmd as _profiles_cmd  # noqa: E402, F401
from . import remediate as _remediate  # noqa: E402, F401
from . import run as _run  # noqa: E402, F401
