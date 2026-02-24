"""CLI entry point for architecture review tool."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import click

from . import deploy as _deploy
from . import kb as _kb
from . import profiles as _profiles
from . import remediate as _remediate
from . import run as _run
from .common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REMEDIATION_FILE,
    DEFAULT_REVIEW_FILE,
    DEFAULT_STATE_FILE,
    EXIT_ERROR,
    EXIT_HIGH_RISK,
    EXIT_MEDIUM_RISK,
    EXIT_SUCCESS,
    configure_logging,
    get_output_dir,
    get_verdict_and_exit_code,
    load_shared_config,
)


def get_version() -> str:
    """Get package version from metadata."""
    try:
        return version("arch-sparring-agent")
    except PackageNotFoundError:
        return "unknown"


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


cli.add_command(_deploy.deploy)
cli.add_command(_deploy.destroy)
cli.add_command(_kb.kb)
cli.add_command(_profiles.profiles)
cli.add_command(_remediate.remediate)
cli.add_command(_run.run)

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REMEDIATION_FILE",
    "DEFAULT_REVIEW_FILE",
    "DEFAULT_STATE_FILE",
    "EXIT_ERROR",
    "EXIT_HIGH_RISK",
    "EXIT_MEDIUM_RISK",
    "EXIT_SUCCESS",
    "cli",
    "configure_logging",
    "get_output_dir",
    "get_verdict_and_exit_code",
    "load_shared_config",
]
