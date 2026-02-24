"""Shared constants and helpers for CLI commands."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from ..config import DEFAULT_MODEL, DEFAULT_REGION, SUPPORTED_MODELS
from ..exceptions import ConfigurationError
from ..infra import SharedConfig, load_from_ssm
from ..review.extraction import extract_verdict

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


def common_options(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Shared ``--region`` and ``-v/--verbose`` options."""
    fn = click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")(fn)
    fn = click.option(
        "--region",
        default=lambda: os.environ.get("AWS_REGION", DEFAULT_REGION),
        help=f"AWS region (default: {DEFAULT_REGION})",
    )(fn)
    return fn


def model_option(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Shared ``--model`` option."""
    fn = click.option(
        "--model",
        type=click.Choice(list(SUPPORTED_MODELS.keys()), case_sensitive=False),
        default=lambda: os.environ.get("ARCH_REVIEW_MODEL", DEFAULT_MODEL),
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )(fn)
    return fn


def configure_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )
    logging.getLogger("arch_sparring_agent").setLevel(level)


def get_output_dir(output_dir: str | None) -> Path:
    """Get output directory, creating if needed."""
    path = Path(output_dir or DEFAULT_OUTPUT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_verdict_and_exit_code(review_text: str) -> tuple[str, int]:
    """Map verdict string to exit code."""
    verdict = extract_verdict(review_text)

    if verdict == "FAIL":
        return "FAIL", EXIT_HIGH_RISK
    if verdict == "PASS WITH CONCERNS":
        return "PASS WITH CONCERNS", EXIT_MEDIUM_RISK
    return "PASS", EXIT_SUCCESS


def load_shared_config(region: str) -> SharedConfig:
    """Load shared config from SSM, exiting with a helpful message on failure."""
    try:
        return load_from_ssm(region)
    except ConfigurationError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)
