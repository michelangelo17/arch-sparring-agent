"""Remediate CLI command."""

from __future__ import annotations

import json
import os
import sys

import click

from ..agents.remediation_agent import create_remediation_agent, run_remediation
from ..config import DEFAULT_MODEL, DEFAULT_REGION, SUPPORTED_MODELS, create_model
from ..exceptions import ArchReviewError
from ..state import ReviewState
from . import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REMEDIATION_FILE,
    DEFAULT_STATE_FILE,
    EXIT_ERROR,
    cli,
    configure_logging,
    get_env_or_default,
    get_output_dir,
)


@cli.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
    help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
)
@click.option(
    "--no-output",
    is_flag=True,
    default=False,
    help="Don't save remediation notes to file",
)
@click.option(
    "--model",
    type=click.Choice(list(SUPPORTED_MODELS.keys()), case_sensitive=False),
    default=lambda: get_env_or_default("ARCH_REVIEW_MODEL", DEFAULT_MODEL),
    help=f"Model to use (default: {DEFAULT_MODEL})",
)
@click.option(
    "--region",
    default=lambda: get_env_or_default("AWS_REGION", DEFAULT_REGION),
    help=f"AWS region (default: {DEFAULT_REGION})",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")
def remediate(output_dir, no_output, model, region, verbose):
    """Discuss and resolve previous review findings.

    Loads state from a previous review and starts an interactive session
    to work through gaps, risks, and recommendations.
    """
    configure_logging(verbose)
    os.environ["AWS_REGION"] = region

    out_path = get_output_dir(output_dir)
    state_path = out_path / DEFAULT_STATE_FILE

    if not state_path.exists():
        click.echo(f"Error: State file not found: {state_path}", err=True)
        click.echo("Run a review first (arch-review run) to generate a state file.", err=True)
        sys.exit(EXIT_ERROR)

    try:
        state = ReviewState.from_file(state_path)
        click.echo(f"Loaded state from: {state_path}")

        bedrock_model = create_model(model)

        agent = create_remediation_agent(state=state, model=bedrock_model, region=region)
        notes = run_remediation(agent, state, output_fn=click.echo)

        if not no_output:
            remediation_path = out_path / DEFAULT_REMEDIATION_FILE
            remediation_path.write_text(notes)
            click.echo(f"\nRemediation notes saved to: {remediation_path}")

    except ArchReviewError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)
