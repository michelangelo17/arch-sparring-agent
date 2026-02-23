"""Run CLI command."""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from ..config import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_LEVEL,
    DEFAULT_REGION,
    REASONING_LEVELS,
    SUPPORTED_MODELS,
)
from ..exceptions import ArchReviewError
from ..extraction import extract_state_from_review
from ..infra import SharedConfig
from ..orchestrator import ReviewOrchestrator
from ..profiles import load_profile
from ..state import ReviewState
from . import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REMEDIATION_FILE,
    DEFAULT_REVIEW_FILE,
    DEFAULT_STATE_FILE,
    EXIT_ERROR,
    cli,
    configure_logging,
    get_env_or_default,
    get_output_dir,
    get_verdict_and_exit_code,
    load_shared_config,
)


def _archive_previous(output_dir: Path) -> None:
    """Archive existing outputs."""
    review_file = output_dir / DEFAULT_REVIEW_FILE
    state_file = output_dir / DEFAULT_STATE_FILE

    if not review_file.exists() and not state_file.exists():
        return

    if state_file.exists():
        try:
            state = ReviewState.from_file(state_file)
            date_str = state.timestamp[:10]
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            date_str = datetime.now().strftime("%Y-%m-%d")
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    history_dir = output_dir / "history" / date_str
    if history_dir.exists():
        time_str = datetime.now().strftime("%H%M%S")
        history_dir = output_dir / "history" / f"{date_str}_{time_str}"

    history_dir.mkdir(parents=True, exist_ok=True)

    for filename in [DEFAULT_REVIEW_FILE, DEFAULT_STATE_FILE, DEFAULT_REMEDIATION_FILE]:
        src = output_dir / filename
        if src.exists():
            shutil.move(str(src), str(history_dir / filename))

    click.echo(f"Archived previous review to: {history_dir}")


@cli.command()
@click.option(
    "--documents-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_DOCUMENTS_DIR", ""),
    help="Directory containing markdown requirements documents",
)
@click.option(
    "--templates-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_TEMPLATES_DIR", ""),
    help="Directory containing CloudFormation templates",
)
@click.option(
    "--diagrams-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_DIAGRAMS_DIR", ""),
    help="Directory containing architecture diagrams",
)
@click.option(
    "--source-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_SOURCE_DIR", ""),
    help="Directory containing Lambda/application source code",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
    help=f"Output directory for all files (default: {DEFAULT_OUTPUT_DIR})",
)
@click.option("--no-history", is_flag=True, default=False, help="Don't archive previous reviews")
@click.option("--no-state", is_flag=True, default=False, help="Don't save state file after review")
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
@click.option(
    "--profile",
    "profile_name",
    default="default",
    help="Review profile (e.g. strict, lightweight, or custom name)",
)
@click.option(
    "--reasoning-level",
    type=click.Choice(REASONING_LEVELS, case_sensitive=False),
    default=lambda: get_env_or_default("ARCH_REVIEW_REASONING_LEVEL", DEFAULT_REASONING_LEVEL),
    help=f"Reasoning effort for analysis agents (default: {DEFAULT_REASONING_LEVEL})",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")
def run(
    documents_dir,
    templates_dir,
    diagrams_dir,
    source_dir,
    output_dir,
    no_history,
    no_state,
    model,
    region,
    profile_name,
    reasoning_level,
    verbose,
):
    """Run an interactive architecture review.

    \b
    Requires infrastructure to be deployed first (arch-review deploy).

    \b
    Output Structure:
      .arch-review/
      +-- review.md           # Latest review
      +-- state.json          # Latest state (for remediation)
      +-- remediation-notes.md
      +-- history/            # Archived previous reviews

    \b
    Examples:
      arch-review run --documents-dir ./docs --templates-dir ./cdk.out --diagrams-dir ./diagrams
      arch-review run --profile strict --reasoning-level medium
    """
    configure_logging(verbose)
    os.environ["AWS_REGION"] = region

    profile_data = load_profile(profile_name)

    if not documents_dir or not Path(documents_dir).is_dir():
        raise click.UsageError("--documents-dir is required and must exist")
    if not templates_dir or not Path(templates_dir).is_dir():
        raise click.UsageError("--templates-dir is required and must exist")
    if not diagrams_dir or not Path(diagrams_dir).is_dir():
        raise click.UsageError("--diagrams-dir is required and must exist")
    if source_dir and not Path(source_dir).is_dir():
        raise click.UsageError("--source-dir must exist if provided")

    shared_config = load_shared_config(region)

    sys.exit(
        _run_review(
            documents_dir=documents_dir,
            templates_dir=templates_dir,
            diagrams_dir=diagrams_dir,
            source_dir=source_dir,
            output_dir=output_dir,
            no_state=no_state,
            should_archive=not no_history,
            model=model,
            shared_config=shared_config,
            reasoning_level=reasoning_level,
            profile=profile_data,
        )
    )


def _run_review(
    documents_dir: str,
    templates_dir: str,
    diagrams_dir: str,
    source_dir: str | None,
    output_dir: str,
    no_state: bool,
    should_archive: bool,
    model: str,
    shared_config: SharedConfig,
    reasoning_level: str = DEFAULT_REASONING_LEVEL,
    profile: dict[str, Any] | None = None,
) -> int:
    """Execute the review and handle output. Returns an exit code."""
    out_path = get_output_dir(output_dir)

    if should_archive:
        _archive_previous(out_path)

    try:
        orchestrator = ReviewOrchestrator.create(
            documents_dir=documents_dir,
            templates_dir=templates_dir,
            diagrams_dir=diagrams_dir,
            shared_config=shared_config,
            source_dir=source_dir or None,
            model_name=model,
            output_fn=click.echo,
            reasoning_level=reasoning_level,
            profile=profile,
        )

        result = orchestrator.run_review()
        verdict, exit_code = get_verdict_and_exit_code(result.review)

        review_path = out_path / DEFAULT_REVIEW_FILE
        review_path.write_text(result.full_session)
        click.echo(f"\nReview saved to: {review_path}")

        if not no_state:
            state = extract_state_from_review(result)
            state_path = out_path / DEFAULT_STATE_FILE
            state.save(state_path)
            click.echo(f"State saved to: {state_path}")

        click.echo(f"\nVerdict: {verdict}")
        return exit_code

    except click.UsageError:
        raise
    except ArchReviewError as e:
        click.echo(f"Error: {e}", err=True)
        return EXIT_ERROR
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        click.echo(traceback.format_exc(), err=True)
        return EXIT_ERROR
