"""CLI entry point for architecture review tool."""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import click

from .agents.remediation_agent import create_remediation_agent, run_remediation
from .config import DEFAULT_REGION, MODEL_ID, get_inference_profile_arn
from .orchestrator import ReviewOrchestrator
from .state import ReviewState, extract_state_from_review

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


def _get_output_dir(output_dir: str | None) -> Path:
    """Get output directory, creating if needed."""
    path = Path(output_dir or DEFAULT_OUTPUT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _archive_previous(output_dir: Path) -> None:
    """Archive existing outputs."""
    review_file = output_dir / DEFAULT_REVIEW_FILE
    state_file = output_dir / DEFAULT_STATE_FILE

    if not review_file.exists() and not state_file.exists():
        return

    # Use date from state file if available, else current date
    if state_file.exists():
        try:
            state = ReviewState.from_file(state_file)
            date_str = state.timestamp[:10]
        except Exception:
            date_str = datetime.now().strftime("%Y-%m-%d")
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Add time suffix if folder exists
    history_dir = output_dir / "history" / date_str
    if history_dir.exists():
        time_str = datetime.now().strftime("%H%M%S")
        history_dir = output_dir / "history" / f"{date_str}_{time_str}"

    history_dir.mkdir(parents=True, exist_ok=True)

    # Move existing files to history
    for filename in [DEFAULT_REVIEW_FILE, DEFAULT_STATE_FILE, DEFAULT_REMEDIATION_FILE]:
        src = output_dir / filename
        if src.exists():
            shutil.move(str(src), str(history_dir / filename))

    click.echo(f"📁 Archived previous review to: {history_dir}")


def _extract_verdict(review_text: str, strict: bool = False) -> tuple[str, int]:
    """Extract verdict from review text and determine exit code."""
    text_lower = review_text.lower()
    has_high_impact = "impact: high" in text_lower or "impact high" in text_lower

    # Check explicit verdict
    if "verdict" in text_lower:
        after_verdict = text_lower.split("verdict")[-1][:50]
        if "fail" in after_verdict:
            return "FAIL", EXIT_HIGH_RISK
        if "pass with concerns" in after_verdict:
            verdict = "FAIL" if strict and has_high_impact else "PASS WITH CONCERNS"
            return verdict, EXIT_HIGH_RISK if verdict == "FAIL" else EXIT_SUCCESS
        return "PASS", EXIT_SUCCESS

    # Fallback: infer from content
    critical_terms = ["critical", "severe", "major vulnerability"]
    if any(term in text_lower for term in critical_terms):
        return "FAIL", EXIT_HIGH_RISK
    if has_high_impact:
        verdict = "FAIL" if strict else "PASS WITH CONCERNS"
        return verdict, EXIT_HIGH_RISK if verdict == "FAIL" else EXIT_SUCCESS

    return "PASS", EXIT_SUCCESS


@click.command()
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
@click.option(
    "--no-history",
    is_flag=True,
    default=False,
    help="Don't archive previous reviews (default in CI mode)",
)
@click.option(
    "--keep-history",
    is_flag=True,
    default=False,
    help="Archive previous reviews even in CI mode",
)
@click.option(
    "--no-state",
    is_flag=True,
    default=False,
    help="Don't save state file after review",
)
@click.option(
    "--remediate",
    is_flag=True,
    default=False,
    help="Enter remediation mode to discuss previous findings",
)
@click.option(
    "--no-remediation-output",
    is_flag=True,
    default=False,
    help="Don't save remediation notes to file",
)
@click.option(
    "--model",
    default=lambda: get_env_or_default("ARCH_REVIEW_MODEL", MODEL_ID),
    help=f"Bedrock model ID (default: {MODEL_ID})",
)
@click.option(
    "--region",
    default=lambda: get_env_or_default("AWS_REGION", DEFAULT_REGION),
    help=f"AWS region (default: {DEFAULT_REGION})",
)
@click.option(
    "--ci",
    is_flag=True,
    default=lambda: get_env_or_default("CI", "").lower() in ("true", "1", "yes"),
    help="CI/CD mode: non-interactive, no history by default",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output results as JSON (implies --ci)",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Strict mode: any High impact risk fails",
)
@click.option(
    "--skip-policy-check",
    is_flag=True,
    default=False,
    help="Skip policy engine enforcement (development only - NOT recommended for production)",
)
def main(
    documents_dir,
    templates_dir,
    diagrams_dir,
    source_dir,
    output_dir,
    no_history,
    keep_history,
    no_state,
    remediate,
    no_remediation_output,
    model,
    region,
    ci,
    json_output,
    strict,
    skip_policy_check,
):
    """
    Architecture Review Sparring Partner

    Analyzes requirements, CloudFormation templates, diagrams, and source code.
    Outputs to .arch-review/ folder with automatic history archiving.

    \b
    Modes:
      Default       - Run full architecture review (archives previous)
      --remediate   - Discuss and resolve previous review findings
      --ci          - Non-interactive CI/CD mode (no archive by default)

    \b
    Output Structure:
      .arch-review/
      ├── review.md           # Latest review
      ├── state.json          # Latest state (for remediation)
      ├── remediation-notes.md
      └── history/            # Archived previous reviews
          └── 2025-12-09/

    \b
    Examples:
      arch-review --documents-dir ./docs --templates-dir ./cdk.out --diagrams-dir ./diagrams
      arch-review --remediate
      arch-review --ci --keep-history  # CI with history archiving
    """
    os.environ["AWS_REGION"] = region
    ci_mode = ci or json_output

    # Determine whether to archive history
    should_archive = keep_history or (not ci_mode and not no_history)

    # Remediation mode
    if remediate:
        _run_remediation_mode(
            output_dir=output_dir,
            no_remediation_output=no_remediation_output,
            model=model,
            region=region,
        )
        return

    # Review mode - validate required dirs
    if not documents_dir or not Path(documents_dir).is_dir():
        raise click.UsageError("--documents-dir is required and must exist")
    if not templates_dir or not Path(templates_dir).is_dir():
        raise click.UsageError("--templates-dir is required and must exist")
    if not diagrams_dir or not Path(diagrams_dir).is_dir():
        raise click.UsageError("--diagrams-dir is required and must exist")
    if source_dir and not Path(source_dir).is_dir():
        raise click.UsageError("--source-dir must exist if provided")

    _run_review_mode(
        documents_dir=documents_dir,
        templates_dir=templates_dir,
        diagrams_dir=diagrams_dir,
        source_dir=source_dir,
        output_dir=output_dir,
        no_state=no_state,
        should_archive=should_archive,
        model=model,
        region=region,
        ci_mode=ci_mode,
        json_output=json_output,
        strict=strict,
        skip_policy_check=skip_policy_check,
    )


def _run_remediation_mode(
    output_dir: str,
    no_remediation_output: bool,
    model: str,
    region: str,
):
    """Run remediation mode."""
    out_path = _get_output_dir(output_dir)
    state_path = out_path / DEFAULT_STATE_FILE

    if not state_path.exists():
        click.echo(f"Error: State file not found: {state_path}", err=True)
        click.echo("Run a review first to generate a state file.", err=True)
        sys.exit(EXIT_ERROR)

    try:
        state = ReviewState.from_file(state_path)
        click.echo(f"✓ Loaded state from: {state_path}")

        inference_profile = get_inference_profile_arn(model, region)
        model_id = inference_profile or model

        agent = create_remediation_agent(state=state, model_id=model_id, region=region)
        notes = run_remediation(agent, state)

        if not no_remediation_output:
            remediation_path = out_path / DEFAULT_REMEDIATION_FILE
            remediation_path.write_text(notes)
            click.echo(f"\n✓ Remediation notes saved to: {remediation_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


def _run_review_mode(
    documents_dir: str,
    templates_dir: str,
    diagrams_dir: str,
    source_dir: str | None,
    output_dir: str,
    no_state: bool,
    should_archive: bool,
    model: str,
    region: str,
    ci_mode: bool,
    json_output: bool,
    strict: bool,
    skip_policy_check: bool = False,
):
    """Run review mode."""
    out_path = _get_output_dir(output_dir)

    # Archive previous review if enabled
    if should_archive:
        _archive_previous(out_path)

    try:
        orchestrator = ReviewOrchestrator(
            documents_dir=documents_dir,
            templates_dir=templates_dir,
            diagrams_dir=diagrams_dir,
            source_dir=source_dir or None,
            model_id=model,
            region=region,
            ci_mode=ci_mode,
            skip_policy_check=skip_policy_check,
        )

        result = orchestrator.run_review()
        verdict, exit_code = _extract_verdict(result["review"], strict=strict)

        if json_output:
            json_result = {
                "review": result["review"],
                "requirements_summary": result.get("requirements_summary", ""),
                "architecture_summary": result.get("architecture_summary", ""),
                "gaps": result.get("gaps", ""),
                "risks": result.get("risks", ""),
                "exit_code": exit_code,
                "verdict": verdict,
                "agents_used": result["agents_used"],
            }
            click.echo(json.dumps(json_result, indent=2))
        else:
            # Save review
            review_path = out_path / DEFAULT_REVIEW_FILE
            full_session = result.get("full_session", result["review"])
            review_path.write_text(full_session)
            click.echo(f"\n✓ Review saved to: {review_path}")

            # Save state
            if not no_state:
                state = extract_state_from_review(result)
                state_path = out_path / DEFAULT_STATE_FILE
                state.save(state_path)
                click.echo(f"✓ State saved to: {state_path}")

            if ci_mode:
                click.echo(f"\n📊 Verdict: {verdict}")
                if exit_code != EXIT_SUCCESS:
                    click.echo(f"⚠️  Exiting with code {exit_code}")

        sys.exit(exit_code)

    except click.UsageError:
        raise
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"error": str(e), "exit_code": EXIT_ERROR}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()
