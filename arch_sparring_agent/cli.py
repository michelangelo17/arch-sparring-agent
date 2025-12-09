"""CLI entry point for architecture review tool."""

import json
import os
import sys
from pathlib import Path

import click

from .agents.remediation_agent import create_remediation_agent, run_remediation
from .config import DEFAULT_REGION, MODEL_ID, get_inference_profile_arn
from .orchestrator import ReviewOrchestrator
from .state import ReviewState, extract_state_from_review, get_default_state_path

# Exit codes
EXIT_SUCCESS = 0
EXIT_HIGH_RISK = 1
EXIT_MEDIUM_RISK = 2
EXIT_ERROR = 3

# Defaults
DEFAULT_OUTPUT = "review.md"
DEFAULT_REMEDIATION_OUTPUT = "remediation-notes.md"


def get_env_or_default(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


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
    help="Directory containing markdown requirements documents (env: ARCH_REVIEW_DOCUMENTS_DIR)",
)
@click.option(
    "--templates-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_TEMPLATES_DIR", ""),
    help="Directory containing CloudFormation templates (env: ARCH_REVIEW_TEMPLATES_DIR)",
)
@click.option(
    "--diagrams-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_DIAGRAMS_DIR", ""),
    help="Directory containing architecture diagrams (env: ARCH_REVIEW_DIAGRAMS_DIR)",
)
@click.option(
    "--source-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_SOURCE_DIR", ""),
    help="Directory containing Lambda/application source code (env: ARCH_REVIEW_SOURCE_DIR)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False),
    default=lambda: get_env_or_default("ARCH_REVIEW_OUTPUT", DEFAULT_OUTPUT),
    help=f"Output file for review session (default: {DEFAULT_OUTPUT})",
)
@click.option(
    "--state-file",
    type=click.Path(file_okay=True, dir_okay=False),
    default=None,
    help="State file path (default: {output}.state.json)",
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
    help="Enter remediation mode. Use --state-file to specify a custom state file.",
)
@click.option(
    "--remediation-output",
    type=click.Path(file_okay=True, dir_okay=False),
    default=lambda: get_env_or_default(
        "ARCH_REVIEW_REMEDIATION_OUTPUT", DEFAULT_REMEDIATION_OUTPUT
    ),
    help=f"Output file for remediation notes (default: {DEFAULT_REMEDIATION_OUTPUT})",
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
    help=f"Bedrock model ID (env: ARCH_REVIEW_MODEL, default: {MODEL_ID})",
)
@click.option(
    "--region",
    default=lambda: get_env_or_default("AWS_REGION", DEFAULT_REGION),
    help=f"AWS region (env: AWS_REGION, default: {DEFAULT_REGION})",
)
@click.option(
    "--ci",
    is_flag=True,
    default=lambda: get_env_or_default("CI", "").lower() in ("true", "1", "yes"),
    help="CI/CD mode: non-interactive with structured output (auto-detected from CI env var)",
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
    help="Strict mode: any High impact risk fails, ignoring agent verdict",
)
def main(
    documents_dir,
    templates_dir,
    diagrams_dir,
    source_dir,
    output,
    state_file,
    no_state,
    remediate,
    remediation_output,
    no_remediation_output,
    model,
    region,
    ci,
    json_output,
    strict,
):
    """
    Architecture Review Sparring Partner

    Analyzes requirements, CloudFormation templates, diagrams, and source code.
    In interactive mode, challenges decisions through sparring.
    In CI mode, generates automated risk assessment.
    In remediation mode, helps address findings from previous reviews.

    \b
    Modes:
      Default       - Run full architecture review
      --remediate   - Discuss and resolve previous review findings
      --ci          - Non-interactive CI/CD mode

    \b
    Examples:
      # Run review with defaults
      arch-review --documents-dir ./docs --templates-dir ./cdk.out --diagrams-dir ./diagrams

      # Custom output location
      arch-review ... --output my-review.md

      # Enter remediation mode (uses default state file)
      arch-review --remediate

      # Remediate specific state file
      arch-review --remediate ./old-review.state.json

    \b
    Environment Variables:
      ARCH_REVIEW_DOCUMENTS_DIR  - Documents directory
      ARCH_REVIEW_TEMPLATES_DIR  - Templates directory
      ARCH_REVIEW_DIAGRAMS_DIR   - Diagrams directory
      ARCH_REVIEW_SOURCE_DIR     - Source code directory (optional)
      ARCH_REVIEW_OUTPUT         - Output file path
      ARCH_REVIEW_MODEL          - Bedrock model ID
      AWS_REGION                 - AWS region
      CI                         - Enable CI mode (true/1/yes)
    """
    os.environ["AWS_REGION"] = region

    # Remediation mode
    if remediate:
        _run_remediation_mode(
            state_file=state_file,
            output=output,
            remediation_output=remediation_output,
            no_remediation_output=no_remediation_output,
            model=model,
            region=region,
        )
        return

    # Review mode - validate required dirs exist
    if not documents_dir or not Path(documents_dir).is_dir():
        raise click.UsageError("--documents-dir is required and must exist for review mode")
    if not templates_dir or not Path(templates_dir).is_dir():
        raise click.UsageError("--templates-dir is required and must exist for review mode")
    if not diagrams_dir or not Path(diagrams_dir).is_dir():
        raise click.UsageError("--diagrams-dir is required and must exist for review mode")
    if source_dir and not Path(source_dir).is_dir():
        raise click.UsageError("--source-dir must exist if provided")

    _run_review_mode(
        documents_dir=documents_dir,
        templates_dir=templates_dir,
        diagrams_dir=diagrams_dir,
        source_dir=source_dir,
        output=output,
        state_file=state_file,
        no_state=no_state,
        model=model,
        region=region,
        ci=ci,
        json_output=json_output,
        strict=strict,
    )


def _run_remediation_mode(
    state_file: str | None,
    output: str,
    remediation_output: str,
    no_remediation_output: bool,
    model: str,
    region: str,
):
    """Run remediation mode."""
    # Determine state file path: use explicit --state-file, or derive from --output
    if state_file:
        state_path = Path(state_file)
    else:
        state_path = get_default_state_path(output)

    if not state_path.exists():
        click.echo(f"Error: State file not found: {state_path}", err=True)
        click.echo("Run a review first, or specify a valid state file path.", err=True)
        sys.exit(EXIT_ERROR)

    try:
        state = ReviewState.from_file(state_path)
        click.echo(f"✓ Loaded state from: {state_path}")

        # Get inference profile ARN for model
        inference_profile = get_inference_profile_arn(model, region)
        model_id = inference_profile or model

        # Create remediation agent (handles its own memory setup)
        agent = create_remediation_agent(state=state, model_id=model_id, region=region)

        # Run session
        notes = run_remediation(agent, state)

        # Save notes if enabled
        if not no_remediation_output and remediation_output:
            Path(remediation_output).write_text(notes)
            click.echo(f"\n✓ Remediation notes saved to: {remediation_output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


def _run_review_mode(
    documents_dir: str,
    templates_dir: str,
    diagrams_dir: str,
    source_dir: str | None,
    output: str,
    state_file: str | None,
    no_state: bool,
    model: str,
    region: str,
    ci: bool,
    json_output: bool,
    strict: bool,
):
    """Run review mode."""
    ci_mode = ci or json_output

    try:
        orchestrator = ReviewOrchestrator(
            documents_dir=documents_dir,
            templates_dir=templates_dir,
            diagrams_dir=diagrams_dir,
            source_dir=source_dir or None,
            model_id=model,
            region=region,
            ci_mode=ci_mode,
        )

        result = orchestrator.run_review()
        verdict, exit_code = _extract_verdict(result["review"], strict=strict)

        # Save outputs
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
            # Save review output
            if output:
                full_session = result.get("full_session", result["review"])
                Path(output).write_text(full_session)
                click.echo(f"\n✓ Review saved to: {output}")

            # Save state file
            if not no_state:
                state = extract_state_from_review(result)
                state_path = Path(state_file) if state_file else get_default_state_path(output)
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
