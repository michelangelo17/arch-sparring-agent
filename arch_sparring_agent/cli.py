"""CLI entry point for architecture review tool."""

import json
import os
import sys
from pathlib import Path

import click

from .config import DEFAULT_REGION, MODEL_ID
from .orchestrator import ReviewOrchestrator

# Exit codes
EXIT_SUCCESS = 0
EXIT_HIGH_RISK = 1
EXIT_MEDIUM_RISK = 2
EXIT_ERROR = 3


def get_env_or_default(env_var: str, default: str) -> str:
    """Get value from environment variable or return default."""
    return os.environ.get(env_var, default)


@click.command()
@click.option(
    "--documents-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_DOCUMENTS_DIR", ""),
    help="Directory containing markdown requirements documents (env: ARCH_REVIEW_DOCUMENTS_DIR)",
)
@click.option(
    "--templates-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_TEMPLATES_DIR", ""),
    help="Directory containing CloudFormation templates (env: ARCH_REVIEW_TEMPLATES_DIR)",
)
@click.option(
    "--diagrams-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_DIAGRAMS_DIR", ""),
    help="Directory containing architecture diagrams (env: ARCH_REVIEW_DIAGRAMS_DIR)",
)
@click.option(
    "--source-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=lambda: get_env_or_default("ARCH_REVIEW_SOURCE_DIR", ""),
    help="Directory containing Lambda/application source code (env: ARCH_REVIEW_SOURCE_DIR)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False),
    default=lambda: get_env_or_default("ARCH_REVIEW_OUTPUT", ""),
    help="Output file for review session (env: ARCH_REVIEW_OUTPUT)",
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
    if not documents_dir:
        raise click.UsageError("--documents-dir is required")
    if not templates_dir:
        raise click.UsageError("--templates-dir is required")
    if not diagrams_dir:
        raise click.UsageError("--diagrams-dir is required")

    ci_mode = ci or json_output
    output = output or None

    try:
        os.environ["AWS_REGION"] = region

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

        # Determine verdict from review (CI mode outputs explicit verdict)
        review_lower = result["review"].lower()
        has_high_impact = "impact: high" in review_lower or "impact high" in review_lower

        if "verdict" in review_lower:
            if "fail" in review_lower.split("verdict")[-1][:50]:
                verdict = "FAIL"
            elif "pass with concerns" in review_lower:
                verdict = "PASS WITH CONCERNS"
            else:
                verdict = "PASS"
        else:
            # Fallback for interactive mode
            if any(term in review_lower for term in ["critical", "severe", "major vulnerability"]):
                verdict = "FAIL"
            elif has_high_impact:
                verdict = "PASS WITH CONCERNS"
            else:
                verdict = "PASS"

        # Strict mode: any high impact = FAIL
        if strict and has_high_impact:
            verdict = "FAIL"

        # Determine exit code based on verdict
        exit_code = EXIT_SUCCESS
        if verdict == "FAIL":
            exit_code = EXIT_HIGH_RISK

        # Output results
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
            if output:
                full_session = result.get("full_session", result["review"])
                Path(output).write_text(full_session)
                click.echo(f"\n✓ Session saved to {output}")

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
