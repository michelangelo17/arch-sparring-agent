"""CLI entry point for architecture review tool."""

import os

import click

from .config import DEFAULT_REGION, MODEL_ID
from .orchestrator import ReviewOrchestrator


@click.command()
@click.option(
    "--documents-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing markdown requirements documents",
)
@click.option(
    "--templates-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing CloudFormation templates or cdk.out/",
)
@click.option(
    "--diagrams-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing architecture diagrams (PNG, JPEG)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False),
    help="Output file for full review session",
)
@click.option(
    "--model",
    default=MODEL_ID,
    help=f"Bedrock model ID (default: {MODEL_ID})",
)
@click.option(
    "--region",
    default=DEFAULT_REGION,
    help=f"AWS region (default: {DEFAULT_REGION})",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Enable Q&A and sparring phases (default: enabled)",
)
def main(documents_dir, templates_dir, diagrams_dir, output, model, region, interactive):
    """
    Architecture Review Sparring Partner

    Analyzes requirements, CloudFormation templates, and diagrams.
    Challenges architectural decisions through interactive sparring.
    """
    try:
        os.environ["AWS_REGION"] = region

        orchestrator = ReviewOrchestrator(
            documents_dir=documents_dir,
            templates_dir=templates_dir,
            diagrams_dir=diagrams_dir,
            model_id=model,
            region=region,
        )

        result = orchestrator.run_review(interactive=interactive)

        if output:
            full_session = result.get("full_session", str(result["review"]))
            with open(output, "w") as f:
                f.write(full_session)
            click.echo(f"\n✓ Session saved to {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e


if __name__ == "__main__":
    main()
