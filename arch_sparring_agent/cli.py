import os

import click

from .config import DEFAULT_REGION, MODEL_ID
from .orchestrator import ReviewOrchestrator


@click.command()
@click.option(
    "--documents-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing markdown documents (problem statement, goals, NFRs, ADRs)",
)
@click.option(
    "--templates-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing CloudFormation templates (can also point to CDK cdk.out/ directory)",
)
@click.option(
    "--diagrams-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing architecture diagrams (images)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False),
    help="Output file for the full review session (e.g., review.md)",
)
@click.option(
    "--model", default=MODEL_ID, help=f"Bedrock model ID (default: {MODEL_ID} - Nova 2 Lite)"
)
@click.option("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
@click.option(
    "--interactive/--no-interactive", default=True, help="Interactive mode (default: enabled)"
)
def main(documents_dir, templates_dir, diagrams_dir, output, model, region, interactive):
    """
    Architecture Review Sparring Partner CLI

    A multi-agent system that conducts comprehensive architecture reviews by
    analyzing requirements documents, CloudFormation templates (including CDK
    synthesized output), and architecture diagrams.

    Example usage:

        arch-review \\
            --documents-dir ./docs \\
            --templates-dir ./templates \\
            --diagrams-dir ./diagrams \\
            -o review.md
    """
    try:
        # Set AWS region environment variable
        os.environ["AWS_REGION"] = region

        # Create orchestrator
        orchestrator = ReviewOrchestrator(
            documents_dir=documents_dir,
            templates_dir=templates_dir,
            diagrams_dir=diagrams_dir,
            model_id=model,
            region=region,
        )

        # Run the review
        result = orchestrator.run_review(interactive=interactive)

        # Save full session to file if requested
        if output:
            full_session = result.get("full_session", str(result["review"]))
            with open(output, "w") as f:
                f.write(full_session)
            click.echo(f"\n✓ Full session saved to {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e


if __name__ == "__main__":
    main()
