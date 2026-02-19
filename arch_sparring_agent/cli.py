"""CLI entry point for architecture review tool."""

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click

from .agents.remediation_agent import create_remediation_agent, run_remediation
from .config import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_LEVEL,
    DEFAULT_REGION,
    REASONING_LEVELS,
    SUPPORTED_MODELS,
)
from .exceptions import ArchReviewError, ConfigurationError
from .infra import SharedConfig, delete_from_ssm, load_from_ssm, save_to_ssm
from .orchestrator import ReviewOrchestrator
from .profiles import get_profile_path, list_profiles, load_profile
from .state import ReviewState, extract_state_from_review, extract_verdict


def _configure_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )
    logging.getLogger("arch_sparring_agent").setLevel(level)


def get_version() -> str:
    """Get package version from metadata."""
    try:
        return version("arch-sparring-agent")
    except PackageNotFoundError:
        return "unknown"


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


def _get_verdict_and_exit_code(review_text: str, strict: bool = False) -> tuple[str, int]:
    """Map verdict string to exit code, applying strict mode logic."""
    verdict = extract_verdict(review_text)

    if verdict == "FAIL":
        return "FAIL", EXIT_HIGH_RISK

    if verdict == "PASS WITH CONCERNS":
        text_lower = review_text.lower()
        has_high_impact = "impact: high" in text_lower or "impact high" in text_lower
        if strict and has_high_impact:
            return "FAIL", EXIT_HIGH_RISK
        return "PASS WITH CONCERNS", EXIT_MEDIUM_RISK

    return "PASS", EXIT_SUCCESS


def _load_shared_config(region: str) -> SharedConfig:
    """Load shared config from SSM, exiting with a helpful message on failure."""
    try:
        return load_from_ssm(region)
    except ConfigurationError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--region",
    default=lambda: get_env_or_default("AWS_REGION", DEFAULT_REGION),
    help=f"AWS region (default: {DEFAULT_REGION})",
)
@click.option(
    "--gateway-name",
    default="ArchReviewGateway",
    help="Name for the Gateway resource",
)
@click.option(
    "--policy-engine-name",
    default="ArchReviewPolicyEngine",
    help="Name for the Policy Engine resource",
)
@click.option(
    "--with-kb", is_flag=True, default=False, help="Provision a Knowledge Base for WAF content"
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")
def deploy(region, gateway_name, policy_engine_name, with_kb, verbose):
    """Deploy shared infrastructure to an AWS account.

    Creates the Gateway, Policy Engine, and Cedar policies. Writes the
    resulting resource IDs to SSM Parameter Store so that ``arch-review run``
    can discover them automatically.

    Pass --with-kb to also create a Bedrock Knowledge Base backed by S3 and
    OpenSearch Serverless for WAF best-practice content.

    Idempotent — safe to run repeatedly.
    """
    _configure_logging(verbose)
    os.environ["AWS_REGION"] = region

    click.echo(f"Deploying arch-review infrastructure in {region}...")

    gateway_arn, gateway_id, engine_id = _deploy_infra(
        region, gateway_name, policy_engine_name
    )

    kb_id = None
    kb_bucket = None

    if with_kb:
        click.echo("\nProvisioning Knowledge Base...")
        from .kb.infra import setup_knowledge_base

        kb_id, kb_bucket = setup_knowledge_base(region=region)
        click.echo(f"  Knowledge Base ID: {kb_id}")
        click.echo(f"  KB Bucket:         {kb_bucket}")

    config = SharedConfig(
        gateway_id=gateway_id,
        gateway_arn=gateway_arn,
        policy_engine_id=engine_id,
        region=region,
        knowledge_base_id=kb_id,
        kb_bucket_name=kb_bucket,
    )
    save_to_ssm(config)

    click.echo("\nDeployment complete.")
    click.echo(f"  Gateway ID:       {gateway_id}")
    click.echo(f"  Policy Engine ID: {engine_id}")
    if kb_id:
        click.echo(f"  Knowledge Base:   {kb_id}")
    click.echo(f"  Config stored in: SSM {_ssm_param_name()}")


def _deploy_infra(
    region: str, gateway_name: str, policy_engine_name: str
) -> tuple[str, str, str]:
    """Run the existing setup logic and return (gateway_arn, gateway_id, engine_id).

    Raises click.ClickException on failure so the CLI exits cleanly.
    """
    from .gateway import setup_gateway
    from .policy import setup_architecture_review_policies

    gateway_arn, gateway_id = setup_gateway(region=region, gateway_name=gateway_name)
    if not gateway_arn or not gateway_id:
        raise click.ClickException("Gateway setup failed. Check logs with -v for details.")

    engine_id = setup_architecture_review_policies(
        region=region,
        policy_engine_name=policy_engine_name,
        gateway_arn=gateway_arn,
        gateway_name=gateway_name,
    )
    if not engine_id:
        raise click.ClickException(
            "Policy Engine / Cedar policy setup failed. Check logs with -v for details."
        )

    return gateway_arn, gateway_id, engine_id


def _ssm_param_name() -> str:
    from .infra import SSM_PARAMETER_NAME
    return SSM_PARAMETER_NAME


# ---------------------------------------------------------------------------
# destroy
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--region",
    default=lambda: get_env_or_default("AWS_REGION", DEFAULT_REGION),
    help=f"AWS region (default: {DEFAULT_REGION})",
)
@click.option(
    "--confirm",
    is_flag=True,
    default=False,
    help="Required to actually destroy resources",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")
def destroy(region, confirm, verbose):
    """Tear down shared infrastructure.

    Reads config from SSM, deletes Cedar policies, Policy Engine, Gateway,
    and the SSM parameter itself.

    Requires --confirm as a safety guard.
    """
    _configure_logging(verbose)
    os.environ["AWS_REGION"] = region

    if not confirm:
        click.echo(
            "This will destroy all arch-review infrastructure in the account.\n"
            "Re-run with --confirm to proceed."
        )
        return

    config = _load_shared_config(region)

    from .gateway import destroy_gateway
    from .policy import destroy_policy_engine

    click.echo(f"Destroying arch-review infrastructure in {region}...")

    if config.knowledge_base_id or config.kb_bucket_name:
        click.echo("Tearing down Knowledge Base...")
        from .kb.infra import destroy_knowledge_base

        destroy_knowledge_base(config.knowledge_base_id, config.kb_bucket_name, region=region)

    destroy_policy_engine(config.policy_engine_id, region=region)
    destroy_gateway(config.gateway_id, region=region)
    delete_from_ssm(region=region)

    click.echo("Destroy complete.")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

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
@click.option(
    "--keep-history", is_flag=True, default=False, help="Archive previous reviews even in CI mode"
)
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
    "--ci",
    is_flag=True,
    default=lambda: get_env_or_default("CI", "").lower() in ("true", "1", "yes"),
    help="CI/CD mode: non-interactive, no history by default",
)
@click.option("--json", "json_output", is_flag=True, help="Output results as JSON (implies --ci)")
@click.option("--strict", is_flag=True, default=False, help="Strict: any High impact risk fails")
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
    keep_history,
    no_state,
    model,
    region,
    ci,
    json_output,
    strict,
    profile_name,
    reasoning_level,
    verbose,
):
    """Run an architecture review.

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
      arch-review run --ci --keep-history --profile strict
    """
    _configure_logging(verbose)
    os.environ["AWS_REGION"] = region
    ci_mode = ci or json_output

    load_profile(profile_name)

    # Validate required dirs
    if not documents_dir or not Path(documents_dir).is_dir():
        raise click.UsageError("--documents-dir is required and must exist")
    if not templates_dir or not Path(templates_dir).is_dir():
        raise click.UsageError("--templates-dir is required and must exist")
    if not diagrams_dir or not Path(diagrams_dir).is_dir():
        raise click.UsageError("--diagrams-dir is required and must exist")
    if source_dir and not Path(source_dir).is_dir():
        raise click.UsageError("--source-dir must exist if provided")

    shared_config = _load_shared_config(region)
    should_archive = keep_history or (not ci_mode and not no_history)

    _run_review(
        documents_dir=documents_dir,
        templates_dir=templates_dir,
        diagrams_dir=diagrams_dir,
        source_dir=source_dir,
        output_dir=output_dir,
        no_state=no_state,
        should_archive=should_archive,
        model=model,
        shared_config=shared_config,
        ci_mode=ci_mode,
        json_output=json_output,
        strict=strict,
        reasoning_level=reasoning_level,
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
    ci_mode: bool,
    json_output: bool,
    strict: bool,
    reasoning_level: str = DEFAULT_REASONING_LEVEL,
):
    """Execute the review and handle output."""
    out_path = _get_output_dir(output_dir)

    if should_archive:
        _archive_previous(out_path)

    try:
        orchestrator = ReviewOrchestrator(
            documents_dir=documents_dir,
            templates_dir=templates_dir,
            diagrams_dir=diagrams_dir,
            shared_config=shared_config,
            source_dir=source_dir or None,
            model_name=model,
            ci_mode=ci_mode,
            output_fn=click.echo,
            reasoning_level=reasoning_level,
        )

        result = orchestrator.run_review()
        verdict, exit_code = _get_verdict_and_exit_code(result["review"], strict=strict)

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
            review_path = out_path / DEFAULT_REVIEW_FILE
            full_session = result.get("full_session", result["review"])
            review_path.write_text(full_session)
            click.echo(f"\nReview saved to: {review_path}")

            if not no_state:
                state = extract_state_from_review(result)
                state_path = out_path / DEFAULT_STATE_FILE
                state.save(state_path)
                click.echo(f"State saved to: {state_path}")

            if ci_mode:
                click.echo(f"\nVerdict: {verdict}")
                if exit_code != EXIT_SUCCESS:
                    click.echo(f"Exiting with code {exit_code}")

        sys.exit(exit_code)

    except click.UsageError:
        raise
    except ArchReviewError as e:
        if json_output:
            click.echo(json.dumps({"error": str(e), "exit_code": EXIT_ERROR}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"error": str(e), "exit_code": EXIT_ERROR}))
        else:
            click.echo(f"Unexpected error: {e}", err=True)
            import traceback

            click.echo(traceback.format_exc(), err=True)
        sys.exit(EXIT_ERROR)


# ---------------------------------------------------------------------------
# remediate
# ---------------------------------------------------------------------------

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
@click.option(
    "--profile",
    "profile_name",
    default="default",
    help="Review profile (e.g. strict, lightweight, or custom name)",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")
def remediate(output_dir, no_output, model, region, profile_name, verbose):
    """Discuss and resolve previous review findings.

    Loads state from a previous review and starts an interactive session
    to work through gaps, risks, and recommendations.
    """
    _configure_logging(verbose)
    os.environ["AWS_REGION"] = region

    load_profile(profile_name)

    out_path = _get_output_dir(output_dir)
    state_path = out_path / DEFAULT_STATE_FILE

    if not state_path.exists():
        click.echo(f"Error: State file not found: {state_path}", err=True)
        click.echo("Run a review first (arch-review run) to generate a state file.", err=True)
        sys.exit(EXIT_ERROR)

    try:
        state = ReviewState.from_file(state_path)
        click.echo(f"Loaded state from: {state_path}")

        model_id = SUPPORTED_MODELS[model]["model_id"]

        agent = create_remediation_agent(state=state, model_id=model_id, region=region)
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


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

@cli.group()
def profiles():
    """Manage review profiles."""


@profiles.command("list")
def profiles_list():
    """List available review profiles."""
    all_profiles = list_profiles()

    for label in ("builtin", "user", "project"):
        names = all_profiles[label]
        if not names:
            continue
        click.echo(f"\n{label.capitalize()} profiles:")
        for name in names:
            path = get_profile_path(name)
            if path:
                import yaml

                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                desc = data.get("description", "")
                click.echo(f"  {name:20s} {desc}")
            else:
                click.echo(f"  {name}")


@profiles.command("show")
@click.argument("name")
def profiles_show(name):
    """Show the contents of a profile."""
    path = get_profile_path(name)
    if not path:
        click.echo(f"Profile '{name}' not found.", err=True)
        sys.exit(EXIT_ERROR)
    click.echo(path.read_text())


# ---------------------------------------------------------------------------
# kb
# ---------------------------------------------------------------------------

@cli.group()
def kb():
    """Manage the WAF Knowledge Base."""


@kb.command("sync")
@click.option(
    "--region",
    default=lambda: get_env_or_default("AWS_REGION", DEFAULT_REGION),
    help=f"AWS region (default: {DEFAULT_REGION})",
)
@click.option(
    "--content-dir",
    type=click.Path(file_okay=False),
    default=".arch-review/waf-content",
    help="Directory for scraped WAF content (default: .arch-review/waf-content)",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")
def kb_sync(region, content_dir, verbose):
    """Scrape WAF docs, upload to S3, and trigger KB ingestion."""
    _configure_logging(verbose)
    os.environ["AWS_REGION"] = region

    config = _load_shared_config(region)
    if not config.knowledge_base_id or not config.kb_bucket_name:
        click.echo(
            "Error: No Knowledge Base found in config. "
            "Deploy with --with-kb first: arch-review deploy --with-kb",
            err=True,
        )
        sys.exit(EXIT_ERROR)

    click.echo("Scraping WAF documentation...")
    from .kb.scraper import scrape_waf

    total = scrape_waf(content_dir)
    click.echo(f"Scraped {total} pages to {content_dir}")

    click.echo("Uploading to S3 and starting ingestion...")
    from .kb.sync import sync_kb

    success = sync_kb(
        content_dir=content_dir,
        kb_id=config.knowledge_base_id,
        bucket_name=config.kb_bucket_name,
        region=region,
    )
    if success:
        click.echo("Knowledge Base sync complete.")
    else:
        click.echo("Knowledge Base sync failed.", err=True)
        sys.exit(EXIT_ERROR)


# ---------------------------------------------------------------------------
# profiles (continued)
# ---------------------------------------------------------------------------

@profiles.command("create")
@click.argument("name")
@click.option(
    "--from",
    "from_profile",
    default="default",
    help="Base profile to copy from (default: default)",
)
def profiles_create(name, from_profile):
    """Create a custom profile by copying an existing one."""
    from .profiles import USER_DIR

    source = get_profile_path(from_profile)
    if not source:
        click.echo(f"Source profile '{from_profile}' not found.", err=True)
        sys.exit(EXIT_ERROR)

    USER_DIR.mkdir(parents=True, exist_ok=True)
    dest = USER_DIR / f"{name}.yaml"

    if dest.exists():
        click.echo(f"Profile '{name}' already exists at {dest}", err=True)
        sys.exit(EXIT_ERROR)

    import yaml

    with open(source) as f:
        data = yaml.safe_load(f) or {}

    data["name"] = name.replace("-", " ").replace("_", " ").title()
    data["description"] = f"Custom profile based on {from_profile}"

    with open(dest, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    click.echo(f"Created profile: {dest}")
    click.echo("Edit this file to customize behavior.")


if __name__ == "__main__":
    cli()
