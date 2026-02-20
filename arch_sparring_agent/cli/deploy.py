"""Deploy and destroy CLI commands."""

import os

import click

from ..config import DEFAULT_REGION
from ..infra import SSM_PARAMETER_NAME, SharedConfig, delete_from_ssm, save_to_ssm
from ..infra.gateway import destroy_gateway, setup_gateway
from ..infra.policy import destroy_policy_engine, setup_architecture_review_policies
from . import (
    cli,
    configure_logging,
    get_env_or_default,
    load_shared_config,
)


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
    S3 Vectors for WAF best-practice content.

    Idempotent — safe to run repeatedly.
    """
    configure_logging(verbose)
    os.environ["AWS_REGION"] = region

    click.echo(f"Deploying arch-review infrastructure in {region}...")

    gateway_arn, gateway_id, engine_id = _deploy_infra(region, gateway_name, policy_engine_name)

    kb_id = None
    kb_bucket = None

    if with_kb:
        click.echo("\nProvisioning Knowledge Base...")
        from ..kb.infra import setup_knowledge_base

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
    click.echo(f"  Config stored in: SSM {SSM_PARAMETER_NAME}")


def _deploy_infra(region: str, gateway_name: str, policy_engine_name: str) -> tuple[str, str, str]:
    """Run the existing setup logic and return (gateway_arn, gateway_id, engine_id).

    Raises click.ClickException on failure so the CLI exits cleanly.
    """
    from ..exceptions import PolicySetupError

    try:
        gateway_arn, gateway_id = setup_gateway(region=region, gateway_name=gateway_name)
    except PolicySetupError as e:
        raise click.ClickException(f"Gateway setup failed: {e}") from e

    try:
        engine_id = setup_architecture_review_policies(
            region=region,
            policy_engine_name=policy_engine_name,
            gateway_arn=gateway_arn,
            gateway_name=gateway_name,
        )
    except PolicySetupError as e:
        raise click.ClickException(f"Policy Engine / Cedar policy setup failed: {e}") from e

    return gateway_arn, gateway_id, engine_id


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
    configure_logging(verbose)
    os.environ["AWS_REGION"] = region

    if not confirm:
        click.echo(
            "This will destroy all arch-review infrastructure in the account.\n"
            "Re-run with --confirm to proceed."
        )
        return

    config = load_shared_config(region)

    click.echo(f"Destroying arch-review infrastructure in {region}...")

    if config.knowledge_base_id or config.kb_bucket_name:
        click.echo("Tearing down Knowledge Base...")
        from ..kb.infra import destroy_knowledge_base

        destroy_knowledge_base(config.knowledge_base_id, config.kb_bucket_name, region=region)

    destroy_policy_engine(config.policy_engine_id, region=region)
    destroy_gateway(config.gateway_id, region=region)
    delete_from_ssm(region=region)

    click.echo("Destroy complete.")
