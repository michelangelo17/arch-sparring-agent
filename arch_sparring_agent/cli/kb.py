"""Knowledge Base CLI subgroup."""

from __future__ import annotations

import sys

import click

from .common import EXIT_ERROR, common_options, configure_logging, load_shared_config


@click.group()
def kb():
    """Manage the WAF Knowledge Base."""


@kb.command("sync")
@click.option(
    "--content-dir",
    type=click.Path(file_okay=False),
    default=".arch-review/waf-content",
    help="Directory for scraped WAF content (default: .arch-review/waf-content)",
)
@common_options
def kb_sync(region, content_dir, verbose):
    """Scrape WAF docs, upload to S3, and trigger KB ingestion."""
    configure_logging(verbose)

    config = load_shared_config(region)
    if not config.knowledge_base_id or not config.kb_bucket_name:
        click.echo(
            "Error: No Knowledge Base found in config. "
            "Deploy with --with-kb first: arch-review deploy --with-kb",
            err=True,
        )
        sys.exit(EXIT_ERROR)

    click.echo("Scraping WAF documentation...")
    from ..kb.scraper import scrape_waf

    total = scrape_waf(content_dir)
    click.echo(f"Scraped {total} pages to {content_dir}")

    click.echo("Uploading to S3 and starting ingestion...")
    from ..kb.sync import sync_kb

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
