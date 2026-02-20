"""Profiles CLI subgroup."""

import sys

import click
import yaml

from ..profiles import BUILTIN_DIR, USER_DIR, get_profile_path, list_profiles
from . import EXIT_ERROR, cli


@cli.group()
def profiles():
    """Manage review profiles."""


@profiles.command("list")
def profiles_list():
    """List available review profiles."""
    from ..profiles import _project_dir

    all_profiles = list_profiles()
    dirs = {"builtin": BUILTIN_DIR, "user": USER_DIR, "project": _project_dir()}

    for label in ("builtin", "user", "project"):
        names = all_profiles[label]
        if not names:
            continue
        click.echo(f"\n{label.capitalize()} profiles:")
        directory = dirs[label]
        for name in names:
            path = directory / f"{name}.yaml"
            if path.is_file():
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
    from ..profiles import USER_DIR

    source = get_profile_path(from_profile)
    if not source:
        click.echo(f"Source profile '{from_profile}' not found.", err=True)
        sys.exit(EXIT_ERROR)

    USER_DIR.mkdir(parents=True, exist_ok=True)
    dest = USER_DIR / f"{name}.yaml"

    if dest.exists():
        click.echo(f"Profile '{name}' already exists at {dest}", err=True)
        sys.exit(EXIT_ERROR)

    with open(source) as f:
        data = yaml.safe_load(f) or {}

    data["name"] = name.replace("-", " ").replace("_", " ").title()
    data["description"] = f"Custom profile based on {from_profile}"

    with open(dest, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    click.echo(f"Created profile: {dest}")
    click.echo("Edit this file to customize behavior.")
