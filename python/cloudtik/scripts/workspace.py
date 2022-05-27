import click
import logging
import urllib

from cloudtik.core._private.workspace.workspace_operator import (
    create_workspace, delete_workspace, update_workspace_firewalls, list_workspace_clusters)
from cloudtik.core._private.cli_logger import (add_click_logging_options, cli_logger)
from cloudtik.scripts.utils import NaturalOrderGroup

logger = logging.getLogger(__name__)


@click.group(cls=NaturalOrderGroup)
def workspace():
    """
    Commands for working with workspace.
    """


@workspace.command()
@click.argument("workspace_config_file", required=True, type=str)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Don't ask for confirmation.")
@click.option(
    "--workspace-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured workspace name.")
@click.option(
    "--no-config-cache",
    is_flag=True,
    default=False,
    help="Disable the local workspace config cache.")
@add_click_logging_options
def create(workspace_config_file, yes, workspace_name, no_config_cache):
    """Create a workspace on cloud using the workspace configuration file."""
    if urllib.parse.urlparse(workspace_config_file).scheme in ("http", "https"):
        try:
            response = urllib.request.urlopen(workspace_config_file, timeout=5)
            content = response.read()
            file_name = workspace_config_file.split("/")[-1]
            with open(file_name, "wb") as f:
                f.write(content)
                workspace_config_file = file_name
        except urllib.error.HTTPError as e:
            cli_logger.warning("{}", str(e))
            cli_logger.warning(
                "Could not download remote cluster configuration file.")

    create_workspace(
        config_file=workspace_config_file,
        yes=yes,
        override_workspace_name=workspace_name,
        no_config_cache=no_config_cache)


@workspace.command()
@click.argument("workspace_config_file", required=True, type=str)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Don't ask for confirmation.")
@click.option(
    "--workspace-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured workspace name.")
@click.option(
    "--delete-managed-storage/--no-delete-managed-storage",
    is_flag=True,
    default=False,
    help="Whether to delete the managed cloud storage")
@add_click_logging_options
def delete(workspace_config_file, yes, workspace_name, delete_managed_storage):
    """Delete a workspace and the associated cloud resources."""
    delete_workspace(workspace_config_file, yes, workspace_name, delete_managed_storage)


@workspace.command()
@click.argument("workspace_config_file", required=True, type=str)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Don't ask for confirmation.")
@click.option(
    "--workspace-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured workspace name.")
@add_click_logging_options
def update_firewalls(workspace_config_file, yes, workspace_name):
    """Update the firewalls for workspace."""
    update_workspace_firewalls(workspace_config_file, yes, workspace_name)


@workspace.command()
@click.argument("workspace_config_file", required=True, type=str)
@click.option(
    "--workspace-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured workspace name.")
@add_click_logging_options
def show_clusters(workspace_config_file, workspace_name):
    """List clusters running in this workspace."""
    list_workspace_clusters(workspace_config_file, workspace_name)


# core commands working on workspace
workspace.add_command(create)
workspace.add_command(delete)
workspace.add_command(update_firewalls)

# commands for workspace info
workspace.add_command(show_clusters)
