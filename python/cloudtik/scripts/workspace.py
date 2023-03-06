import copy
import click
import logging
import urllib

from cloudtik.core._private.workspace.workspace_operator import (
    create_workspace, delete_workspace, update_workspace_firewalls, list_workspace_clusters, show_status,
    show_workspace_info, show_managed_cloud_storage, show_managed_cloud_storage_uri)
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
@click.option(
    "--delete-in-completed/--no-delete-in-completed",
    is_flag=True,
    default=True,
    help="Delete in completed workspace if exists.")
@add_click_logging_options
def create(workspace_config_file, yes, workspace_name, no_config_cache,
           delete_in_completed):
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
        no_config_cache=no_config_cache,
        delete_in_completed=delete_in_completed)


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
def status(workspace_config_file, workspace_name):
    """Show workspace status."""
    show_status(workspace_config_file, workspace_name)


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


@workspace.command()
@click.argument("workspace_config_file", required=True, type=str)
@click.option(
    "--workspace-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured workspace name.")
@click.option(
    "--managed-storage",
    is_flag=True,
    default=False,
    help="Get the managed cloud storage for workspace.")
@click.option(
    "--managed-storage-uri",
    is_flag=True,
    default=False,
    help="Get the managed cloud storage uri for Hadoop.")
@add_click_logging_options
def info(workspace_config_file, workspace_name, managed_storage, managed_storage_uri):
    """Show workspace summary information."""
    if managed_storage:
        return show_managed_cloud_storage(workspace_config_file, workspace_name)

    if managed_storage_uri:
        return show_managed_cloud_storage_uri(workspace_config_file, workspace_name)

    show_workspace_info(
        workspace_config_file,
        workspace_name)


def _add_command_alias(command, name, hidden):
    new_command = copy.deepcopy(command)
    new_command.hidden = hidden
    workspace.add_command(new_command, name=name)


# core commands working on workspace
workspace.add_command(create)
workspace.add_command(delete)
workspace.add_command(update_firewalls)

# commands for workspace info
workspace.add_command(status)
workspace.add_command(show_clusters)
_add_command_alias(show_clusters, name="list-clusters", hidden=True)
workspace.add_command(info)
