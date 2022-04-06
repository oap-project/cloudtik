import click
import logging
import urllib

from cloudtik.core._private.workspace.workspace_operator import (
    create_or_update_workspace, delete_workspace)
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
    "--no-workspace-config-cache",
    is_flag=True,
    default=False,
    help="Disable the local workspace config cache.")
@add_click_logging_options
def create(workspace_config_file, yes, workspace_name, no_workspace_config_cache):
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

    create_or_update_workspace(
        config_file=workspace_config_file,
        yes=yes,
        override_workspace_name=workspace_name,
        no_workspace_config_cache=no_workspace_config_cache)


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
def delete(workspace_config_file, yes, workspace_name):
    """Delete a workspace and the associated cloud resources."""
    delete_workspace(workspace_config_file, yes, workspace_name)


# core commands working on workspace
workspace.add_command(create)
workspace.add_command(delete)
