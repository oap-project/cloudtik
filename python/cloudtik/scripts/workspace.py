import click
import logging

from cloudtik.core._private.cli_logger import (add_click_logging_options,
                                                cli_logger, cf)

logger = logging.getLogger(__name__)


@click.group()
def workspace():
    """
    Commands for working with Workspace.
    """
    pass


@workspace.command()
@click.argument("workspace_config_file", required=True, type=str)
@click.option(
    "--workspace-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured workspace name.")
@add_click_logging_options
def create(workspace_config_file, workspace_name):
    """Create a Workspace on Cloud based on the workspace configuration file."""
    # TODO: Implement creating of workspace based on cloud provider.
    pass


@workspace.command()
@click.argument("workspace_config_file", required=True, type=str)
@click.option(
    "--workspace-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured workspace name.")
@add_click_logging_options
def delete(workspace_config_file, workspace_name):
    """Delete the workspace and associated Cloud resources."""
    # TODO: Implement deleting of workspace based on cloud provider.
    pass

