import logging
import traceback

import click

from cloudtik.core._private.cli_logger import (add_click_logging_options,
                                               cli_logger)
from cloudtik.core._private.cluster.cluster_operator import (
    start_node_from_head, stop_node_from_head)
from cloudtik.scripts.utils import NaturalOrderGroup

logger = logging.getLogger(__name__)


@click.group(cls=NaturalOrderGroup)
def runtime():
    """
    Commands for runtime service control
    """
    pass


@runtime.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@click.option(
    "--no-config-cache",
    is_flag=True,
    default=False,
    help="Disable the local cluster config cache.")
@click.option(
    "--head",
    required=False,
    type=str,
    default=None,
    help="Start runtime for head node")
@click.option(
    "--node-ip",
    required=False,
    type=str,
    default=None,
    help="The node ip address of the node to run start commands")
@click.option(
    "--all-nodes",
    is_flag=True,
    default=True,
    help="Whether to execute start commands to all nodes.")
@click.option(
    "--parallel/--no-parallel", is_flag=True, default=True, help="Whether the run the commands on nodes in parallel.")
@add_click_logging_options
def start(cluster_config_file, cluster_name, no_config_cache,
          node_ip, all_nodes, parallel):
    """Manually start the node and runtime services on head or worker node."""
    try:
        # attach to the worker node
        start_node_from_head(
            cluster_config_file,
            node_ip,
            all_nodes,
            cluster_name,
            no_config_cache=no_config_cache,
            parallel=parallel)
    except RuntimeError as re:
        cli_logger.error("Start node failed. " + str(re))
        if cli_logger.verbosity == 0:
            cli_logger.print("For more details, please run with -v flag.")
        else:
            traceback.print_exc()


@runtime.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@click.option(
    "--no-config-cache",
    is_flag=True,
    default=False,
    help="Disable the local cluster config cache.")
@click.option(
    "--node-ip",
    required=False,
    type=str,
    default=None,
    help="The node ip address of the node to stop")
@click.option(
    "--all-nodes",
    is_flag=True,
    default=False,
    help="Whether to execute stop commands to all nodes.")
@click.option(
    "--parallel/--no-parallel", is_flag=True, default=True, help="Whether the run the commands on nodes in parallel.")
@add_click_logging_options
def stop(cluster_config_file, cluster_name, no_config_cache,
         node_ip, all_nodes, parallel):
    """Manually run stop commands on head or worker nodes."""
    try:
        # attach to the worker node
        stop_node_from_head(
            cluster_config_file,
            node_ip,
            all_nodes,
            cluster_name,
            no_config_cache=no_config_cache,
            parallel=parallel)
    except RuntimeError as re:
        cli_logger.error("Stop node failed. " + str(re))
        if cli_logger.verbosity == 0:
            cli_logger.print("For more details, please run with -v flag.")
        else:
            traceback.print_exc()


runtime.add_command(start)
runtime.add_command(stop)
