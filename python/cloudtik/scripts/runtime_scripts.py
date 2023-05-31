import logging
import os
import sys
import traceback
from shlex import quote

import click

from cloudtik.core._private.cli_logger import (add_click_logging_options,
                                               cli_logger)
from cloudtik.core._private.cluster.cluster_operator import (
    start_node_from_head, stop_node_from_head)
from cloudtik.core._private.runtime_factory import _get_runtime_home
from cloudtik.core._private.utils import run_bash_scripts, run_system_command, with_script_args
from cloudtik.scripts.utils import NaturalOrderGroup, add_command_alias

logger = logging.getLogger(__name__)


RUNTIME_SCRIPTS_PATH = "scripts"
RUNTIME_INSTALL_SCRIPT_NAME = "install"
RUNTIME_CONFIGURE_SCRIPT_NAME = "configure"
RUNTIME_SERVICES_SCRIPT_NAME = "services"


def _get_runtime_script_path(runtime_type: str, script_name: str):
    runtime_home = _get_runtime_home(runtime_type)
    return os.path.join(runtime_home, RUNTIME_SCRIPTS_PATH, script_name)


def _run_runtime_bash_script(script_path, command, head, script_args):
    args = [command] or []
    if head:
        args += ["--head"]
    run_args = " ".join(args)
    run_bash_scripts(script_path, run_args, script_args)


def _run_runtime_python_script(script_path, command, head, script_args):
    args = [command] or []
    if head:
        args += ["--head"]
    run_args = " ".join(args)

    cmds = [
        sys.executable, "-u",
        quote(script_path),
    ]
    if run_args:
        cmds += [run_args]
    with_script_args(cmds, script_args)
    final_cmd = " ".join(cmds)

    run_system_command(final_cmd)


def _run_runtime_script(
        runtime_type, command, head, script_args, script_name):
    # search for either bash script or python script with the name to run
    bash_script_name = script_name + ".sh"
    install_script_path = _get_runtime_script_path(
        runtime_type, bash_script_name)
    if os.path.exists(install_script_path):
        _run_runtime_bash_script(
            install_script_path, command, head, script_args)

    python_script_name = script_name + ".py"
    install_script_path = _get_runtime_script_path(
        runtime_type, python_script_name)
    if os.path.exists(install_script_path):
        _run_runtime_python_script(
            install_script_path, command, head, script_args)


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
    "--node-ip",
    required=False,
    type=str,
    default=None,
    help="The node ip address of the node to run start commands")
@click.option(
    "--all-nodes/--no-all-nodes",
    is_flag=True,
    default=True,
    help="Whether to execute start commands to all nodes.")
@click.option(
    "--runtimes",
    required=False,
    type=str,
    default=None,
    help="The runtimes to start. Comma separated list.")
@click.option(
    "--parallel/--no-parallel", is_flag=True, default=True, help="Whether the run the commands on nodes in parallel.")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Don't ask for confirmation.")
@add_click_logging_options
def start(cluster_config_file, cluster_name, no_config_cache,
          node_ip, all_nodes, runtimes, parallel, yes):
    """Manually start the node and runtime services on head or worker node."""
    try:
        # attach to the worker node
        start_node_from_head(
            cluster_config_file,
            node_ip=node_ip,
            all_nodes=all_nodes,
            runtimes=runtimes,
            override_cluster_name=cluster_name,
            no_config_cache=no_config_cache,
            parallel=parallel,
            yes=yes)
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
    "--all-nodes/--no-all-nodes",
    is_flag=True,
    default=True,
    help="Whether to execute stop commands to all nodes.")
@click.option(
    "--runtimes",
    required=False,
    type=str,
    default=None,
    help="The runtimes to start. Comma separated list.")
@click.option(
    "--parallel/--no-parallel", is_flag=True, default=True, help="Whether the run the commands on nodes in parallel.")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Don't ask for confirmation.")
@add_click_logging_options
def stop(cluster_config_file, cluster_name, no_config_cache,
         node_ip, all_nodes, runtimes, parallel, yes):
    """Manually run stop commands on head or worker nodes."""
    try:
        # attach to the worker node
        stop_node_from_head(
            cluster_config_file,
            node_ip=node_ip,
            all_nodes=all_nodes,
            runtimes=runtimes,
            override_cluster_name=cluster_name,
            no_config_cache=no_config_cache,
            parallel=parallel,
            yes=yes)
    except RuntimeError as re:
        cli_logger.error("Stop node failed. " + str(re))
        if cli_logger.verbosity == 0:
            cli_logger.print("For more details, please run with -v flag.")
        else:
            traceback.print_exc()


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("runtime", required=True, type=str)
@click.option(
    "--head",
    is_flag=True,
    default=False,
    help="provide this argument for the head node")
@click.argument("script_args", nargs=-1)
def install(runtime, head, script_args):
    _run_runtime_script(
        runtime, head, script_args,
        RUNTIME_INSTALL_SCRIPT_NAME)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("runtime", required=True, type=str)
@click.option(
    "--head",
    is_flag=True,
    default=False,
    help="provide this argument for the head node")
@click.argument("script_args", nargs=-1)
def configure(runtime, head, script_args):
    _run_runtime_script(
        runtime, None, head, script_args,
        RUNTIME_CONFIGURE_SCRIPT_NAME)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("runtime", required=True, type=str)
@click.argument("command", required=True, type=str)
@click.option(
    "--head",
    is_flag=True,
    default=False,
    help="provide this argument for the head node")
@click.argument("script_args", nargs=-1)
def services(runtime, command, head, script_args):
    _run_runtime_script(
        runtime, command, head, script_args,
        RUNTIME_SERVICES_SCRIPT_NAME)


def runtime_add_command_alias(command, name, hidden):
    add_command_alias(runtime, command, name, hidden)


runtime.add_command(start)
runtime_add_command_alias(start, name="restart", hidden=True)
runtime.add_command(stop)

runtime.add_command(install)
runtime.add_command(configure)
runtime.add_command(services)
