import json
import logging
import sys
from typing import Optional

import click

from cloudtik.core._private import services
from cloudtik.core._private.cli_logger import (add_click_logging_options,
                                               cli_logger)
from cloudtik.core._private.cluster.cluster_operator import (
    debug_status_string, get_cluster_dump_archive_on_head,
    RUN_ENV_TYPES, teardown_cluster_on_head, cluster_process_status_on_head, rsync_node_on_head, attach_node_on_head,
    exec_node_on_head,
    start_node_on_head, stop_node_on_head, kill_node_on_head, scale_cluster_on_head,
    _wait_for_ready, _get_worker_node_ips, _get_head_node_ip,
    _show_cluster_status, _monitor_cluster, _show_cluster_info, _show_worker_cpus, _show_worker_memory)
from cloudtik.core._private.constants import CLOUDTIK_REDIS_DEFAULT_PASSWORD, \
    CLOUDTIK_KV_NAMESPACE_HEALTHCHECK
from cloudtik.core._private.state import kv_store
from cloudtik.core._private.state.kv_store import kv_initialize_with_address
from cloudtik.core._private.config_utils import CLOUDTIK_CLUSTER_SCALING_ERROR, \
    CLOUDTIK_CLUSTER_SCALING_STATUS, decode_cluster_scaling_time, is_alive_time, get_head_bootstrap_config, \
    load_head_cluster_config
from cloudtik.scripts.utils import NaturalOrderGroup, add_command_alias

logger = logging.getLogger(__name__)


@click.group(cls=NaturalOrderGroup)
def head():
    """
    Commands running on head node only.
    """
    pass


@head.command()
@click.option(
    "--node-ip",
    "-n",
    required=True,
    type=str,
    help="The node ip to attach to.")
@click.option(
    "--screen", is_flag=True, default=False, help="Run the command in screen.")
@click.option(
    "--tmux", is_flag=True, default=False, help="Run the command in tmux.")
@click.option(
    "--new", "-N", is_flag=True, help="Force creation of a new screen.")
@click.option(
    "--port-forward",
    "-p",
    required=False,
    multiple=True,
    type=int,
    help="Port to forward. Use this multiple times to forward multiple ports.")
@click.option(
    "--host", is_flag=True, default=False, help="Attach to the host even running with docker.")
@add_click_logging_options
def attach(node_ip, screen, tmux, new, port_forward, host):
    """Attach to worker node from head."""
    port_forward = [(port, port) for port in list(port_forward)]
    attach_node_on_head(node_ip,
                        screen,
                        tmux,
                        new,
                        port_forward,
                        force_to_host=host)


@head.command()
@click.argument("cmd", required=True, type=str)
@click.option(
    "--node-ip",
    "-n",
    required=False,
    type=str,
    default=None,
    help="The node ip to operate on.")
@click.option(
    "--all-nodes/--no-all-nodes",
    is_flag=True,
    default=False,
    help="Whether to execute on all nodes.")
@click.option(
    "--run-env",
    required=False,
    type=click.Choice(RUN_ENV_TYPES),
    default="auto",
    help="Choose whether to execute this command in a container or directly on"
    " the cluster head. Only applies when docker is configured in the YAML.")
@click.option(
    "--screen", is_flag=True, default=False, help="Run the command in screen.")
@click.option(
    "--tmux", is_flag=True, default=False, help="Run the command in tmux.")
@click.option(
    "--port-forward",
    "-p",
    required=False,
    multiple=True,
    type=int,
    help="Port to forward. Use this multiple times to forward multiple ports.")
@click.option(
    "--with-output",
    is_flag=True,
    default=False,
    help="Whether to capture command output.")
@click.option(
    "--parallel/--no-parallel", is_flag=True, default=True, help="Whether the run the commands on nodes in parallel.")
@add_click_logging_options
def exec(cmd, node_ip, all_nodes, run_env, screen, tmux, port_forward, with_output, parallel):
    """Execute command on the worker node from head."""
    port_forward = [(port, port) for port in list(port_forward)]
    exec_node_on_head(
        node_ip,
        all_nodes,
        cmd,
        run_env,
        screen,
        tmux,
        port_forward,
        with_output=with_output,
        parallel=parallel)


@head.command()
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Don't ask for confirmation.")
@click.option(
    "--cpus",
    required=False,
    type=int,
    help="Specify the number of cpus of the cluster.")
@click.option(
    "--nodes",
    required=False,
    type=int,
    help="Specify the number of nodes of the cluster.")
@add_click_logging_options
def scale(yes, cpus, nodes):
    """Scale the cluster with a specific number cpus or nodes."""
    scale_cluster_on_head(yes, cpus, nodes)


@head.command()
@click.argument("source", required=False, type=str)
@click.argument("target", required=False, type=str)
@click.option(
    "--node-ip",
    "-n",
    required=False,
    type=str,
    default=None,
    help="The worker node ip to rsync up.")
@click.option(
    "--all-workers/--no-all-workers",
    is_flag=True,
    default=False,
    help="Whether to sync the file to all workers.")
@add_click_logging_options
def rsync_up(source, target, node_ip, all_workers):
    """Rsync up specific file from or to the worker node."""
    rsync_node_on_head(
        source,
        target,
        down=False,
        node_ip=node_ip,
        all_workers=all_workers)


@head.command()
@click.argument("source", required=False, type=str)
@click.argument("target", required=False, type=str)
@click.option(
    "--node-ip",
    "-n",
    required=True,
    type=str,
    help="The worker node ip to rsync from.")
@add_click_logging_options
def rsync_down(source, target, node_ip):
    """Rsync down specific file from or to the worker node."""
    rsync_node_on_head(
        source,
        target,
        down=True,
        node_ip=node_ip,
        all_workers=False)


@head.command()
@add_click_logging_options
def status():
    """Show cluster summary status."""
    config = load_head_cluster_config()
    _show_cluster_status(config)


@head.command()
@click.option(
    "--worker-cpus",
    is_flag=True,
    default=False,
    help="Get the total number of cpus for workers.")
@click.option(
    "--worker-memory",
    is_flag=True,
    default=False,
    help="Get the total memory for workers.")
@add_click_logging_options
def info(worker_cpus, worker_memory):
    """Show cluster summary information and useful links to use the cluster."""
    cluster_config_file = get_head_bootstrap_config()
    config = load_head_cluster_config()

    if worker_cpus:
        return _show_worker_cpus(config)

    if worker_memory:
        return _show_worker_memory(config)

    _show_cluster_info(config, cluster_config_file)


@head.command()
@click.option(
    "--public",
    is_flag=True,
    default=False,
    help="Return public ip if there is one")
@add_click_logging_options
def head_ip(public):
    """Return the head node IP of a cluster."""
    try:
        config = load_head_cluster_config()
        head_node_ip = _get_head_node_ip(config, public)
        click.echo(head_node_ip)
    except RuntimeError as re:
        cli_logger.error("Get head IP failed. " + str(re))
        raise re


@head.command()
@add_click_logging_options
@click.option(
    "--runtime",
    required=False,
    type=str,
    default=None,
    help="Get the worker ips for specific runtime.")
def worker_ips(runtime):
    """Return the list of worker IPs of a cluster."""
    config = load_head_cluster_config()
    workers = _get_worker_node_ips(config, runtime=runtime)
    if len(workers) > 0:
        click.echo("\n".join(workers))


@head.command()
@click.option(
    "--lines",
    required=False,
    default=100,
    type=int,
    help="Number of lines to tail.")
@click.option(
    "--file-type",
    required=False,
    type=str,
    default=None,
    help="The type of information to check: log, out, err")
@add_click_logging_options
def monitor(lines, file_type):
    """Tails the monitor logs of a cluster."""
    config = load_head_cluster_config()
    _monitor_cluster(config, lines, file_type=file_type)


@head.command()
@click.option(
    "--keep-min-workers",
    is_flag=True,
    default=False,
    help="Retain the minimal amount of workers specified in the config.")
@click.option(
    "--hard",
    is_flag=True,
    default=False,
    help="Stop the cluster nodes by without running stop commands.")
@click.option(
    "--indent-level",
    required=False,
    default=None,
    type=int,
    hidden=True,
    help="The indent level for showing messages during this command.")
@add_click_logging_options
def teardown(keep_min_workers, hard, indent_level):
    """Tear down a cluster."""
    if indent_level is not None:
        with cli_logger.indented_by(indent_level):
            teardown_cluster_on_head(keep_min_workers, hard=hard)
    else:
        teardown_cluster_on_head(keep_min_workers, hard=hard)


@head.command()
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Don't ask for confirmation.")
@click.option(
    "--hard",
    is_flag=True,
    default=False,
    help="Terminates node by directly delete the instances")
@click.option(
    "--node-ip",
    required=False,
    type=str,
    default=None,
    help="The node ip address of the node to kill")
@click.option(
    "--indent-level",
    required=False,
    default=None,
    type=int,
    hidden=True,
    help="The indent level for showing messages during this command.")
@add_click_logging_options
def kill_node(yes, hard, node_ip, indent_level):
    """Kills a specified worker node and a random worker node."""

    def do_kill_node():
        killed_node_ip = kill_node_on_head(
            yes, hard, node_ip)
        if killed_node_ip:
            click.echo("Killed node with IP " + killed_node_ip)

    if indent_level is not None:
        with cli_logger.indented_by(indent_level):
            do_kill_node()
    else:
        do_kill_node()


@head.command()
@click.option(
    "--min-workers",
    required=False,
    default=None,
    type=int,
    help="The min workers to wait for being ready.")
@click.option(
    "--timeout",
    required=False,
    default=None,
    type=int,
    help="The maximum number of seconds to wait.")
@add_click_logging_options
def wait_for_ready(min_workers, timeout):
    """Wait for the minimum number of workers to be ready."""
    config = load_head_cluster_config()
    _wait_for_ready(config, min_workers, timeout)


@head.command()
@click.option(
    "--address",
    required=False,
    type=str,
    help="Override the address to connect to.")
@click.option(
    "--redis-password",
    required=False,
    type=str,
    default=CLOUDTIK_REDIS_DEFAULT_PASSWORD,
    help="Connect with redis password.")
@add_click_logging_options
def debug_status(address, redis_password):
    """Print cluster status, including autoscaling info."""
    if not address:
        address = services.get_address_to_use_or_die()
    kv_initialize_with_address(address, redis_password)
    status = kv_store.kv_get(
        CLOUDTIK_CLUSTER_SCALING_STATUS)
    error = kv_store.kv_get(
        CLOUDTIK_CLUSTER_SCALING_ERROR)
    print(debug_status_string(status, error))


@head.command()
@click.option(
    "--address",
    required=False,
    type=str,
    help="Override the address to connect to.")
@click.option(
    "--redis-password",
    required=False,
    type=str,
    default=CLOUDTIK_REDIS_DEFAULT_PASSWORD,
    help="Connect with redis password.")
@add_click_logging_options
def process_status(address, redis_password):
    """Show cluster process status."""
    cluster_process_status_on_head(
        address, redis_password)


@head.command()
@click.option(
    "--address",
    required=False,
    type=str,
    help="Override the address to connect to.")
@click.option(
    "--redis-password",
    required=False,
    type=str,
    default=CLOUDTIK_REDIS_DEFAULT_PASSWORD,
    help="Connect with redis password.")
@click.option(
    "--component",
    required=False,
    type=str,
    help="Health check for a specific component. Currently supports: "
    "[None]")
@add_click_logging_options
def health_check(address, redis_password, component):
    """
    Health check a cluster or a specific component. Exit code 0 is healthy.
    """

    if not address:
        address = services.get_address_to_use_or_die()
    else:
        address = services.address_to_ip(address)

    redis_client = kv_initialize_with_address(address, redis_password)

    if not component:
        # If no component is specified, we are health checking the core. If
        # client creation or ping fails, we will still exit with a non-zero
        # exit code.
        redis_client.ping()

        try:
            # check cluster controller live status through scaling status time
            status = kv_store.kv_get(CLOUDTIK_CLUSTER_SCALING_STATUS)
            if not status:
                cli_logger.print("Cluster is not healthy! No status reported.")
                sys.exit(1)

            report_time = decode_cluster_scaling_time(status)
            time_ok = is_alive_time(report_time)
            if not time_ok:
                cli_logger.print("Cluster is not healthy! Last status time {}", report_time)
                sys.exit(1)

            cli_logger.print("Cluster is healthy.")
            sys.exit(0)
        except Exception as e:
            cli_logger.error("Health check failed." + str(e))
            pass
        sys.exit(1)

    report_str = kv_store.kv_get(
        component, namespace=CLOUDTIK_KV_NAMESPACE_HEALTHCHECK)
    if not report_str:
        # Status was never updated
        cli_logger.print("{} is not healthy! No status reported.", component)
        sys.exit(1)

    report = json.loads(report_str)
    report_time = float(report["time"])
    time_ok = is_alive_time(report_time)
    if not time_ok:
        cli_logger.print("{} is not healthy! Last status time {}",
                         component, report_time)
        sys.exit(1)

    cli_logger.print("{} is healthy.", component)
    sys.exit(0)


@head.command()
@click.option(
    "--host",
    "-h",
    required=False,
    type=str,
    help="Single or list of hosts, separated by comma.")
@click.option(
    "--stream",
    "-S",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
    help="If True, will stream the binary archive contents to stdout")
@click.option(
    "--output",
    "-o",
    required=False,
    type=str,
    default=None,
    help="Output file.")
@click.option(
    "--logs/--no-logs",
    is_flag=True,
    default=True,
    help="Collect logs from session dir")
@click.option(
    "--debug-state/--no-debug-state",
    is_flag=True,
    default=True,
    help="Collect debug_state.txt from log dir")
@click.option(
    "--pip/--no-pip",
    is_flag=True,
    default=True,
    help="Collect installed pip packages")
@click.option(
    "--processes/--no-processes",
    is_flag=True,
    default=True,
    help="Collect info on running processes")
@click.option(
    "--processes-verbose/--no-processes-verbose",
    is_flag=True,
    default=True,
    help="Increase process information verbosity")
@click.option(
    "--tempfile",
    "-T",
    required=False,
    type=str,
    default=None,
    help="Temporary file to use")
@click.option(
    "--verbosity",
    required=False,
    default=None,
    type=int,
    help="The integer verbosity to set.")
@add_click_logging_options
def cluster_dump(host: Optional[str] = None,
                 stream: bool = False,
                 output: Optional[str] = None,
                 logs: bool = True,
                 debug_state: bool = True,
                 pip: bool = True,
                 processes: bool = True,
                 processes_verbose: bool = False,
                 tempfile: Optional[str] = None,
                 verbosity: int = None):
    """Collect cluster data and package into an archive on head.

        Usage:

            cloudtik head cluster-dump[--stream/--output file]

        This script is called on head node to fetch the cluster data.
        """
    if verbosity is not None:
        cli_logger.set_verbosity(verbosity)

    get_cluster_dump_archive_on_head(
        host=host,
        stream=stream,
        output=output,
        logs=logs,
        debug_state=debug_state,
        pip=pip,
        processes=processes,
        processes_verbose=processes_verbose,
        tempfile=tempfile)


@click.group(cls=NaturalOrderGroup)
def runtime():
    """
    Commands running on head for runtime control
    """
    pass


@runtime.command()
@click.option(
    "--node-ip",
    "-n",
    required=False,
    type=str,
    default=None,
    help="The node ip on which to execute start commands.")
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
    "--indent-level",
    required=False,
    default=None,
    type=int,
    hidden=True,
    help="The indent level for showing messages during this command.")
@click.option(
    "--parallel/--no-parallel", is_flag=True, default=True, help="Whether the run the commands on nodes in parallel.")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Don't ask for confirmation.")
@add_click_logging_options
def start(node_ip, all_nodes, runtimes, indent_level, parallel, yes):
    """Run start commands on the specific node or all nodes."""
    def do_start_node():
        start_node_on_head(node_ip=node_ip, all_nodes=all_nodes,
                           runtimes=runtimes, parallel=parallel, yes=yes)

    if indent_level is not None:
        with cli_logger.indented_by(indent_level):
            do_start_node()
    else:
        do_start_node()


@runtime.command()
@click.option(
    "--node-ip",
    "-n",
    required=False,
    type=str,
    default=None,
    help="The node ip on which to execute start commands.")
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
    "--indent-level",
    required=False,
    default=None,
    type=int,
    hidden=True,
    help="The indent level for showing messages during this command.")
@click.option(
    "--parallel/--no-parallel", is_flag=True, default=True, help="Whether the run the commands on nodes in parallel.")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Don't ask for confirmation.")
@add_click_logging_options
def stop(node_ip, all_nodes, runtimes, indent_level, parallel, yes):
    """Run stop commands on the specific node or all nodes."""
    def do_stop_node():
        stop_node_on_head(node_ip=node_ip, all_nodes=all_nodes,
                          runtimes=runtimes, parallel=parallel, yes=yes)

    if indent_level is not None:
        with cli_logger.indented_by(indent_level):
            do_stop_node()
    else:
        do_stop_node()


def runtime_add_command_alias(command, name, hidden):
    add_command_alias(runtime, command, name, hidden)


runtime.add_command(start)
runtime_add_command_alias(start, name="restart", hidden=True)
runtime.add_command(stop)

# commands running on head node
head.add_command(attach)
head.add_command(exec)
head.add_command(scale)

head.add_command(rsync_up)
head.add_command(rsync_down)

head.add_command(status)
head.add_command(info)
head.add_command(worker_ips)
head.add_command(monitor)

head.add_command(teardown)
head.add_command(kill_node)

# runtime commands
head.add_command(runtime)

head.add_command(debug_status)
head.add_command(process_status)
head.add_command(health_check)
head.add_command(cluster_dump)
