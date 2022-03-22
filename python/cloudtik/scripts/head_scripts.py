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
    exec_worker_on_head, show_cluster_info, show_cluster_status, monitor_cluster, get_worker_node_ips,
    start_node_on_head)
from cloudtik.core._private.constants import CLOUDTIK_REDIS_DEFAULT_PASSWORD, \
    CLOUDTIK_KV_NAMESPACE_HEALTHCHECK
from cloudtik.core._private.state import kv_store
from cloudtik.core._private.state.kv_store import kv_initialize_with_address
from cloudtik.core._private.utils import CLOUDTIK_CLUSTER_SCALING_ERROR, \
    CLOUDTIK_CLUSTER_SCALING_STATUS, decode_cluster_scaling_time, is_alive_time, get_head_bootstrap_config

logger = logging.getLogger(__name__)


@click.group()
def head():
    """
    Commands running on head node only.
    """
    pass


@head.command()
@click.option(
    "--keep-min-workers",
    is_flag=True,
    default=False,
    help="Retain the minimal amount of workers specified in the config.")
@add_click_logging_options
def teardown(keep_min_workers):
    """Tear down a cluster."""
    teardown_cluster_on_head(keep_min_workers)


@head.command()
@click.option(
    "--node-ip",
    "-n",
    required=True,
    type=str,
    help="The node internal ip on which to execute start commands.")
@click.option(
    "--all-nodes",
    is_flag=True,
    default=False,
    help="Whether to execute start commands to all nodes.")
@add_click_logging_options
def start_node(node_ip, all_nodes):
    """Run start commands on the specific node."""
    start_node_on_head(
        node_ip=node_ip, all_nodes=all_nodes)


@head.command()
@click.argument("source", required=False, type=str)
@click.argument("target", required=False, type=str)
@click.option(
    "--node-ip",
    "-n",
    required=True,
    type=str,
    help="The worker node internal ip to rsync from or to.")
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
@click.argument("source", required=False, type=str)
@click.argument("target", required=False, type=str)
@click.option(
    "--node-ip",
    "-n",
    required=False,
    type=str,
    default=None,
    help="The worker node internal ip to rsync from or to.")
@click.option(
    "--all-workers",
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
@click.option(
    "--node-ip",
    "-n",
    required=True,
    type=str,
    help="The node internal ip to attach to.")
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
@add_click_logging_options
def attach(node_ip, screen, tmux, new, port_forward):
    """Attach to worker node from head."""
    port_forward = [(port, port) for port in list(port_forward)]
    attach_node_on_head(node_ip,
                        screen,
                        tmux,
                        new,
                        port_forward)


@head.command()
@click.argument("cmd", required=True, type=str)
@click.option(
    "--node-ip",
    "-n",
    required=True,
    type=str,
    help="The node internal ip to attach to.")
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
@add_click_logging_options
def exec(cmd, node_ip, run_env, screen, tmux, port_forward):
    """Execute command on the worker node from head."""
    port_forward = [(port, port) for port in list(port_forward)]
    exec_worker_on_head(node_ip,
                        cmd,
                        run_env,
                        screen,
                        tmux,
                        port_forward)


@head.command()
@add_click_logging_options
def worker_ips():
    """Return the list of worker IPs of a cluster."""
    cluster_config_file = get_head_bootstrap_config()
    workers = get_worker_node_ips(cluster_config_file, None)
    if len(workers) == 0:
        click.echo("No worker found.")
    else:
        click.echo("\n".join(workers))


@head.command()
@add_click_logging_options
def info():
    """Show cluster summary information and useful links to use the cluster."""
    cluster_config_file = get_head_bootstrap_config()
    show_cluster_info(cluster_config_file, None)


@head.command()
@add_click_logging_options
def status():
    """Show cluster summary status."""
    cluster_config_file = get_head_bootstrap_config()
    show_cluster_status(cluster_config_file, None)


@head.command()
@click.option(
    "--lines",
    required=False,
    default=100,
    type=int,
    help="Number of lines to tail.")
@add_click_logging_options
def monitor(lines):
    """Tails the monitor logs of a cluster."""
    cluster_config_file = get_head_bootstrap_config()
    monitor_cluster(cluster_config_file, lines, None)


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
@add_click_logging_options
def cluster_dump(host: Optional[str] = None,
                 stream: bool = False,
                 output: Optional[str] = None,
                 logs: bool = True,
                 debug_state: bool = True,
                 pip: bool = True,
                 processes: bool = True,
                 processes_verbose: bool = False,
                 tempfile: Optional[str] = None):
    """Collect cluster data and package into an archive on head.

        Usage:

            cloudtik head cluster-dump[--stream/--output file]

        This script is called on head node to fetch the cluster data.
        """
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


@head.command()
@click.option(
    "--address",
    required=False,
    type=str,
    help="Override the address to connect to.")
@click.option(
    "--redis_password",
    required=False,
    type=str,
    default=CLOUDTIK_REDIS_DEFAULT_PASSWORD,
    help="Connect with redis_password.")
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
    "--redis_address", required=True, type=str, help="the address to redis", default="127.0.0.1")
@add_click_logging_options
def process_status(redis_address):
    """Show cluster process status."""
    cluster_process_status_on_head(
        redis_address)


@head.command()
@click.option(
    "--address",
    required=False,
    type=str,
    help="Override the address to connect to.")
@click.option(
    "--redis_password",
    required=False,
    type=str,
    default=CLOUDTIK_REDIS_DEFAULT_PASSWORD,
    help="Connect  with redis_password.")
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


# commands running on head node
head.add_command(teardown)
head.add_command(start_node)

head.add_command(rsync_down)
head.add_command(rsync_up)
head.add_command(attach)
head.add_command(exec)

head.add_command(worker_ips)
head.add_command(info)
head.add_command(status)
head.add_command(monitor)

head.add_command(cluster_dump)
head.add_command(debug_status)
head.add_command(process_status)
head.add_command(health_check)
