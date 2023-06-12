import logging
import os
import subprocess
import sys
from socket import socket
from typing import Optional

import click
import psutil

from cloudtik.core._private import services
from cloudtik.core._private.cli_logger import (add_click_logging_options,
                                               cli_logger, cf)
from cloudtik.core._private.cluster.cluster_operator import (
    dump_local)
from cloudtik.core._private.constants import CLOUDTIK_PROCESSES, \
    CLOUDTIK_REDIS_DEFAULT_PASSWORD, \
    CLOUDTIK_DEFAULT_PORT
from cloudtik.core._private.core_utils import get_cloudtik_temp_dir
from cloudtik.core._private.node.node_services import NodeServicesStarter
from cloudtik.core._private.parameter import StartParams
from cloudtik.core._private.resource_spec import ResourceSpec
from cloudtik.core._private.utils import parse_resources_json, run_script
from cloudtik.scripts.utils import NaturalOrderGroup

logger = logging.getLogger(__name__)


@click.group(cls=NaturalOrderGroup)
def node():
    """
    Commands running on node local only.
    """
    pass


@node.command()
@click.option(
    "--node-ip-address",
    required=False,
    type=str,
    help="the IP address of this node")
@click.option(
    "--address", required=False, type=str, help="the address to use for this node")
@click.option(
    "--port",
    type=int,
    required=False,
    help=f"the port of the head redis process. If not provided, defaults to "
    f"{CLOUDTIK_DEFAULT_PORT}; if port is set to 0, we will"
    f" allocate an available port.")
@click.option(
    "--head",
    is_flag=True,
    default=False,
    help="provide this argument for the head node")
@click.option(
    "--redis-password",
    required=False,
    hidden=True,
    type=str,
    default=CLOUDTIK_REDIS_DEFAULT_PASSWORD,
    help="If provided, secure Redis ports with this password")
@click.option(
    "--redis-shard-ports",
    required=False,
    hidden=True,
    type=str,
    help="the port to use for the Redis shards other than the "
    "primary Redis shard")
@click.option(
    "--redis-max-memory",
    required=False,
    hidden=True,
    type=int,
    help="The max amount of memory (in bytes) to allow redis to use. Once the "
    "limit is exceeded, redis will start LRU eviction of entries. This only "
    "applies to the sharded redis tables (task, object, and profile tables). "
    "By default this is capped at 10GB but can be set higher.")
@click.option(
    "--memory",
    required=False,
    hidden=True,
    type=int,
    help="The amount of memory (in bytes) to make available to workers. "
    "By default, this is set to the available memory on the node.")
@click.option(
    "--num-cpus",
    required=False,
    type=int,
    help="the number of CPUs on this node")
@click.option(
    "--num-gpus",
    required=False,
    type=int,
    help="the number of GPUs on this node")
@click.option(
    "--resources",
    required=False,
    default="{}",
    type=str,
    help="a JSON serialized dictionary mapping resource name to "
    "resource quantity")
@click.option(
    "--cluster-scaling-config",
    required=False,
    type=str,
    help="the file that contains the autoscaling config")
@click.option(
    "--temp-dir",
    hidden=True,
    default=None,
    help="manually specify the root temporary dir of the Cloudtik process")
@click.option(
    "--metrics-export-port",
    type=int,
    hidden=True,
    default=None,
    help="the port to use to expose metrics through a "
    "Prometheus endpoint.")
@click.option(
    "--no-redirect-output",
    is_flag=True,
    default=False,
    help="do not redirect non-worker stdout and stderr to files")
@click.option(
    "--runtimes",
    required=False,
    type=str,
    default=None,
    hidden=True,
    help="Runtimes enabled for process monitoring purposes")
@click.option(
    "--no-controller",
    is_flag=True,
    hidden=True,
    default=False,
    help="If True, the cluster controller will not be started on head",
)
@add_click_logging_options
def start(node_ip_address, address, port, head,
          redis_password, redis_shard_ports, redis_max_memory,
          memory, num_cpus, num_gpus, resources,
          cluster_scaling_config, temp_dir, metrics_export_port,
          no_redirect_output, runtimes, no_controller):
    """Start the main daemon processes on the local machine."""
    # Convert hostnames to numerical IP address.
    if node_ip_address is not None:
        node_ip_address = services.address_to_ip(node_ip_address)
    redirect_output = None if not no_redirect_output else True

    resources = parse_resources_json(resources)

    start_params = StartParams(
        node_ip_address=node_ip_address,
        memory=memory,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        resources=resources,
        temp_dir=temp_dir,
        metrics_export_port=metrics_export_port,
        redirect_output=redirect_output,
        runtimes=runtimes,
        no_controller=no_controller,
    )
    if head:
        # Use default if port is none, allocate an available port if port is 0
        if port is None:
            port = CLOUDTIK_DEFAULT_PORT

        if port == 0:
            with socket() as s:
                s.bind(("", 0))
                port = s.getsockname()[1]

        num_redis_shards = None
        # Start on the head node.
        if redis_shard_ports is not None and address is None:
            redis_shard_ports = redis_shard_ports.split(",")
            # Infer the number of Redis shards from the ports if the number is
            # not provided.
            num_redis_shards = len(redis_shard_ports)

        # Get the node IP address if one is not provided.
        start_params.update_if_absent(
            node_ip_address=services.get_node_ip_address())
        cli_logger.labeled_value("Local node IP", start_params.node_ip_address)
        start_params.update_if_absent(
            redis_port=port,
            redis_shard_ports=redis_shard_ports,
            redis_max_memory=redis_max_memory,
            num_redis_shards=num_redis_shards,
            redis_max_clients=None,
            cluster_scaling_config=cluster_scaling_config,
        )

        # Fail early when starting a new cluster when one is already running
        if address is None:
            default_address = f"{start_params.node_ip_address}:{port}"
            redis_addresses = services.find_redis_address(default_address)
            if len(redis_addresses) > 0:
                raise ConnectionError(
                    f"CloudTik is already running at {default_address}. "
                    f"Please specify a different port using the `--port`"
                    f" command to `cloudtik node start`.")

        node = NodeServicesStarter(
            start_params, head=True, shutdown_at_exit=False, spawn_reaper=False)

        redis_address = node.redis_address
        if temp_dir is None:
            # Default temp directory.
            temp_dir = get_cloudtik_temp_dir()
        # Using the user-supplied temp dir unblocks
        # users who can't write to the default temp.
        os.makedirs(temp_dir, exist_ok=True)
        current_cluster_path = os.path.join(temp_dir, "cloudtik_current_cluster")
        # TODO: Consider using the custom temp_dir for this file across the
        # code base.
        with open(current_cluster_path, "w") as f:
            print(redis_address, file=f)
    else:
        # Start on a non-head node.
        redis_address = None
        if address is not None:
            (redis_address, redis_address_ip,
             redis_address_port) = services.validate_redis_address(address)
        if not (port is None):
            cli_logger.abort("`{}` should not be specified without `{}`.",
                             cf.bold("--port"), cf.bold("--head"))

            raise Exception("If --head is not passed in, --port is not "
                            "allowed.")
        if redis_shard_ports is not None:
            cli_logger.abort("`{}` should not be specified without `{}`.",
                             cf.bold("--redis-shard-ports"), cf.bold("--head"))

            raise Exception("If --head is not passed in, --redis-shard-ports "
                            "is not allowed.")
        if redis_address is None:
            cli_logger.abort("`{}` is required unless starting with `{}`.",
                             cf.bold("--address"), cf.bold("--head"))

            raise Exception("If --head is not passed in, --address must "
                            "be provided.")

        # Wait for the Redis server to be started. And throw an exception if we
        # can't connect to it.
        services.wait_for_redis_to_start(
            redis_address_ip, redis_address_port, password=redis_password)

        # Create a Redis client.
        redis_client = services.create_redis_client(
            redis_address, password=redis_password)

        # Check that the version information on this node matches the version
        # information that the cluster was started with.
        services.check_version_info(redis_client)

        # Get the node IP address if one is not provided.
        start_params.update_if_absent(
            node_ip_address=services.get_node_ip_address(redis_address))

        cli_logger.labeled_value("Local node IP", start_params.node_ip_address)

        start_params.update(redis_address=redis_address)
        node = NodeServicesStarter(
            start_params, head=False, shutdown_at_exit=False, spawn_reaper=False)

    startup_msg = "CloudTik runtime started."
    cli_logger.success(startup_msg)
    cli_logger.flush()


@node.command()
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="If set, will send SIGKILL instead of SIGTERM.")
@add_click_logging_options
def stop(force):
    """Stop CloudTik processes on the local machine."""

    is_linux = sys.platform.startswith("linux")
    processes_to_kill = CLOUDTIK_PROCESSES

    process_infos = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            process_infos.append((proc, proc.name(), proc.cmdline()))
        except psutil.Error:
            pass

    total_found = 0
    total_stopped = 0
    stopped = []
    for keyword, filter_by_cmd, _, _ in processes_to_kill:
        if filter_by_cmd and is_linux and len(keyword) > 15:
            # getting here is an internal bug, so we do not use cli_logger
            msg = ("The filter string should not be more than {} "
                   "characters. Actual length: {}. Filter: {}").format(
                       15, len(keyword), keyword)
            raise ValueError(msg)

        found = []
        for candidate in process_infos:
            proc, proc_cmd, proc_args = candidate
            corpus = (proc_cmd
                      if filter_by_cmd else subprocess.list2cmdline(proc_args))
            if keyword in corpus:
                found.append(candidate)

        for proc, proc_cmd, proc_args in found:
            total_found += 1

            proc_string = str(subprocess.list2cmdline(proc_args))
            try:
                if force:
                    proc.kill()
                else:
                    # TODO: On Windows, this is forceful termination.
                    # We don't want CTRL_BREAK_EVENT, because that would
                    # terminate the entire process group. What to do?
                    proc.terminate()

                if force:
                    cli_logger.verbose("Killed `{}` {} ", cf.bold(proc_string),
                                       cf.dimmed("(via SIGKILL)"))
                else:
                    cli_logger.verbose("Send termination request to `{}` {}",
                                       cf.bold(proc_string),
                                       cf.dimmed("(via SIGTERM)"))

                total_stopped += 1
                stopped.append(proc)
            except psutil.NoSuchProcess:
                cli_logger.verbose(
                    "Attempted to stop `{}`, but process was already dead.",
                    cf.bold(proc_string))
                total_stopped += 1
            except (psutil.Error, OSError) as ex:
                cli_logger.error("Could not terminate `{}` due to {}",
                                 cf.bold(proc_string), str(ex))

    if total_found == 0:
        cli_logger.print("Did not find any active processes.")
    else:
        if total_stopped == total_found:
            cli_logger.success("Stopped all {} processes.", total_stopped)
        else:
            cli_logger.warning(
                "Stopped only {} out of {} processes. "
                "Set `{}` to see more details.", total_stopped, total_found,
                cf.bold("-v"))
            cli_logger.warning("Try running the command again, or use `{}`.",
                               cf.bold("--force"))

    try:
        os.remove(
            os.path.join(get_cloudtik_temp_dir(),
                         "cloudtik_current_cluster"))
    except OSError:
        # This just means the file doesn't exist.
        pass
    # Wait for the processes to actually stop.
    psutil.wait_procs(stopped, timeout=2)


@node.command(context_settings={"ignore_unknown_options": True})
@click.argument("script", required=True, type=str)
@click.argument("script_args", nargs=-1)
def run(script, script_args):
    """Runs a built-in script (bash or python or a registered command).

    If you want to execute any commands or user scripts, use exec or submit.
    """
    run_script(script, script_args)


@node.command()
@click.option(
    "--cpu",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
    help="Show total CPU available in the current environment - considering docker or K8S.")
@click.option(
    "--memory",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
    help="Show total memory in the current environment - considering docker or K8S.")
@click.option(
    "--in-mb",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
    help="Show total memory in MB.")
def resources(cpu, memory, in_mb):
    """Show system resource information of the node"""
    resource_spec = ResourceSpec().resolve(is_head=False, available_memory=False)
    if cpu:
        click.echo(resource_spec.num_cpus)
    elif memory:
        if in_mb:
            memory_in_mb = int(resource_spec.memory / (1024 * 1024))
            click.echo(memory_in_mb)
        else:
            click.echo(resource_spec.memory)
    else:
        static_resources = resource_spec.to_resource_dict()
        click.echo(static_resources)


@node.command()
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
    help="Collect debug_state.txt from session dir")
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
    "--runtimes",
    required=False,
    type=str,
    default=None,
    help="The list of runtimes to collect logs from")
@click.option(
    "--silent",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
    help="Whether print a informational message.")
@add_click_logging_options
def dump(
        stream: bool = False,
        output: Optional[str] = None,
        logs: bool = True,
        debug_state: bool = True,
        pip: bool = True,
        processes: bool = True,
        processes_verbose: bool = False,
        tempfile: Optional[str] = None,
        runtimes: str = None,
        silent: bool = False):
    """Collect local data and package into an archive.

    Usage:

        cloudtik node dump [--stream/--output file]

    This script is called on remote nodes to fetch their data.
    """
    dump_local(
        stream=stream,
        output=output,
        logs=logs,
        debug_state=debug_state,
        pip=pip,
        processes=processes,
        processes_verbose=processes_verbose,
        tempfile=tempfile,
        runtimes=runtimes,
        silent=silent)


# core commands running on head and worker node
node.add_command(start)
node.add_command(stop)
node.add_command(run)
node.add_command(resources)

# utility commands running on head or worker node for dump local data
node.add_command(dump)
