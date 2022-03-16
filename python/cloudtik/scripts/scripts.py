from typing import Optional

import click
import copy
import json
import logging
import os
import subprocess
import sys
import time
import urllib
import urllib.parse
from socket import socket

import psutil

from cloudtik.core._private import services, utils, logging_utils
from cloudtik.core._private.state import kv_store, control_state
from cloudtik.core._private.parameter import StartParams
from cloudtik.core._private.node.node_services import NodeServicesStarter

from cloudtik.core._private.cluster.cluster_operator import (
    attach_cluster, exec_cluster, create_or_update_cluster, monitor_cluster,
    rsync, teardown_cluster, get_head_node_ip, kill_node, get_worker_node_ips,
    get_cluster_dump_archive, decode_debug_status, get_local_dump_archive, get_cluster_dump_archive_on_head,
    show_cluster_info, show_cluster_status, RUN_ENV_TYPES, start_proxy, stop_proxy, cluster_debug_status,
    cluster_health_check, teardown_cluster_on_head, cluster_process_status_on_head, cluster_process_status)
from cloudtik.core._private.constants import CLOUDTIK_PROCESSES, \
    CLOUDTIK_REDIS_DEFAULT_PASSWORD, \
    CLOUDTIK_KV_NAMESPACE_HEALTHCHECK, \
    CLOUDTIK_DEFAULT_PORT
from cloudtik.core._private import constants

from cloudtik.core._private.utils import CLOUDTIK_CLUSTER_SCALING_ERROR, \
    CLOUDTIK_CLUSTER_SCALING_STATUS
from cloudtik.core._private.cli_logger import (add_click_logging_options,
                                                cli_logger, cf)
from cloudtik.scripts.workspace import workspace

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--logging-level",
    required=False,
    default=constants.LOGGER_LEVEL,
    type=str,
    help=constants.LOGGER_LEVEL_HELP)
@click.option(
    "--logging-format",
    required=False,
    default=constants.LOGGER_FORMAT,
    type=str,
    help=constants.LOGGER_FORMAT_HELP)
@click.version_option()
def cli(logging_level, logging_format):
    level = logging.getLevelName(logging_level.upper())
    logging_utils.setup_logger(level, logging_format)
    cli_logger.set_format(format_tmpl=logging_format)


@cli.command(hidden=True)
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
@add_click_logging_options
def start(node_ip_address, address, port, head,
          redis_password, redis_shard_ports, redis_max_memory,
          memory, num_cpus, num_gpus, resources,
          cluster_scaling_config, temp_dir, metrics_export_port,
          no_redirect_output):
    """Start the main daemon processes on the local machine."""
    # Convert hostnames to numerical IP address.
    if node_ip_address is not None:
        node_ip_address = services.address_to_ip(node_ip_address)
    redirect_output = None if not no_redirect_output else True

    try:
        resources = json.loads(resources)
    except Exception:
        cli_logger.error("`{}` is not a valid JSON string.",
                         cf.bold("--resources"))
        cli_logger.abort(
            "Valid values look like this: `{}`",
            cf.bold("--resources='{\"CustomResource3\": 1, "
                    "\"CustomResource2\": 2}'"))

        raise Exception("Unable to parse the --resources argument using "
                        "json.loads. Try using a format like\n\n"
                        "    --resources='{\"CustomResource1\": 3, "
                        "\"CustomReseource2\": 2}'")

    start_params = StartParams(
        node_ip_address=node_ip_address,
        memory=memory,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        resources=resources,
        temp_dir=temp_dir,
        metrics_export_port=metrics_export_port,
        redirect_output=redirect_output)
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
                    f" command to `cloudtik start`.")

        node = NodeServicesStarter(
            start_params, head=True, shutdown_at_exit=False, spawn_reaper=False)

        redis_address = node.redis_address
        if temp_dir is None:
            # Default temp directory.
            temp_dir = utils.get_user_temp_dir()
        # Using the user-supplied temp dir unblocks on-prem
        # users who can't write to the default temp.
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

    cli_logger.newline()
    startup_msg = "CloudTik runtime started."
    cli_logger.success("-" * len(startup_msg))
    cli_logger.success(startup_msg)
    cli_logger.success("-" * len(startup_msg))
    cli_logger.newline()
    cli_logger.print("To terminate CloudTik runtime, run")
    cli_logger.print(cf.bold("  cloudtik stop"))
    cli_logger.flush()


@cli.command(hidden=True)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="If set, will send SIGKILL instead of SIGTERM.")
@add_click_logging_options
def stop(force):
    """Stop CloudTik processes manually on the local machine."""

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
            os.path.join(utils.get_user_temp_dir(),
                         "cloudtik_current_cluster"))
    except OSError:
        # This just means the file doesn't exist.
        pass
    # Wait for the processes to actually stop.
    psutil.wait_procs(stopped, timeout=2)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--min-workers",
    required=False,
    type=int,
    help="Override the configured min worker node count for the cluster.")
@click.option(
    "--max-workers",
    required=False,
    type=int,
    help="Override the configured max worker node count for the cluster.")
@click.option(
    "--no-restart",
    is_flag=True,
    default=False,
    help=("Whether to skip restarting services during the update. "
          "This avoids interrupting running jobs."))
@click.option(
    "--restart-only",
    is_flag=True,
    default=False,
    help=("Whether to skip running setup commands and only restart. "
          "This cannot be used with 'no-restart'."))
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Don't ask for confirmation.")
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@click.option(
    "--workspace-name",
    required=False,
    type=str,
    help="Override the workspace which will provide network service.")
@click.option(
    "--no-config-cache",
    is_flag=True,
    default=False,
    help="Disable the local cluster config cache.")
@click.option(
    "--redirect-command-output",
    is_flag=True,
    default=False,
    help="Whether to redirect command output to a file.")
@click.option(
    "--use-login-shells/--use-normal-shells",
    is_flag=True,
    default=True,
    help=("We uses login shells (bash --login -i) to run cluster commands "
          "by default. If your workflow is compatible with normal shells, "
          "this can be disabled for a better user experience."))
@add_click_logging_options
def up(cluster_config_file, min_workers, max_workers, no_restart, restart_only,
       yes, cluster_name, workspace_name, no_config_cache, redirect_command_output,
       use_login_shells):
    """Create or update a cluster."""
    if restart_only or no_restart:
        cli_logger.doassert(restart_only != no_restart,
                            "`{}` is incompatible with `{}`.",
                            cf.bold("--restart-only"), cf.bold("--no-restart"))
        assert restart_only != no_restart, "Cannot set both 'restart_only' " \
            "and 'no_restart' at the same time!"

    if urllib.parse.urlparse(cluster_config_file).scheme in ("http", "https"):
        try:
            response = urllib.request.urlopen(cluster_config_file, timeout=5)
            content = response.read()
            file_name = cluster_config_file.split("/")[-1]
            with open(file_name, "wb") as f:
                f.write(content)
            cluster_config_file = file_name
        except urllib.error.HTTPError as e:
            cli_logger.warning("{}", str(e))
            cli_logger.warning(
                "Could not download remote cluster configuration file.")
    create_or_update_cluster(
        config_file=cluster_config_file,
        override_min_workers=min_workers,
        override_max_workers=max_workers,
        no_restart=no_restart,
        restart_only=restart_only,
        yes=yes,
        override_cluster_name=cluster_name,
        override_workspace_name=workspace_name,
        no_config_cache=no_config_cache,
        redirect_command_output=redirect_command_output,
        use_login_shells=use_login_shells)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Don't ask for confirmation.")
@click.option(
    "--workers-only",
    is_flag=True,
    default=False,
    help="Only destroy the workers.")
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@click.option(
    "--keep-min-workers",
    is_flag=True,
    default=False,
    help="Retain the minimal amount of workers specified in the config.")
@add_click_logging_options
def down(cluster_config_file, yes, workers_only, cluster_name,
         keep_min_workers):
    """Tear down a cluster."""
    teardown_cluster(cluster_config_file, yes, workers_only, cluster_name,
                     keep_min_workers, True)


@cli.command(hidden=True)
@click.argument("cluster_config_file", required=True, type=str)
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
    help="Terminates the node via node provider (defaults to a 'soft kill'"
    " which terminates the system but does not actually delete the instances).")
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
def kill_random_node(cluster_config_file, yes, hard, cluster_name):
    """Kills a random node. For testing purposes only."""
    click.echo("Killed node with IP " +
               kill_node(cluster_config_file, yes, hard, cluster_name))


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--lines",
    required=False,
    default=100,
    type=int,
    help="Number of lines to tail.")
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@add_click_logging_options
def monitor(cluster_config_file, lines, cluster_name):
    """Tails the monitor logs of a cluster."""
    monitor_cluster(cluster_config_file, lines, cluster_name)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--start",
    is_flag=True,
    default=False,
    help="Start the cluster if needed.")
@click.option(
    "--screen", is_flag=True, default=False, help="Run the command in screen.")
@click.option(
    "--tmux", is_flag=True, default=False, help="Run the command in tmux.")
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
    "--new", "-N", is_flag=True, help="Force creation of a new screen.")
@click.option(
    "--port-forward",
    "-p",
    required=False,
    multiple=True,
    type=int,
    help="Port to forward. Use this multiple times to forward multiple ports.")
@add_click_logging_options
def attach(cluster_config_file, start, screen, tmux, cluster_name,
           no_config_cache, new, port_forward):
    """Create or attach to SH session to a cluster."""
    port_forward = [(port, port) for port in list(port_forward)]
    attach_cluster(
        cluster_config_file,
        start,
        screen,
        tmux,
        cluster_name,
        no_config_cache=no_config_cache,
        new=new,
        port_forward=port_forward)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--no-config-cache",
    is_flag=True,
    default=False,
    help="Disable the local cluster config cache.")
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@add_click_logging_options
def enable_local_access(cluster_config_file, no_config_cache, cluster_name):
    """Enable local SOCKS5 proxy to the cluster through SSH tunnel forwarding to the head."""
    start_proxy(
        cluster_config_file,
        override_cluster_name=cluster_name,
        no_config_cache=no_config_cache)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@add_click_logging_options
def disable_local_access(cluster_config_file,cluster_name):
    """Disable the local SOCKS5 proxy to the cluster."""
    stop_proxy(cluster_config_file,cluster_name)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.argument("source", required=False, type=str)
@click.argument("target", required=False, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@add_click_logging_options
def rsync_down(cluster_config_file, source, target, cluster_name):
    """Download specific files from a cluster."""
    rsync(cluster_config_file, source, target, cluster_name, down=True)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.argument("source", required=False, type=str)
@click.argument("target", required=False, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@click.option(
    "--all-nodes",
    "-A",
    is_flag=True,
    required=False,
    help="Upload to all nodes (workers and head).")
@add_click_logging_options
def rsync_up(cluster_config_file, source, target, cluster_name, all_nodes):
    """Upload specific files to a cluster."""
    if all_nodes:
        cli_logger.warning(
            "WARNING: the `all_nodes` option is deprecated and will be "
            "removed in the future. "
            "Rsync to worker nodes is not reliable since workers may be "
            "added during autoscaling. Please use the `file_mounts` "
            "feature instead for consistent file sync in autoscaling clusters")

    rsync(
        cluster_config_file,
        source,
        target,
        cluster_name,
        down=False,
        all_nodes=all_nodes)


@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--stop",
    is_flag=True,
    default=False,
    help="Stop the cluster after the command finishes running.")
@click.option(
    "--start",
    is_flag=True,
    default=False,
    help="Start the cluster if needed.")
@click.option(
    "--screen",
    is_flag=True,
    default=False,
    help="Run the command in a screen.")
@click.option(
    "--tmux", is_flag=True, default=False, help="Run the command in tmux.")
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
    "--port-forward",
    "-p",
    required=False,
    multiple=True,
    type=int,
    help="Port to forward. Use this multiple times to forward multiple ports.")
@click.argument("script", required=True, type=str)
@click.argument("script_args", nargs=-1)
@add_click_logging_options
def submit(cluster_config_file, screen, tmux, stop, start, cluster_name,
           no_config_cache, port_forward, script, script_args):
    """Uploads and runs a script on the specified cluster.

    The script is automatically synced to the following location:

        os.path.join("~", os.path.basename(script))

    Example:
        >>> cloudtik submit [CLUSTER.YAML] experiment.py -- --smoke-test
    """
    cli_logger.doassert(not (screen and tmux),
                        "`{}` and `{}` are incompatible.", cf.bold("--screen"),
                        cf.bold("--tmux"))

    assert not (screen and tmux), "Can specify only one of `screen` or `tmux`."
    assert not script_args, "Use -- --arg1 --arg2 for script args."

    if start:
        create_or_update_cluster(
            config_file=cluster_config_file,
            override_min_workers=None,
            override_max_workers=None,
            no_restart=False,
            restart_only=False,
            yes=True,
            override_cluster_name=cluster_name,
            no_config_cache=no_config_cache,
            redirect_command_output=False,
            use_login_shells=True)
    target = os.path.basename(script)
    target = os.path.join("~", target)
    rsync(
        cluster_config_file,
        script,
        target,
        cluster_name,
        no_config_cache=no_config_cache,
        down=False)

    command_parts = ["python", target]
    if script_args:
        command_parts += list(script_args)

    port_forward = [(port, port) for port in list(port_forward)]
    cmd = " ".join(command_parts)
    exec_cluster(
        cluster_config_file,
        cmd=cmd,
        run_env="docker",
        screen=screen,
        tmux=tmux,
        stop=stop,
        start=False,
        override_cluster_name=cluster_name,
        no_config_cache=no_config_cache,
        port_forward=port_forward)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.argument("cmd", required=True, type=str)
@click.option(
    "--run-env",
    required=False,
    type=click.Choice(RUN_ENV_TYPES),
    default="auto",
    help="Choose whether to execute this command in a container or directly on"
    " the cluster head. Only applies when docker is configured in the YAML.")
@click.option(
    "--stop",
    is_flag=True,
    default=False,
    help="Stop the cluster after the command finishes running.")
@click.option(
    "--start",
    is_flag=True,
    default=False,
    help="Start the cluster if needed.")
@click.option(
    "--screen",
    is_flag=True,
    default=False,
    help="Run the command in a screen.")
@click.option(
    "--tmux", is_flag=True, default=False, help="Run the command in tmux.")
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
    "--port-forward",
    "-p",
    required=False,
    multiple=True,
    type=int,
    help="Port to forward. Use this multiple times to forward multiple ports.")
@add_click_logging_options
def exec(cluster_config_file, cmd, run_env, screen, tmux, stop, start,
         cluster_name, no_config_cache, port_forward):
    """Execute a command via SSH on a cluster."""
    port_forward = [(port, port) for port in list(port_forward)]

    exec_cluster(
        cluster_config_file,
        cmd=cmd,
        run_env=run_env,
        screen=screen,
        tmux=tmux,
        stop=stop,
        start=start,
        override_cluster_name=cluster_name,
        no_config_cache=no_config_cache,
        port_forward=port_forward,
        _allow_uninitialized_state=True)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
def get_head_ip(cluster_config_file, cluster_name):
    """Return the head node IP of a cluster."""
    click.echo(get_head_node_ip(cluster_config_file, cluster_name))


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
def get_worker_ips(cluster_config_file, cluster_name):
    """Return the list of worker IPs of a cluster."""
    worker_ips = get_worker_node_ips(cluster_config_file, cluster_name)
    click.echo("\n".join(worker_ips))


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@add_click_logging_options
def info(cluster_config_file, cluster_name):
    """Show cluster summary information and useful links to use the cluster."""
    show_cluster_info(
        cluster_config_file,
        cluster_name)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@add_click_logging_options
def status(cluster_config_file, cluster_name):
    """Show cluster summary status."""
    show_cluster_status(
        cluster_config_file,
        cluster_name)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@add_click_logging_options
def process_status(cluster_config_file, cluster_name):
    """Show process status of cluster nodes."""
    cluster_process_status(
        cluster_config_file,
        cluster_name)


@cli.command(hidden=True)
@click.option(
    "--redis_address", required=True, type=str, help="the address to redis", default="127.0.0.1")
@add_click_logging_options
def process_status_on_head(redis_address):
    """Show cluster process status."""
    cluster_process_status_on_head(
        redis_address)


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@add_click_logging_options
def debug_status(cluster_config_file, cluster_name):
    """Show debug status of cluster scaling."""
    cluster_debug_status(cluster_config_file, cluster_name)


@cli.command(hidden=True)
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
def debug_status_on_head(address, redis_password):
    """Print cluster status, including autoscaling info."""
    if not address:
        address = services.get_address_to_use_or_die()
    redis_client = services.create_redis_client(
        address, redis_password)
    state_client = control_state.StateClient.create_from_redis(
        redis_client)
    kv_store.kv_initialize(state_client)
    status = kv_store.kv_get(
        CLOUDTIK_CLUSTER_SCALING_STATUS)
    error = kv_store.kv_get(
        CLOUDTIK_CLUSTER_SCALING_ERROR)
    print(decode_debug_status(status, error))


@cli.command(hidden=True)
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
def local_dump(stream: bool = False,
               output: Optional[str] = None,
               logs: bool = True,
               debug_state: bool = True,
               pip: bool = True,
               processes: bool = True,
               processes_verbose: bool = False,
               tempfile: Optional[str] = None):
    """Collect local data and package into an archive.

    Usage:

        cloudtik local-dump [--stream/--output file]

    This script is called on remote nodes to fetch their data.
    """
    # This may stream data to stdout, so no printing here
    get_local_dump_archive(
        stream=stream,
        output=output,
        logs=logs,
        debug_state=debug_state,
        pip=pip,
        processes=processes,
        processes_verbose=processes_verbose,
        tempfile=tempfile)


@cli.command(hidden=True)
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
def cluster_dump_on_head(host: Optional[str] = None,
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

            cloudtik local-dump [--stream/--output file]

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


@cli.command()
@click.argument("cluster_config_file", required=False, type=str)
@click.option(
    "--host",
    "-h",
    required=False,
    type=str,
    help="Single or list of hosts, separated by comma.")
@click.option(
    "--ssh-user",
    "-U",
    required=False,
    type=str,
    default=None,
    help="Username of the SSH user.")
@click.option(
    "--ssh-key",
    "-K",
    required=False,
    type=str,
    default=None,
    help="Path to the SSH key file.")
@click.option(
    "--docker",
    "-d",
    required=False,
    type=str,
    default=None,
    help="Name of the docker container, if applicable.")
@click.option(
    "--local",
    "-L",
    required=False,
    type=bool,
    is_flag=True,
    default=None,
    help="Also include information about the local node.")
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
def cluster_dump(cluster_config_file: Optional[str] = None,
                 host: Optional[str] = None,
                 ssh_user: Optional[str] = None,
                 ssh_key: Optional[str] = None,
                 docker: Optional[str] = None,
                 local: Optional[bool] = None,
                 output: Optional[str] = None,
                 logs: bool = True,
                 debug_state: bool = True,
                 pip: bool = True,
                 processes: bool = True,
                 processes_verbose: bool = False,
                 tempfile: Optional[str] = None):
    """Get log data from one or more nodes.

    Best used with cluster configs:

        cloudtik cluster-dump [cluster.yaml]

    Include the --local flag to also collect and include data from the
    local node.

    Missing fields will be tried to be auto-filled.

    You can also manually specify a list of hosts using the
    ``--host <host1,host2,...>`` parameter.
    """
    archive_path = get_cluster_dump_archive(
        cluster_config_file=cluster_config_file,
        host=host,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        docker=docker,
        local=local,
        output=output,
        logs=logs,
        debug_state=debug_state,
        pip=pip,
        processes=processes,
        processes_verbose=processes_verbose,
        tempfile=tempfile)
    if archive_path:
        click.echo(f"Created archive: {archive_path}")
    else:
        click.echo("Could not create archive.")


@cli.command()
@click.argument("cluster_config_file", required=True, type=str)
@click.option(
    "--cluster-name",
    "-n",
    required=False,
    type=str,
    help="Override the configured cluster name.")
@add_click_logging_options
def health_check(cluster_config_file, cluster_name):
    """Do cluster health check."""
    cluster_health_check(cluster_config_file, cluster_name)


@cli.command(hidden=True)
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
def health_check_on_head(address, redis_password, component):
    """
    This is NOT a public api.

    Health check a cluster or a specific component. Exit code 0 is healthy.
    """

    if not address:
        address = services.get_address_to_use_or_die()
    else:
        address = services.address_to_ip(address)
    redis_client = services.create_redis_client(
        address, redis_password)

    if not component:
        # If no component is specified, we are health checking the core. If
        # client creation or ping fails, we will still exit with a non-zero
        # exit code.
        redis_client.ping()
        """
        try:
            # TODO: check head and worker status through control state such as heartbeat
            
        except Exception:
            pass
        """
        sys.exit(1)
    state_client = control_state.StateClient.create_from_redis(
        redis_client)
    kv_store.kv_initialize(state_client)
    report_str = kv_store.kv_get(
        component, namespace=CLOUDTIK_KV_NAMESPACE_HEALTHCHECK)
    if not report_str:
        # Status was never updated
        sys.exit(1)

    report = json.loads(report_str)

    # TODO: We probably shouldn't rely on time here, but cloud providers
    # have very well synchronized NTP servers, so this should be fine in
    # practice.
    cur_time = time.time()
    report_time = float(report["time"])

    # If the status is too old, the service has probably already died.
    delta = cur_time - report_time
    time_ok = delta < constants.HEALTHCHECK_EXPIRATION_S

    if time_ok:
        sys.exit(0)
    else:
        sys.exit(1)


@cli.command(hidden=True)
@click.option(
    "--keep-min-workers",
    is_flag=True,
    default=False,
    help="Retain the minimal amount of workers specified in the config.")
@add_click_logging_options
def teardown_on_head(keep_min_workers):
    """Tear down a cluster."""
    teardown_cluster_on_head(keep_min_workers)


def add_command_alias(command, name, hidden):
    new_command = copy.deepcopy(command)
    new_command.hidden = hidden
    cli.add_command(new_command, name=name)


# core commands running on head and worker node
cli.add_command(start)
cli.add_command(stop)

# commands running on working node for handling a cluster
cli.add_command(up)
cli.add_command(down)

cli.add_command(attach)
cli.add_command(exec)
cli.add_command(submit)

cli.add_command(rsync_down)
add_command_alias(rsync_down, name="rsync_down", hidden=True)
cli.add_command(rsync_up)
add_command_alias(rsync_up, name="rsync_up", hidden=True)

cli.add_command(enable_local_access)
cli.add_command(disable_local_access)

# commands running on working node for information and status
cli.add_command(get_head_ip)
add_command_alias(get_head_ip, name="get_head_ip", hidden=True)
cli.add_command(get_worker_ips)
add_command_alias(get_worker_ips, name="get_worker_ips", hidden=True)

cli.add_command(info)
cli.add_command(status)
cli.add_command(process_status)
cli.add_command(monitor)

# commands running on working node for debug
cli.add_command(cluster_dump)
add_command_alias(cluster_dump, name="cluster_dump", hidden=True)

cli.add_command(kill_random_node)
add_command_alias(kill_random_node, name="kill_random_node", hidden=True)

cli.add_command(debug_status)
cli.add_command(health_check)

# utility commands running on head or worker node for dump local data
cli.add_command(local_dump)
add_command_alias(local_dump, name="local_dump", hidden=True)

# utility commands running on head node
cli.add_command(teardown_on_head)
cli.add_command(cluster_dump_on_head)
cli.add_command(debug_status_on_head)
cli.add_command(process_status_on_head)
cli.add_command(health_check_on_head)

# workspace commands
cli.add_command(workspace)


def main():
    return cli()


if __name__ == "__main__":
    main()
