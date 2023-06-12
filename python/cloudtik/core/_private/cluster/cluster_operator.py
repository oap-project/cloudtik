import copy
import datetime
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib
import urllib.parse
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import prettytable as pt
import psutil
import yaml

from cloudtik.core import tags
from cloudtik.core._private import services, constants
from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.cluster.cluster_config import _load_cluster_config, _bootstrap_config, try_logging_config
from cloudtik.core._private.cluster.cluster_tunnel_request import request_tunnel_to_head
from cloudtik.core._private.cluster.cluster_utils import create_node_updater_for_exec
from cloudtik.core._private.cluster.resource_demand_scheduler import ResourceDict, \
    get_node_type_counts, get_unfulfilled_for_bundles
from cloudtik.core._private.core_utils import kill_process_tree, double_quote, get_cloudtik_temp_dir, get_free_port, \
    memory_to_gb, memory_to_gb_string
from cloudtik.core._private.job_waiter.job_waiter_factory import create_job_waiter
from cloudtik.core._private.runtime_factory import _get_runtime_cls
from cloudtik.core._private.services import validate_redis_address
from cloudtik.core._private.state import kv_store
from cloudtik.core.job_waiter import JobWaiter

try:  # py3
    from shlex import quote
except ImportError:  # py2
    from pipes import quote

from cloudtik.core._private.state.kv_store import kv_put, kv_initialize_with_address, kv_get

from cloudtik.core.node_provider import NodeProvider
from cloudtik.core._private.constants import \
    CLOUDTIK_RESOURCE_REQUESTS, \
    MAX_PARALLEL_SHUTDOWN_WORKERS, \
    CLOUDTIK_REDIS_DEFAULT_PASSWORD, CLOUDTIK_CLUSTER_STATUS_STOPPED, CLOUDTIK_CLUSTER_STATUS_RUNNING, \
    CLOUDTIK_RUNTIME_NAME, CLOUDTIK_KV_NAMESPACE_HEALTHCHECK
from cloudtik.core._private.utils import hash_runtime_conf, \
    hash_launch_conf, get_proxy_process_file, get_safe_proxy_process, \
    get_head_working_ip, get_node_cluster_ip, is_use_internal_ip, \
    get_attach_command, is_alive_time, is_docker_enabled, get_proxy_bind_address_to_show, \
    with_runtime_environment_variables, get_nodes_info, \
    sum_worker_cpus, sum_worker_memory, get_runtime_services, get_enabled_runtimes, \
    with_node_ip_environment_variables, run_in_parallel_on_nodes, get_commands_to_run, \
    cluster_booting_completed, load_head_cluster_config, get_runnable_command, get_cluster_uri, \
    with_head_node_ip_environment_variables, get_verified_runtime_list, get_commands_of_runtimes, \
    is_node_in_completed_status, check_for_single_worker_type, \
    get_node_specific_commands_of_runtimes, _get_node_specific_runtime_config, \
    RUNTIME_CONFIG_KEY, DOCKER_CONFIG_KEY, get_running_head_node, \
    get_nodes_for_runtime, with_script_args, encrypt_config, convert_nodes_to_resource, \
    HeadNotRunningError, get_cluster_head_ip, get_command_session_name, ParallelTaskSkipped, \
    CLOUDTIK_CLUSTER_SCALING_STATUS, decode_cluster_scaling_time, RUNTIME_TYPES_CONFIG_KEY, get_node_info, \
    NODE_INFO_NODE_IP, get_cpus_of_node_info, _sum_min_workers, get_memory_of_node_info, sum_worker_gpus, \
    sum_nodes_resource, get_gpus_of_node_info, get_resource_of_node_info, get_resource_info_of_node_type, \
    get_worker_node_type, save_server_process, get_resource_requests_for, _get_head_resource_requests, \
    get_resource_list_str, with_verbose_option, run_script

from cloudtik.core._private.providers import _get_node_provider, _NODE_PROVIDERS
from cloudtik.core.tags import (
    CLOUDTIK_TAG_NODE_KIND, CLOUDTIK_TAG_LAUNCH_CONFIG, CLOUDTIK_TAG_NODE_NAME,
    NODE_KIND_WORKER, NODE_KIND_HEAD, CLOUDTIK_TAG_USER_NODE_TYPE,
    STATUS_UNINITIALIZED, STATUS_UP_TO_DATE, CLOUDTIK_TAG_NODE_STATUS, STATUS_UPDATE_FAILED, CLOUDTIK_TAG_NODE_NUMBER,
    CLOUDTIK_TAG_HEAD_NODE_NUMBER)
from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.node.node_updater import NodeUpdaterThread
from cloudtik.core._private.event_system import (CreateClusterEvent, global_event_system)
from cloudtik.core._private.log_timer import LogTimer
from cloudtik.core._private.cluster.cluster_dump import Archive, \
    GetParameters, Node, _get_nodes_to_dump, \
    add_archive_for_remote_nodes, get_all_local_data, \
    add_archive_for_cluster_nodes, add_archive_for_local_node
from cloudtik.core._private.state.control_state import ControlState

from cloudtik.core._private.cluster.cluster_metrics import ClusterMetricsSummary
from cloudtik.core._private.cluster.cluster_scaler import ClusterScalerSummary
from cloudtik.core._private.utils import format_info_string

logger = logging.getLogger(__name__)

RUN_ENV_TYPES = ["auto", "host", "docker"]

POLL_INTERVAL = 5

Port_forward = Union[Tuple[int, int], List[Tuple[int, int]]]

NUM_TEARDOWN_CLUSTER_STEPS_BASE = 2


# The global shared CLI call context
_cli_call_context = CallContext()


def cli_call_context() -> CallContext:
    return _cli_call_context


def decode_cluster_scaling_status(status):
    status = status.decode("utf-8")
    as_dict = json.loads(status)
    time = datetime.datetime.fromtimestamp(as_dict["time"])
    cluster_metrics_summary = ClusterMetricsSummary(**as_dict["cluster_metrics_report"])
    scaler_summary = ClusterScalerSummary(**as_dict["cluster_scaler_report"])
    return time, cluster_metrics_summary, scaler_summary


def debug_status_string(status, error) -> str:
    """Return a debug string for the cluster scaler."""
    if not status:
        status = "No cluster status."
    else:
        time, cluster_metrics_summary, scaler_summary = decode_cluster_scaling_status(status)
        status = format_info_string(cluster_metrics_summary, scaler_summary, time=time)
    if error:
        status += "\n"
        status += error.decode("utf-8")
    return status


def request_resources(num_cpus: Optional[int] = None,
                      num_gpus: Optional[int] = None,
                      resources: Optional[Dict[str, int]] = None,
                      bundles: Optional[List[dict]] = None,
                      config: Dict[str, Any] = None) -> None:
    to_request = []
    if num_cpus:
        to_request += get_resource_requests_for(
            config, constants.CLOUDTIK_RESOURCE_CPU, num_cpus)
    if num_gpus:
        to_request += get_resource_requests_for(
            config, constants.CLOUDTIK_RESOURCE_GPU, num_gpus)
    elif resources:
        if resources:
            for resource_name, resource_amount in resources.items():
                to_request += get_resource_requests_for(
                    config, resource_name, resource_amount)

    _request_resources(resources=to_request, bundles=bundles)


def _request_resources(resources: Optional[List[dict]] = None,
                       bundles: Optional[List[dict]] = None) -> None:
    """Remotely request some CPU or GPU resources from the cluster scaler.

    This function is to be called e.g. on a node before submitting a bunch of
    jobs to ensure that resources rapidly become available.

    Args:
        resources (List[ResourceDict]): Scale the cluster to ensure this number of CPUs/GPUs are
            available. This request is persistent until another call to
            request_resources() is made.
        bundles (List[ResourceDict]): Scale the cluster to ensure this set of
            resource shapes can fit. This request is persistent until another
            call to request_resources() is made.
    """
    to_request = []
    if resources:
        to_request += resources
    if bundles:
        to_request += bundles
    request_time = time.time()
    resource_requests = {
        "request_time": request_time,
        "requests": to_request
    }
    kv_put(
        CLOUDTIK_RESOURCE_REQUESTS,
        json.dumps(resource_requests),
        overwrite=True)


def create_or_update_cluster(
        config_file: str,
        call_context: CallContext,
        override_min_workers: Optional[int],
        override_max_workers: Optional[int],
        no_restart: bool,
        restart_only: bool,
        yes: bool,
        override_cluster_name: Optional[str] = None,
        override_workspace_name: Optional[str] = None,
        no_config_cache: bool = False,
        redirect_command_output: Optional[bool] = False,
        use_login_shells: bool = True) -> Dict[str, Any]:
    """Creates or updates an scaling cluster from a config json."""
    _cli_logger = call_context.cli_logger

    def handle_yaml_error(e):
        _cli_logger.error("Cluster config invalid")
        _cli_logger.newline()
        _cli_logger.error("Failed to load YAML file " + cf.bold("{}"),
                         config_file)
        _cli_logger.newline()
        with _cli_logger.verbatim_error_ctx("PyYAML error:"):
            _cli_logger.error(e)
        _cli_logger.abort()

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f.read())
    except FileNotFoundError:
        _cli_logger.abort(
            "Provided cluster configuration file ({}) does not exist",
            cf.bold(config_file))
    except yaml.parser.ParserError as e:
        handle_yaml_error(e)
        raise
    except yaml.scanner.ScannerError as e:
        handle_yaml_error(e)
        raise

    # TODO: validate file_mounts, ssh keys, etc.
    importer = _NODE_PROVIDERS.get(config["provider"]["type"])
    if not importer:
        _cli_logger.abort(
            "Unknown provider type " + cf.bold("{}") + "\n"
            "Available providers are: {}", config["provider"]["type"],
            _cli_logger.render_list([
                k for k in _NODE_PROVIDERS.keys()
                if _NODE_PROVIDERS[k] is not None
            ]))

    printed_overrides = False

    def handle_cli_override(key, override):
        if override is not None:
            if key in config:
                nonlocal printed_overrides
                printed_overrides = True
                cli_logger.warning(
                    "`{}` override provided on the command line.\n"
                    "  Using " + cf.bold("{}") + cf.dimmed(
                        " [configuration file has " + cf.bold("{}") + "]"),
                    key, override, config[key])
            config[key] = override

    handle_cli_override("min_workers", override_min_workers)
    handle_cli_override("max_workers", override_max_workers)
    handle_cli_override("cluster_name", override_cluster_name)
    handle_cli_override("workspace_name", override_workspace_name)

    if printed_overrides:
        _cli_logger.newline()

    _cli_logger.labeled_value("Cluster", config["cluster_name"])
    workspace_name = config.get("workspace_name")
    if workspace_name:
        _cli_logger.labeled_value("Workspace", workspace_name)
    _cli_logger.labeled_value("Runtimes", ", ".join(get_enabled_runtimes(config)))

    _cli_logger.newline()
    config = _bootstrap_config(config, no_config_cache=no_config_cache,
                               init_config_cache=True)
    _create_or_update_cluster(
        config,
        call_context=call_context,
        no_restart=no_restart,
        restart_only=restart_only,
        yes=yes,
        redirect_command_output=redirect_command_output,
        use_login_shells=use_login_shells
    )

    if is_proxy_needed(config):
        # start proxy and bind to localhost
        _cli_logger.newline()
        with _cli_logger.group("Starting SOCKS5 proxy..."):
            _start_proxy(config, True, "localhost")

    provider = _get_node_provider(config["provider"], config["cluster_name"])
    head_node = _get_running_head_node(config)
    show_useful_commands(call_context=call_context,
                         config=config,
                         provider=provider,
                         head_node=head_node,
                         config_file=config_file,
                         override_cluster_name=override_cluster_name)
    return config


def _create_or_update_cluster(
        config: Dict[str, Any],
        call_context: CallContext,
        no_restart: bool,
        restart_only: bool,
        yes: bool,
        redirect_command_output: Optional[bool] = False,
        use_login_shells: bool = True):
    global_event_system.execute_callback(
        get_cluster_uri(config),
        CreateClusterEvent.up_started,
        {"cluster_config": config})

    call_context.set_using_login_shells(use_login_shells)
    if not use_login_shells:
        call_context.set_allow_interactive(False)
    if redirect_command_output is None:
        # Do not redirect by default.
        call_context.set_output_redirected(False)
    else:
        call_context.set_output_redirected(redirect_command_output)

    try_logging_config(config)
    get_or_create_head_node(config, call_context, no_restart, restart_only,
                            yes)


def teardown_cluster(config_file: str, yes: bool, workers_only: bool,
                     override_cluster_name: Optional[str],
                     keep_min_workers: bool,
                     proxy_stop: bool = False,
                     hard: bool = False) -> None:
    """Destroys all nodes of a cluster described by a config json."""
    config = _load_cluster_config(config_file, override_cluster_name)

    cli_logger.confirm(yes, "Are you sure that you want to shut down cluster {}?",
                       config["cluster_name"], _abort=True)
    cli_logger.newline()
    with cli_logger.group("Shutting down cluster: {}", config["cluster_name"]):
        _teardown_cluster(config,
                          call_context=cli_call_context(),
                          workers_only=workers_only,
                          keep_min_workers=keep_min_workers,
                          proxy_stop=proxy_stop,
                          hard=hard)

    cli_logger.success("Successfully shut down cluster: {}.", config["cluster_name"])


def _teardown_cluster(config: Dict[str, Any],
                      call_context: CallContext,
                      workers_only: bool = False,
                      keep_min_workers: bool = False,
                      proxy_stop: bool = False,
                      hard: bool = False) -> None:
    current_step = 1
    total_steps = NUM_TEARDOWN_CLUSTER_STEPS_BASE
    if proxy_stop and is_proxy_needed(config):
        total_steps += 1
    if not hard:
        total_steps += 1
        if not workers_only:
            total_steps += 2

    if proxy_stop and is_proxy_needed(config):
        with cli_logger.group(
                "Stopping proxy",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _stop_proxy(config)
    if not hard:
        if not workers_only:
            with cli_logger.group(
                    "Requesting head to stop controller services",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                try:
                    _stop_node_from_head(
                        config,
                        call_context=call_context,
                        node_ip=None, all_nodes=False,
                        runtimes=[CLOUDTIK_RUNTIME_NAME],
                        indent_level=2)
                except Exception as e:
                    cli_logger.verbose_error("{}", str(e))
                    cli_logger.warning(
                        "Exception occurred when stopping controller services "
                        "(use -v to show details).")
                    cli_logger.warning(
                        "Ignoring the exception and "
                        "attempting to shut down the cluster nodes anyway.")

        # Running teardown cluster process on head first. But we allow this to fail.
        # Since head node problem should not prevent cluster tear down
        with cli_logger.group(
                "Requesting head to stop workers",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _teardown_workers_from_head(
                config,
                call_context,
                keep_min_workers=keep_min_workers)

        if not workers_only:
            with cli_logger.group(
                    "Requesting head to stop head services",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                try:
                    _stop_node_from_head(
                        config,
                        call_context=call_context,
                        node_ip=None, all_nodes=False,
                        indent_level=2)
                except Exception as e:
                    cli_logger.verbose_error("{}", str(e))
                    cli_logger.warning(
                        "Exception occurred when stopping head services "
                        "(use -v to show details).")
                    cli_logger.warning(
                        "Ignoring the exception and "
                        "attempting to shut down the cluster nodes anyway.")

    with cli_logger.group(
            "Stopping head and remaining nodes",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        provider = _get_node_provider(config["provider"], config["cluster_name"])
        # Since head node has down the workers shutdown
        # We continue shutdown the head and remaining workers
        teardown_cluster_nodes(config,
                               call_context=call_context,
                               provider=provider,
                               workers_only=workers_only,
                               keep_min_workers=keep_min_workers,
                               on_head=False,
                               hard=hard)

    with cli_logger.group(
            "Clean up the cluster",
            _numbered=("[]", current_step, total_steps)):
        current_step += 1
        cleanup_cluster(config,
                        call_context=call_context,
                        provider=provider)


def _teardown_workers_from_head(
        config: Dict[str, Any],
        call_context: CallContext,
        keep_min_workers: bool = False,
        hard: bool = False
):
    cmd = "cloudtik head teardown --yes"
    if keep_min_workers:
        cmd += " --keep-min-workers"
    if hard:
        cmd += " --hard"
    cmd += " --indent-level={}".format(2)

    try:
        _exec_cmd_on_cluster(config,
                             call_context=call_context,
                             cmd=cmd)
    except Exception as e:
        cli_logger.verbose_error("{}", str(e))
        cli_logger.warning(
            "Exception occurred when requesting head to stop the workers "
            "(use -v to show details).")
        cli_logger.warning(
            "Ignoring the exception and "
            "attempting to shut down the cluster nodes anyway.")


def cleanup_cluster(config: Dict[str, Any],
                    call_context: CallContext,
                    provider: NodeProvider):
    call_context.cli_logger.print("Cleaning up cluster resources...")
    provider.cleanup_cluster(config)
    call_context.cli_logger.print(cf.bold("Successfully cleaned up other cluster resources."))


def teardown_cluster_nodes(config: Dict[str, Any],
                           call_context: CallContext,
                           provider: NodeProvider,
                           workers_only: bool,
                           keep_min_workers: bool,
                           on_head: bool,
                           hard: bool = False):
    _cli_logger = call_context.cli_logger

    def remaining_nodes():
        workers = provider.non_terminated_nodes({
            CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER
        })

        if keep_min_workers:
            min_workers = _sum_min_workers(config)
            _cli_logger.print(
                "{} random worker nodes will not be shut down. " +
                cf.dimmed("(due to {})"), cf.bold(min_workers),
                cf.bold("--keep-min-workers"))

            workers = random.sample(workers, len(workers) - min_workers)

        head = provider.non_terminated_nodes({
            CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD
        })

        # todo: it's weird to kill the head node but not all workers
        if workers_only:
            if not on_head:
                _cli_logger.print(
                    "The head node will not be shut down. " +
                    cf.dimmed("(due to {})"), cf.bold("--workers-only"))

            return head, workers

        return head, head + workers

    # Loop here to check that both the head and worker nodes are actually
    #   really gone
    head, A = remaining_nodes()

    current_step = 1
    total_steps = 1

    if not hard:
        total_steps += 1
        # first stop the services on the nodes
        if on_head and len(head) > 0:
            total_steps += 1

            # Only do this for workers on head
            head_node = head[0]

            # Step 1:
            with _cli_logger.group(
                    "Stopping services for worker nodes...",
                    _numbered=("()", current_step, total_steps)):
                current_step += 1
                _do_stop_node_on_head(
                    config=config,
                    call_context=call_context,
                    provider=provider,
                    head_node=head_node,
                    node_head=None,
                    node_workers=A,
                    parallel=True
                )

        # Step 2: Running termination
        with _cli_logger.group(
                "Running termination for nodes...",
                _numbered=("()", current_step, total_steps)):
            current_step += 1
            _run_termination_on_nodes(
                config=config,
                call_context=call_context,
                provider=provider,
                workers_only=workers_only,
                on_head=on_head,
                head=head,
                nodes=A
            )

    # Step 3
    node_type = "workers" if workers_only else "nodes"
    with _cli_logger.group(
            "Terminating {}...".format(node_type),
            _numbered=("()", current_step, total_steps)):
        current_step += 1
        with LogTimer("teardown_cluster: done."):
            while A:
                provider.terminate_nodes(A)

                _cli_logger.print(
                    "Requested {} {} to shut down.",
                    cf.bold(len(A)), node_type,
                    _tags=dict(interval="1s"))

                time.sleep(POLL_INTERVAL)  # todo: interval should be a variable
                head, A = remaining_nodes()
                _cli_logger.print("{} {} remaining after {} second(s).",
                                 cf.bold(len(A)), node_type, POLL_INTERVAL)
            _cli_logger.print(cf.bold("No {} remaining."), node_type)


def _run_termination_on_nodes(
        config: Dict[str, Any],
        call_context: CallContext,
        provider: NodeProvider,
        workers_only: bool,
        on_head: bool,
        head: List[str],
        nodes: List[str]):
    use_internal_ip = True if on_head else False

    def run_termination(node_id, call_context):
        try:
            updater = create_node_updater_for_exec(
                config=config,
                call_context=call_context,
                node_id=node_id,
                provider=provider,
                start_commands=[],
                is_head_node=False,
                use_internal_ip=use_internal_ip)
            updater.cmd_executor.run_terminate()
        except Exception:
            raise RuntimeError(f"Run termination failed on {node_id}") from None

    _cli_logger = call_context.cli_logger
    if on_head or not workers_only:
        if on_head:
            _cli_logger.print("Running termination on workers...")
            running_nodes = nodes
        else:
            _cli_logger.print("Running termination on head...")
            running_nodes = head
        # This is to ensure that the parallel SSH calls below do not mess with
        # the users terminal.
        run_in_parallel_on_nodes(run_termination,
                                 call_context=call_context,
                                 nodes=running_nodes,
                                 max_workers=MAX_PARALLEL_SHUTDOWN_WORKERS)
        _cli_logger.print(cf.bold("Done running termination."))


def kill_node_from_head(config_file: str, yes: bool, hard: bool,
                        override_cluster_name: Optional[str],
                        node_ip: str = None):
    """Kills a specified or a random worker."""
    config = _load_cluster_config(config_file, override_cluster_name)
    call_context = cli_call_context()
    if node_ip:
        cli_logger.confirm(yes, "Node {} will be killed.", node_ip, _abort=True)
    else:
        cli_logger.confirm(yes, "A random node will be killed.", _abort=True)

    killed_node_ip = _kill_node_from_head(
        config,
        call_context=call_context,
        node_ip=node_ip,
        hard=hard)
    if killed_node_ip and hard:
        click.echo("Killed node with IP " + killed_node_ip)


def _kill_node_from_head(config: Dict[str, Any],
                         call_context: CallContext,
                         node_ip: str = None,
                         hard: bool = False) -> Optional[str]:
    if node_ip is None:
        provider = _get_node_provider(config["provider"], config["cluster_name"])
        nodes = provider.non_terminated_nodes({
            CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER
        })
        if not nodes:
            cli_logger.print("No worker nodes launched.")
            return None
        node = random.choice(nodes)
        node_ip = get_node_cluster_ip(provider, node)

    if hard:
        return _kill_node(config, call_context=call_context, node_ip=node_ip, hard=hard)

    # soft kill, we need to do on head
    cmds = [
        "cloudtik",
        "head",
        "kill-node",
        "--yes",
    ]
    cmds += ["--node-ip={}".format(node_ip)]

    with_verbose_option(cmds, call_context)
    final_cmd = " ".join(cmds)
    _exec_cmd_on_cluster(config,
                         call_context=call_context,
                         cmd=final_cmd)
    return node_ip


def kill_node_on_head(yes, hard, node_ip: str = None):
    # Since this is running on head, the bootstrap config must exist
    config = load_head_cluster_config()
    call_context = cli_call_context()
    if not yes:
        if node_ip:
            cli_logger.confirm(yes, "Node {} will be killed.", node_ip, _abort=True)
        else:
            cli_logger.confirm(yes, "A random node will be killed.", _abort=True)

    return _kill_node(config, call_context=call_context, node_ip=node_ip, hard=hard)


def _kill_node(config: Dict[str, Any],
               call_context: CallContext,
               hard: bool,
               node_ip: str = None):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    _cli_logger = call_context.cli_logger
    if node_ip:
        node = provider.get_node_id(node_ip, use_internal_ip=True)
        if not node:
            _cli_logger.error("No node with the specified node ip - {} found.", node_ip)
            return None
    else:
        nodes = provider.non_terminated_nodes({
            CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER
        })
        if not nodes:
            _cli_logger.print("No worker nodes found.")
            return None
        node = random.choice(nodes)
        node_ip = get_node_cluster_ip(provider, node)

    if not hard:
        # execute runtime stop command
        _stop_node_on_head(
            config, call_context=call_context,
            node_ip=node_ip, all_nodes=False)

    # terminate the node
    _cli_logger.print("Shutdown " + cf.bold("{}:{}"), node, node_ip)
    provider.terminate_node(node)
    time.sleep(POLL_INTERVAL)

    return node_ip


def monitor_cluster(config_file: str, num_lines: int,
                    override_cluster_name: Optional[str] = None,
                    file_type: str = None) -> None:
    config = _load_cluster_config(config_file, override_cluster_name)
    _monitor_cluster(config, num_lines, file_type)


def _monitor_cluster(config: Dict[str, Any],
                     num_lines: int,
                     file_type: str = None) -> None:
    """Tails the controller logs of a cluster."""

    call_context = cli_call_context()
    cmd = f"tail -n {num_lines} -f /tmp/cloudtik/session_latest/logs/cloudtik_cluster_controller"
    if file_type and file_type != "":
        cmd += f".{file_type}"
    else:
        cmd += "*"

    _exec_cluster(
        config,
        call_context=call_context,
        cmd=cmd,
        run_env="auto",
        screen=False,
        tmux=False,
        stop=False,
        start=False,
        port_forward=None)


def get_or_create_head_node(config: Dict[str, Any],
                            call_context: CallContext,
                            no_restart: bool,
                            restart_only: bool,
                            yes: bool,
                            _provider: Optional[NodeProvider] = None,
                            _runner: ModuleType = subprocess) -> None:
    """Create the cluster head node, which in turn creates the workers."""
    _cli_logger = call_context.cli_logger

    global_event_system.execute_callback(
        get_cluster_uri(config),
        CreateClusterEvent.cluster_booting_started)
    provider = (_provider or _get_node_provider(config["provider"],
                                                config["cluster_name"]))

    config = copy.deepcopy(config)
    head_node_tags = {
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD,
    }
    nodes = provider.non_terminated_nodes(head_node_tags)
    if len(nodes) > 0:
        head_node = nodes[0]
    else:
        head_node = None

    if not head_node:
        _cli_logger.confirm(
            yes,
            "No head node found. "
            "Launching a new cluster.",
            _abort=True)

    if head_node:
        if restart_only:
            _cli_logger.confirm(
                yes,
                "Updating cluster configuration and "
                "restarting the cluster runtime. "
                "Setup commands will not be run due to `{}`.\n",
                cf.bold("--restart-only"),
                _abort=True)
        elif no_restart:
            _cli_logger.print(
                "Cluster runtime will not be restarted due "
                "to `{}`.", cf.bold("--no-restart"))
            _cli_logger.confirm(
                yes,
                "Updating cluster configuration and "
                "running setup commands.",
                _abort=True)
        else:
            _cli_logger.print(
                "Updating cluster configuration and running full setup.")
            _cli_logger.confirm(
                yes,
                cf.bold("Cluster runtime will be restarted."),
                _abort=True)

    _cli_logger.newline()

    # The "head_node" config stores only the internal head node specific
    # configuration values generated in the runtime, for example IAM
    head_node_config = copy.deepcopy(config.get("head_node", {}))
    head_node_resources = None
    head_node_type = config.get("head_node_type")
    if head_node_type:
        head_node_tags[CLOUDTIK_TAG_USER_NODE_TYPE] = head_node_type
        head_config = config["available_node_types"][head_node_type]
        head_node_config.update(head_config["node_config"])

        # Not necessary to keep in sync with node_launcher.py
        # Keep in sync with cluster_scaler.py _node_resources
        head_node_resources = head_config.get("resources")

    launch_hash = hash_launch_conf(head_node_config, config["auth"])
    creating_new_head = _should_create_new_head(head_node, launch_hash,
                                                head_node_type, provider)
    if creating_new_head:
        with _cli_logger.group("Acquiring an up-to-date head node"):
            global_event_system.execute_callback(
                get_cluster_uri(config),
                CreateClusterEvent.acquiring_new_head_node)
            if head_node is not None:
                _cli_logger.confirm(
                    yes, "Relaunching the head node.", _abort=True)

                provider.terminate_node(head_node)
                _cli_logger.print("Terminated head node {}", head_node)

            head_node_tags[CLOUDTIK_TAG_LAUNCH_CONFIG] = launch_hash
            head_node_tags[CLOUDTIK_TAG_NODE_NAME] = "cloudtik-{}-head".format(
                config["cluster_name"])
            head_node_tags[CLOUDTIK_TAG_NODE_STATUS] = STATUS_UNINITIALIZED
            head_node_tags[CLOUDTIK_TAG_NODE_NUMBER] = str(CLOUDTIK_TAG_HEAD_NODE_NUMBER)
            provider.create_node(head_node_config, head_node_tags, 1)
            _cli_logger.print("Launched a new head node")

            start = time.time()
            head_node = None
            with _cli_logger.group("Fetching the new head node"):
                while True:
                    if time.time() - start > 50:
                        _cli_logger.abort("Head node fetch timed out. "
                                          "Failed to create head node.")
                    nodes = provider.non_terminated_nodes(head_node_tags)
                    if len(nodes) == 1:
                        head_node = nodes[0]
                        break
                    time.sleep(POLL_INTERVAL)
            _cli_logger.newline()

    global_event_system.execute_callback(
        get_cluster_uri(config),
        CreateClusterEvent.head_node_acquired)

    with _cli_logger.group(
            "Setting up head node",
            _numbered=("<>", 1, 1),
            # cf.bold(provider.node_tags(head_node)[CLOUDTIK_TAG_NODE_NAME]),
            _tags=dict()):  # add id, ARN to tags?

        # TODO: right now we always update the head node even if the
        # hash matches.
        # We could prompt the user for what they want to do here.
        # No need to pass in cluster_sync_files because we use this
        # hash to set up the head node
        (runtime_hash,
         file_mounts_contents_hash,
         runtime_hash_for_node_types) = hash_runtime_conf(
            file_mounts=config["file_mounts"],
            cluster_synced_files=None,
            extra_objs=config)
        # Even we don't need controller on head, we still need config and cluster keys on head
        # because head depends a lot on the cluster config file and cluster keys to do cluster
        # operations and connect to the worker.

        # Return remote_config_file to avoid prematurely closing it.
        config, remote_config_file = _set_up_config_for_head_node(
            config, provider, no_restart)
        _cli_logger.print("Prepared bootstrap config")

        if restart_only:
            # Docker may re-launch nodes, requiring setup
            # commands to be rerun.
            if is_docker_enabled(config):
                setup_commands = get_commands_to_run(config, "head_setup_commands")
            else:
                setup_commands = []
            start_commands = get_commands_to_run(config, "head_start_commands")
        # If user passed in --no-restart and we're not creating a new head,
        # omit start commands.
        elif no_restart and not creating_new_head:
            setup_commands = get_commands_to_run(config, "head_setup_commands")
            start_commands = []
        else:
            setup_commands = get_commands_to_run(config, "head_setup_commands")
            start_commands = get_commands_to_run(config, "head_start_commands")

        initialization_commands = get_commands_to_run(config, "head_initialization_commands")
        updater = NodeUpdaterThread(
            config=config,
            call_context=call_context,
            node_id=head_node,
            provider_config=config["provider"],
            provider=provider,
            auth_config=config["auth"],
            cluster_name=config["cluster_name"],
            file_mounts=config["file_mounts"],
            initialization_commands=initialization_commands,
            setup_commands=setup_commands,
            start_commands=start_commands,
            process_runner=_runner,
            runtime_hash=runtime_hash,
            file_mounts_contents_hash=file_mounts_contents_hash,
            is_head_node=True,
            node_resources=head_node_resources,
            rsync_options={
                "rsync_exclude": config.get("rsync_exclude"),
                "rsync_filter": config.get("rsync_filter")
            },
            docker_config=config.get(DOCKER_CONFIG_KEY),
            restart_only=restart_only,
            runtime_config=config.get(RUNTIME_CONFIG_KEY))
        updater.start()
        updater.join()

        # Refresh the node cache so we see the external ip if available
        provider.non_terminated_nodes(head_node_tags)

        if updater.exitcode != 0:
            msg = "Failed to setup head node."
            if call_context.is_call_from_api():
                raise RuntimeError(msg)
            else:
                _cli_logger.abort(msg)

    global_event_system.execute_callback(
        get_cluster_uri(config),
        CreateClusterEvent.cluster_booting_completed, {
            "head_node_id": head_node,
        })

    cluster_booting_completed(config, head_node)

    _cli_logger.newline()
    successful_msg = "Successfully started cluster: {}.".format(config["cluster_name"])
    _cli_logger.success("-" * len(successful_msg))
    _cli_logger.success(successful_msg)
    _cli_logger.success("-" * len(successful_msg))


def _should_create_new_head(head_node_id: Optional[str], new_launch_hash: str,
                            new_head_node_type: str,
                            provider: NodeProvider) -> bool:
    """Decides whether a new head node needs to be created.

    We need a new head if at least one of the following holds:
    (a) There isn't an existing head node
    (b) The user-submitted head node_config differs from the existing head
        node's node_config.
    (c) The user-submitted head node_type key differs from the existing head
        node's node_type.

    Args:
        head_node_id (Optional[str]): head node id if a head exists, else None
        new_launch_hash (str): hash of current user-submitted head config
        new_head_node_type (str): current user-submitted head node-type key

    Returns:
        bool: True if a new head node should be launched, False otherwise
    """
    if not head_node_id:
        # No head node exists, need to create it.
        return True

    # Pull existing head's data.
    head_tags = provider.node_tags(head_node_id)
    current_launch_hash = head_tags.get(CLOUDTIK_TAG_LAUNCH_CONFIG)
    current_head_type = head_tags.get(CLOUDTIK_TAG_USER_NODE_TYPE)

    # Compare to current head
    hashes_mismatch = new_launch_hash != current_launch_hash
    types_mismatch = new_head_node_type != current_head_type

    new_head_required = hashes_mismatch or types_mismatch

    # Warn user
    if new_head_required:
        with cli_logger.group(
                "Currently running head node is out-of-date with cluster "
                "configuration"):

            if hashes_mismatch:
                cli_logger.print("Current hash is {}, expected {}",
                                 cf.bold(current_launch_hash),
                                 cf.bold(new_launch_hash))

            if types_mismatch:
                cli_logger.print("Current head node type is {}, expected {}",
                                 cf.bold(current_head_type),
                                 cf.bold(new_head_node_type))

    return new_head_required


def _set_up_config_for_head_node(config: Dict[str, Any],
                                 provider: NodeProvider,
                                 no_restart: bool) ->\
        Tuple[Dict[str, Any], Any]:
    """Prepares autoscaling config and, if needed, ssh key, to be mounted onto
    the head node for use by the cluster scaler.

    Returns the modified config and the temporary config file that will be
    mounted onto the head node.
    """
    # Rewrite the auth config so that the head
    # node can update the workers
    remote_config = copy.deepcopy(config)

    # Set bootstrapped mark
    remote_config["bootstrapped"] = True

    # drop proxy options if they exist, otherwise
    # head node won't be able to connect to workers
    remote_config["auth"].pop("ssh_proxy_command", None)
    remote_config["auth"].pop("ssh_public_key", None)

    if "ssh_private_key" in config["auth"]:
        remote_key_path = "~/cloudtik_bootstrap_key.pem"
        remote_config["auth"]["ssh_private_key"] = remote_key_path

    # Adjust for new file locations
    new_mounts = {}
    for remote_path in config["file_mounts"]:
        new_mounts[remote_path] = remote_path
    remote_config["file_mounts"] = new_mounts
    remote_config["no_restart"] = no_restart

    # Now inject the rewritten config and SSH key into the head node
    config_temp_dir = os.path.join(get_cloudtik_temp_dir(), "configs")
    os.makedirs(config_temp_dir, exist_ok=True)
    remote_config_file = tempfile.NamedTemporaryFile(
        "w", dir=config_temp_dir, prefix="cloudtik-bootstrap-")

    config["file_mounts"].update({
        "~/cloudtik_bootstrap_config.yaml": remote_config_file.name
    })

    if "ssh_private_key" in config["auth"]:
        config["file_mounts"].update({
            remote_key_path: config["auth"]["ssh_private_key"],
        })

    remote_config = provider.prepare_for_head_node(config, remote_config)
    remote_config = encrypt_config(remote_config)

    remote_config_file.write(json.dumps(remote_config))
    remote_config_file.flush()
    return config, remote_config_file


def attach_cluster(config_file: str,
                   use_screen: bool,
                   use_tmux: bool,
                   override_cluster_name: Optional[str],
                   no_config_cache: bool = False,
                   new: bool = False,
                   port_forward: Optional[Port_forward] = None,
                   force_to_host: bool = False) -> None:
    """Attaches to a screen for the specified cluster.

    Arguments:
        config_file: path to the cluster yaml
        use_screen: whether to use screen as multiplexer
        use_tmux: whether to use tmux as multiplexer
        override_cluster_name: set the name of the cluster
        no_config_cache: no use config cache
        new: whether to force a new screen
        port_forward ( (int,int) or list[(int,int)] ): port(s) to forward
        force_to_host: Force attaching to host
    """
    config = _load_cluster_config(config_file, override_cluster_name,
                                  no_config_cache=no_config_cache)
    call_context = cli_call_context()
    cmd = get_attach_command(use_screen, use_tmux, new)
    run_env = "auto"
    if force_to_host:
        run_env = "host"

    _exec_cluster(
        config,
        call_context=call_context,
        cmd=cmd,
        run_env=run_env,
        screen=False,
        tmux=False,
        stop=False,
        start=False,
        port_forward=port_forward,
        _allow_uninitialized_state=True)


def _exec_cluster(config: Dict[str, Any],
                  call_context: CallContext,
                  *,
                  cmd: str = None,
                  run_env: str = "auto",
                  screen: bool = False,
                  tmux: bool = False,
                  stop: bool = False,
                  start: bool = False,
                  port_forward: Optional[Port_forward] = None,
                  with_output: bool = False,
                  _allow_uninitialized_state: bool = False,
                  job_waiter: Optional[JobWaiter] = None,
                  session_name: Optional[str] = None) -> str:
    """Runs a command on the specified cluster.

    Arguments:
        cmd: command to run
        run_env: whether to run the command on the host or in a container.
            Select between "auto", "host" and "docker"
        screen: whether to run in a screen
        tmux: whether to run in a tmux session
        stop: whether to stop the cluster after command run
        start: whether to start the cluster if it isn't up
        port_forward ( (int, int) or list[(int, int)] ): port(s) to forward
        _allow_uninitialized_state: whether to execute on an uninitialized head
            node.
        job_waiter: The waiter object to check the job is done
        session_name: The session name of the job
    """
    assert not (screen and tmux), "Can specify only one of `screen` or `tmux`."
    assert run_env in RUN_ENV_TYPES, "--run_env must be in {}".format(
        RUN_ENV_TYPES)

    # TODO (haifeng): we may not need to set_allow_interactive explicitly here
    # We default this to True to maintain backwards-compatibility
    # In the future we would want to support disabling login-shells
    # and interactivity.
    call_context.set_allow_interactive(True)

    use_internal_ip = config.get("bootstrapped", False)
    head_node = _get_running_head_node_ex(
        config,
        call_context=call_context,
        create_if_needed=start,
        _allow_uninitialized_state=_allow_uninitialized_state)

    provider = _get_node_provider(config["provider"], config["cluster_name"])
    updater = create_node_updater_for_exec(
        config=config,
        call_context=call_context,
        node_id=head_node,
        provider=provider,
        start_commands=[],
        is_head_node=True,
        use_internal_ip=use_internal_ip)

    if cmd and stop:
        # if no job waiter defined, we shut down at the end
        if job_waiter is None:
            cmd = "; ".join([
                cmd, "cloudtik head runtime stop --runtimes=cloudtik --no-all-nodes --yes",
                "cloudtik head teardown --yes"
            ])

    # Only when there is no job waiter we hold the tmux or screen session
    hold_session = False if job_waiter else True
    if not session_name:
        session_name = get_command_session_name(cmd, time.time_ns())

    result = _exec(
        updater,
        cmd,
        screen,
        tmux,
        port_forward=port_forward,
        with_output=with_output,
        run_env=run_env,
        session_name=session_name,
        hold_session=hold_session)

    # if a job waiter is specified, we always wait for its completion.
    if job_waiter is not None:
        job_waiter.wait_for_completion(head_node, cmd, session_name)

    # if the cmd is not run with screen or tmux
    # or in the future we can check the screen or tmux session completion
    # we do tear down here
    if cmd and stop:
        if (job_waiter is not None) or (not screen and not tmux):
            # for either job waiter case or command run in foreground case
            _teardown_cluster(
                config=config, call_context=call_context)
    return result


def _exec(updater: NodeUpdaterThread,
          cmd: Optional[str] = None,
          screen: bool = False,
          tmux: bool = False,
          port_forward: Optional[Port_forward] = None,
          with_output: bool = False,
          run_env: str = "auto",
          shutdown_after_run: bool = False,
          exit_on_fail: bool = False,
          session_name: str = None,
          hold_session: bool = True) -> str:
    if cmd:
        if screen:
            if not session_name:
                session_name = get_command_session_name(cmd, time.time_ns())
            wrapped_cmd = [
                "screen", "-S", session_name, "-L", "-dm", "bash", "-c",
                quote(cmd + "; exec bash") if hold_session else quote(cmd)
            ]
            cmd = " ".join(wrapped_cmd)
        elif tmux:
            if not session_name:
                session_name = get_command_session_name(cmd, time.time_ns())
            wrapped_cmd = [
                "tmux", "new", "-s", session_name, "-d", "bash", "-c",
                quote(cmd + "; exec bash") if hold_session else quote(cmd)
            ]
            cmd = " ".join(wrapped_cmd)
    exec_out = updater.cmd_executor.run(
        cmd,
        exit_on_fail=exit_on_fail,
        port_forward=port_forward,
        with_output=with_output,
        run_env=run_env,
        shutdown_after_run=shutdown_after_run)
    if with_output:
        return exec_out.decode(encoding="utf-8")
    else:
        return exec_out


def _rsync(config: Dict[str, Any],
           call_context: CallContext,
           source: Optional[str],
           target: Optional[str],
           down: bool,
           node_ip: Optional[str] = None,
           all_nodes: bool = False,
           use_internal_ip: bool = False,
           _runner: ModuleType = subprocess) -> None:
    if bool(source) != bool(target):
        cli_logger.abort(
            "Expected either both a source and a target, or neither.")

    if node_ip and all_nodes:
        cli_logger.abort("Cannot provide both node_ip and 'all_nodes'.")

    assert bool(source) == bool(target), (
        "Must either provide both or neither source and target.")

    is_file_mount = False
    if source and target:
        for remote_mount in config.get("file_mounts", {}).keys():
            if (source if down else target).startswith(remote_mount):
                is_file_mount = True
                break

    provider = _get_node_provider(config["provider"], config["cluster_name"])

    def rsync_to_node(node_id, source, target, is_head_node):
        updater = create_node_updater_for_exec(
            config=config,
            call_context=call_context,
            node_id=node_id,
            provider=provider,
            start_commands=[],
            is_head_node=is_head_node,
            process_runner=_runner,
            use_internal_ip=use_internal_ip)
        if down:
            rsync = updater.rsync_down
        else:
            rsync = updater.rsync_up

        if source and target:
            # print rsync progress for single file rsync
            if cli_logger.verbosity > 0:
                call_context.set_output_redirected(False)
                call_context.set_rsync_silent(False)
            rsync(source, target, is_file_mount)
        else:
            updater.sync_file_mounts(rsync)

    head_node = _get_running_head_node(config)
    if not node_ip:
        # No node specified, rsync with head or rsync up with all nodes
        rsync_to_node(head_node, source, target, is_head_node=True)
        if not down and all_nodes:
            # rsync up with all workers
            source_for_target = target
            if os.path.isdir(source):
                source_for_target = source_for_target.rstrip("/")
                source_for_target += "/."

            rsync_to_node_from_head(config,
                                    call_context=call_context,
                                    source=source_for_target, target=target, down=False,
                                    node_ip=None, all_workers=all_nodes)
    else:
        # for the cases that specified sync up or down with specific node
        # both source and target must be specified
        if not source or not target:
            cli_logger.abort("Need to specify both source and target when rsync with specific node")

        target_base = os.path.basename(target)
        target_on_head = tempfile.mktemp(prefix=f"{target_base}_")
        if down:
            # rsync down
            # first run rsync from head with the specific node
            rsync_to_node_from_head(config,
                                    call_context=call_context,
                                    source=source, target=target_on_head, down=True,
                                    node_ip=node_ip)
            # then rsync local node with the head
            if source[-1] == "/":
                target_on_head += "/."
            rsync_to_node(head_node, target_on_head, target, is_head_node=True)
        else:
            # rsync up
            # first rsync local node with head
            rsync_to_node(head_node, source, target_on_head, is_head_node=True)

            # then rsync from head to the specific node
            if os.path.isdir(source):
                target_on_head += "/."
            rsync_to_node_from_head(config,
                                    call_context=call_context,
                                    source=target_on_head, target=target, down=False,
                                    node_ip=node_ip)


def rsync_to_node_from_head(config: Dict[str, Any],
                            call_context: CallContext,
                            source: str,
                            target: str,
                            down: bool,
                            node_ip: str = None,
                            all_workers: bool = False
                            ) -> None:
    """Exec the rsync on head command to do rsync with the target worker"""
    cmds = [
        "cloudtik",
        "head",
    ]
    if down:
        cmds += ["rsync-down"]
    else:
        cmds += ["rsync-up"]
    if source and target:
        cmds += [quote(source)]
        cmds += [quote(target)]
    if node_ip:
        cmds += ["--node-ip={}".format(node_ip)]

    if not down:
        if all_workers:
            cmds += ["--all-workers"]
        else:
            cmds += ["--no-all-workers"]

    with_verbose_option(cmds, call_context)
    final_cmd = " ".join(cmds)
    _exec_cmd_on_cluster(config,
                         call_context=call_context,
                         cmd=final_cmd)


def rsync_node_on_head(config: Dict[str, Any],
                       call_context: CallContext,
                       source: str,
                       target: str,
                       down: bool,
                       node_ip: str = None,
                       all_workers: bool = False):
    provider = _get_node_provider(config["provider"], config["cluster_name"])

    is_file_mount = False
    if source and target:
        for remote_mount in config.get("file_mounts", {}).keys():
            if (source if down else target).startswith(remote_mount):
                is_file_mount = True
                break

    def rsync_to_node(node_id, source, target):
        updater = create_node_updater_for_exec(
            config=config,
            call_context=call_context,
            node_id=node_id,
            provider=provider,
            start_commands=[],
            is_head_node=False,
            process_runner=subprocess,
            use_internal_ip=True)
        if down:
            rsync = updater.rsync_down
        else:
            rsync = updater.rsync_up

        if source and target:
            if down:
                # rsync down, expand user for target (on head) if it is not handled
                target = os.path.expanduser(target)
            else:
                # rsync up, expand user for source (on head) if it is not handled
                source = os.path.expanduser(source)

            # print rsync progress for single file rsync
            if cli_logger.verbosity > 0:
                call_context.set_output_redirected(False)
                call_context.set_rsync_silent(False)
            rsync(source, target, is_file_mount)
        else:
            updater.sync_file_mounts(rsync)

    nodes = []
    if node_ip:
        nodes = [provider.get_node_id(node_ip, use_internal_ip=True)]
    else:
        # either node_ip or all_workers be set
        if all_workers:
            nodes.extend(_get_worker_nodes(config))

    for node_id in nodes:
        rsync_to_node(node_id, source, target)


def get_worker_cpus(config, provider):
    return get_worker_resource(
        config, provider, constants.CLOUDTIK_RESOURCE_CPU)


def get_worker_gpus(config, provider):
    return get_worker_resource(
        config, provider, constants.CLOUDTIK_RESOURCE_GPU)


def get_worker_memory(config, provider):
    return get_worker_resource(
        config, provider, constants.CLOUDTIK_RESOURCE_MEMORY)


def get_worker_resource(config, provider, resource_name):
    workers = _get_worker_nodes(config)
    workers_info = get_nodes_info(provider, workers, True, config["available_node_types"])
    return sum_nodes_resource(workers_info, resource_name)


def get_cpus_per_worker(config, provider):
    return get_resource_per_worker(
        config, provider, constants.CLOUDTIK_RESOURCE_CPU)


def get_gpus_per_worker(config, provider):
    return get_resource_per_worker(
        config, provider, constants.CLOUDTIK_RESOURCE_GPU)


def get_memory_per_worker(config, provider):
    return get_resource_per_worker(
        config, provider, constants.CLOUDTIK_RESOURCE_MEMORY)


def get_resource_per_worker(config, provider, resource_name):
    # Assume all the worker nodes are the same
    node_type = get_worker_node_type(config)
    available_node_types = config["available_node_types"]
    resource_info = get_resource_info_of_node_type(
        node_type, available_node_types)
    return get_resource_of_node_info(resource_info, resource_name)


def get_sockets_per_worker(config, provider):
    workers = _get_worker_nodes(config)
    if not workers:
        return None
    call_context = cli_call_context()

    get_sockets_cmd = "lscpu | grep Socket | awk '{print $2}'"
    sockets_per_worker = exec_cmd_on_head(config=config,
                                          call_context=call_context,
                                          provider=provider,
                                          node_id=workers[0],
                                          cmd=get_sockets_cmd,
                                          with_output=True)
    return sockets_per_worker.strip()


def get_head_node_ip(config_file: str,
                     override_cluster_name: Optional[str] = None,
                     public: bool = False) -> str:
    """Returns head node IP for given configuration file if exists."""
    config = _load_cluster_config(config_file, override_cluster_name)
    return _get_head_node_ip(config=config,
                             public=public)


def get_worker_node_ips(config_file: str,
                        override_cluster_name: Optional[str] = None,
                        runtime: str = None,
                        node_status: str = None
                        ) -> List[str]:
    """Returns worker node IPs for given configuration file."""
    config = _load_cluster_config(config_file, override_cluster_name)
    return _get_worker_node_ips(config, runtime, node_status=node_status)


def _get_head_node_ip(config: Dict[str, Any], public: bool = False) -> str:
    return get_cluster_head_ip(config, public)


def _get_worker_node_ips(
        config: Dict[str, Any], runtime: str = None,
        node_status: str = None) -> List[str]:
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    nodes = provider.non_terminated_nodes({
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER
    })

    if runtime is not None:
        # Filter the nodes for the specific runtime only
        nodes = get_nodes_for_runtime(config, nodes, runtime)

    if node_status:
        nodes = _get_nodes_in_status(provider, nodes, node_status)

    return [get_node_cluster_ip(provider, node) for node in nodes]


def _get_nodes_in_status(provider, nodes: List[str], node_status: str) -> List[str]:
    return [node for node in nodes if is_node_in_status(provider, node, node_status)]


def is_node_in_status(provider, node: str, node_status: str):
    node_info = provider.get_node_info(node)
    return True if node_status == node_info.get(CLOUDTIK_TAG_NODE_STATUS) else False


def _get_worker_nodes(config: Dict[str, Any]) -> List[str]:
    """Returns worker node ids for given configuration."""
    # Technically could be reused in get_worker_node_ips
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    return provider.non_terminated_nodes({CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER})


def _get_running_head_node_ex(
        config: Dict[str, Any],
        call_context: CallContext = None,
        create_if_needed: bool = False,
        _provider: Optional[NodeProvider] = None,
        _allow_uninitialized_state: bool = False,
) -> str:
    """Get a valid, running head node.
    Args:
        config (Dict[str, Any]): Cluster Config dictionary
        call_context (CallContext): The call context if create_if_needed is true
        create_if_needed (bool): Create a head node if one is not present.
        _provider (NodeProvider): [For testing], a Node Provider to use.
        _allow_uninitialized_state (bool): Whether to return a head node that
            is not 'UP TO DATE'. This is used to allow `cloudtik attach` and
            `cloudtik exec` to debug a cluster in a bad state.

    """
    provider = _provider or _get_node_provider(config["provider"],
                                               config["cluster_name"])
    head_node_tags = {
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD,
    }
    nodes = provider.non_terminated_nodes(head_node_tags)
    head_node = None
    _backup_head_node = None
    for node in nodes:
        node_state = provider.node_tags(node).get(CLOUDTIK_TAG_NODE_STATUS)
        if node_state == STATUS_UP_TO_DATE:
            head_node = node
        else:
            _backup_head_node = node
            cli_logger.warning(f"Head node ({node}) is in state {node_state}.")

    if head_node is not None:
        return head_node
    elif create_if_needed:
        if call_context is None:
            raise RuntimeError("You need to pass a CallContext for creating a cluster.")
        get_or_create_head_node(
            config,
            call_context=call_context,
            restart_only=False,
            no_restart=False,
            yes=True)
        # NOTE: `_allow_uninitialized_state` is forced to False if
        # `create_if_needed` is set to True. This is to ensure that the
        # commands executed after creation occur on an actually running
        # cluster.
        return _get_running_head_node(
            config,
            _allow_uninitialized_state=False)
    else:
        if _allow_uninitialized_state and _backup_head_node is not None:
            cli_logger.warning(
                f"The head node being returned: {_backup_head_node} is not "
                "`up-to-date`. If you are not debugging a startup issue "
                "it is recommended to restart this cluster.")

            return _backup_head_node
        raise HeadNotRunningError("Head node of cluster {} not found!".format(
            config["cluster_name"]))


def _get_running_head_node(
        config: Dict[str, Any],
        _provider: Optional[NodeProvider] = None,
        _allow_uninitialized_state: bool = False,
) -> str:
    """Get a valid, running head node. Raise error if no running head
    Args:
        config (Dict[str, Any]): Cluster Config dictionary
        _provider (NodeProvider): [For testing], a Node Provider to use.
        _allow_uninitialized_state (bool): Whether to return a head node that
            is not 'UP TO DATE'. This is used to allow `cloudtik attach` and
            `cloudtik exec` to debug a cluster in a bad state.

    """
    return get_running_head_node(config=config, _provider=_provider,
                                 _allow_uninitialized_state=_allow_uninitialized_state)


def dump_local(stream: bool = False,
               output: Optional[str] = None,
               logs: bool = True,
               debug_state: bool = True,
               pip: bool = True,
               processes: bool = True,
               processes_verbose: bool = False,
               tempfile: Optional[str] = None,
               runtimes: str = None,
               silent: bool = False) -> Optional[str]:
    if stream and output:
        raise ValueError(
            "You can only use either `--output` or `--stream`, but not both.")
    runtime_list = runtimes.split(",") if runtimes and len(runtimes) > 0 else None
    parameters = GetParameters(
        logs=logs,
        debug_state=debug_state,
        pip=pip,
        processes=processes,
        processes_verbose=processes_verbose,
        runtimes=runtime_list)

    with Archive(file=tempfile) as archive:
        get_all_local_data(archive, parameters)

    tmp = archive.file

    if stream:
        with open(tmp, "rb") as fp:
            os.write(1, fp.read())
        os.remove(tmp)
        return None

    target = output or os.path.join(os.getcwd(), os.path.basename(tmp))
    shutil.move(tmp, target)

    if not silent:
        cli_logger.print(f"Created local data archive at {target}")

    return target


def dump_cluster_on_head(
        config: Dict[str, Any],
        call_context: CallContext,
        hosts: Optional[str] = None,
        stream: bool = False,
        output: Optional[str] = None,
        logs: bool = True,
        debug_state: bool = True,
        pip: bool = True,
        processes: bool = True,
        processes_verbose: bool = False,
        temp_file: Optional[str] = None,
        silent: bool = False) -> Optional[str]:
    if stream and output:
        raise ValueError(
            "You can only use either `--output` or `--stream`, but not both.")

    if not stream and not silent:
        _print_cluster_dump_warning(
            call_context,
            logs, debug_state, pip, processes)

    head, workers, = _get_nodes_to_dump(config, hosts)

    head_node = Node(
        node_id=head[0],
        host=head[1],
        is_head=True) if head is not None else None

    worker_nodes = [
        Node(
            node_id=worker[0],
            host=worker[1]) for worker in workers
    ]

    if head_node is None and not worker_nodes:
        cli_logger.error(
            "No nodes found. Specify with `--host` or by passing a "
            "cluster config to `--cluster`.")
        return None

    parameters = GetParameters(
        logs=logs,
        debug_state=debug_state,
        pip=pip,
        processes=processes,
        processes_verbose=processes_verbose,
        runtimes=get_enabled_runtimes(config))

    with Archive(file=temp_file) as archive:
        if head_node:
            # dump local head node
            add_archive_for_local_node(
                archive, head_node, parameters)

        if worker_nodes:
            add_archive_for_remote_nodes(
                config, call_context,
                archive, remote_nodes=worker_nodes, parameters=parameters)

    tmp = archive.file

    if stream:
        with open(tmp, "rb") as fp:
            os.write(1, fp.read())
        os.remove(tmp)
        return None

    if not output:
        cluster_name = config["cluster_name"]
        filename = f"{cluster_name}_" \
                   f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.tar.gz"
        target = os.path.join(os.getcwd(), filename)
    else:
        target = os.path.expanduser(output)
    shutil.move(tmp, target)
    cli_logger.print(f"Created cluster dump archive: {target}")
    return target


def _print_cluster_dump_warning(
        call_context: CallContext, logs, debug_state, pip, processes):
    content_str = ""
    if logs:
        content_str += \
            "  - The logfiles of your session\n" \
            "    This usually includes Python outputs (stdout/stderr)\n"

    if debug_state:
        content_str += \
            "  - Debug state information on your cluster \n" \
            "    e.g. number of workers, drivers, objects, etc.\n"

    if pip:
        content_str += "  - Your installed Python packages (`pip freeze`)\n"

    if processes:
        content_str += \
            "  - Information on your running processes\n" \
            "    This includes command line arguments\n"

    call_context.cli_logger.warning(
        "You are about to create a cluster dump. This will collect data from "
        "cluster nodes.\n\n"
        "The dump will contain this information:\n\n"
        f"{content_str}\n"
        f"If you are concerned about leaking private information, extract "
        f"the archive and inspect its contents before sharing it with "
        f"anyone.")


def dump_cluster(
        config: Dict[str, Any],
        call_context: CallContext,
        hosts: Optional[str] = None,
        head_only: Optional[bool] = None,
        output: Optional[str] = None,
        logs: bool = True,
        debug_state: bool = True,
        pip: bool = True,
        processes: bool = True,
        processes_verbose: bool = False,
        tempfile: Optional[str] = None,
        silent: bool = False):
    # Inform the user what kind of logs are collected (before actually
    # collecting, so they can abort)
    if not silent:
        _print_cluster_dump_warning(
            call_context,
            logs, debug_state, pip, processes)

    head, workers = _get_nodes_to_dump(config, hosts)

    head_node = Node(
            node_id=head[0],
            host=head[1],
            is_head=True) if head is not None else None

    worker_nodes = [
        Node(node_id=worker[0], host=worker[1]) for worker in workers
    ]

    _cli_logger = call_context.cli_logger
    if not head_node and not worker_nodes:
        _cli_logger.print(
            f"No matched cluster nodes to dump.")
        return

    parameters = GetParameters(
        logs=logs,
        debug_state=debug_state,
        pip=pip,
        processes=processes,
        processes_verbose=processes_verbose,
        runtimes=get_enabled_runtimes(config))

    with Archive(file=tempfile) as archive:
        add_archive_for_cluster_nodes(
            config, call_context,
            archive,
            head_node=head_node, worker_nodes=worker_nodes,
            parameters=parameters, head_only=head_only)

    if not output:
        cluster_name = config["cluster_name"]
        filename = f"{cluster_name}_" \
                   f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.tar.gz"
        output = os.path.join(os.getcwd(), filename)
    else:
        output = os.path.expanduser(output)

    shutil.move(archive.file, output)

    if not silent:
        _cli_logger.print(
            f"Created cluster dump archive: {output}")


def _show_worker_cpus(config: Dict[str, Any]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    worker_cpus = get_worker_cpus(config, provider)
    cli_logger.print(worker_cpus)


def _show_worker_gpus(config: Dict[str, Any]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    worker_gpus = get_worker_gpus(config, provider)
    cli_logger.print(worker_gpus)


def _show_worker_memory(config: Dict[str, Any]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    memory_in_gb = memory_to_gb_string(get_worker_memory(config, provider))
    cli_logger.print(memory_in_gb)


def _show_cpus_per_worker(config: Dict[str, Any]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    cpus_per_worker = get_cpus_per_worker(config, provider)
    cli_logger.print(cpus_per_worker)


def _show_gpus_per_worker(config: Dict[str, Any]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    gpus_per_worker = get_gpus_per_worker(config, provider)
    cli_logger.print(gpus_per_worker)


def _show_sockets_per_worker(config: Dict[str, Any]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    sockets_per_worker = get_sockets_per_worker(config, provider)
    cli_logger.print(sockets_per_worker)


def _show_memory_per_worker(config: Dict[str, Any]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    memory_per_worker = get_memory_per_worker(config, provider)
    # convert to GB and print
    cli_logger.print(memory_to_gb(memory_per_worker))


def _show_total_workers(config: Dict[str, Any]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    cluster_info = _get_cluster_info(config, provider)
    worker_count = cluster_info["total-workers"]
    cli_logger.print(worker_count)


def show_info(
        config: Dict[str, Any], config_file,
        worker_cpus, worker_gpus, worker_memory,
        cpus_per_worker, gpus_per_worker, memory_per_worker,
        sockets_per_worker, total_workers):
    if worker_cpus:
        return _show_worker_cpus(config)

    if worker_gpus:
        return _show_worker_gpus(config)

    if worker_memory:
        return _show_worker_memory(config)

    if cpus_per_worker:
        return _show_cpus_per_worker(config)

    if gpus_per_worker:
        return _show_gpus_per_worker(config)

    if memory_per_worker:
        return _show_memory_per_worker(config)

    if sockets_per_worker:
        return _show_sockets_per_worker(config)

    if total_workers:
        return _show_total_workers(config)

    _show_cluster_info(config, config_file)


def show_cluster_info(config_file: str,
                      override_cluster_name: Optional[str] = None) -> None:
    """Shows the cluster information for given configuration file."""
    config = _load_cluster_config(config_file, override_cluster_name)
    _show_cluster_info(config, config_file, override_cluster_name)


def _show_cluster_info(config: Dict[str, Any],
                       config_file: str,
                       override_cluster_name: Optional[str] = None):
    provider = _get_node_provider(config["provider"], config["cluster_name"])

    cluster_info = _get_cluster_info(config, provider)

    cli_logger.print(cf.bold("Cluster {} is: {}"), config["cluster_name"], cluster_info["status"])
    if cluster_info["status"] == CLOUDTIK_CLUSTER_STATUS_STOPPED:
        return

    # Check the running worker nodes
    worker_count = cluster_info["total-workers"]
    cli_logger.print(cf.bold("{} worker(s) are running"), worker_count)
    cli_logger.print(cf.bold("{} worker(s) are ready"), cluster_info["total-workers-ready"])

    cli_logger.newline()
    cli_logger.print(cf.bold("Runtimes: {}"), ", ".join(cluster_info["runtimes"]))

    cli_logger.newline()
    cli_logger.print(cf.bold("The total worker CPUs: {}."), cluster_info["total-worker-cpus"])
    cli_logger.print(
        cf.bold("The total worker memory: {}."),
        memory_to_gb_string(cluster_info["total-worker-memory"]))

    head_node = cluster_info["head-id"]
    show_useful_commands(call_context=cli_call_context(),
                         config=config,
                         provider=provider,
                         head_node=head_node,
                         config_file=config_file,
                         override_cluster_name=override_cluster_name)


def show_useful_commands(call_context: CallContext,
                         config: Dict[str, Any],
                         provider: NodeProvider,
                         head_node: str,
                         config_file: str,
                         override_cluster_name: Optional[str] = None
                         ) -> None:
    _cli_logger = call_context.cli_logger
    if override_cluster_name:
        modifiers = " --cluster-name={}".format(quote(override_cluster_name))
        cluster_name = override_cluster_name
    else:
        modifiers = ""
        cluster_name = config["cluster_name"]

    _cli_logger.newline()
    private_key_file = config["auth"].get("ssh_private_key")
    public_key_file = config["auth"].get("ssh_public_key")
    if private_key_file is not None or public_key_file is not None:
        with _cli_logger.group("Key information:"):
            if private_key_file is not None:
                _cli_logger.labeled_value("Cluster private key file", private_key_file)
                _cli_logger.print("Please keep the cluster private key file safe.")

            if public_key_file is not None:
                _cli_logger.print("Cluster public key file: {}", public_key_file)

    _cli_logger.newline()
    with _cli_logger.group("Useful commands:"):
        config_file = os.path.abspath(config_file)

        with _cli_logger.group("Check cluster status with:"):
            _cli_logger.print(
                cf.bold("cloudtik status {}{}"), config_file, modifiers)

        with _cli_logger.group("Execute command on cluster with:"):
            _cli_logger.print(
                cf.bold("cloudtik exec {}{} [command]"), config_file, modifiers)

        with _cli_logger.group("Connect to a terminal on the cluster head:"):
            _cli_logger.print(
                cf.bold("cloudtik attach {}{}"), config_file, modifiers)

        with _cli_logger.group("Upload files or folders to cluster:"):
            _cli_logger.print(
                cf.bold("cloudtik rsync-up {}{} [source] [target]"), config_file, modifiers)

        with _cli_logger.group("Download files or folders from cluster:"):
            _cli_logger.print(
                cf.bold("cloudtik rsync-down {}{} [source] [target]"), config_file, modifiers)

        with _cli_logger.group("Submit job to cluster to run with:"):
            _cli_logger.print(
                cf.bold("cloudtik submit {}{} [job-file.(py|sh|scala)] "), config_file, modifiers)

        with _cli_logger.group("Monitor cluster with:"):
            _cli_logger.print(
                cf.bold("cloudtik monitor {}{}"), config_file, modifiers)

    _cli_logger.newline()
    with _cli_logger.group("Useful addresses:"):
        proxy_process_file = get_proxy_process_file(cluster_name)
        pid, address, port = get_safe_proxy_process(proxy_process_file)
        if pid is not None:
            bind_address_show = get_proxy_bind_address_to_show(address)
            with _cli_logger.group("The SOCKS5 proxy to access the cluster Web UI from local browsers:"):
                _cli_logger.print(
                    cf.bold("{}:{}"),
                    bind_address_show, port)

        head_node_cluster_ip = get_node_cluster_ip(provider, head_node)

        runtime_services = get_runtime_services(config.get(RUNTIME_CONFIG_KEY), head_node_cluster_ip)
        sorted_runtime_services = sorted(runtime_services.items(), key=lambda kv: kv[1]["name"])
        for service_id, runtime_service in sorted_runtime_services:
            with _cli_logger.group(runtime_service["name"] + ":"):
                if "info" in runtime_service:
                    service_desc = "{}, {}".format(runtime_service["url"], runtime_service["info"])
                else:
                    service_desc = runtime_service["url"]
                _cli_logger.print(cf.bold(service_desc))


def show_cluster_status(config_file: str,
                        override_cluster_name: Optional[str] = None
                        ) -> None:
    config = _load_cluster_config(config_file, override_cluster_name)
    _show_cluster_status(config)


def _show_cluster_status(config: Dict[str, Any]) -> None:
    nodes_info = _get_cluster_nodes_info(config)

    tb = pt.PrettyTable()
    tb.field_names = ["node-id", "node-ip", "node-type", "node-status", "instance-type",
                      "public-ip", "instance-status"]
    for node_info in nodes_info:
        tb.add_row([node_info["node_id"], node_info[NODE_INFO_NODE_IP], node_info[CLOUDTIK_TAG_NODE_KIND],
                    node_info[CLOUDTIK_TAG_NODE_STATUS], node_info["instance_type"], node_info["public_ip"],
                    node_info["instance_status"]
                    ])

    nodes_ready = _get_node_number_in_status(nodes_info, STATUS_UP_TO_DATE)
    cli_logger.print(cf.bold("Total {} nodes. {} nodes are ready"), len(nodes_info), nodes_ready)
    cli_logger.print(tb)


def _get_node_number_in_status(node_info_list, status):
    num_nodes = 0
    for node_info in node_info_list:
        if status == node_info.get(CLOUDTIK_TAG_NODE_STATUS):
            num_nodes += 1
    return num_nodes


def _get_nodes_info_in_status(node_info_list, status):
    return [node_info for node_info in node_info_list if status == node_info.get(CLOUDTIK_TAG_NODE_STATUS)]


def _get_cluster_nodes_info(config: Dict[str, Any]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    nodes = provider.non_terminated_nodes({})
    return _get_sorted_nodes_info(provider, nodes)


def _get_sorted_nodes_info(provider, nodes):
    nodes_info = get_nodes_info(provider, nodes)

    # sort nodes info based on node type and then node ip for workers
    def node_info_sort(node_info):
        node_ip = node_info[NODE_INFO_NODE_IP]
        if node_ip is None:
            node_ip = ""

        return node_info[CLOUDTIK_TAG_NODE_KIND] + node_ip

    nodes_info.sort(key=node_info_sort)
    return nodes_info


def _get_cluster_info(config: Dict[str, Any],
                      provider: NodeProvider = None,
                      simple_config: bool = False) -> Dict[str, Any]:
    if provider is None:
        provider = _get_node_provider(config["provider"], config["cluster_name"])

    cluster_info = {
        "name": config["cluster_name"]
    }

    if not simple_config:
        cluster_info["runtimes"] = get_enabled_runtimes(config)

    # Check whether the head node is running
    try:
        head_node = _get_running_head_node(config)
    except HeadNotRunningError:
        head_node = None

    if head_node is None:
        cluster_info["status"] = CLOUDTIK_CLUSTER_STATUS_STOPPED
        return cluster_info

    cluster_info["status"] = CLOUDTIK_CLUSTER_STATUS_RUNNING
    cluster_info["head-id"] = head_node

    head_ssh_ip = get_head_working_ip(config, provider, head_node)
    cluster_info["head-ssh-ip"] = head_ssh_ip

    # Check the running worker nodes
    workers = _get_worker_nodes(config)
    worker_count = len(workers)

    if not simple_config:
        workers_info = get_nodes_info(provider, workers,
                                      True, config["available_node_types"])
    else:
        workers_info = get_nodes_info(provider, workers)

    # get working nodes which are ready
    worker_nodes_ready = _get_nodes_info_in_status(workers_info, STATUS_UP_TO_DATE)
    workers_ready = len(worker_nodes_ready)
    workers_failed = _get_node_number_in_status(workers_info, STATUS_UPDATE_FAILED)

    cluster_info["total-workers"] = worker_count
    cluster_info["total-workers-ready"] = workers_ready
    cluster_info["total-workers-failed"] = workers_failed

    if not simple_config:
        worker_cpus = sum_worker_cpus(workers_info)
        worker_gpus = sum_worker_gpus(workers_info)
        worker_memory = sum_worker_memory(workers_info)

        any_worker_info = workers_info[0] if workers_info else None
        cpus_per_worker = get_cpus_of_node_info(any_worker_info)
        gpus_per_worker = get_gpus_of_node_info(any_worker_info)
        memory_per_worker = get_memory_of_node_info(any_worker_info)

        cluster_info["total-worker-cpus"] = worker_cpus
        cluster_info["total-worker-gpus"] = worker_gpus
        cluster_info["total-worker-memory"] = worker_memory

        cluster_info["total-worker-cpus-ready"] = sum_worker_cpus(worker_nodes_ready)
        cluster_info["total-worker-gpus-ready"] = sum_worker_gpus(worker_nodes_ready)
        cluster_info["total-worker-memory-ready"] = sum_worker_memory(worker_nodes_ready)

        cluster_info["cpus-per-worker"] = cpus_per_worker
        cluster_info["gpus-per-worker"] = gpus_per_worker
        cluster_info["memory-per-worker"] = memory_per_worker

    if not simple_config:
        default_cloud_storage = get_default_cloud_storage(config)
        if default_cloud_storage:
            cluster_info["default-cloud-storage"] = default_cloud_storage

        default_cloud_database = get_default_cloud_database(config)
        if default_cloud_database:
            cluster_info["default-cloud-database"] = default_cloud_database

    return cluster_info


def confirm(msg: str, yes: bool) -> Optional[bool]:
    return None if yes else click.confirm(msg, abort=True)


def is_proxy_needed(config):
    # A flag to force proxy start and stop in any case
    provider = config["provider"]
    if provider.get("proxy_internal_ips", False):
        return True
    return False if is_use_internal_ip(config) else True


def start_ssh_proxy(config_file: str,
                    override_cluster_name: Optional[str] = None,
                    no_config_cache: bool = False,
                    bind_address: str = None):
    config = _load_cluster_config(config_file, override_cluster_name,
                                  no_config_cache=no_config_cache)

    if not is_proxy_needed(config):
        cli_logger.print(cf.bold(
            "SOCKS5 proxy is not needed. With use_internal_ips is True, "
            "you can access the cluster directly."))
        return

    # Warning about bind_address
    if bind_address is None or bind_address == "":
        cli_logger.warning("The SOCKS5 proxy will be bound on localhost of this node. "
                           "Use --bind-address to specify to bind on a specific address if you want.")

    _start_proxy(config, restart=True, bind_address=bind_address)


def _start_proxy(config: Dict[str, Any],
                 restart: bool = False,
                 bind_address: str = None):
    cluster_name = config["cluster_name"]
    proxy_process_file = get_proxy_process_file(cluster_name)
    pid, address, port = get_safe_proxy_process(proxy_process_file)
    if pid is not None:
        if restart:
            # stop the proxy first
            _stop_proxy(config)
        else:
            cli_logger.print(cf.bold(
                "The SOCKS5 proxy to the cluster {} is already running."),
                cluster_name)
            bind_address_to_show = get_proxy_bind_address_to_show(address)
            cli_logger.print(cf.bold(
                "To access the cluster from local tools, please configure the SOCKS5 proxy with {}:{}."),
                bind_address_to_show, port)
            return

    provider = _get_node_provider(config["provider"], config["cluster_name"])
    # Check whether the head node is running
    try:
        head_node = _get_running_head_node(config)
        head_node_ip = get_head_working_ip(config, provider, head_node)
    except HeadNotRunningError:
        cli_logger.print(cf.bold("Cluster {} is not running."), cluster_name)
        return

    pid, address, port = _start_proxy_process(head_node_ip, config, bind_address)
    cli_logger.print(cf.bold(
        "The SOCKS5 proxy to the cluster {} has been started."),
        cluster_name)
    bind_address_to_show = get_proxy_bind_address_to_show(bind_address)
    cli_logger.print(cf.bold(
        "To access the cluster from local tools, please configure the SOCKS5 proxy with {}:{}."),
        bind_address_to_show, port)


def _start_proxy_process(head_node_ip, config,
                         bind_address: str = None):
    proxy_process_file = get_proxy_process_file(config["cluster_name"])
    cmd = "ssh -o \'StrictHostKeyChecking no\'"

    auth_config = config["auth"]
    ssh_proxy_command = auth_config.get("ssh_proxy_command", None)
    ssh_private_key = auth_config.get("ssh_private_key", None)
    ssh_user = auth_config["ssh_user"]
    ssh_port = auth_config.get("ssh_port", None)

    if not bind_address:
        proxy_port = get_free_port(
            '127.0.0.1', constants.DEFAULT_PROXY_PORT)
    else:
        proxy_port = get_free_port(
            bind_address, constants.DEFAULT_PROXY_PORT)
    if ssh_private_key:
        cmd += " -i {}".format(ssh_private_key)
    if ssh_proxy_command:
        cmd += " -o ProxyCommand=\'{}\'".format(ssh_proxy_command)
    if ssh_port:
        cmd += " -p {}".format(ssh_port)
    if not bind_address:
        bind_string = "{}".format(proxy_port)
    else:
        bind_string = "{}:{}".format(bind_address, proxy_port)

    cmd += " -D {} -C -N {}@{}".format(bind_string, ssh_user, head_node_ip)

    cli_logger.verbose("Running `{}`", cf.bold(cmd))
    p = subprocess.Popen(cmd, shell=True, stderr=subprocess.DEVNULL)

    proxy_process = {"pid": p.pid, "bind_address": bind_address, "port": proxy_port}
    save_server_process(proxy_process_file, proxy_process)
    return p.pid, bind_address, proxy_port


def stop_ssh_proxy(config_file: str,
                   override_cluster_name: Optional[str] = None):
    config = _load_cluster_config(config_file, override_cluster_name)

    if not is_proxy_needed(config):
        cli_logger.print(cf.bold(
            "SOCKS5 proxy is not needed. With use_internal_ips is True, "
            "you can access the cluster directly."))
        return

    _stop_proxy(config)


def _stop_proxy(config: Dict[str, Any]):
    cluster_name = config["cluster_name"]

    proxy_process_file = get_proxy_process_file(cluster_name)
    pid, address, port = get_safe_proxy_process(proxy_process_file)
    if pid is None:
        cli_logger.print(
            cf.bold("The SOCKS5 proxy of cluster {} was not started."), cluster_name)
        return

    kill_process_tree(pid)
    save_server_process(proxy_process_file, {})
    cli_logger.print(
        cf.bold("Successfully stopped the SOCKS5 proxy of cluster {}."), cluster_name)


def exec_cmd_on_cluster(config_file: str,
                        cmd: str,
                        override_cluster_name: Optional[str],
                        no_config_cache: bool = False):
    config = _load_cluster_config(config_file, override_cluster_name,
                                  no_config_cache=no_config_cache)
    call_context = cli_call_context()
    _exec_cluster(
        config,
        call_context=call_context,
        cmd=cmd,
        run_env="auto",
        screen=False,
        tmux=False,
        stop=False,
        start=False,
        port_forward=None,
        with_output=False,
        _allow_uninitialized_state=False)


def _exec_cmd_on_cluster(config: Dict[str, Any],
                         call_context: CallContext,
                         cmd: str):
    _exec_cluster(
        config,
        call_context=call_context,
        cmd=cmd,
        run_env="auto",
        screen=False,
        tmux=False,
        stop=False,
        start=False,
        port_forward=None,
        with_output=False,
        _allow_uninitialized_state=False)


def cluster_debug_status(config_file: str,
                         override_cluster_name: Optional[str],
                         no_config_cache: bool = False) -> None:
    """Return the debug status of a cluster scaling from head node"""

    cmd = f"cloudtik head debug-status"
    exec_cmd_on_cluster(config_file, cmd,
                        override_cluster_name, no_config_cache)


def cluster_health_check(config_file: str,
                         override_cluster_name: Optional[str],
                         no_config_cache: bool = False,
                         with_details=False) -> None:
    """Do a health check on head node and return the results"""

    cmd = f"cloudtik head health-check"
    if with_details:
        cmd += " --with-details"
    exec_cmd_on_cluster(config_file, cmd,
                        override_cluster_name, no_config_cache)


def teardown_cluster_on_head(yes: bool = False,
                             keep_min_workers: bool = False,
                             hard: bool = False) -> None:
    # Since this is running on head, the bootstrap config must exist
    config = load_head_cluster_config()

    if not yes:
        cli_logger.confirm(yes, "Are you sure that you want to shut down all workers of {}?",
                           config["cluster_name"], _abort=True)
        cli_logger.newline()

    call_context = cli_call_context()
    provider = _get_node_provider(config["provider"], config["cluster_name"])

    teardown_cluster_nodes(config,
                           call_context=call_context,
                           provider=provider,
                           workers_only=True,
                           keep_min_workers=keep_min_workers,
                           on_head=True,
                           hard=hard)


def _get_combined_runtimes(config, provider, nodes, runtimes):
    valid_runtime_set = {CLOUDTIK_RUNTIME_NAME}
    for node in nodes:
        runtime_config = _get_node_specific_runtime_config(
            config, provider, node)
        runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
        for runtime_type in runtime_types:
            valid_runtime_set.add(runtime_type)

    if runtimes:
        runtime_filtered = set()
        filtering_runtime_list = runtimes.split(",")
        for filtering_runtime in filtering_runtime_list:
            if filtering_runtime in valid_runtime_set:
                runtime_filtered.add(filtering_runtime)

        return runtime_filtered
    else:
        return valid_runtime_set


def _get_sorted_combined_runtimes(config, provider, nodes, runtimes):
    combined_runtimes = _get_combined_runtimes(config, provider, nodes, runtimes)

    # sort and put cloudtik at the first
    if CLOUDTIK_RUNTIME_NAME in combined_runtimes:
        combined_runtimes.discard(CLOUDTIK_RUNTIME_NAME)
        sorted_runtimes = sorted(combined_runtimes)
        sorted_runtimes.insert(0, CLOUDTIK_RUNTIME_NAME)
        return sorted_runtimes
    else:
        return sorted(combined_runtimes)


def get_runtime_processes(runtime_type):
    if runtime_type == CLOUDTIK_RUNTIME_NAME:
        return constants.CLOUDTIK_PROCESSES

    runtime_cls = _get_runtime_cls(runtime_type)
    return runtime_cls.get_processes()


def cluster_process_status_on_head(
        redis_address, redis_password, runtimes):
    config = load_head_cluster_config()
    provider = _get_node_provider(config["provider"], config["cluster_name"])

    control_state = ControlState()
    _, redis_ip_address, redis_port = validate_redis_address(redis_address)
    control_state.initialize_control_state(redis_ip_address, redis_port,
                                           redis_password)
    node_table = control_state.get_node_table()
    raw_node_states = node_table.get_all().values()
    node_states = _index_node_states(raw_node_states)
    cli_logger.print(cf.bold("Total {} live nodes reported."), len(node_states))

    # checking worker node process
    nodes = provider.non_terminated_nodes({
        CLOUDTIK_TAG_NODE_STATUS: STATUS_UP_TO_DATE
    })
    nodes_info = _get_sorted_nodes_info(provider, nodes)

    combined_runtimes = _get_sorted_combined_runtimes(
        config, provider, nodes, runtimes)

    for runtime in combined_runtimes:
        runtime_processes = get_runtime_processes(runtime)
        if not runtime_processes:
            continue

        cli_logger.newline()
        cli_logger.print(cf.bold("Processes status for: {}."), runtime)

        tb = pt.PrettyTable()
        field_names = ["node-ip", "node-type"]
        for process_meta in runtime_processes:
            field_names.append(process_meta[2])
        tb.field_names = field_names

        for node_info in nodes_info:
            node_ip = node_info[NODE_INFO_NODE_IP]
            node_state = node_states.get(node_ip)
            if node_state:
                process_info = node_state["process"]
            else:
                process_info = {}

            row = [node_ip, node_info[CLOUDTIK_TAG_NODE_KIND]]
            for process_meta in runtime_processes:
                process_name = process_meta[2]
                row.append(process_info.get(process_name, "-"))
            tb.add_row(row)

        cli_logger.print(tb)


def cluster_process_status(config_file: str,
                           override_cluster_name: Optional[str],
                           no_config_cache: bool = False,
                           runtimes=None) -> None:
    """List process status on head node and return the results"""

    cmd = f"cloudtik head process-status"
    if runtimes:
        cmd += " --runtimes=" + quote(runtimes)

    exec_cmd_on_cluster(config_file, cmd,
                        override_cluster_name, no_config_cache)


def exec_on_nodes(
        config: Dict[str, Any],
        call_context: CallContext,
        node_ip: str,
        all_nodes: bool = False,
        cmd: str = None,
        run_env: str = "auto",
        screen: bool = False,
        tmux: bool = False,
        stop: bool = False,
        start: bool = False,
        force_update: bool = False,
        wait_for_workers: bool = False,
        min_workers: Optional[int] = None,
        wait_timeout: Optional[int] = None,
        port_forward: Optional[Port_forward] = None,
        with_output: bool = False,
        parallel: bool = True,
        yes: bool = False,
        job_waiter_name: Optional[str] = None) -> str:
    if not node_ip and not all_nodes:
        if (start or stop) and not yes:
            cli_logger.confirm(
                yes,
                "You are about to start or stop cluster {} for this operation. "
                "Are you sure that you want to continue?",
                config["cluster_name"], _abort=True)
            cli_logger.newline()

        # create job waiter, None if no job waiter specified
        job_waiter = _create_job_waiter(
            config, call_context, job_waiter_name)

        _start_cluster_and_wait_for_workers(
            config=config,
            call_context=call_context,
            start=start,
            force_update=force_update,
            wait_for_workers=wait_for_workers,
            min_workers=min_workers,
            wait_timeout=wait_timeout,
        )

        return _exec_cluster(
            config,
            call_context=call_context,
            cmd=cmd,
            run_env=run_env,
            screen=screen,
            tmux=tmux,
            stop=stop,
            start=False,
            port_forward=port_forward,
            with_output=with_output,
            _allow_uninitialized_state=True,
            job_waiter=job_waiter)
    else:
        return _exec_node_from_head(
            config,
            call_context=call_context,
            node_ip=node_ip,
            all_nodes=all_nodes,
            cmd=cmd,
            run_env=run_env,
            screen=screen,
            tmux=tmux,
            wait_for_workers=wait_for_workers,
            min_workers=min_workers,
            wait_timeout=wait_timeout,
            port_forward=port_forward,
            with_output=with_output,
            parallel=parallel,
            job_waiter_name=job_waiter_name)


def _exec_node_from_head(config: Dict[str, Any],
                         call_context: CallContext,
                         node_ip: str,
                         all_nodes: bool = False,
                         cmd: str = None,
                         run_env: str = "auto",
                         screen: bool = False,
                         tmux: bool = False,
                         wait_for_workers: bool = False,
                         min_workers: Optional[int] = None,
                         wait_timeout: Optional[int] = None,
                         port_forward: Optional[Port_forward] = None,
                         with_output: bool = False,
                         parallel: bool = True,
                         job_waiter_name: Optional[str] = None) -> str:

    # execute exec on head with the cmd
    cmds = [
        "cloudtik",
        "head",
        "exec",
    ]
    cmds += [quote(cmd)]
    if node_ip:
        cmds += ["--node-ip={}".format(node_ip)]
    if all_nodes:
        cmds += ["--all-nodes"]
    else:
        cmds += ["--no-all-nodes"]
    if run_env:
        cmds += ["--run-env={}".format(run_env)]
    if screen:
        cmds += ["--screen"]
    if tmux:
        cmds += ["--tmux"]
    if wait_for_workers:
        cmds += ["--wait-for-workers"]
        if min_workers:
            cmds += ["--min-workers={}".format(min_workers)]
        if wait_timeout:
            cmds += ["--wait-timeout={}".format(wait_timeout)]
    if parallel:
        cmds += ["--parallel"]
    else:
        cmds += ["--no-parallel"]

    if job_waiter_name:
        cmds += ["--job-waiter={}".format(job_waiter_name)]

    # TODO (haifeng): handle port forward and with_output for two state cases
    with_verbose_option(cmds, call_context)
    final_cmd = " ".join(cmds)

    job_waiter = _create_job_waiter(
        config, call_context, job_waiter_name)
    return _exec_cluster(
        config,
        call_context=call_context,
        cmd=final_cmd,
        run_env="auto",
        screen=screen if job_waiter else False,
        tmux=tmux if job_waiter else False,
        stop=False,
        start=False,
        port_forward=port_forward,
        with_output=with_output,
        _allow_uninitialized_state=False,
        job_waiter=job_waiter)


def attach_worker(config_file: str,
                  node_ip: str,
                  use_screen: bool = False,
                  use_tmux: bool = False,
                  override_cluster_name: Optional[str] = None,
                  no_config_cache: bool = False,
                  new: bool = False,
                  port_forward: Optional[Port_forward] = None,
                  force_to_host: bool = False) -> None:
    """Attaches to a screen for the specified cluster.

    Arguments:
        config_file: path to the cluster yaml
        node_ip: the node internal IP to attach
        use_screen: whether to use screen as multiplexer
        use_tmux: whether to use tmux as multiplexer
        override_cluster_name: set the name of the cluster
        no_config_cache: no use config cache
        new: whether to force a new screen
        port_forward ( (int,int) or list[(int,int)] ): port(s) to forward
        force_to_host: Whether attach to host even running with docker
    """
    config = _load_cluster_config(config_file, override_cluster_name,
                                  no_config_cache=no_config_cache)
    call_context = cli_call_context()
    # execute attach on head
    cmds = [
        "cloudtik",
        "head",
        "attach",
    ]
    cmds += ["--node-ip={}".format(node_ip)]
    if use_screen:
        cmds += ["--screen"]
    if use_tmux:
        cmds += ["--tmux"]
    if new:
        cmds += ["--new"]
    if force_to_host:
        cmds += ["--host"]

    # TODO (haifeng): handle port forward for two state cases
    with_verbose_option(cmds, call_context)
    final_cmd = " ".join(cmds)

    _exec_cluster(
        config,
        call_context=call_context,
        cmd=final_cmd,
        run_env="auto",
        screen=False,
        tmux=False,
        stop=False,
        start=False,
        port_forward=port_forward,
        _allow_uninitialized_state=False)


def exec_cmd_on_head(config,
                     call_context: CallContext,
                     provider,
                     node_id: str,
                     cmd: str = None,
                     run_env: str = "auto",
                     screen: bool = False,
                     tmux: bool = False,
                     port_forward: Optional[Port_forward] = None,
                     with_output: bool = False,
                     job_waiter_name: Optional[str] = None) -> str:
    """Runs a command on the specified node from head."""

    assert not (screen and tmux), "Can specify only one of `screen` or `tmux`."
    assert run_env in RUN_ENV_TYPES, "--run_env must be in {}".format(
        RUN_ENV_TYPES)

    # create job waiter, None if no job waiter specified
    job_waiter = _create_job_waiter(
        config, call_context, job_waiter_name)

    updater = create_node_updater_for_exec(
        config=config,
        call_context=call_context,
        node_id=node_id,
        provider=provider,
        start_commands=[],
        is_head_node=False,
        use_internal_ip=True)

    hold_session = False if job_waiter else True
    session_name = get_command_session_name(cmd, time.time_ns())
    result = _exec(
        updater,
        cmd,
        screen,
        tmux,
        port_forward=port_forward,
        with_output=with_output,
        run_env=run_env,
        shutdown_after_run=False,
        session_name=session_name,
        hold_session=hold_session)

    # if a job waiter is specified, we always wait for its completion.
    if job_waiter is not None:
        job_waiter.wait_for_completion(node_id, cmd, session_name)

    return result


def attach_node_on_head(node_ip: str,
                        use_screen: bool,
                        use_tmux: bool,
                        new: bool = False,
                        port_forward: Optional[Port_forward] = None,
                        force_to_host: bool = False):
    config = load_head_cluster_config()
    call_context = cli_call_context()
    provider = _get_node_provider(config["provider"], config["cluster_name"])

    if not node_ip:
        cli_logger.error("Node IP must be specified to attach node!")
        return

    node_id = provider.get_node_id(node_ip, use_internal_ip=True)
    if not node_id:
        cli_logger.error("No node with the specified node ip - {} found.", node_ip)
        return

    cmd = get_attach_command(use_screen, use_tmux, new)
    run_env = "auto"
    if force_to_host:
        run_env = "host"

    exec_cmd_on_head(
        config,
        call_context=call_context,
        provider=provider,
        node_id=node_id,
        cmd=cmd,
        run_env=run_env,
        screen=False,
        tmux=False,
        port_forward=port_forward)


def _exec_node_on_head(
        config: Dict[str, Any],
        call_context: CallContext,
        node_ip: str,
        all_nodes: bool = False,
        cmd: str = None,
        run_env: str = "auto",
        screen: bool = False,
        tmux: bool = False,
        wait_for_workers: bool = False,
        min_workers: Optional[int] = None,
        wait_timeout: Optional[int] = None,
        port_forward: Optional[Port_forward] = None,
        with_output: bool = False,
        parallel: bool = True,
        job_waiter_name: Optional[str] = None):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    head_node = _get_running_head_node(config, _provider=provider)

    # wait for workers if needed
    if wait_for_workers:
        _wait_for_ready(
            config=config, call_context=call_context,
            min_workers=min_workers, timeout=wait_timeout)

    node_head, node_workers = get_nodes_of(
        config, provider=provider, head_node=head_node,
        node_ip=node_ip, all_nodes=all_nodes)
    nodes = [node_head] if node_head else []
    nodes += node_workers

    def run_exec_cmd_on_head(node_id, call_context):
        return exec_cmd_on_head(
            config,
            call_context=call_context,
            provider=provider,
            node_id=node_id, cmd=cmd,
            run_env=run_env,
            screen=screen, tmux=tmux,
            port_forward=port_forward,
            with_output=with_output,
            job_waiter_name=job_waiter_name)

    total_nodes = len(nodes)
    if total_nodes == 1:
        return run_exec_cmd_on_head(
            node_id=nodes[0], call_context=call_context)

    if parallel and total_nodes > 1:
        cli_logger.print("Executing on {} nodes in parallel...", total_nodes)
        run_in_parallel_on_nodes(run_exec_cmd_on_head,
                                 call_context=call_context,
                                 nodes=nodes)
    else:
        for i, node_id in enumerate(nodes):
            node_ip = provider.internal_ip(node_id)
            with cli_logger.group(
                    "Executing on node: {}", node_ip,
                    _numbered=("()", i + 1, total_nodes)):
                run_exec_cmd_on_head(node_id=node_id, call_context=call_context)


def start_node_on_head(node_ip: str = None,
                       all_nodes: bool = False,
                       runtimes: str = None,
                       parallel: bool = True,
                       yes: bool = False):
    # Since this is running on head, the bootstrap config must exist
    config = load_head_cluster_config()
    call_context = cli_call_context()

    if not yes:
        cli_logger.confirm(yes, "Are you sure that you want to perform the start operation?", _abort=True)
        cli_logger.newline()

    _start_node_on_head(
        config=config,
        call_context=call_context,
        node_ip=node_ip,
        all_nodes=all_nodes,
        runtimes=runtimes,
        parallel=parallel
    )


def _start_node_on_head(config: Dict[str, Any],
                        call_context: CallContext,
                        node_ip: str = None,
                        all_nodes: bool = False,
                        runtimes: str = None,
                        parallel: bool = True):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    runtime_list = get_verified_runtime_list(config, runtimes) if runtimes else None

    head_node = _get_running_head_node(config, _provider=provider)
    node_head, node_workers = get_nodes_of(
        config, provider=provider, head_node=head_node,
        node_ip=node_ip, all_nodes=all_nodes)

    _do_start_node_on_head(
        config=config,
        call_context=call_context,
        provider=provider,
        head_node=head_node,
        node_head=node_head,
        node_workers=node_workers,
        runtimes=runtime_list,
        parallel=parallel
    )


def _do_start_node_on_head(
        config: Dict[str, Any],
        call_context: CallContext,
        provider: NodeProvider,
        head_node: str,
        node_head: Optional[str],
        node_workers: List[str],
        runtimes: Optional[List[str]] = None,
        parallel: bool = True):
    head_node_ip = provider.internal_ip(head_node)

    def start_single_node_on_head(node_id, call_context):
        if not is_node_in_completed_status(provider, node_id):
            node_ip = provider.internal_ip(node_id)
            raise ParallelTaskSkipped("Skip starting node {} as it is in setting up.".format(node_ip))

        runtime_config = _get_node_specific_runtime_config(
            config, provider, node_id)
        node_envs = with_runtime_environment_variables(
            runtime_config, config=config, provider=provider, node_id=node_id)

        is_head_node = False
        if node_id == head_node:
            is_head_node = True

        if is_head_node:
            start_commands = get_commands_of_runtimes(config, "head_start_commands",
                                                      runtimes=runtimes)
            node_runtime_envs = with_node_ip_environment_variables(
                call_context, head_node_ip, provider, node_id)
        else:
            start_commands = get_node_specific_commands_of_runtimes(
                config, provider, node_id=node_id,
                command_key="worker_start_commands", runtimes=runtimes)
            node_runtime_envs = with_node_ip_environment_variables(
                call_context, None, provider, node_id)
            node_runtime_envs = with_head_node_ip_environment_variables(
                head_node_ip, node_runtime_envs)

        updater = create_node_updater_for_exec(
            config=config,
            call_context=call_context,
            node_id=node_id,
            provider=provider,
            start_commands=[],
            is_head_node=is_head_node,
            use_internal_ip=True,
            runtime_config=runtime_config)

        node_envs.update(node_runtime_envs)
        updater.exec_commands("Starting", start_commands, node_envs)

    _cli_logger = call_context.cli_logger

    # First start on head if needed
    if node_head:
        with _cli_logger.group(
                "Starting on head: {}", head_node_ip):
            try:
                start_single_node_on_head(node_head, call_context=call_context)
            except ParallelTaskSkipped as e:
                _cli_logger.print(str(e))

    total_workers = len(node_workers)
    if parallel and total_workers > 1:
        _cli_logger.print("Starting on {} workers in parallel...", total_workers)
        run_in_parallel_on_nodes(start_single_node_on_head,
                                 call_context=call_context,
                                 nodes=node_workers)
    else:
        for i, node_id in enumerate(node_workers):
            node_ip = provider.internal_ip(node_id)
            with _cli_logger.group(
                    "Starting on worker: {}", node_ip,
                    _numbered=("()", i + 1, total_workers)):
                try:
                    start_single_node_on_head(node_id, call_context=call_context)
                except ParallelTaskSkipped as e:
                    _cli_logger.print(str(e))


def start_node_from_head(config_file: str,
                         node_ip: str,
                         all_nodes: bool,
                         runtimes: Optional[str] = None,
                         override_cluster_name: Optional[str] = None,
                         no_config_cache: bool = False,
                         indent_level: int = None,
                         parallel: bool = True,
                         yes: bool = False):
    """Execute start node command on head."""
    config = _load_cluster_config(config_file, override_cluster_name,
                                  no_config_cache=no_config_cache)
    call_context = cli_call_context()
    runtime_list = get_verified_runtime_list(config, runtimes) if runtimes else None

    cli_logger.confirm(yes, "Are you sure that you want to perform the start operation?", _abort=True)
    cli_logger.newline()

    _start_node_from_head(
        config, call_context=call_context,
        node_ip=node_ip, all_nodes=all_nodes, runtimes=runtime_list,
        indent_level=indent_level, parallel=parallel)


def _start_node_from_head(config: Dict[str, Any],
                          call_context: CallContext,
                          node_ip: Optional[str],
                          all_nodes: bool,
                          runtimes: Optional[List[str]] = None,
                          indent_level: int = None,
                          parallel: bool = True):
    cmds = [
        "cloudtik",
        "head",
        "runtime",
        "start",
        "--yes",
    ]
    if node_ip:
        cmds += ["--node-ip={}".format(node_ip)]
    if all_nodes:
        cmds += ["--all-nodes"]
    else:
        cmds += ["--no-all-nodes"]
    if runtimes:
        runtime_arg = ",".join(runtimes)
        cmds += ["--runtimes={}".format(quote(runtime_arg))]
    if indent_level:
        cmds += ["--indent-level={}".format(indent_level)]
    if parallel:
        cmds += ["--parallel"]
    else:
        cmds += ["--no-parallel"]

    with_verbose_option(cmds, call_context)
    final_cmd = " ".join(cmds)

    _exec_cmd_on_cluster(config,
                         call_context=call_context,
                         cmd=final_cmd)


def stop_node_from_head(config_file: str,
                        node_ip: str,
                        all_nodes: bool,
                        runtimes: Optional[str] = None,
                        override_cluster_name: Optional[str] = None,
                        no_config_cache: bool = False,
                        indent_level: int = None,
                        parallel: bool = True,
                        yes: bool = False):
    """Execute stop node command on head."""

    config = _load_cluster_config(config_file, override_cluster_name,
                                  no_config_cache=no_config_cache)
    call_context = cli_call_context()
    runtime_list = get_verified_runtime_list(config, runtimes) if runtimes else None

    cli_logger.confirm(yes, "Are you sure that you want to perform the stop operation?", _abort=True)
    cli_logger.newline()

    _stop_node_from_head(
        config, call_context=call_context,
        node_ip=node_ip, all_nodes=all_nodes, runtimes=runtime_list,
        indent_level=indent_level, parallel=parallel)


def _stop_node_from_head(config: Dict[str, Any],
                         call_context: CallContext,
                         node_ip: Optional[str],
                         all_nodes: bool,
                         runtimes: Optional[List[str]] = None,
                         indent_level: int = None,
                         parallel: bool = True):
    cmds = [
        "cloudtik",
        "head",
        "runtime",
        "stop",
        "--yes",
    ]
    if node_ip:
        cmds += ["--node-ip={}".format(node_ip)]
    if all_nodes:
        cmds += ["--all-nodes"]
    else:
        cmds += ["--no-all-nodes"]
    if runtimes:
        runtime_arg = ",".join(runtimes)
        cmds += ["--runtimes={}".format(quote(runtime_arg))]
    if indent_level:
        cmds += ["--indent-level={}".format(indent_level)]
    if parallel:
        cmds += ["--parallel"]
    else:
        cmds += ["--no-parallel"]

    with_verbose_option(cmds, call_context)
    final_cmd = " ".join(cmds)

    _exec_cmd_on_cluster(config,
                         call_context=call_context,
                         cmd=final_cmd)


def get_nodes_of(config,
                 provider,
                 head_node,
                 node_ip: str = None,
                 all_nodes: bool = False):
    node_head = None
    node_workers = []
    if not node_ip:
        if head_node:
            node_head = head_node

        if all_nodes:
            node_workers.extend(_get_worker_nodes(config))
    else:
        node_id = provider.get_node_id(node_ip, use_internal_ip=True)
        if not node_id:
            cli_logger.error("No node with the specified node ip - {} found.", node_ip)
            return
        if head_node == node_id:
            node_head = node_id
        else:
            node_workers = [node_id]
    return node_head, node_workers


def stop_node_on_head(node_ip: str = None,
                      all_nodes: bool = False,
                      runtimes: Optional[str] = None,
                      parallel: bool = True,
                      yes: bool = False):
    config = load_head_cluster_config()
    call_context = cli_call_context()
    if not yes:
        cli_logger.confirm(yes, "Are you sure that you want to perform the stop operation?", _abort=True)
        cli_logger.newline()

    _stop_node_on_head(
        config=config,
        call_context=call_context,
        node_ip=node_ip,
        all_nodes=all_nodes,
        runtimes=runtimes,
        parallel=parallel
    )


def _stop_node_on_head(config: Dict[str, Any],
                       call_context: CallContext,
                       node_ip: str = None,
                       all_nodes: bool = False,
                       runtimes: Optional[str] = None,
                       parallel: bool = True):
    # Since this is running on head, the bootstrap config must exist
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    runtime_list = get_verified_runtime_list(config, runtimes) if runtimes else None
    head_node = _get_running_head_node(config, _provider=provider,
                                       _allow_uninitialized_state=True)
    node_head, node_workers = get_nodes_of(
        config, provider=provider, head_node=head_node,
        node_ip=node_ip, all_nodes=all_nodes)

    _do_stop_node_on_head(
        config=config,
        call_context=call_context,
        provider=provider,
        head_node=head_node,
        node_head=node_head,
        node_workers=node_workers,
        runtimes=runtime_list,
        parallel=parallel
    )


def _do_stop_node_on_head(
        config: Dict[str, Any],
        call_context: CallContext,
        provider: NodeProvider,
        head_node: str,
        node_head: Optional[str],
        node_workers: List[str],
        runtimes: Optional[List[str]] = None,
        parallel: bool = True):
    head_node_ip = provider.internal_ip(head_node)

    def stop_single_node_on_head(node_id, call_context):
        if not is_node_in_completed_status(provider, node_id):
            node_ip = provider.internal_ip(node_id)
            raise ParallelTaskSkipped("Skip stopping node {} as it is in setting up.".format(node_ip))

        runtime_config = _get_node_specific_runtime_config(
            config, provider, node_id)
        node_envs = with_runtime_environment_variables(
            runtime_config, config=config, provider=provider, node_id=node_id)

        is_head_node = False
        if node_id == head_node:
            is_head_node = True

        if is_head_node:
            stop_commands = get_commands_of_runtimes(config, "head_stop_commands",
                                                     runtimes=runtimes)
            node_runtime_envs = with_node_ip_environment_variables(
                call_context, head_node_ip, provider, node_id)
        else:
            stop_commands = get_node_specific_commands_of_runtimes(
                config, provider, node_id=node_id,
                command_key="worker_stop_commands", runtimes=runtimes)
            node_runtime_envs = with_node_ip_environment_variables(
                call_context, None, provider, node_id)
            node_runtime_envs = with_head_node_ip_environment_variables(
                head_node_ip, node_runtime_envs)

        if not stop_commands:
            return

        updater = create_node_updater_for_exec(
            config=config,
            call_context=call_context,
            node_id=node_id,
            provider=provider,
            start_commands=[],
            is_head_node=is_head_node,
            use_internal_ip=True,
            runtime_config=runtime_config)

        node_envs.update(node_runtime_envs)
        updater.exec_commands("Stopping", stop_commands, node_envs)

    _cli_logger = call_context.cli_logger

    total_workers = len(node_workers)
    total_steps = 0
    if node_head:
        total_steps += 1
    if parallel and total_workers > 1:
        total_steps += 1
    else:
        total_steps += total_workers
    current_step = 0

    # First stop the head service
    if node_head:
        with _cli_logger.group(
                "Stopping on head: {}", head_node_ip,
                _numbered=("()", current_step + 1, total_steps)):
            try:
                current_step += 1
                stop_single_node_on_head(node_head, call_context=call_context)
            except ParallelTaskSkipped as e:
                _cli_logger.print(str(e))

    if parallel and total_workers > 1:
        with _cli_logger.group(
                "Stopping on {} workers in parallel...", total_workers,
                _numbered=("()", current_step + 1, total_steps)):
            current_step += 1
            run_in_parallel_on_nodes(stop_single_node_on_head,
                                     call_context=call_context,
                                     nodes=node_workers)
    else:
        for i, node_id in enumerate(node_workers):
            node_ip = provider.internal_ip(node_id)
            with _cli_logger.group(
                    "Stopping on worker: {}", node_ip,
                    _numbered=("()", current_step + i + 1, total_steps)):
                try:
                    stop_single_node_on_head(node_id, call_context=call_context)
                except ParallelTaskSkipped as e:
                    _cli_logger.print(str(e))


def _get_scale_resource_desc(
        cpus: int, gpus: int,
        workers: int, worker_type: Optional[str] = None,
        resources: Optional[Dict[str, int]] = None,
        bundles: Optional[List[dict]] = None
):
    def append_resource_item(resource_string, resource_item):
        if resource_string:
            resource_string += f" and {resource_item}"
        else:
            resource_string = f"{resource_item}"
        return resource_string

    resource_desc = ""
    if cpus:
        resource_desc = append_resource_item(
            resource_desc, f"{cpus} worker CPUs")
    if gpus:
        resource_desc = append_resource_item(
            resource_desc, f"{gpus} worker GPUs")
    if workers:
        resource_desc = append_resource_item(
            resource_desc,
            f"{workers} {worker_type} workers" if worker_type else f"{workers} workers")
    if resources:
        resource_desc = append_resource_item(
            resource_desc, resources)
    if bundles:
        resource_desc = append_resource_item(
            resource_desc, bundles)

    return resource_desc


def scale_cluster(config_file: str, yes: bool, override_cluster_name: Optional[str],
                  cpus: int, gpus: int,
                  workers: int, worker_type: Optional[str] = None,
                  resources: Optional[Dict[str, int]] = None,
                  bundles: Optional[List[dict]] = None,
                  up_only: bool = False):
    config = _load_cluster_config(config_file, override_cluster_name)
    call_context = cli_call_context()
    resource_desc = _get_scale_resource_desc(
        cpus=cpus, gpus=gpus,
        workers=workers, worker_type=worker_type,
        resources=resources, bundles=bundles,
    )
    cli_logger.confirm(yes, "Are you sure that you want to scale cluster {} to {}?",
                       config["cluster_name"], resource_desc, _abort=True)
    cli_logger.newline()

    _scale_cluster(config,
                   call_context=call_context,
                   cpus=cpus, gpus=gpus,
                   workers=workers, worker_type=worker_type,
                   resources=resources,
                   bundles=bundles,
                   up_only=up_only)


def _check_scale_parameters(
    cpus: int, gpus: int, workers: int = None,
    resources: Optional[Dict[str, int]] = None,
    bundles: Optional[List[dict]] = None,
):
    if not (cpus or gpus or workers or resources or bundles):
        raise ValueError("Need specify either 'cpus', `gpus`, `workers`, `resources` or `bundles`.")


def _scale_cluster(config: Dict[str, Any],
                   call_context: CallContext,
                   cpus: int, gpus: int, workers: int = None,
                   worker_type: Optional[str] = None,
                   resources: Optional[Dict[str, int]] = None,
                   bundles: Optional[List[dict]] = None,
                   up_only: bool = False):
    _check_scale_parameters(
        cpus=cpus, gpus=gpus,
        workers=workers,
        resources=resources,
        bundles=bundles
    )

    # send the head the resource request
    scale_cluster_from_head(
        config,
        call_context=call_context,
        cpus=cpus, gpus=gpus,
        workers=workers, worker_type=worker_type,
        resources=resources,
        bundles=bundles,
        up_only=up_only)


def scale_cluster_from_head(
        config: Dict[str, Any],
        call_context: CallContext,
        cpus: int, gpus: int,
        workers: int = None, worker_type: Optional[str] = None,
        resources: Optional[Dict[str, int]] = None,
        bundles: Optional[List[dict]] = None,
        up_only: bool = False):
    # Make a request to head to scale the cluster
    cmds = [
        "cloudtik",
        "head",
        "scale",
        "--yes",
    ]
    if cpus:
        cmds += ["--cpus={}".format(cpus)]
    if gpus:
        cmds += ["--gpus={}".format(gpus)]
    if workers:
        cmds += ["--workers={}".format(workers)]
    if worker_type:
        cmds += ["--worker-type={}".format(worker_type)]
    if resources:
        resources_list = get_resource_list_str(resources)
        cmds += ["--resources={}".format(
            quote(resources_list))]
    if bundles:
        # json dump
        bundles_json = json.dumps(bundles)
        cmds += ["--bundles={}".format(
            quote(bundles_json))]
    if up_only:
        cmds += ["--up-only"]

    with_verbose_option(cmds, call_context)
    final_cmd = " ".join(cmds)
    _exec_cmd_on_cluster(config,
                         call_context=call_context,
                         cmd=final_cmd)


def scale_cluster_on_head(yes: bool, cpus: int, gpus: int,
                          workers: int, worker_type: Optional[str] = None,
                          resources: Optional[Dict[str, int]] = None,
                          bundles: Optional[List[dict]] = None,
                          up_only: bool = False):
    config = load_head_cluster_config()
    call_context = cli_call_context()
    if not yes:
        resource_desc = _get_scale_resource_desc(
            cpus=cpus, gpus=gpus,
            workers=workers, worker_type=worker_type,
            resources=resources, bundles=bundles,
        )
        cli_logger.confirm(yes, "Are you sure that you want to scale cluster {} to {}?",
                           config["cluster_name"], resource_desc, _abort=True)
        cli_logger.newline()

    _scale_cluster_on_head(
        config=config,
        call_context=call_context,
        cpus=cpus,
        gpus=gpus,
        workers=workers,
        worker_type=worker_type,
        resources=resources,
        bundles=bundles,
        up_only=up_only
    )


def _scale_cluster_on_head(
        config: Dict[str, Any],
        call_context: CallContext,
        cpus: int,
        gpus: int,
        workers: int,
        worker_type: Optional[str] = None,
        resources: Optional[Dict[str, int]] = None,
        bundles: Optional[List[dict]] = None,
        up_only: bool = False):
    _check_scale_parameters(
        cpus=cpus, gpus=gpus,
        workers=workers,
        resources=resources,
        bundles=bundles,
    )

    all_resources = {}
    # Calculate nodes request to the number of cpus
    if workers:
        # if nodes specified, we need to check there is only one worker type defined
        if not worker_type:
            worker_type = check_for_single_worker_type(config)

        resource_amount = convert_nodes_to_resource(
            config, workers, worker_type, worker_type)
        if not resource_amount:
            raise RuntimeError("Not be able to convert number of workers to worker node resources.")
        all_resources[worker_type] = resource_amount
    if resources:
        all_resources.update(resources)

    address = services.get_address_to_use_or_die()
    kv_initialize_with_address(address, CLOUDTIK_REDIS_DEFAULT_PASSWORD)

    if up_only:
        # check the existing resources,
        # if it is already larger than the requests, no need to make the requests
        if _is_resource_satisfied(
                config, call_context,
                cpus, gpus,
                resources=all_resources,
                bundles=bundles):
            cli_logger.print("Resources request already satisfied. Skip scaling operation.")
            return

    request_resources(
        num_cpus=cpus, num_gpus=gpus,
        resources=all_resources,
        bundles=bundles,
        config=config)


def _get_resource_requests():
    data = kv_get(CLOUDTIK_RESOURCE_REQUESTS)
    if data:
        try:
            resource_requests = json.loads(data)
            requests = resource_requests.get("requests")
            return requests
        except Exception:
            # improve to handle error
            return None
    return None


def _get_requested_resource(config, requested_resources, resource_id):
    requested = 0
    for requested_resource in requested_resources:
        requested += requested_resource.get(resource_id, 0)

    # remove head amount
    head_resource_requests = _get_head_resource_requests(
        config, resource_id)
    for head_resource_request in head_resource_requests:
        requested -= head_resource_request.get(resource_id, 0)
    return requested


def _get_cluster_unfulfilled_for_bundles(
        config: Dict[str, Any],
        bundles: List[ResourceDict]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    node_types = config["available_node_types"]
    workers = provider.non_terminated_nodes({
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER
    })
    node_type_counts = get_node_type_counts(
        provider, workers, {}, node_types)
    return get_unfulfilled_for_bundles(
        bundles, node_types, node_type_counts)


def _is_bundles_fulfilled(
        config: Dict[str, Any],
        bundles: List[dict]):
    unfulfilled = _get_cluster_unfulfilled_for_bundles(
        config, bundles)
    if unfulfilled:
        return False
    return True


def _is_resource_satisfied(
        config: Dict[str, Any],
        call_context: CallContext,
        cpus: int,
        gpus: Optional[int] = None,
        resources: Optional[Dict[str, int]] = None,
        bundles: Optional[List[dict]] = None,
):
    # check two things
    # 1. whether the request resources already larger
    requested_resources = _get_resource_requests()
    if requested_resources:
        if cpus and _get_requested_resource(
                config, requested_resources, constants.CLOUDTIK_RESOURCE_CPU) < cpus:
            return False
        if gpus and _get_requested_resource(
                config, requested_resources, constants.CLOUDTIK_RESOURCE_GPU) < gpus:
            return False
        if resources:
            for resource_name, resource_amount in resources.items():
                if _get_requested_resource(
                        config, requested_resources, resource_name) < resource_amount:
                    return False

    # 2. whether running cluster resources already satisfied
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    if cpus and get_worker_cpus(config, provider) < cpus:
        return False
    if gpus and get_worker_gpus(config, provider) < gpus:
        return False

    if resources:
        for resource_name, resource_amount in resources.items():
            if get_worker_resource(
                    config, provider, resource_name) < resource_amount:
                return False

    # check the bundles satisfied,
    # this is not 100% accurate because it doesn't consider nodes that are launching
    if bundles:
        if not _is_bundles_fulfilled(config, bundles):
            return False
    return True


def _start_cluster_and_wait_for_workers(
        config: Dict[str, Any],
        call_context: CallContext,
        start: bool = False,
        force_update: bool = False,
        wait_for_workers: bool = False,
        min_workers: Optional[int] = None,
        wait_timeout: Optional[int] = None):

    try:
        head_node = _get_running_head_node_ex(
            config,
            call_context=call_context,
            create_if_needed=False,
            _allow_uninitialized_state=False)
    except HeadNotRunningError:
        head_node = None

    if start:
        if head_node is None or force_update:
            _create_or_update_cluster(
                config=config,
                call_context=call_context,
                no_restart=False,
                restart_only=False,
                yes=True,
                redirect_command_output=False,
                use_login_shells=True)
    else:
        if head_node is None:
            raise RuntimeError("Cluster {} is not running.".format(config["cluster_name"]))

    if wait_for_workers:
        _wait_for_ready(
            config=config, call_context=call_context,
            min_workers=min_workers, timeout=wait_timeout)


def submit_and_exec(
        config: Dict[str, Any],
        call_context: CallContext,
        script: str,
        script_args,
        screen: bool = False,
        tmux: bool = False,
        stop: bool = False,
        start: bool = False,
        force_update: bool = False,
        wait_for_workers: bool = False,
        min_workers: Optional[int] = None,
        wait_timeout: Optional[int] = None,
        port_forward: Optional[Port_forward] = None,
        with_output: bool = False,
        yes: bool = False,
        job_waiter_name: Optional[str] = None,
        job_log: bool = False,
        runtime: Optional[str] = None,
        runtime_options: Optional[List[str]] = None,
):
    def prepare_submit_command():
        target_name = os.path.basename(script)
        target = os.path.join("~", "user", "jobs", target_name)

        # Create the "user/jobs" and "user/logs" folder before do upload
        cmd_mkdir = "mkdir -p ~/user/jobs; mkdir -p ~/user/logs"
        _exec_cmd_on_cluster(
            config,
            call_context=call_context,
            cmd=cmd_mkdir
        )
        cmds = []
        if urllib.parse.urlparse(script).scheme in ("http", "https"):
            cmds = ["wget", quote(script), "-O", f"~/user/jobs/{target_name};"]
        else:
            # upload the script to cluster
            _rsync(
                config,
                call_context=call_context,
                source=script,
                target=target,
                down=False)

        # Use new target with $HOME instead of ~ for exec
        target = os.path.join("$HOME", "user", "jobs", target_name)
        if runtime is not None:
            runtime_commands = get_runnable_command(
                config.get(RUNTIME_CONFIG_KEY), target, runtime, runtime_options)
            if runtime_commands is None:
                cli_logger.abort("Runtime {} doesn't how to execute your file: {}", runtime, script)
            cmds += runtime_commands
        elif target_name.endswith(".py"):
            cmds += ["python", double_quote(target)]
        elif target_name.endswith(".sh"):
            cmds += ["bash", double_quote(target)]
        else:
            runtime_commands = get_runnable_command(config.get(RUNTIME_CONFIG_KEY), target)
            if runtime_commands is None:
                cli_logger.abort("We don't how to execute your file: {}", script)
            cmds += runtime_commands

        with_script_args(cmds, script_args)

        # If user uses screen or tmux and job waiter is used
        # which means to not hold the tmux or screen session, we redirect log with the session name
        user_cmd = " ".join(cmds)
        session_name = get_command_session_name(user_cmd, time.time_ns())
        if job_log or ((screen or tmux) and job_waiter_name is not None):
            redirect_output = f">$HOME/user/logs/{session_name}.log"
            cmds += [redirect_output, "2>&1"]

        return cmds, session_name

    _exec_with_prepare(
        config,
        call_context=call_context,
        prepare=prepare_submit_command,
        prepare_args=(),
        screen=screen,
        tmux=tmux,
        stop=stop,
        start=start,
        force_update=force_update,
        wait_for_workers=wait_for_workers,
        min_workers=min_workers,
        wait_timeout=wait_timeout,
        port_forward=port_forward,
        with_output=with_output,
        yes=yes,
        job_waiter_name=job_waiter_name,
    )


def _run_script(
        config: Dict[str, Any],
        call_context: CallContext,
        script: str,
        script_args,
        screen: bool = False,
        tmux: bool = False,
        stop: bool = False,
        start: bool = False,
        force_update: bool = False,
        wait_for_workers: bool = False,
        min_workers: Optional[int] = None,
        wait_timeout: Optional[int] = None,
        port_forward: Optional[Port_forward] = None,
        with_output: bool = False,
        yes: bool = False,
        job_waiter_name: Optional[str] = None,
        job_log: bool = False
):
    def prepare_run_script():
        cmds = ["cloudtik", "node", "run", script]
        with_script_args(cmds, script_args)

        # If user uses screen or tmux and job waiter is used
        # which means to not hold the tmux or screen session, we redirect log with the session name
        user_cmd = " ".join(cmds)
        session_name = get_command_session_name(user_cmd, time.time_ns())
        if job_log or ((screen or tmux) and job_waiter_name is not None):
            redirect_output = f">$HOME/user/logs/{session_name}.log"
            cmds += [redirect_output, "2>&1"]
        return cmds, session_name

    _exec_with_prepare(
        config,
        call_context=call_context,
        prepare=prepare_run_script,
        prepare_args=(),
        screen=screen,
        tmux=tmux,
        stop=stop,
        start=start,
        force_update=force_update,
        wait_for_workers=wait_for_workers,
        min_workers=min_workers,
        wait_timeout=wait_timeout,
        port_forward=port_forward,
        with_output=with_output,
        yes=yes,
        job_waiter_name=job_waiter_name,
    )


def _exec_with_prepare(
        config: Dict[str, Any],
        call_context: CallContext,
        prepare,
        prepare_args=(),
        screen: bool = False,
        tmux: bool = False,
        stop: bool = False,
        start: bool = False,
        force_update: bool = False,
        wait_for_workers: bool = False,
        min_workers: Optional[int] = None,
        wait_timeout: Optional[int] = None,
        port_forward: Optional[Port_forward] = None,
        with_output: bool = False,
        yes: bool = False,
        job_waiter_name: Optional[str] = None
):
    assert not (screen and tmux), "Can specify only one of `screen` or `tmux`."

    if (start or stop) and not yes:
        cli_logger.confirm(
            yes,
            "You are about to start or stop cluster {} for this operation. "
            "Are you sure that you want to continue?",
            config["cluster_name"], _abort=True)
        cli_logger.newline()

    # create job waiter, None if no job waiter specified
    job_waiter = _create_job_waiter(
        config, call_context, job_waiter_name)

    _start_cluster_and_wait_for_workers(
        config=config,
        call_context=call_context,
        start=start,
        force_update=force_update,
        wait_for_workers=wait_for_workers,
        min_workers=min_workers,
        wait_timeout=wait_timeout,
    )
    cmds, session_name = prepare(
        *prepare_args,
    )

    cmd = " ".join(cmds)
    return _exec_cluster(
        config,
        call_context=call_context,
        cmd=cmd,
        screen=screen,
        tmux=tmux,
        stop=stop,
        start=False,
        port_forward=port_forward,
        with_output=with_output,
        job_waiter=job_waiter,
        session_name=session_name)


def _run_script_on_head(
        config: Dict[str, Any],
        call_context: CallContext,
        script: str,
        script_args: Optional[List[str]] = None,
        wait_for_workers: bool = False,
        min_workers: Optional[int] = None,
        wait_timeout: Optional[int] = None,
        with_output: bool = False):
    # wait for workers if needed
    if wait_for_workers:
        _wait_for_ready(
            config=config, call_context=call_context,
            min_workers=min_workers, timeout=wait_timeout)

    return run_script(
        script, script_args,
        with_output=with_output)


def _get_workers_ready(config: Dict[str, Any], provider):
    workers = _get_worker_nodes(config)
    workers_info = get_nodes_info(provider, workers)

    # get working nodes which are ready
    workers_ready = _get_node_number_in_status(workers_info, STATUS_UP_TO_DATE)
    return workers_ready


def _wait_for_ready(config: Dict[str, Any],
                    call_context: CallContext,
                    min_workers: int = None,
                    timeout: int = None) -> None:
    if min_workers is None:
        min_workers = _sum_min_workers(config)

    if timeout is None:
        timeout = constants.CLOUDTIK_WAIT_FOR_CLUSTER_READY_TIMEOUT_S

    provider = _get_node_provider(config["provider"], config["cluster_name"])
    workers_ready = _get_workers_ready(config, provider)
    if workers_ready >= min_workers:
        return

    interval = constants.CLOUDTIK_WAIT_FOR_CLUSTER_READY_INTERVAL_S
    start_time = time.time()
    while time.time() - start_time < timeout:
        workers_ready = _get_workers_ready(config, provider)
        if workers_ready >= min_workers:
            return
        else:
            call_context.cli_logger.print("Waiting for workers to be ready: {}/{} ({} seconds)...", workers_ready, min_workers, interval)
            time.sleep(interval)
    raise TimeoutError("Timed out while waiting for workers to be ready: {}/{}".format(workers_ready, min_workers))


def _create_job_waiter(
        config: Dict[str, Any],
        call_context: CallContext,
        job_waiter_name: Optional[str] = None) -> Optional[JobWaiter]:
    return create_job_waiter(config, job_waiter_name)


def get_default_cloud_storage(
        config: Dict[str, Any]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    return provider.get_default_cloud_storage()


def get_default_cloud_database(
        config: Dict[str, Any]):
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    return provider.get_default_cloud_database()


def do_health_check(
        address, redis_password, component, with_details):
    if not address:
        redis_address = services.get_address_to_use_or_die()
    else:
        redis_address = services.address_to_ip(address)

    if not component:
        do_core_health_check(
            redis_address, redis_password, with_details)
    else:
        do_component_health_check(
            redis_address, redis_password, component, with_details)


def do_component_health_check(
        redis_address, redis_password, component, with_details=False):
    kv_initialize_with_address(redis_address, redis_password)
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


def do_core_health_check(redis_address, redis_password, with_details=False):
    redis_client = kv_initialize_with_address(redis_address, redis_password)

    check_ok = True
    try:
        # We are health checking the core. If
        # client creation or ping fails, we will still exit with a non-zero
        # exit code.
        redis_client.ping()

        # check cluster controller live status through scaling status time
        status = kv_store.kv_get(CLOUDTIK_CLUSTER_SCALING_STATUS)
        if not status:
            cli_logger.warning("No scaling status reported from the Cluster Controller.")
            check_ok = False
        else:
            report_time = decode_cluster_scaling_time(status)
            time_ok = is_alive_time(report_time)
            if not time_ok:
                cli_logger.warning("Last scaling status is too old. Status time: {}", report_time)
                check_ok = False

        # check the process status
        failed_nodes = do_nodes_health_check(
            redis_address, redis_password, with_details)
        if failed_nodes:
            cli_logger.warning("{} nodes are not healthy.", len(failed_nodes))
            check_ok = False
    except Exception as e:
        cli_logger.error("Health check failed." + str(e))
        check_ok = False
        pass

    if check_ok:
        cli_logger.success("Cluster is healthy.")
        sys.exit(0)
    else:
        cli_logger.error("Cluster is not healthy. Please check the details above.")
        sys.exit(1)


def _index_node_states(raw_node_states):
    node_states = {}
    if raw_node_states:
        for raw_node_state in raw_node_states:
            node_state = eval(raw_node_state)
            if not is_alive_time(node_state.get("last_heartbeat_time", 0)):
                continue
            node_states[node_state["node_ip"]] = node_state
    return node_states


def do_nodes_health_check(redis_address, redis_password, with_details=False):
    config = load_head_cluster_config()
    provider = _get_node_provider(config["provider"], config["cluster_name"])

    failed_nodes = {}
    # Check whether the head node is running
    try:
        head_node = _get_running_head_node(config)
    except HeadNotRunningError:
        head_node = None

    if head_node is None:
        cli_logger.warning("Head node is not running.")
        failed_nodes[head_node] = head_node
        return failed_nodes

    control_state = ControlState()
    _, redis_ip_address, redis_port = validate_redis_address(redis_address)
    control_state.initialize_control_state(redis_ip_address, redis_port,
                                           redis_password)
    node_table = control_state.get_node_table()
    raw_node_states = node_table.get_all().values()
    node_states = _index_node_states(raw_node_states)

    # checking head node processes
    head_node_info = get_node_info(provider, head_node)
    node_process_ok = check_node_processes(
        config, provider, head_node_info,
        node_states, with_details)
    if not node_process_ok:
        cli_logger.warning("Head node is not healthy. One or more process are not running.")
        failed_nodes[head_node] = head_node

    # checking worker node process
    workers = provider.non_terminated_nodes({
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER,
        CLOUDTIK_TAG_NODE_STATUS: STATUS_UP_TO_DATE
    })
    workers_info = _get_sorted_nodes_info(provider, workers)
    for worker_info in workers_info:
        worker = worker_info["node"]
        node_process_ok = check_node_processes(
            config, provider, worker_info,
            node_states, with_details)
        if not node_process_ok:
            cli_logger.warning("Worker node {} is not healthy. One or more process are not running.",
                             worker_info[NODE_INFO_NODE_IP])
            failed_nodes[worker] = worker

    return failed_nodes


def is_process_status_healthy(process_status):
    if process_status and (
            process_status == psutil.STATUS_RUNNING or
            process_status == psutil.STATUS_SLEEPING or
            process_status == psutil.STATUS_IDLE or
            process_status == psutil.STATUS_WAKING):
        return True
    return False


def check_node_processes(
        config, provider, node_info, node_states, with_details=False):
    node_process_ok = True
    node_ip = node_info[NODE_INFO_NODE_IP]
    node_kind = node_info[CLOUDTIK_TAG_NODE_KIND]
    node_kind_name = "Worker"
    if node_kind == NODE_KIND_HEAD:
        node_kind_name = "Head"
    node_state = node_states.get(node_ip)
    if not node_state:
        cli_logger.warning("No states reported for {} node: {}.",
                         node_kind, node_ip)
        return False

    process_info = node_state["process"]
    unhealthy_processes = []

    tb = pt.PrettyTable()
    tb.field_names = ["process-name", "process-status", "runtime"]
    tb.align = "l"

    # Check core processes
    runtime_type = CLOUDTIK_RUNTIME_NAME
    core_processes = constants.CLOUDTIK_PROCESSES
    for process_meta in core_processes:
        process_name = process_meta[2]
        process_kind = process_meta[3]
        if process_kind != node_kind and process_kind != "node":
            continue

        process_status = process_info.get(process_name)
        if not is_process_status_healthy(process_status):
            unhealthy_processes.append(process_name)
        tb.add_row(
            [process_name, process_info.get(process_name, "-"), runtime_type])

    runtime_config = _get_node_specific_runtime_config(
        config, provider, node_info["node"])

    runtime_types = runtime_config.get(RUNTIME_TYPES_CONFIG_KEY, [])
    for runtime_type in runtime_types:
        runtime_cls = _get_runtime_cls(runtime_type)
        runtime_processes = runtime_cls.get_processes()
        if not runtime_processes:
            continue

        for process_meta in runtime_processes:
            process_name = process_meta[2]
            process_kind = process_meta[3]
            if process_kind != node_kind and process_kind != "node":
                continue

            process_status = process_info.get(process_name)
            if not is_process_status_healthy(process_status):
                unhealthy_processes.append(process_name)
            tb.add_row(
                [process_name, process_info.get(process_name, "-"), runtime_type])

    if unhealthy_processes:
        cli_logger.warning(
            "{} ({}) has {} unhealthy processes: {}.",
            node_kind_name, node_ip,
            len(unhealthy_processes), unhealthy_processes)
        node_process_ok = False
    else:
        cli_logger.success(
            "{} ({}) is healthy.",
            node_kind_name, node_ip)

    if with_details:
        cli_logger.print("Process details:", node_kind_name, node_ip)
        cli_logger.print(tb)
        cli_logger.newline()

    return node_process_ok


def cluster_resource_metrics(
        config_file: str,
        override_cluster_name: Optional[str],
        no_config_cache: bool = False) -> None:
    """Show cluster resource metrics from head node"""

    cmd = f"cloudtik head resource-metrics"
    exec_cmd_on_cluster(config_file, cmd,
                        override_cluster_name, no_config_cache)


def cluster_resource_metrics_on_head(
        redis_address, redis_password):
    config = load_head_cluster_config()
    _, redis_ip_address, redis_port = validate_redis_address(redis_address)
    call_context = cli_call_context()

    show_cluster_metrics(
        config=config,
        call_context=call_context,
        redis_port=redis_port,
        redis_password=redis_password,
        on_head=True
    )


def show_cluster_metrics(
        config: Dict[str, Any],
        call_context: CallContext,
        redis_port, redis_password,
        on_head=False):
    resource_metrics = get_cluster_metrics(
        config=config,
        redis_port=redis_port,
        redis_password=redis_password,
        on_head=on_head)

    _cli_logger = call_context.cli_logger

    # print the metrics
    cli_logger.print(cf.bold("Cluster resource metrics (workers):"))
    cli_logger.labeled_value("Total Cores", resource_metrics["total_cpus"])
    cli_logger.labeled_value("Used Cores", resource_metrics["used_cpus"])
    cli_logger.labeled_value("Free Cores", resource_metrics["available_cpus"])
    cli_logger.labeled_value("CPU Load", resource_metrics["cpu_load"])
    cli_logger.labeled_value("Total Memory", memory_to_gb_string(resource_metrics["total_memory"]))
    cli_logger.labeled_value("Used Memory", memory_to_gb_string(resource_metrics["used_memory"]))
    cli_logger.labeled_value("Free Memory", memory_to_gb_string(resource_metrics["available_memory"]))
    cli_logger.labeled_value("Memory Load", resource_metrics["memory_load"])

    cli_logger.newline()
    cli_logger.print(cf.bold("Node resource metrics:"))

    tb = pt.PrettyTable()
    tb.field_names = ["node-ip", "node-type",
                      "Total Cores", "Used Cores", "Free Cores", "CPU Load",
                      "Total Mem", "Used Mem", "Free Mem", "Mem Load"
                      ]
    tb.align = "l"

    nodes_resource_metrics = resource_metrics["nodes"]

    def node_resource_sort(node_resource):
        node_ip = node_resource["node_ip"]
        return node_resource["node_type"] + node_ip

    nodes_resource_metrics.sort(key=node_resource_sort)

    for node_resource_metrics in nodes_resource_metrics:
        tb.add_row(
            [node_resource_metrics["node_ip"], node_resource_metrics["node_type"],
             node_resource_metrics["total_cpus"], node_resource_metrics["used_cpus"],
             node_resource_metrics["available_cpus"], node_resource_metrics["cpu_load"],
             memory_to_gb_string(node_resource_metrics["total_memory"]),
             memory_to_gb_string(node_resource_metrics["used_memory"]),
             memory_to_gb_string(node_resource_metrics["available_memory"]),
             node_resource_metrics["memory_load"]
             ])
    cli_logger.print(tb)


def get_cluster_metrics(
        config: Dict[str, Any],
        redis_port, redis_password,
        on_head=False
):
    def get_node_states(ip_address, port):
        control_state = ControlState()
        control_state.initialize_control_state(
            ip_address, port, redis_password)
        node_metrics_table = control_state.get_node_metrics_table()
        return node_metrics_table.get_all().values()

    node_metrics_rows = request_tunnel_to_head(
        config=config,
        target_port=redis_port,
        on_head=on_head,
        request_fn=get_node_states
    )

    # Organize the node resource state
    nodes_metrics = _get_nodes_metrics(node_metrics_rows)
    nodes_resource_metrics = get_nodes_resource_metrics(
        nodes_metrics
    )
    resource_metrics = get_cluster_resource_metrics(
        nodes_metrics
    )
    resource_metrics["nodes"] = nodes_resource_metrics
    return resource_metrics


def _get_nodes_metrics(node_metrics_rows):
    nodes_metrics = []
    for node_metrics_row in node_metrics_rows:
        node_metrics = json.loads(node_metrics_row)
        node_id = node_metrics["node_id"]
        node_ip = node_metrics["node_ip"]
        if not node_id or not node_ip:
            continue

        # Filter out the stale record in the node table
        last_metrics_time = node_metrics.get("metrics_time", 0)
        delta = time.time() - last_metrics_time
        if delta >= constants.CLOUDTIK_HEARTBEAT_TIMEOUT_S:
            continue

        metrics = node_metrics.get("metrics")
        if not metrics:
            continue

        nodes_metrics.append(node_metrics)

    return nodes_metrics


def get_nodes_resource_metrics(nodes_metrics):
    nodes_resource_metrics = []
    for node_metrics in nodes_metrics:
        node_ip = node_metrics["node_ip"]
        node_type = node_metrics["node_type"]
        metrics = node_metrics["metrics"]

        cpu_counts = metrics.get("cpus")
        total_cpus = cpu_counts[0]

        load_avg = metrics.get("load_avg")
        load_avg_per_cpu = load_avg[1]
        load_avg_per_cpu_1 = load_avg_per_cpu[0]
        load_avg_all_1 = load_avg[0][0]
        used_cpus = min(math.ceil(load_avg_all_1), total_cpus)

        memory = metrics.get("mem")
        (total_memory, available_memory, percent_memory, used_memory) = memory

        node_resource_metrics = {
            "node_ip": node_ip,
            "node_type": node_type,
            "total_cpus": total_cpus,
            "used_cpus": used_cpus,
            "available_cpus": max(0, total_cpus - used_cpus),
            "cpu_load": load_avg_per_cpu_1,
            "total_memory": total_memory,
            "used_memory": used_memory,
            "available_memory": max(0, total_memory - used_memory),
            "memory_load": percent_memory,
        }

        nodes_resource_metrics.append(node_resource_metrics)
    return nodes_resource_metrics


def get_cluster_resource_metrics(nodes_metrics):
    cluster_total_cpus = 0
    cluster_used_cpus = 0
    cluster_total_memory = 0
    cluster_used_memory = 0
    cluster_load_avg_all_1 = 0.0
    for node_metrics in nodes_metrics:
        # filter out the head node
        if node_metrics["node_type"] == tags.NODE_KIND_HEAD:
            continue

        metrics = node_metrics["metrics"]

        cpu_counts = metrics.get("cpus")
        total_cpus = cpu_counts[0]

        load_avg = metrics.get("load_avg")
        load_avg_all = load_avg[0]
        load_avg_all_1 = load_avg_all[0]

        memory = metrics.get("mem")
        (total_memory, available_memory, percent_memory, used_memory) = memory

        cluster_total_cpus += total_cpus
        cluster_used_cpus += min(math.ceil(load_avg_all_1), total_cpus)
        cluster_load_avg_all_1 += load_avg_all_1
        cluster_total_memory += total_memory
        cluster_used_memory += used_memory

    cluster_cpu_load_1 = 0.0
    if cluster_total_cpus > 0:
        cluster_cpu_load_1 = round(cluster_load_avg_all_1 / cluster_total_cpus, 2)

    cluster_memory_load = 0.0
    if cluster_total_memory > 0:
        cluster_memory_load = round(cluster_used_memory / cluster_total_memory, 2)
    return {
        "total_cpus": cluster_total_cpus,
        "used_cpus": cluster_used_cpus,
        "available_cpus": max(0, cluster_total_cpus - cluster_used_cpus),
        "cpu_load": cluster_cpu_load_1,
        "total_memory": cluster_total_memory,
        "used_memory": cluster_used_memory,
        "available_memory": max(0, cluster_total_memory - cluster_used_memory),
        "memory_load": cluster_memory_load,
    }
