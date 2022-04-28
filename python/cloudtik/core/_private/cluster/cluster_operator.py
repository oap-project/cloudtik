import copy
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import subprocess
import tempfile
import time
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union
import prettytable as pt

import click
import yaml

from cloudtik.core._private import services

try:  # py3
    from shlex import quote
except ImportError:  # py2
    from pipes import quote

from cloudtik.core._private.state.kv_store import kv_put, kv_initialize_with_address

from cloudtik.core.node_provider import NodeProvider
from cloudtik.core._private.constants import \
    CLOUDTIK_RESOURCE_REQUEST_CHANNEL, \
    MAX_PARALLEL_SHUTDOWN_WORKERS, \
    CLOUDTIK_DEFAULT_PORT, \
    CLOUDTIK_REDIS_DEFAULT_PASSWORD
from cloudtik.core._private.utils import validate_config, hash_runtime_conf, \
    hash_launch_conf, prepare_config, get_free_port, \
    get_proxy_info_file, get_safe_proxy_process_info, \
    get_head_working_ip, get_node_cluster_ip, is_use_internal_ip, get_head_bootstrap_config, \
    get_attach_command, is_alive_time, is_docker_enabled, get_proxy_bind_address_to_show, \
    kill_process_tree, with_runtime_environment_variables, verify_config, runtime_prepare_config, get_nodes_info, \
    sum_worker_cpus, sum_worker_memory, get_useful_runtime_urls, get_enabled_runtimes, \
    with_head_node_ip, with_node_ip_environment_variables, run_in_paralell_on_nodes, cluster_booting_completed

from cloudtik.core._private.providers import _get_node_provider, \
    _NODE_PROVIDERS, _PROVIDER_PRETTY_NAMES
from cloudtik.core.tags import (
    CLOUDTIK_TAG_NODE_KIND, CLOUDTIK_TAG_LAUNCH_CONFIG, CLOUDTIK_TAG_NODE_NAME,
    NODE_KIND_WORKER, NODE_KIND_HEAD, CLOUDTIK_TAG_USER_NODE_TYPE,
    STATUS_UNINITIALIZED, STATUS_UP_TO_DATE, CLOUDTIK_TAG_NODE_STATUS)
from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core._private.node.node_updater import NodeUpdaterThread
from cloudtik.core._private.command_executor import set_using_login_shells, \
    set_rsync_silent
from cloudtik.core._private.event_system import (CreateClusterEvent, global_event_system)
from cloudtik.core._private.log_timer import LogTimer
from cloudtik.core._private.cluster.cluster_dump import Archive, \
    GetParameters, Node, _info_from_params, \
    create_archive_for_remote_nodes, get_all_local_data, \
    create_archive_for_cluster_nodes
from cloudtik.core._private.state.control_state import ControlState
from cloudtik.core._private.debug import log_once

import cloudtik.core._private.subprocess_output_util as cmd_output_util
from cloudtik.core._private.cluster.cluster_metrics import ClusterMetricsSummary
from cloudtik.core._private.cluster.cluster_scaler import ClusterScalerSummary
from cloudtik.core._private.utils import format_info_string

logger = logging.getLogger(__name__)

RUN_ENV_TYPES = ["auto", "host", "docker"]

POLL_INTERVAL = 5

Port_forward = Union[Tuple[int, int], List[Tuple[int, int]]]

NUM_TEARDOWN_CLUSTER_STEPS_BASE = 2


def try_logging_config(config: Dict[str, Any]) -> None:
    if config["provider"]["type"] == "aws":
        from cloudtik.providers._private.aws.config import log_to_cli
        log_to_cli(config)


def try_get_log_state(provider_config: Dict[str, Any]) -> Optional[dict]:
    if provider_config["type"] == "aws":
        from cloudtik.providers._private.aws.config import get_log_state
        return get_log_state()
    return None


def try_reload_log_state(provider_config: Dict[str, Any],
                         log_state: dict) -> None:
    if not log_state:
        return
    if provider_config["type"] == "aws":
        from cloudtik.providers._private.aws.config import reload_log_state
        return reload_log_state(log_state)


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
                      bundles: Optional[List[dict]] = None) -> None:
    """Remotely request some CPU or GPU resources from the cluster scaler.

    This function is to be called e.g. on a node before submitting a bunch of
    jobs to ensure that resources rapidly become available.

    Args:
        num_cpus (int): Scale the cluster to ensure this number of CPUs are
            available. This request is persistent until another call to
            request_resources() is made.
        bundles (List[ResourceDict]): Scale the cluster to ensure this set of
            resource shapes can fit. This request is persistent until another
            call to request_resources() is made.
    """
    # TODO (haifeng): handle resource request
    to_request = []
    if num_cpus:
        to_request += [{"CPU": 1}] * num_cpus
    if bundles:
        to_request += bundles
    kv_put(
        CLOUDTIK_RESOURCE_REQUEST_CHANNEL,
        json.dumps(to_request),
        overwrite=True)


def create_or_update_cluster(
        config_file: str,
        override_min_workers: Optional[int],
        override_max_workers: Optional[int],
        no_restart: bool,
        restart_only: bool,
        yes: bool,
        override_cluster_name: Optional[str] = None,
        override_workspace_name: Optional[str] = None,
        no_config_cache: bool = False,
        redirect_command_output: Optional[bool] = False,
        use_login_shells: bool = True,
        no_controller_on_head: bool = False) -> Dict[str, Any]:
    """Creates or updates an scaling cluster from a config json."""
    # no_controller_on_head is an internal flag used by the K8s operator.
    # If True, prevents autoscaling config sync to the  head during cluster
    # creation. See pull #13720.
    set_using_login_shells(use_login_shells)
    if not use_login_shells:
        cmd_output_util.set_allow_interactive(False)
    if redirect_command_output is None:
        # Do not redirect by default.
        cmd_output_util.set_output_redirected(False)
    else:
        cmd_output_util.set_output_redirected(redirect_command_output)

    def handle_yaml_error(e):
        cli_logger.error("Cluster config invalid")
        cli_logger.newline()
        cli_logger.error("Failed to load YAML file " + cf.bold("{}"),
                         config_file)
        cli_logger.newline()
        with cli_logger.verbatim_error_ctx("PyYAML error:"):
            cli_logger.error(e)
        cli_logger.abort()

    try:
        config = yaml.safe_load(open(config_file).read())
    except FileNotFoundError:
        cli_logger.abort(
            "Provided cluster configuration file ({}) does not exist",
            cf.bold(config_file))
    except yaml.parser.ParserError as e:
        handle_yaml_error(e)
        raise
    except yaml.scanner.ScannerError as e:
        handle_yaml_error(e)
        raise
    global_event_system.execute_callback(CreateClusterEvent.up_started,
                                         {"cluster_config": config})

    # todo: validate file_mounts, ssh keys, etc.

    importer = _NODE_PROVIDERS.get(config["provider"]["type"])
    if not importer:
        cli_logger.abort(
            "Unknown provider type " + cf.bold("{}") + "\n"
            "Available providers are: {}", config["provider"]["type"],
            cli_logger.render_list([
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
        cli_logger.newline()

    cli_logger.labeled_value("Cluster", config["cluster_name"])
    workspace_name = config.get("workspace_name")
    if workspace_name:
        cli_logger.labeled_value("Workspace", workspace_name)
    cli_logger.labeled_value("Runtimes", ", ".join(get_enabled_runtimes(config)))

    cli_logger.newline()
    config = _bootstrap_config(config, no_config_cache=no_config_cache,
                               init_config_cache=True)

    try_logging_config(config)
    get_or_create_head_node(config, config_file, no_restart, restart_only, yes,
                            override_cluster_name, no_controller_on_head)
    return config


CONFIG_CACHE_VERSION = 1


def _bootstrap_config(config: Dict[str, Any],
                      no_config_cache: bool = False,
                      init_config_cache: bool = False) -> Dict[str, Any]:
    # Check if bootstrapped, return if it is the case
    if config.get("bootstrapped", False):
        return config

    config = prepare_config(config)
    # NOTE: multi-node-type cluster scaler is guaranteed to be in use after this.

    hasher = hashlib.sha1()
    hasher.update(json.dumps([config], sort_keys=True).encode("utf-8"))
    cache_key = os.path.join(tempfile.gettempdir(),
                             "cloudtik-config-{}".format(hasher.hexdigest()))

    if os.path.exists(cache_key) and not no_config_cache:
        config_cache = json.loads(open(cache_key).read())
        if config_cache.get("_version", -1) == CONFIG_CACHE_VERSION:
            # todo: is it fine to re-resolve? afaik it should be.
            # we can have migrations otherwise or something
            # but this seems overcomplicated given that resolving is
            # relatively cheap
            try_reload_log_state(config_cache["config"]["provider"],
                                 config_cache.get("provider_log_info"))

            if log_once("_printed_cached_config_warning"):
                cli_logger.verbose_warning(
                    "Loaded cached provider configuration "
                    "from " + cf.bold("{}"), cache_key)
                cli_logger.verbose_warning(
                    "If you experience issues with "
                    "the cloud provider, try re-running "
                    "the command with {}.", cf.bold("--no-config-cache"))

            return config_cache["config"]
        else:
            cli_logger.warning(
                "Found cached cluster config "
                "but the version " + cf.bold("{}") + " "
                "(expected " + cf.bold("{}") + ") does not match.\n"
                "This is normal if cluster launcher was updated.\n"
                "Config will be re-resolved.",
                config_cache.get("_version", "none"), CONFIG_CACHE_VERSION)

    importer = _NODE_PROVIDERS.get(config["provider"]["type"])
    if not importer:
        raise NotImplementedError("Unsupported provider {}".format(
            config["provider"]))

    provider_cls = importer(config["provider"])

    cli_logger.print("Checking {} environment settings",
                     _PROVIDER_PRETTY_NAMES.get(config["provider"]["type"]))
    try:
        config = provider_cls.fillout_available_node_types_resources(config)
        config = runtime_prepare_config(config.get("runtime"), config)
    except Exception as exc:
        if cli_logger.verbosity > 2:
            logger.exception("Failed to detect node resources.")
        else:
            cli_logger.warning(
                f"Failed to detect node resources: {str(exc)}. "
                "You can see full stack trace with higher verbosity.")

    try:
        # NOTE: if `resources` field is missing, validate_config for providers
        # other than AWS and Kubernetes will fail (the schema error will ask
        # the user to manually fill the resources) as we currently support
        # autofilling resources for AWS and Kubernetes only.
        validate_config(config)
    except (ModuleNotFoundError, ImportError):
        cli_logger.abort(
            "Not all dependencies were found. Please "
            "update your install command.")

    resolved_config = provider_cls.bootstrap_config(config)

    # add a verify step
    verify_config(resolved_config)

    if not no_config_cache or init_config_cache:
        with open(cache_key, "w") as f:
            config_cache = {
                "_version": CONFIG_CACHE_VERSION,
                "provider_log_info": try_get_log_state(
                    resolved_config["provider"]),
                "config": resolved_config
            }
            f.write(json.dumps(config_cache))
    return resolved_config


def _load_cluster_config(config_file: str,
                         override_cluster_name: Optional[str] = None,
                         need_bootstrap: bool = True,
                         no_config_cache: bool = False) -> Dict[str, Any]:
    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name
    if need_bootstrap:
        config = _bootstrap_config(config, no_config_cache=no_config_cache)
    return config


def teardown_cluster(config_file: str, yes: bool, workers_only: bool,
                     override_cluster_name: Optional[str],
                     keep_min_workers: bool,
                     proxy_stop: bool = False) -> None:
    """Destroys all nodes of a cluster described by a config json."""
    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name

    config = _bootstrap_config(config)

    cli_logger.confirm(yes, "Are you sure that you want to shut down cluster {}?",
                       config["cluster_name"], _abort=True)
    cli_logger.newline()
    with cli_logger.group("Shutting down cluster: {}", config["cluster_name"]):
        _teardown_cluster(config_file, config,
                          workers_only=workers_only,
                          override_cluster_name=override_cluster_name,
                          keep_min_workers=keep_min_workers,
                          proxy_stop=proxy_stop)

    cli_logger.success("Successfully shut down cluster: {}.", config["cluster_name"])


def _teardown_cluster(config_file: str, config: Dict[str, Any],
                      workers_only: bool,
                      override_cluster_name: Optional[str],
                      keep_min_workers: bool,
                      proxy_stop: bool = False) -> None:
    current_step = 1
    total_steps = NUM_TEARDOWN_CLUSTER_STEPS_BASE
    if proxy_stop:
        total_steps += 1
    if not workers_only:
        total_steps += 1

    if proxy_stop:
        with cli_logger.group(
                "Stopping proxy",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            _stop_proxy(config)

    if not workers_only:
        with cli_logger.group(
                "Requesting head to stop head services",
                _numbered=("[]", current_step, total_steps)):
            current_step += 1
            try:
                stop_node_from_head(config_file,
                                    node_ip=None, all_nodes=False,
                                    override_cluster_name=override_cluster_name,
                                    indent_level=1)
            except Exception as e:
                cli_logger.verbose_error("{}", str(e))
                cli_logger.warning(
                    "Exception occurred when stopping head services "
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
        cmd = "cloudtik head teardown"
        if keep_min_workers:
            cmd += " --keep-min-workers"

        try:
            exec_cmd_on_cluster(config_file,
                                cmd,
                                override_cluster_name)
        except Exception as e:
            cli_logger.verbose_error("{}", str(e))
            cli_logger.warning(
                "Exception occurred when requesting head to stop the workers "
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
        teardown_cluster_nodes(config, provider,
                               workers_only, keep_min_workers,
                               False)


def teardown_cluster_nodes(config: Dict[str, Any],
                           provider: NodeProvider,
                           workers_only: bool,
                           keep_min_workers: bool,
                           on_head: bool):
    use_internal_ip = True if on_head else False

    def remaining_nodes():
        workers = provider.non_terminated_nodes({
            CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER
        })

        if keep_min_workers:
            min_workers = config.get("min_workers", 0)
            cli_logger.print(
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
                cli_logger.print(
                    "The head node will not be shut down. " +
                    cf.dimmed("(due to {})"), cf.bold("--workers-only"))

            return head, workers

        return head, head + workers

    def run_docker_stop(node, container_name):
        try:
            updater = create_node_updater_for_exec(
                config=config,
                node_id=node,
                provider=provider,
                start_commands=[],
                is_head_node=False,
                use_internal_ip=use_internal_ip)

            _exec(
                updater,
                f"docker stop {container_name}",
                with_output=False,
                run_env="host")
        except Exception:
            cli_logger.warning(f"Docker stop failed on {node}")

    # Loop here to check that both the head and worker nodes are actually
    #   really gone
    head, A = remaining_nodes()

    docker_enabled = is_docker_enabled(config)
    if docker_enabled and (on_head or not workers_only):
        container_name = config.get("docker", {}).get("container_name")
        if on_head:
            cli_logger.print("Stopping docker containers on workers.")
            container_nodes = A
        else:
            cli_logger.print("Stopping docker container on head.")
            container_nodes = head
        # This is to ensure that the parallel SSH calls below do not mess with
        # the users terminal.
        output_redir = cmd_output_util.is_output_redirected()
        cmd_output_util.set_output_redirected(True)
        allow_interactive = cmd_output_util.does_allow_interactive()
        cmd_output_util.set_allow_interactive(False)

        with ThreadPoolExecutor(
                max_workers=MAX_PARALLEL_SHUTDOWN_WORKERS) as executor:
            for node in container_nodes:
                executor.submit(
                    run_docker_stop, node=node, container_name=container_name)
        cmd_output_util.set_output_redirected(output_redir)
        cmd_output_util.set_allow_interactive(allow_interactive)
    node_type = "workers" if workers_only else "nodes"
    with LogTimer("teardown_cluster: done."):
        while A:
            provider.terminate_nodes(A)

            cli_logger.print(
                "Requested {} {} to shut down.",
                cf.bold(len(A)), node_type,
                _tags=dict(interval="1s"))

            time.sleep(POLL_INTERVAL)  # todo: interval should be a variable
            head, A = remaining_nodes()
            cli_logger.print("{} {} remaining after {} second(s).",
                             cf.bold(len(A)), node_type, POLL_INTERVAL)
        cli_logger.success("No {} remaining.", node_type)


def kill_node_from_head(config_file: str, yes: bool, hard: bool,
                        override_cluster_name: Optional[str],
                        node_ip: str = None) -> Optional[str]:
    """Kills a specified or a random worker."""

    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name
    config = _bootstrap_config(config)

    if node_ip:
        cli_logger.confirm(yes, "Node {} will be killed.", node_ip, _abort=True)
    else:
        cli_logger.confirm(yes, "A random node will be killed.", _abort=True)

    if hard:
        return _kill_node(config, hard, node_ip)

    # soft kill, we need to do on head
    cmds = [
        "cloudtik",
        "head",
        "kill-node",
        "--yes",
    ]
    if node_ip:
        cmds += ["--node-ip={}".format(node_ip)]
    final_cmd = " ".join(cmds)
    exec_cmd_on_cluster(config_file, final_cmd,
                        override_cluster_name)
    return None


def kill_node_on_head(yes, hard, node_ip: str = None):
    # Since this is running on head, the bootstrap config must exist
    cluster_config_file = get_head_bootstrap_config()
    config = yaml.safe_load(open(cluster_config_file).read())

    if not yes:
        if node_ip:
            cli_logger.confirm(yes, "Node {} will be killed.", node_ip, _abort=True)
        else:
            cli_logger.confirm(yes, "A random node will be killed.", _abort=True)

    return _kill_node(config, hard, node_ip)


def _kill_node(config, hard, node_ip: str = None):
    provider = _get_node_provider(config["provider"], config["cluster_name"])

    if node_ip:
        node = provider.get_node_id(node_ip, use_internal_ip=True)
        if not node:
            cli_logger.error("No node with the specified node ip - {} found.", node_ip)
            return None
    else:
        nodes = provider.non_terminated_nodes({
            CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER
        })
        if not nodes:
            cli_logger.print("No worker nodes detected.")
            return None
        node = random.choice(nodes)
        node_ip = get_node_cluster_ip(provider, node)

    if not hard:
        # execute stop-node command
        stop_node_on_head(node_ip, False)

    # terminate the node
    cli_logger.print("Shutdown " + cf.bold("{}:{}"), node, node_ip)
    provider.terminate_node(node)
    time.sleep(POLL_INTERVAL)

    return node_ip


def monitor_cluster(cluster_config_file: str, num_lines: int,
                    override_cluster_name: Optional[str] = None,
                    file_type: str = None) -> None:
    """Tails the controller logs of a cluster."""
    cmd = f"tail -n {num_lines} -f /tmp/cloudtik/session_latest/logs/cloudtik_cluster_controller"
    if file_type and file_type != "":
        cmd += f".{file_type}"
    else:
        cmd += "*"

    exec_cluster(
        cluster_config_file,
        cmd=cmd,
        run_env="auto",
        screen=False,
        tmux=False,
        stop=False,
        start=False,
        override_cluster_name=override_cluster_name,
        port_forward=None)


def warn_about_bad_start_commands(start_commands: List[str],
                                 no_controller_on_head: bool = False) -> None:
    start_cmds = list(filter(lambda x: "cloudtik node-start" in x, start_commands))
    if len(start_cmds) == 0:
        cli_logger.warning(
            "CloudTik will not be started because `{}` is not in `{}`.",
            cf.bold("cloudtik node-start"), cf.bold("head_start_commands"))

    cluster_scaling_config_in_start_cmd = any(
        "cluster-scaling-config" in x for x in start_cmds)
    if not (cluster_scaling_config_in_start_cmd or no_controller_on_head):
        cli_logger.warning(
            "The head node will not launch any workers because "
            "`{}` does not have `{}` set.\n"
            "Potential fix: add `{}` to the `{}` command under `{}`.",
            cf.bold("cloudtik node-start"), cf.bold("--cluster-scaling-config"),
            cf.bold("--cluster-scaling-config=~/cloudtik_bootstrap_config.yaml"),
            cf.bold("cloudtik node-start"), cf.bold("head_start_commands"))


def get_or_create_head_node(config: Dict[str, Any],
                            printable_config_file: str,
                            no_restart: bool,
                            restart_only: bool,
                            yes: bool,
                            override_cluster_name: Optional[str],
                            no_controller_on_head: bool = False,
                            _provider: Optional[NodeProvider] = None,
                            _runner: ModuleType = subprocess) -> None:
    """Create the cluster head node, which in turn creates the workers."""
    global_event_system.execute_callback(
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
        cli_logger.confirm(
            yes,
            "No head node found. "
            "Launching a new cluster.",
            _abort=True)

    if head_node:
        if restart_only:
            cli_logger.confirm(
                yes,
                "Updating cluster configuration and "
                "restarting the cluster runtime. "
                "Setup commands will not be run due to `{}`.\n",
                cf.bold("--restart-only"),
                _abort=True)
        elif no_restart:
            cli_logger.print(
                "Cluster runtime will not be restarted due "
                "to `{}`.", cf.bold("--no-restart"))
            cli_logger.confirm(
                yes,
                "Updating cluster configuration and "
                "running setup commands.",
                _abort=True)
        else:
            cli_logger.print(
                "Updating cluster configuration and running full setup.")
            cli_logger.confirm(
                yes,
                cf.bold("Cluster runtime will be restarted."),
                _abort=True)

    cli_logger.newline()

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
        with cli_logger.group("Acquiring an up-to-date head node"):
            global_event_system.execute_callback(
                CreateClusterEvent.acquiring_new_head_node)
            if head_node is not None:
                cli_logger.confirm(
                    yes, "Relaunching the head node.", _abort=True)

                provider.terminate_node(head_node)
                cli_logger.print("Terminated head node {}", head_node)

            head_node_tags[CLOUDTIK_TAG_LAUNCH_CONFIG] = launch_hash
            head_node_tags[CLOUDTIK_TAG_NODE_NAME] = "cloudtik-{}-head".format(
                config["cluster_name"])
            head_node_tags[CLOUDTIK_TAG_NODE_STATUS] = STATUS_UNINITIALIZED
            provider.create_node(head_node_config, head_node_tags, 1)
            cli_logger.print("Launched a new head node")

            start = time.time()
            head_node = None
            with cli_logger.group("Fetching the new head node"):
                while True:
                    if time.time() - start > 50:
                        cli_logger.abort("Head node fetch timed out. "
                                         "Failed to create head node.")
                    nodes = provider.non_terminated_nodes(head_node_tags)
                    if len(nodes) == 1:
                        head_node = nodes[0]
                        break
                    time.sleep(POLL_INTERVAL)
            cli_logger.newline()

    global_event_system.execute_callback(CreateClusterEvent.head_node_acquired)

    with cli_logger.group(
            "Setting up head node",
            _numbered=("<>", 1, 1),
            # cf.bold(provider.node_tags(head_node)[CLOUDTIK_TAG_NODE_NAME]),
            _tags=dict()):  # add id, ARN to tags?

        # TODO(ekl) right now we always update the head node even if the
        # hash matches.
        # We could prompt the user for what they want to do here.
        # No need to pass in cluster_sync_files because we use this
        # hash to set up the head node
        (runtime_hash, file_mounts_contents_hash) = hash_runtime_conf(
            config["file_mounts"], None, config)

        if not no_controller_on_head:
            # Return remote_config_file to avoid prematurely closing it.
            config, remote_config_file = _set_up_config_for_head_node(
                config, provider, no_restart)
            cli_logger.print("Prepared bootstrap config")

        if restart_only:
            # Docker may re-launch nodes, requiring setup
            # commands to be rerun.
            if is_docker_enabled(config):
                setup_commands = config["head_setup_commands"]
            else:
                setup_commands = []
            start_commands = config["head_start_commands"]
        # If user passed in --no-restart and we're not creating a new head,
        # omit start commands.
        elif no_restart and not creating_new_head:
            setup_commands = config["head_setup_commands"]
            start_commands = []
        else:
            setup_commands = config["head_setup_commands"]
            start_commands = config["head_start_commands"]

        if not no_restart:
            warn_about_bad_start_commands(start_commands,
                                         no_controller_on_head)

        initialization_commands = config["initialization_commands"]
        updater = NodeUpdaterThread(
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
            docker_config=config.get("docker"),
            restart_only=restart_only,
            runtime_config=config.get("runtime"))
        updater.start()
        updater.join()

        # Refresh the node cache so we see the external ip if available
        provider.non_terminated_nodes(head_node_tags)

        if updater.exitcode != 0:
            # todo: this does not follow the mockup and is not good enough
            cli_logger.abort("Failed to setup head node.")
            sys.exit(1)

    global_event_system.execute_callback(
        CreateClusterEvent.cluster_booting_completed, {
            "head_node_id": head_node,
        })

    cluster_booting_completed(config, head_node)

    if not is_use_internal_ip(config):
        # start proxy and bind to localhost
        cli_logger.newline()
        with cli_logger.group("Starting SOCKS5 proxy..."):
            _start_proxy(printable_config_file, config,
                         True, "localhost")

    cli_logger.newline()
    successful_msg = "Successfully started cluster: {}.".format(config["cluster_name"])
    cli_logger.success("-" * len(successful_msg))
    cli_logger.success(successful_msg)
    cli_logger.success("-" * len(successful_msg))

    show_useful_commands(printable_config_file,
                         config,
                         provider,
                         head_node,
                         updater,
                         override_cluster_name)


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

    if "ssh_private_key" in config["auth"]:
        remote_key_path = "~/cloudtik_bootstrap_key.pem"
        remote_config["auth"]["ssh_private_key"] = remote_key_path

    # Adjust for new file locations
    new_mounts = {}
    for remote_path in config["file_mounts"]:
        new_mounts[remote_path] = remote_path
    remote_config["file_mounts"] = new_mounts
    remote_config["no_restart"] = no_restart

    remote_config = provider.prepare_for_head_node(remote_config)

    # Now inject the rewritten config and SSH key into the head node
    remote_config_file = tempfile.NamedTemporaryFile(
        "w", prefix="cloudtik-bootstrap-")
    remote_config_file.write(json.dumps(remote_config))
    remote_config_file.flush()
    config["file_mounts"].update({
        "~/cloudtik_bootstrap_config.yaml": remote_config_file.name
    })

    if "ssh_private_key" in config["auth"]:
        config["file_mounts"].update({
            remote_key_path: config["auth"]["ssh_private_key"],
        })

    return config, remote_config_file


def attach_cluster(config_file: str,
                   start: bool,
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
        start: whether to start the cluster if it isn't up
        use_screen: whether to use screen as multiplexer
        use_tmux: whether to use tmux as multiplexer
        override_cluster_name: set the name of the cluster
        new: whether to force a new screen
        port_forward ( (int,int) or list[(int,int)] ): port(s) to forward
    """

    cmd = get_attach_command(use_screen, use_tmux, new)
    run_env = "auto"
    if force_to_host:
        run_env = "host"

    exec_cluster(
        config_file,
        cmd=cmd,
        run_env=run_env,
        screen=False,
        tmux=False,
        stop=False,
        start=start,
        override_cluster_name=override_cluster_name,
        no_config_cache=no_config_cache,
        port_forward=port_forward,
        _allow_uninitialized_state=True)


def exec_cluster(config_file: str,
                 *,
                 cmd: str = None,
                 run_env: str = "auto",
                 screen: bool = False,
                 tmux: bool = False,
                 stop: bool = False,
                 start: bool = False,
                 override_cluster_name: Optional[str] = None,
                 no_config_cache: bool = False,
                 port_forward: Optional[Port_forward] = None,
                 with_output: bool = False,
                 _allow_uninitialized_state: bool = False) -> str:
    """Runs a command on the specified cluster.

    Arguments:
        config_file: path to the cluster yaml
        cmd: command to run
        run_env: whether to run the command on the host or in a container.
            Select between "auto", "host" and "docker"
        screen: whether to run in a screen
        tmux: whether to run in a tmux session
        stop: whether to stop the cluster after command run
        start: whether to start the cluster if it isn't up
        override_cluster_name: set the name of the cluster
        port_forward ( (int, int) or list[(int, int)] ): port(s) to forward
        _allow_uninitialized_state: whether to execute on an uninitialized head
            node.
    """
    assert not (screen and tmux), "Can specify only one of `screen` or `tmux`."
    assert run_env in RUN_ENV_TYPES, "--run_env must be in {}".format(
        RUN_ENV_TYPES)
    # TODO(rliaw): We default this to True to maintain backwards-compat.
    # In the future we would want to support disabling login-shells
    # and interactivity.
    cmd_output_util.set_allow_interactive(True)

    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name
    config = _bootstrap_config(config, no_config_cache=no_config_cache)
    use_internal_ip = config.get("bootstrapped", False)

    head_node = _get_running_head_node(
        config,
        config_file,
        override_cluster_name,
        create_if_needed=start,
        _allow_uninitialized_state=_allow_uninitialized_state)

    provider = _get_node_provider(config["provider"], config["cluster_name"])
    updater = create_node_updater_for_exec(
        config=config,
        node_id=head_node,
        provider=provider,
        start_commands=[],
        is_head_node=True,
        use_internal_ip=use_internal_ip)
    shutdown_after_run = False
    if cmd and stop:
        cmd = "; ".join([
            cmd, "cloudtik head stop-node",
            "cloudtik down ~/cloudtik_bootstrap_config.yaml --yes --workers-only"
        ])
        shutdown_after_run = True

    result = _exec(
        updater,
        cmd,
        screen,
        tmux,
        port_forward=port_forward,
        with_output=with_output,
        run_env=run_env,
        shutdown_after_run=shutdown_after_run)
    if tmux or screen:
        attach_command_parts = ["cloudtik attach", config_file]
        if override_cluster_name is not None:
            attach_command_parts.append(
                "--cluster-name={}".format(override_cluster_name))
        if tmux:
            attach_command_parts.append("--tmux")
        elif screen:
            attach_command_parts.append("--screen")

        attach_command = " ".join(attach_command_parts)
        cli_logger.print("Run `{}` to check command status.",
                         cf.bold(attach_command))
    return result


def _exec(updater: NodeUpdaterThread,
          cmd: Optional[str] = None,
          screen: bool = False,
          tmux: bool = False,
          port_forward: Optional[Port_forward] = None,
          with_output: bool = False,
          run_env: str = "auto",
          shutdown_after_run: bool = False) -> str:
    if cmd:
        if screen:
            wrapped_cmd = [
                "screen", "-L", "-dm", "bash", "-c",
                quote(cmd + "; exec bash")
            ]
            cmd = " ".join(wrapped_cmd)
        elif tmux:
            # TODO: Consider providing named session functionality
            wrapped_cmd = [
                "tmux", "new", "-d", "bash", "-c",
                quote(cmd + "; exec bash")
            ]
            cmd = " ".join(wrapped_cmd)
    return updater.cmd_executor.run(
        cmd,
        exit_on_fail=True,
        port_forward=port_forward,
        with_output=with_output,
        run_env=run_env,
        shutdown_after_run=shutdown_after_run)


def rsync(config_file: str,
          source: Optional[str],
          target: Optional[str],
          override_cluster_name: Optional[str],
          down: bool,
          ip_address: Optional[str] = None,
          all_nodes: bool = False,
          use_internal_ip: bool = False,
          no_config_cache: bool = False,
          should_bootstrap: bool = True,
          _runner: ModuleType = subprocess) -> None:
    """Rsyncs files.

    Arguments:
        config_file: path to the cluster yaml
        source: source dir
        target: target dir
        override_cluster_name: set the name of the cluster
        down: whether we're syncing remote -> local
        ip_address (str): Address of node. Raise Exception
            if both ip_address and 'all_nodes' are provided.
        all_nodes (bool): Whether the rsync up the files to all nodes
        use_internal_ip (bool): Whether the provided ip_address is
            public or private.
        should_bootstrap: whether to bootstrap cluster config before syncing
    """
    if bool(source) != bool(target):
        cli_logger.abort(
            "Expected either both a source and a target, or neither.")

    if ip_address and all_nodes:
        cli_logger.abort("Cannot provide both ip_address and 'all_nodes'.")

    assert bool(source) == bool(target), (
        "Must either provide both or neither source and target.")

    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name
    if should_bootstrap:
        config = _bootstrap_config(config, no_config_cache=no_config_cache)

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
                cmd_output_util.set_output_redirected(False)
                set_rsync_silent(False)
            rsync(source, target, is_file_mount)
        else:
            updater.sync_file_mounts(rsync)

    head_node = _get_running_head_node(
        config, config_file, override_cluster_name, create_if_needed=False)
    if not ip_address:
        rsync_to_node(head_node, source, target, is_head_node=True)
        if not down and all_nodes:
            rsync_to_node_from_head(config_file, override_cluster_name,
                                    target, target, False,
                                    None, all_nodes)
    else:
        # for the cases that specified sync up or down with specific node
        # both source and target must be specified
        if not source or not target:
            cli_logger.abort("Need to specify both source and target when rsync with specific node")

        target_base = os.path.basename(target)
        target_on_head = tempfile.mktemp(prefix=f"{target_base}_")
        if down:
            # first run rsync on head
            rsync_to_node_from_head(config_file, override_cluster_name,
                                    source, target_on_head, True,
                                    ip_address)
            rsync_to_node(head_node, target_on_head, target, is_head_node=True)
        else:
            # First rsync with head
            rsync_to_node(head_node, source, target_on_head, is_head_node=True)
            rsync_to_node_from_head(config_file, override_cluster_name,
                                    target_on_head, target, False,
                                    ip_address)


def rsync_to_node_from_head(cluster_config_file: str,
                            override_cluster_name: Optional[str],
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
    if all_workers:
        cmds += ["--all-workers"]

    final_cmd = " ".join(cmds)
    exec_cmd_on_cluster(cluster_config_file, final_cmd, override_cluster_name)


def rsync_node_on_head(source: str,
                       target: str,
                       down: bool,
                       node_ip: str = None,
                       all_workers: bool = False):
    # Since this is running on head, the bootstrap config must exist
    cluster_config_file = get_head_bootstrap_config()
    config = yaml.safe_load(open(cluster_config_file).read())
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
            # print rsync progress for single file rsync
            if cli_logger.verbosity > 0:
                cmd_output_util.set_output_redirected(False)
                set_rsync_silent(False)
            rsync(source, target, is_file_mount)
        else:
            updater.sync_file_mounts(rsync)

    nodes = []
    if node_ip:
        nodes = [provider.get_node_id(node_ip, use_internal_ip=True)]
    else:
        # either node_ip or all_workers be set
        if all_workers:
            nodes.extend(_get_worker_nodes(config, None))

    for node_id in nodes:
        rsync_to_node(node_id, source, target)


def get_worker_cpus(config, provider):
    workers = _get_worker_nodes(config, None)
    workers_info = get_nodes_info(provider, workers, True, config["available_node_types"])
    return sum_worker_cpus(workers_info)


def get_worker_memory(config, provider):
    workers = _get_worker_nodes(config, None)
    workers_info = get_nodes_info(provider, workers, True, config["available_node_types"])
    return sum_worker_memory(workers_info)


def get_head_node_ip(config_file: str,
                     override_cluster_name: Optional[str] = None) -> str:
    """Returns head node IP for given configuration file if exists."""

    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name

    return _get_head_node_ip(config, config_file, override_cluster_name)


def get_worker_node_ips(config_file: str,
                        override_cluster_name: Optional[str] = None
                        ) -> List[str]:
    """Returns worker node IPs for given configuration file."""

    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name

    return _get_worker_node_ips(config)


def _get_head_node_ip(config: Dict[str, Any],
                      config_file: str,
                      override_cluster_name: Optional[str] = None) -> str:
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    head_node = _get_running_head_node(config, config_file,
                                       override_cluster_name)
    return get_head_working_ip(config, provider, head_node)


def _get_worker_node_ips(config: Dict[str, Any]) -> List[str]:
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    nodes = provider.non_terminated_nodes({
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER
    })
    return [get_node_cluster_ip(provider, node) for node in nodes]


def _get_worker_nodes(config: Dict[str, Any],
                      override_cluster_name: Optional[str]) -> List[str]:
    """Returns worker node ids for given configuration."""
    # todo: technically could be reused in get_worker_node_ips
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name

    provider = _get_node_provider(config["provider"], config["cluster_name"])
    return provider.non_terminated_nodes({CLOUDTIK_TAG_NODE_KIND: NODE_KIND_WORKER})


def _get_running_head_node(
        config: Dict[str, Any],
        printable_config_file: str,
        override_cluster_name: Optional[str],
        create_if_needed: bool = False,
        _provider: Optional[NodeProvider] = None,
        _allow_uninitialized_state: bool = False,
) -> str:
    """Get a valid, running head node.
    Args:
        config (Dict[str, Any]): Cluster Config dictionary
        printable_config_file (str): Used for printing formatted CLI commands.
        override_cluster_name (str): Passed to `get_or_create_head_node` to
            override the cluster name present in `config`.
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
        get_or_create_head_node(
            config,
            printable_config_file=printable_config_file,
            restart_only=False,
            no_restart=False,
            yes=True,
            override_cluster_name=override_cluster_name)
        # NOTE: `_allow_uninitialized_state` is forced to False if
        # `create_if_needed` is set to True. This is to ensure that the
        # commands executed after creation occur on an actually running
        # cluster.
        return _get_running_head_node(
            config,
            printable_config_file,
            override_cluster_name,
            create_if_needed=False,
            _allow_uninitialized_state=False)
    else:
        if _allow_uninitialized_state and _backup_head_node is not None:
            cli_logger.warning(
                f"The head node being returned: {_backup_head_node} is not "
                "`up-to-date`. If you are not debugging a startup issue "
                "it is recommended to restart this head node with: {}",
                cf.bold(f"  cloudtik down  {printable_config_file}"))

            return _backup_head_node
        raise RuntimeError("Head node of cluster {} not found!".format(
            config["cluster_name"]))


def get_local_dump_archive(stream: bool = False,
                           output: Optional[str] = None,
                           logs: bool = True,
                           debug_state: bool = True,
                           pip: bool = True,
                           processes: bool = True,
                           processes_verbose: bool = False,
                           tempfile: Optional[str] = None) -> Optional[str]:
    if stream and output:
        raise ValueError(
            "You can only use either `--output` or `--stream`, but not both.")

    parameters = GetParameters(
        logs=logs,
        debug_state=debug_state,
        pip=pip,
        processes=processes,
        processes_verbose=processes_verbose)

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
    cli_logger.print(f"Created local data archive at {target}")

    return target


def get_cluster_dump_archive_on_head(
                             host: Optional[str] = None,
                             stream: bool = False,
                             output: Optional[str] = None,
                             logs: bool = True,
                             debug_state: bool = True,
                             pip: bool = True,
                             processes: bool = True,
                             processes_verbose: bool = False,
                             tempfile: Optional[str] = None) -> Optional[str]:
    if stream and output:
        raise ValueError(
            "You can only use either `--output` or `--stream`, but not both.")

    # Parse arguments (e.g. fetch info from cluster config)
    cluster_config_file, head_node_ip, workers, ssh_user, ssh_key, docker, cluster_name = \
        _info_from_params(None, host, None, None, None, False)

    nodes = [
        Node(
            host=h,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            docker_container=docker) for h in workers
    ]

    if not nodes:
        cli_logger.error(
            "No nodes found. Specify with `--host` or by passing a "
            "cluster config to `--cluster`.")
        return None

    parameters = GetParameters(
        logs=logs,
        debug_state=debug_state,
        pip=pip,
        processes=processes,
        processes_verbose=processes_verbose)

    with Archive(file=tempfile) as archive:
        create_archive_for_remote_nodes(
            archive, remote_nodes=nodes, parameters=parameters)

    tmp = archive.file

    if stream:
        with open(tmp, "rb") as fp:
            os.write(1, fp.read())
        os.remove(tmp)
        return None

    target = output or os.path.join(os.getcwd(), os.path.basename(tmp))
    shutil.move(tmp, target)
    cli_logger.print(f"Created local data archive at {target}")

    return target


def get_cluster_dump_archive(cluster_config_file: Optional[str] = None,
                             host: Optional[str] = None,
                             ssh_user: Optional[str] = None,
                             ssh_key: Optional[str] = None,
                             docker: Optional[str] = None,
                             head_only: Optional[str] = None,
                             output: Optional[str] = None,
                             logs: bool = True,
                             debug_state: bool = True,
                             pip: bool = True,
                             processes: bool = True,
                             processes_verbose: bool = False,
                             tempfile: Optional[str] = None) -> Optional[str]:
    # Inform the user what kind of logs are collected (before actually
    # collecting, so they can abort)
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

    cli_logger.warning(
        "You are about to create a cluster dump. This will collect data from "
        "cluster nodes.\n\n"
        "The dump will contain this information:\n\n"
        f"{content_str}\n"
        f"If you are concerned about leaking private information, extract "
        f"the archive and inspect its contents before sharing it with "
        f"anyone.")

    # Parse arguments (e.g. fetch info from cluster config)
    cluster_config_file, head_node_ip, workers, ssh_user, ssh_key, docker, cluster_name = \
        _info_from_params(cluster_config_file, host, ssh_user, ssh_key, docker, True)

    if not head_node_ip:
        cli_logger.error(
            "No head node found. Cluster may not be running. ")
        return None

    head_node = Node(
            host=head_node_ip,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            docker_container=docker,
            is_head=True)

    parameters = GetParameters(
        logs=logs,
        debug_state=debug_state,
        pip=pip,
        processes=processes,
        processes_verbose=processes_verbose)

    with Archive(file=tempfile) as archive:
        create_archive_for_cluster_nodes(
            archive, head_node=head_node, parameters=parameters, head_only=head_only)

    if not output:
        if cluster_name:
            filename = f"{cluster_name}_" \
                       f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.tar.gz"
        else:
            filename = f"collected_logs_" \
                       f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.tar.gz"
        output = os.path.join(os.getcwd(), filename)
    else:
        output = os.path.expanduser(output)

    shutil.move(archive.file, output)
    return output


def show_worker_cpus(config_file: str,
                     override_cluster_name: Optional[str] = None) -> None:
    config = _load_cluster_config(config_file, override_cluster_name)
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    worker_cpus = get_worker_cpus(config, provider)
    cli_logger.print(cf.bold(worker_cpus))


def show_worker_memory(config_file: str,
                       override_cluster_name: Optional[str] = None) -> None:
    config = _load_cluster_config(config_file, override_cluster_name)
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    memory_in_gb = int(get_worker_memory(config, provider))
    cli_logger.print(cf.bold("{}GB"), memory_in_gb)


def show_cluster_info(config_file: str,
                      override_cluster_name: Optional[str] = None) -> None:
    """Shows the cluster information for given configuration file."""
    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name

    config = _bootstrap_config(config, no_config_cache=False)
    provider = _get_node_provider(config["provider"], config["cluster_name"])

    head_node = None
    # Check whether the head node is running
    try:
        head_node = _get_running_head_node(config, config_file,
                                           override_cluster_name)
        head_node_ip = get_head_working_ip(config, provider, head_node)
    except Exception:
        head_node_ip = None

    if head_node_ip is None:
        cli_logger.print(cf.bold("Cluster {} is not running."), config["cluster_name"])
        return

    # Check the running worker nodes
    head_count = 1
    workers = _get_worker_nodes(config, None)
    worker_count = len(workers)

    cli_logger.print(cf.bold("Cluster {}:"), config["cluster_name"])
    cli_logger.print(cf.bold("{} head and {} worker(s) are running"),
                     head_count, worker_count)

    cli_logger.newline()
    cli_logger.print(cf.bold("Runtimes: {}"), ", ".join(get_enabled_runtimes(config)))

    workers_info = get_nodes_info(provider, workers,
                                  True, config["available_node_types"])
    worker_cpus = sum_worker_cpus(workers_info)
    worker_memory = sum_worker_memory(workers_info)
    cli_logger.newline()
    cli_logger.print(cf.bold("The total worker CPUs: {}."), worker_cpus)
    cli_logger.print(cf.bold("The total worker memory: {}GB."), worker_memory)

    if head_node is None:
        return

    updater = create_node_updater_for_exec(
        config=config,
        node_id=head_node,
        provider=provider,
        start_commands=[],
        is_head_node=False,
        use_internal_ip=False)

    show_useful_commands(config_file,
                         config,
                         provider,
                         head_node,
                         updater,
                         override_cluster_name)


def show_useful_commands(printable_config_file: str,
                         config: Dict[str, Any],
                         provider: NodeProvider,
                         head_node: str,
                         updater: NodeUpdaterThread,
                         override_cluster_name: Optional[str] = None
                         ) -> None:
    if override_cluster_name:
        modifiers = " --cluster-name={}".format(quote(override_cluster_name))
        cluster_name = override_cluster_name
    else:
        modifiers = ""
        cluster_name = config["cluster_name"]

    cli_logger.newline()
    with cli_logger.group("Key information:"):
        private_key_file = config["auth"].get("ssh_private_key", "")
        cli_logger.print("Cluster private key file: {}", private_key_file)
        cli_logger.print("Please keep the cluster private key file safe.")

    cli_logger.newline()
    with cli_logger.group("Useful commands:"):
        printable_config_file = os.path.abspath(printable_config_file)

        with cli_logger.group("Check cluster status with:"):
            cli_logger.print(
                cf.bold("cloudtik status {}{}"), printable_config_file, modifiers)

        with cli_logger.group("Execute command on cluster with:"):
            cli_logger.print(
                cf.bold("cloudtik exec {}{} [command]"), printable_config_file, modifiers)

        with cli_logger.group("Connect to a terminal on the cluster head:"):
            cli_logger.print(
                cf.bold("cloudtik attach {}{}"), printable_config_file, modifiers)

        with cli_logger.group("Upload files or folders to cluster:"):
            cli_logger.print(
                cf.bold("cloudtik rsync-up {}{} [source] [target]"), printable_config_file, modifiers)

        with cli_logger.group("Download files or folders from cluster:"):
            cli_logger.print(
                cf.bold("cloudtik rsync-down {}{} [source] [target]"), printable_config_file, modifiers)

        with cli_logger.group("Submit job to cluster to run with:"):
            cli_logger.print(
                cf.bold("cloudtik submit {}{} [job-file.(py|sh|scala)] "), printable_config_file, modifiers)

        with cli_logger.group("Monitor cluster with:"):
            cli_logger.print(
                cf.bold("cloudtik monitor {}{}"), printable_config_file, modifiers)

    cli_logger.newline()
    with cli_logger.group("Useful addresses:"):
        proxy_info_file = get_proxy_info_file(cluster_name)
        pid, address, port = get_safe_proxy_process_info(proxy_info_file)
        if pid is not None:
            bind_address_show = get_proxy_bind_address_to_show(address)
            with cli_logger.group("The SOCKS5 proxy to access the cluster Web UI from local browsers:"):
                cli_logger.print(
                    cf.bold("{}:{}"),
                    bind_address_show, port)

        head_node_cluster_ip = get_node_cluster_ip(provider, head_node)

        runtime_urls = get_useful_runtime_urls(config.get("runtime"), head_node_cluster_ip)
        for runtime_url in runtime_urls:
            with cli_logger.group(runtime_url["name"] + ":"):
                cli_logger.print(runtime_url["url"])


def show_cluster_status(config_file: str,
                        override_cluster_name: Optional[str] = None
                        ) -> None:
    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name
    config = _bootstrap_config(config, no_config_cache=False)

    provider = _get_node_provider(config["provider"], config["cluster_name"])
    nodes = provider.non_terminated_nodes({})
    nodes_info = get_nodes_info(provider, nodes)

    # sort nodes info based on node type and then node ip for workers
    def node_info_sort(node_info):
        node_ip = node_info["private_ip"]
        if node_ip is None:
            node_ip = ""

        return node_info["cloudtik-node-kind"] + node_ip

    nodes_info.sort(key=node_info_sort)

    tb = pt.PrettyTable()
    tb.field_names = ["node-id", "node-ip", "node-type", "node-status", "instance-type",
                      "public-ip", "instance-status"]
    for node_info in nodes_info:
        tb.add_row([node_info["node_id"], node_info["private_ip"], node_info["cloudtik-node-kind"],
                    node_info["cloudtik-node-status"], node_info["instance_type"], node_info["public_ip"],
                    node_info["instance_status"]
                    ])

    def get_nodes_ready(node_info_list):
        nodes_ready = 0
        for node_info in node_info_list:
            if STATUS_UP_TO_DATE == node_info["cloudtik-node-status"]:
                nodes_ready += 1
        return nodes_ready

    nodes_ready = get_nodes_ready(nodes_info)
    cli_logger.print(cf.bold("Total {} nodes. {} nodes are ready"), len(nodes_info), nodes_ready)
    cli_logger.print(tb)


def confirm(msg: str, yes: bool) -> Optional[bool]:
    return None if yes else click.confirm(msg, abort=True)


def start_proxy(config_file: str,
                override_cluster_name: Optional[str] = None,
                no_config_cache: bool = False,
                bind_address: str = None):
    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name
    config = _bootstrap_config(config, no_config_cache=no_config_cache)

    if is_use_internal_ip(config):
        cli_logger.print(cf.bold(
            "SOCKS5 proxy is not needed. With use_internal_ips is True, you can access the cluster directly."),)
        return

    # Warning about bind_address
    if bind_address is None or bind_address == "":
        cli_logger.warning("The SOCKS5 proxy will be bound on localhost of this node. "
                           "Use --bind-address to specify to bind on a specific address if you want.")

    _start_proxy(config_file, config,
                 restart=True, bind_address=bind_address)


def _start_proxy(config_file: str, config: Dict[str, Any],
                 restart: bool = False,
                 bind_address: str = None):
    cluster_name = config["cluster_name"]
    proxy_info_file = get_proxy_info_file(cluster_name)
    pid, address, port = get_safe_proxy_process_info(proxy_info_file)
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
        head_node = _get_running_head_node(config, config_file, cluster_name)
        head_node_ip = get_head_working_ip(config, provider, head_node)
    except Exception:
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
    proxy_info_file = get_proxy_info_file(config["cluster_name"])

    auth_config = config["auth"]
    ssh_proxy_command = auth_config.get("ssh_proxy_command", None)
    ssh_private_key = auth_config.get("ssh_private_key", None)
    ssh_user = auth_config["ssh_user"]
    proxy_port = get_free_port()
    cmd = "ssh -o \'StrictHostKeyChecking no\'"
    if ssh_private_key:
        cmd += " -i {}".format(ssh_private_key)
    if ssh_proxy_command:
        cmd += " -o ProxyCommand=\'{}\'".format(ssh_proxy_command)

    if bind_address is None or bind_address == "":
        bind_string = "{}".format(proxy_port)
    else:
        bind_string = "{}:{}".format(bind_address, proxy_port)

    cmd += " -D {} -C -N {}@{}".format(bind_string, ssh_user, head_node_ip)

    cli_logger.verbose("Running `{}`", cf.bold(cmd))
    p = subprocess.Popen(cmd, shell=True)
    if os.path.exists(proxy_info_file):
        process_info = json.loads(open(proxy_info_file).read())
    else:
        process_info = {}
    process_info["proxy"] = {"pid": p.pid, "bind_address": bind_address, "port": proxy_port}
    with open(proxy_info_file, "w") as f:
        f.write(json.dumps(process_info))
    return p.pid, bind_address, proxy_port


def stop_proxy(config_file: str,
               override_cluster_name: Optional[str] = None):
    config = yaml.safe_load(open(config_file).read())

    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name

    _stop_proxy(config)


def _stop_proxy(config: Dict[str, Any]):
    cluster_name = config["cluster_name"]

    proxy_info_file = get_proxy_info_file(cluster_name)
    pid, address, port = get_safe_proxy_process_info(proxy_info_file)
    if pid is None:
        cli_logger.print(cf.bold("The SOCKS5 proxy of cluster {} was not started."), cluster_name)
        return

    kill_process_tree(pid)
    with open(proxy_info_file, "w") as f:
        f.write(json.dumps({"proxy": {}}))
    cli_logger.print(cf.bold("Successfully stopped the SOCKS5 proxy of cluster {}."), cluster_name)


def exec_cmd_on_cluster(cluster_config_file: str,
                        cmd: str,
                        override_cluster_name: Optional[str],
                        no_config_cache: bool = False):
    exec_cluster(
        cluster_config_file,
        cmd=cmd,
        run_env="auto",
        screen=False,
        tmux=False,
        stop=False,
        start=False,
        override_cluster_name=override_cluster_name,
        no_config_cache=no_config_cache,
        port_forward=None,
        with_output=False,
        _allow_uninitialized_state=False)


def cluster_debug_status(cluster_config_file: str,
                         override_cluster_name: Optional[str]) -> None:
    """Return the debug status of a cluster scaling from head node"""

    cmd = f"cloudtik head debug-status"
    exec_cmd_on_cluster(cluster_config_file, cmd, override_cluster_name)


def cluster_health_check(cluster_config_file: str,
                         override_cluster_name: Optional[str]) -> None:
    """Do a health check on head node and return the results"""

    cmd = f"cloudtik head health-check"
    exec_cmd_on_cluster(cluster_config_file, cmd, override_cluster_name)


def teardown_cluster_on_head(keep_min_workers: bool) -> None:
    # Since this is running on head, the bootstrap config must exist
    cluster_config_file = get_head_bootstrap_config()
    config = yaml.safe_load(open(cluster_config_file).read())
    provider = _get_node_provider(config["provider"], config["cluster_name"])

    teardown_cluster_nodes(config, provider,
                           True, keep_min_workers,
                           True)


def cluster_process_status_on_head(redis_address):
    control_state = ControlState()
    control_state.initialize_control_state(redis_address, CLOUDTIK_DEFAULT_PORT,
                                           CLOUDTIK_REDIS_DEFAULT_PASSWORD)
    node_table = control_state.get_node_table()

    tb = pt.PrettyTable()
    tb.field_names = ["node-ip", "node-type", "n-controller", "n-manager", "l-monitor",
                      "c-controller", "r-manager", "r-server"]
    all_nodes = node_table.get_all().values()
    nodes_info = []
    for value in all_nodes:
        node_info = eval(value)
        if not is_alive_time(node_info.get("last_heartbeat_time", 0)):
            continue
        nodes_info.append(node_info)

    # sort nodes info based on node type and then node ip for workers
    def node_info_sort(node_info):
        return node_info["node_type"] + node_info["resource"]["ip"]
    nodes_info.sort(key=node_info_sort)

    for node_info in nodes_info:
        process_info = node_info["process"]
        tb.add_row([node_info["resource"]["ip"], node_info["node_type"],
                    process_info["NodeController"], process_info["NodeManager"], process_info["LogMonitor"],
                    process_info["ClusterController"], process_info["ResourceManager"], process_info["RedisServer"]
                    ])
    cli_logger.print("Total {} live nodes reported.", len(nodes_info))
    cli_logger.print(tb)


def cluster_process_status(cluster_config_file: str,
                           override_cluster_name: Optional[str]) -> None:
    """Do a health check on head node and return the results"""

    cmd = f"cloudtik head process-status"
    exec_cmd_on_cluster(cluster_config_file, cmd, override_cluster_name)


def exec_node_from_head(config_file: str,
                        node_ip: str,
                        all_nodes: bool = False,
                        cmd: str = None,
                        run_env: str = "auto",
                        screen: bool = False,
                        tmux: bool = False,
                        override_cluster_name: Optional[str] = None,
                        no_config_cache: bool = False,
                        port_forward: Optional[Port_forward] = None,
                        parallel: bool = True) -> None:
    """Attaches to a screen for the specified cluster.

    Arguments:
        config_file: path to the cluster yaml
        node_ip: the node internal IP to operate
        all_nodes: Run the operation on all the nodes
        cmd: command to run
        run_env: whether to run the command on the host or in a container.
            Select between "auto", "host" and "docker"
        screen: whether to use screen as multiplexer
        tmux: whether to use tmux as multiplexer
        override_cluster_name: set the name of the cluster
        port_forward ( (int,int) or list[(int,int)] ): port(s) to forward
        parallel: Whether to run on nodes in parallel
    """

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
    if run_env:
        cmds += ["--run-env={}".format(run_env)]
    if screen:
        cmds += ["--screen"]
    if tmux:
        cmds += ["--tmux"]
    if parallel:
        cmds += ["--parallel"]
    else:
        cmds += ["--no-parallel"]

    # TODO (haifeng): handle port forward for two state cases
    final_cmd = " ".join(cmds)

    exec_cluster(
        config_file,
        cmd=final_cmd,
        run_env="auto",
        screen=False,
        tmux=False,
        stop=False,
        start=False,
        override_cluster_name=override_cluster_name,
        no_config_cache=no_config_cache,
        port_forward=port_forward,
        _allow_uninitialized_state=False)


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
        new: whether to force a new screen
        port_forward ( (int,int) or list[(int,int)] ): port(s) to forward
        force_to_host: Whether attach to host even running with docker
    """

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
    final_cmd = " ".join(cmds)

    exec_cluster(
        config_file,
        cmd=final_cmd,
        run_env="auto",
        screen=False,
        tmux=False,
        stop=False,
        start=False,
        override_cluster_name=override_cluster_name,
        no_config_cache=no_config_cache,
        port_forward=port_forward,
        _allow_uninitialized_state=False)


def exec_cmd_on_head(config,
                     provider,
                     node_id: str,
                     cmd: str = None,
                     run_env: str = "auto",
                     screen: bool = False,
                     tmux: bool = False,
                     port_forward: Optional[Port_forward] = None,
                     with_output: bool = False) -> str:
    """Runs a command on the specified node from head."""

    assert not (screen and tmux), "Can specify only one of `screen` or `tmux`."
    assert run_env in RUN_ENV_TYPES, "--run_env must be in {}".format(
        RUN_ENV_TYPES)

    # TODO(rliaw): We default this to True to maintain backwards-compat.
    # In the future we would want to support disabling login-shells
    # and interactivity.
    cmd_output_util.set_allow_interactive(True)

    updater = create_node_updater_for_exec(
        config=config,
        node_id=node_id,
        provider=provider,
        start_commands=[],
        is_head_node=False,
        use_internal_ip=True)

    result = _exec(
        updater,
        cmd,
        screen,
        tmux,
        port_forward=port_forward,
        with_output=with_output,
        run_env=run_env,
        shutdown_after_run=False)

    return result


def attach_node_on_head(node_ip: str,
                        use_screen: bool,
                        use_tmux: bool,
                        new: bool = False,
                        port_forward: Optional[Port_forward] = None,
                        force_to_host: bool = False):
    cluster_config_file = get_head_bootstrap_config()
    config = yaml.safe_load(open(cluster_config_file).read())
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
        provider,
        node_id=node_id,
        cmd=cmd,
        run_env=run_env,
        screen=False,
        tmux=False,
        port_forward=port_forward)


def exec_node_on_head(
                     node_ip: str,
                     all_nodes: bool = False,
                     cmd: str = None,
                     run_env: str = "auto",
                     screen: bool = False,
                     tmux: bool = False,
                     port_forward: Optional[Port_forward] = None,
                     parallel: bool = True):
    cluster_config_file = get_head_bootstrap_config()
    config = yaml.safe_load(open(cluster_config_file).read())
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    head_node = _get_running_head_node(config, cluster_config_file,
                                       None, _provider=provider)

    nodes = get_nodes_of(config, provider, head_node,
                         node_ip, all_nodes)

    def run_exec_cmd_on_head(node_id):
        exec_cmd_on_head(
            config, provider,
            node_id=node_id, cmd=cmd,
            run_env=run_env,
            screen=screen, tmux=tmux,
            port_forward=port_forward)

    if parallel and len(nodes) > 1:
        run_in_paralell_on_nodes(run_exec_cmd_on_head, nodes)
    else:
        for node_id in nodes:
            run_exec_cmd_on_head(node_id=node_id)


def create_node_updater_for_exec(config,
                                 node_id,
                                 provider,
                                 start_commands,
                                 is_head_node: bool = False,
                                 use_internal_ip: bool = False,
                                 process_runner: ModuleType = subprocess):
    updater = NodeUpdaterThread(
        node_id=node_id,
        provider_config=config["provider"],
        provider=provider,
        auth_config=config["auth"],
        cluster_name=config["cluster_name"],
        file_mounts=config["file_mounts"],
        initialization_commands=[],
        setup_commands=[],
        start_commands=start_commands,
        runtime_hash="",
        file_mounts_contents_hash="",
        is_head_node=is_head_node,
        process_runner=process_runner,
        use_internal_ip=use_internal_ip,
        rsync_options={
            "rsync_exclude": config.get("rsync_exclude"),
            "rsync_filter": config.get("rsync_filter")
        },
        docker_config=config.get("docker"),
        runtime_config=config.get("runtime"))
    return updater


def start_node_on_head(node_ip: str = None,
                       all_nodes: bool = False,
                       parallel: bool = True):
    # Since this is running on head, the bootstrap config must exist
    cluster_config_file = get_head_bootstrap_config()
    config = yaml.safe_load(open(cluster_config_file).read())
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    head_node = _get_running_head_node(config, cluster_config_file,
                                       None, _provider=provider)
    head_node_ip = provider.internal_ip(head_node)
    runtime_envs = with_runtime_environment_variables(
        config.get("runtime"), provider)

    nodes = get_nodes_of(config, provider, head_node,
                         node_ip, all_nodes)

    def start_single_node_on_head(node_id):
        is_head_node = False
        if node_id == head_node:
            is_head_node = True

        if is_head_node:
            start_commands = config["head_start_commands"]
            node_runtime_envs = with_node_ip_environment_variables(head_node_ip, provider, node_id)
        else:
            start_commands = with_head_node_ip(config["worker_start_commands"], head_node_ip)
            node_runtime_envs = with_node_ip_environment_variables(None, provider, node_id)

        updater = create_node_updater_for_exec(
            config=config,
            node_id=node_id,
            provider=provider,
            start_commands=start_commands,
            is_head_node=is_head_node,
            use_internal_ip=True)

        node_runtime_envs.update(runtime_envs)
        updater._exec_start_commands(node_runtime_envs)

    if parallel and len(nodes) > 1:
        run_in_paralell_on_nodes(start_single_node_on_head, nodes)
    else:
        for node_id in nodes:
            start_single_node_on_head(node_id)


def start_node_from_head(config_file: str,
                         node_ip: str,
                         all_nodes: bool,
                         override_cluster_name: Optional[str] = None,
                         no_config_cache: bool = False,
                         indent_level: int = None,
                         parallel: bool = True):
    """Execute start node command on head."""

    # execute attach on head
    cmds = [
        "cloudtik",
        "head",
        "start-node",
    ]
    if node_ip:
        cmds += ["--node-ip={}".format(node_ip)]
    if all_nodes:
        cmds += ["--all-nodes"]
    if indent_level:
        cmds += ["--indent-level={}".format(indent_level)]
    if parallel:
        cmds += ["--parallel"]
    else:
        cmds += ["--no-parallel"]
    final_cmd = " ".join(cmds)

    exec_cmd_on_cluster(config_file, final_cmd,
                        override_cluster_name,
                        no_config_cache=no_config_cache)


def stop_node_from_head(config_file: str,
                        node_ip: str,
                        all_nodes: bool,
                        override_cluster_name: Optional[str] = None,
                        no_config_cache: bool = False,
                        indent_level: int = None,
                        parallel: bool = True):
    """Execute stop node command on head."""

    # execute attach on head
    cmds = [
        "cloudtik",
        "head",
        "stop-node",
    ]
    if node_ip:
        cmds += ["--node-ip={}".format(node_ip)]
    if all_nodes:
        cmds += ["--all-nodes"]
    if indent_level:
        cmds += ["--indent-level={}".format(indent_level)]
    if parallel:
        cmds += ["--parallel"]
    else:
        cmds += ["--no-parallel"]
    final_cmd = " ".join(cmds)

    exec_cmd_on_cluster(config_file, final_cmd,
                        override_cluster_name,
                        no_config_cache=no_config_cache)


def get_nodes_of(config,
                 provider,
                 head_node,
                 node_ip: str = None,
                 all_nodes: bool = False):
    if not node_ip:
        if head_node:
            nodes = [head_node]
        else:
            nodes = []
        if all_nodes:
            nodes.extend(_get_worker_nodes(config, None))
    else:
        node_id = provider.get_node_id(node_ip, use_internal_ip=True)
        if not node_id:
            cli_logger.error("No node with the specified node ip - {} found.", node_ip)
            return
        nodes = [node_id]
    return nodes


def stop_node_on_head(node_ip: str = None,
                      all_nodes: bool = False,
                      parallel: bool = True):
    # Since this is running on head, the bootstrap config must exist
    cluster_config_file = get_head_bootstrap_config()
    config = yaml.safe_load(open(cluster_config_file).read())
    provider = _get_node_provider(config["provider"], config["cluster_name"])
    head_node = _get_running_head_node(config, cluster_config_file,
                                       None, _provider=provider,
                                       _allow_uninitialized_state=True)
    head_node_ip = provider.internal_ip(head_node)
    runtime_envs = with_runtime_environment_variables(
        config.get("runtime"), provider)

    nodes = get_nodes_of(config, provider, head_node,
                         node_ip, all_nodes)

    def stop_single_node_on_head(node_id):
        is_head_node = False
        if node_id == head_node:
            is_head_node = True

        if is_head_node:
            stop_commands = config["head_stop_commands"]
            node_runtime_envs = with_node_ip_environment_variables(head_node_ip, provider, node_id)
        else:
            stop_commands = with_head_node_ip(config["worker_stop_commands"], head_node_ip)
            node_runtime_envs = with_node_ip_environment_variables(None, provider, node_id)

        if not stop_commands:
            return

        updater = create_node_updater_for_exec(
            config=config,
            node_id=node_id,
            provider=provider,
            start_commands=[],
            is_head_node=is_head_node,
            use_internal_ip=True)

        node_runtime_envs.update(runtime_envs)
        updater.exec_commands(stop_commands, node_runtime_envs)

    if parallel and len(nodes) > 1:
        run_in_paralell_on_nodes(stop_single_node_on_head, nodes)
    else:
        for node_id in nodes:
            stop_single_node_on_head(node_id)


def scale_cluster(config_file: str, yes: bool, override_cluster_name: Optional[str],
                  cpus: int, nodes: int):
    assert not (cpus and nodes), "Can specify only one of `cpus` or `nodes`."
    assert (cpus or nodes), "Need specify either `cpus` or `nodes`."

    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config["cluster_name"] = override_cluster_name
    config = _bootstrap_config(config)

    resource_string = f"{cpus} CPUs" if cpus else f"{nodes} nodes"
    cli_logger.confirm(yes, "Are you sure that you want to scale cluster {} to {}?",
                       config["cluster_name"], resource_string, _abort=True)
    cli_logger.newline()

    # send the head the resource request
    scale_cluster_from_head(config_file, override_cluster_name,
                            cpus, nodes)


def scale_cluster_from_head(config_file: str, override_cluster_name: Optional[str],
                            cpus: int, nodes: int):
    # Make a request to head to scale the cluster
    cmds = [
        "cloudtik",
        "head",
        "scale",
        "--yes",
    ]
    if cpus:
        cmds += ["--cpus={}".format(cpus)]
    if nodes:
        cmds += ["--nodes={}".format(nodes)]

    final_cmd = " ".join(cmds)
    exec_cmd_on_cluster(config_file, final_cmd,
                        override_cluster_name)


def scale_cluster_on_head(yes: bool, cpus: int, nodes: int):
    assert not (cpus and nodes), "Can specify only one of `cpus` or `nodes`."
    assert (cpus or nodes), "Need specify either `cpus` or `nodes`."

    cluster_config_file = get_head_bootstrap_config()
    config = yaml.safe_load(open(cluster_config_file).read())

    if not yes:
        resource_string = f"{cpus} CPUs" if cpus else f"{nodes} nodes"
        cli_logger.confirm(yes, "Are you sure that you want to scale cluster {} to {}?",
                           config["cluster_name"], resource_string, _abort=True)
        cli_logger.newline()

    # Calculate nodes request to the number of cpus
    if nodes:
        cpus = convert_nodes_to_cpus(config, nodes)
        if cpus == 0:
            cli_logger.abort("Unknown to convert number of nodes to number of CPUs.")

    try:
        address = services.get_address_to_use_or_die()
        kv_initialize_with_address(address, CLOUDTIK_REDIS_DEFAULT_PASSWORD)

        request_resources(num_cpus=cpus)
    except Exception as e:
        cli_logger.abort("Error happened when making the scale cluster request.", exc=e)


def convert_nodes_to_cpus(config: Dict[str, Any], nodes: int) -> int:
    available_node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    for node_type in available_node_types:
        if node_type != head_node_type:
            resources = available_node_types[node_type].get("resources", {})
            cpu_total = resources.get("CPU", 0)
            if cpu_total > 0:
                return nodes * cpu_total

    return 0
