import copy
import json
import os
import subprocess
import uuid
from typing import Any, Optional
from typing import Dict
import logging
import psutil
import tempfile


from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.utils import AUTH_CONFIG_KEY, DOCKER_CONFIG_KEY, \
    FILE_MOUNTS_CONFIG_KEY, exec_with_output, is_docker_enabled
from cloudtik.core._private.resource_spec import ResourceSpec
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME

import cloudtik.core._private.utils as utils

logger = logging.getLogger(__name__)


DEFAULT_SSH_SERVER_PORT = 3371
TAG_WORKSPACE_NAME = "workspace-name"


def _get_cluster_bridge_address(provider_config):
    bridge_address = provider_config.get("bridge_address")
    if bridge_address:
        # within container
        # Add the default port if not specified
        if ":" not in bridge_address:
            bridge_address += (":{}".format(DEFAULT_SSH_SERVER_PORT))

    return bridge_address


def bootstrap_local(config):
    workspace_name = config.get("workspace_name")
    if not workspace_name:
        raise RuntimeError("Workspace name is not specified in cluster configuration.")

    config["provider"]["workspace_name"] = workspace_name

    # only do this for docker workspace
    if is_docker_enabled(config):
        config = _configure_bridge_address(config)
        config = _configure_auth(config)
        config = _configure_docker(config)
        config = _configure_file_mounts(config)
    else:
        config = _configure_workers(config)

    return config


def _configure_bridge_address(config):
    workspace_name = config.get("workspace_name")
    bridge_address = get_workspace_bridge_address(workspace_name)
    if not bridge_address:
        raise RuntimeError("Workspace bridge SSH is not running. Please update workspace.")

    config["provider"]["bridge_address"] = bridge_address
    return config


def get_workspace_bridge_address(workspace_name):
    sshd_pid = _find_ssh_server_process_for_workspace(workspace_name)
    if sshd_pid is None:
        return None
    process_file = get_ssh_server_process_file(workspace_name)
    _, server_address, server_port = _get_ssh_server_process(process_file)
    if not server_address:
        return None
    return "{}:{}".format(server_address, server_port)


def _configure_auth(config):
    # Configure SSH access, using the workspace control key pair
    ssh_private_key_file = _get_ssh_control_key_file(
        config["workspace_name"])
    auth_config = config[AUTH_CONFIG_KEY]
    auth_config["ssh_private_key"] = ssh_private_key_file

    # copy auth to provider section
    config["provider"][AUTH_CONFIG_KEY] = copy.deepcopy(auth_config)
    return config


def _configure_docker(config):
    docker_config = config[DOCKER_CONFIG_KEY]
    rootless = is_rootless_docker()
    if not rootless:
        docker_config["docker_with_sudo"] = True

    # configure docker network with the workspace network
    docker_config["network"] = _get_network_name(
        config["workspace_name"])
    # copy docker to provider section
    config["provider"][DOCKER_CONFIG_KEY] = copy.deepcopy(
        config[DOCKER_CONFIG_KEY])

    # copy node specific docker config to node_config
    for key, node_type in config["available_node_types"].items():
        if DOCKER_CONFIG_KEY in node_type:
            node_config = node_type["node_config"]
            node_config[DOCKER_CONFIG_KEY] = node_type[DOCKER_CONFIG_KEY]

    return config


def _configure_file_mounts(config):
    # copy docker to provider section
    config["provider"][FILE_MOUNTS_CONFIG_KEY] = copy.deepcopy(
        config[FILE_MOUNTS_CONFIG_KEY])
    return config


def _configure_workers(config):
    # for local host mode, there is only one head node, no workers
    for key, node_type in config["available_node_types"].items():
        if key != config["head_node_type"]:
            if node_type.get("min_workers", 0) > 0:
                # print a warning
                cli_logger.warning(
                    "Local with host mode doesn't allow additional workers. Reset min_workers to 0.")
                node_type["min_workers"] = 0

    return config


def prepare_local(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare local cluster config for ingestion by cluster launcher and scaler.
    """
    return config


def get_host_scheduler_lock_path() -> str:
    return os.path.join(
        utils.get_user_temp_dir(), "cloudtik-local-host-scheduler.lock")


def _get_host_scheduler_data_path()-> str:
    return os.path.join(_get_data_path(), "host")


def get_host_scheduler_state_path() -> str:
    return os.path.join(
        _get_host_scheduler_data_path(), "cloudtik-host-scheduler.state")


def get_docker_scheduler_data_path() -> str:
    return os.path.join(_get_data_path(), "docker")


def get_docker_cluster_data_path(workspace, cluster_name) -> str:
    return os.path.join(
        get_docker_scheduler_data_path(), workspace, cluster_name)


def get_docker_scheduler_lock_path(workspace, cluster_name) -> str:
    return os.path.join(
        get_docker_cluster_data_path(workspace, cluster_name),
        get_docker_scheduler_lock_file_name())


def get_docker_scheduler_state_path(workspace, cluster_name) -> str:
    return os.path.join(
        get_docker_cluster_data_path(workspace, cluster_name),
        get_docker_scheduler_state_file_name())


def get_docker_scheduler_lock_file_name() -> str:
    return "cloudtik-docker-scheduler.lock"


def get_docker_scheduler_state_file_name() -> str:
    return "cloudtik-docker-scheduler.state"


def _get_request_instance_type(node_config):
    if "instance_type" not in node_config:
        raise ValueError("Invalid node request. 'instance_type' is required.")

    return node_config["instance_type"]


def set_node_types_resources(
            config: Dict[str, Any]):
    # Update the instance information to node type
    available_node_types = config["available_node_types"]
    for node_type in available_node_types:
        instance_type = available_node_types[node_type]["node_config"].get("instance_type", {})
        resource_spec = ResourceSpec().resolve(available_memory=True)
        detected_resources = {}

        num_cpus = instance_type.get("CPU", 0)
        if not num_cpus:
            # use the current host CPU number
            num_cpus = resource_spec.num_cpus
        detected_resources["CPU"] = num_cpus

        num_gpus = instance_type.get("GPU", 0)
        if not num_gpus:
            # use the current host GPU number
            num_gpus = resource_spec.num_gpus
        if num_gpus > 0:
            detected_resources["GPU"] = num_gpus

        memory_gb = instance_type.get("memory", 0)
        memory_total_in_bytes = int(memory_gb) * 1024 * 1024 * 1024
        if not memory_total_in_bytes:
            # use the current host memory
            memory_total_in_bytes = resource_spec.memory
        detected_resources["memory"] = memory_total_in_bytes

        detected_resources.update(
            available_node_types[node_type].get("resources", {}))
        if detected_resources != \
                available_node_types[node_type].get("resources", {}):
            available_node_types[node_type][
                "resources"] = detected_resources
            logger.debug("Updating the resources of {} to {}.".format(
                node_type, detected_resources))


def get_cluster_name_from_node(node_info) -> Optional[str]:
    for key, value in node_info.items():
        if key == CLOUDTIK_TAG_CLUSTER_NAME:
            return value
    return None


def post_prepare_local(config: Dict[str, Any]) -> Dict[str, Any]:
    config = fill_available_node_types_resources(config)
    return config


def fill_available_node_types_resources(
        cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fills out missing "resources" field for available_node_types."""
    set_node_types_resources(cluster_config)
    return cluster_config


def is_rootless_docker():
    # run docker image list for a test
    try:
        exec_with_output("docker image list")
        return True
    except:
        return False


def with_sudo(docker_cmd):
    # check whether we need to run as sudo based whether we run rootless docker
    if is_rootless_docker():
        return docker_cmd
    return "sudo " + docker_cmd


def _get_expanded_path(path):
    return os.path.expanduser(path)


def _safe_remove_file(file_to_remove):
    if os.path.exists(file_to_remove):
        os.remove(file_to_remove)


def _get_network_name(workspace_name):
    workspace_name = workspace_name.replace("-", "_")
    return "cloudtik_{}".format(workspace_name)


def _get_bridge_interface_name(workspace_name):
    interface_suffix = str(uuid.uuid3(uuid.NAMESPACE_OID, workspace_name))[:8]
    return "tik-{}".format(interface_suffix)


def _get_sshd_config_file(workspace_name):
    return _get_expanded_path(
        "~/.ssh/cloudtik-{}-sshd_config".format(workspace_name))


def _get_ssh_control_key_file(workspace_name):
    return _get_expanded_path(
        "~/.ssh/cloudtik-{}-control_key".format(workspace_name))


def _get_authorized_keys_file(workspace_name):
    return _get_expanded_path(
        "~/.ssh/cloudtik-{}-authorized_keys".format(workspace_name))


def _get_host_key_file(workspace_name):
    return _get_expanded_path(
        "~/.ssh/cloudtik-{}-host_key".format(workspace_name))


def _get_data_path():
    return _get_expanded_path("~/.cloudtik/local")


def _make_sure_data_path():
    data_path = _get_data_path()
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)


def get_ssh_server_process_file(workspace_name: str):
    ssh_server_process_file = os.path.join(
        tempfile.gettempdir(), "cloudtik-{}-sshd".format(workspace_name))
    return ssh_server_process_file


def _get_ssh_server_process(ssh_server_process_file: str):
    if os.path.exists(ssh_server_process_file):
        process_info = json.loads(open(ssh_server_process_file).read())
        if process_info.get("process") and process_info["process"].get("pid"):
            ssh_server_process = process_info["process"]
            return (ssh_server_process["pid"],
                    ssh_server_process.get("bind_address"),
                    ssh_server_process["port"])
    return None, None, None


def _find_ssh_server_process_for_workspace(workspace_name):
    sshd_config = _get_sshd_config_file(workspace_name)
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            args = subprocess.list2cmdline(proc.cmdline())
            cmd_name = proc.name()
            if "sshd" in cmd_name and sshd_config in args:
                return proc.pid
        except psutil.Error:
            pass
    return None
