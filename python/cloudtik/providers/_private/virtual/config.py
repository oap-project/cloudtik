import copy
import os
import subprocess
import uuid
from typing import Any, Optional
from typing import Dict
import logging
import psutil

from cloudtik.core._private.core_utils import get_memory_in_bytes, get_cloudtik_temp_dir, exec_with_output
from cloudtik.core._private.utils import AUTH_CONFIG_KEY, DOCKER_CONFIG_KEY, \
    FILE_MOUNTS_CONFIG_KEY, get_head_service_ports
from cloudtik.core._private.resource_spec import ResourceSpec
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME

import cloudtik.core._private.utils as utils

logger = logging.getLogger(__name__)


DEFAULT_SSH_SERVER_PORT = 3371
MAX_PORT_MAPPING_BASE_RETRY = 20


def _get_provider_bridge_address(provider_config):
    bridge_address = provider_config.get("bridge_address")
    if bridge_address:
        # within container
        # Add the default port if not specified
        if ":" not in bridge_address:
            bridge_address += (":{}".format(DEFAULT_SSH_SERVER_PORT))

    return bridge_address


def bootstrap_virtual(config):
    workspace_name = config.get("workspace_name")
    if not workspace_name:
        raise RuntimeError("Workspace name is not specified in cluster configuration.")

    config["provider"]["workspace_name"] = workspace_name

    # only do this for docker workspace
    config = _configure_docker(config)
    config = _configure_bridge_address(config)
    config = _configure_auth(config)
    config = _configure_docker_of_node_types(config)
    config = _configure_port_mappings(config)
    config = _configure_file_mounts(config)
    config = _configure_shared_memory_ratio(config)

    return config


def bootstrap_virtual_for_api(config):
    workspace_name = config.get("workspace_name")
    if not workspace_name:
        raise RuntimeError("Workspace name is not specified.")

    config["provider"]["workspace_name"] = workspace_name

    config = _configure_docker(config)
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

    # The bridge_address in provider has been set
    bridge_ssh_port = int(config["provider"]["bridge_address"].split(":")[1])
    auth_config["ssh_port"] = bridge_ssh_port

    # copy auth to provider section
    config["provider"][AUTH_CONFIG_KEY] = copy.deepcopy(auth_config)
    return config


def _configure_docker(config):
    if DOCKER_CONFIG_KEY not in config:
        config[DOCKER_CONFIG_KEY] = {}
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
    return config


def _configure_docker_of_node_types(config):
    # copy node specific docker config to node_config
    for key, node_type in config["available_node_types"].items():
        if DOCKER_CONFIG_KEY in node_type:
            node_config = node_type["node_config"]
            node_config[DOCKER_CONFIG_KEY] = copy.deepcopy(
                node_type[DOCKER_CONFIG_KEY])
    return config


def _is_port_mapping_base_usable(
        port_mapping_base, service_ports, existing_port_mapping):
    for port_name in service_ports:
        port_config = service_ports[port_name]
        host_port = port_mapping_base + port_config["port"]
        if host_port in existing_port_mapping:
            return False
    return True


def _get_port_mapping_base(provider, service_ports):
    port_mapping_base = provider.get("port_mapping_base")
    if port_mapping_base is not None:
        return port_mapping_base

    existing_port_mapping = _get_existing_port_mapping()
    port_mapping_base = 0
    retry = 0
    while retry < MAX_PORT_MAPPING_BASE_RETRY:
        if _is_port_mapping_base_usable(
                port_mapping_base, service_ports, existing_port_mapping):
            return port_mapping_base
        port_mapping_base += 1000

    raise RuntimeError("Failed to find a free port mapping base. "
                       "You need specific port_mapping_base in provider configuration.")


def _get_mapping(mapping):
    # 172.18.0.1:80->80/tcp
    # find last : and end with /
    start = mapping.rfind(":")
    if start < 0:
        return None
    start += 1
    end = mapping.rfind("/")
    if end < 0 or end <= start:
        return None
    port_to_port = mapping[start:end]
    ports = port_to_port.split("->")
    if len(ports) != 2:
        return None
    return int(ports[0]), int(ports[1])


def _get_existing_port_mapping():
    # Choose smartly a base on local host based clusters we are starting
    docker_cmd = "docker container list --format '{{.Ports}}'"
    cmd = with_sudo(docker_cmd)
    output = exec_with_output(
        cmd
    ).decode().strip()
    if not output:
        return {}
    existing_port_mappings = {}
    lines = output.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        mappings = line.split(",")
        for mapping in mappings:
            mapping_pair = _get_mapping(mapping)
            if mapping_pair is not None:
                host_port, container_port= mapping_pair
                existing_port_mappings[host_port] = container_port
    return existing_port_mappings


def _configure_port_mappings(config):
    provider = config["provider"]
    # configure port mappings for head node
    runtime_config = config.get("runtime", {})
    service_ports = get_head_service_ports(runtime_config)

    port_mapping_base = _get_port_mapping_base(
        provider, service_ports)
    # The bridge_address in provider has been set
    host_ip = provider["bridge_address"].split(":")[0]
    provider["port_mapping_base"] = port_mapping_base

    node_types = config["available_node_types"]
    head_node_type = config["head_node_type"]
    node_config = node_types[head_node_type]["node_config"]

    port_mappings = {}
    for port_name in service_ports:
        port_config = service_ports[port_name]
        container_port = port_config["port"]
        host_port = port_mapping_base + container_port
        port_mappings[str(container_port)] = "{}:{}".format(host_ip, host_port)
    if port_mappings:
        node_config["port_mappings"] = port_mappings
    return config


def _configure_file_mounts(config):
    state_path = get_virtual_cluster_data_path(
        config.get("workspace_name"), config.get("cluster_name"))
    exec_with_output(
        "mkdir -p '{path}' && chmod -R 777 '{path}'".format(path=state_path))

    config["provider"]["state_path"] = state_path

    for key, node_type in config["available_node_types"].items():
        node_config = node_type["node_config"]
        data_dirs = node_config.get("data_dirs")
        if data_dirs:
            for data_dir in data_dirs:
                # the bootstrap process has updated the permission
                exec_with_output(f"chmod -R 777 '{data_dir}'")

    # copy docker to provider section
    config["provider"][FILE_MOUNTS_CONFIG_KEY] = copy.deepcopy(
        config[FILE_MOUNTS_CONFIG_KEY])
    return config


def _configure_shared_memory_ratio(config):
    # configure shared memory ratio to node config for each type
    runtime_config = config.get(utils.RUNTIME_CONFIG_KEY)
    if not runtime_config:
        return config
    for node_type, node_type_config in config["available_node_types"].items():
        shared_memory_ratio = utils.get_runtime_shared_memory_ratio(
            runtime_config, config, node_type)
        if shared_memory_ratio != 0:
            node_config = node_type_config["node_config"]
            node_config["shared_memory_ratio"] = shared_memory_ratio

    return config


def prepare_virtual(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare cluster config for ingestion by cluster launcher and scaler.
    """
    return config


def get_virtual_cluster_data_path(workspace, cluster_name) -> str:
    return os.path.join(
        _get_data_path(), workspace, cluster_name)


def get_virtual_scheduler_lock_path(workspace, cluster_name) -> str:
    return os.path.join(
        get_cloudtik_temp_dir(),
        get_virtual_scheduler_lock_file_name(workspace, cluster_name))


def get_virtual_scheduler_state_path(workspace, cluster_name) -> str:
    return os.path.join(
        get_virtual_cluster_data_path(workspace, cluster_name),
        get_virtual_scheduler_state_file_name())


def get_virtual_scheduler_lock_file_name(workspace, cluster_name) -> str:
    return "cloudtik-virtual-scheduler-{}-{}.lock".format(
            workspace, cluster_name)


def get_virtual_scheduler_state_file_name() -> str:
    return "cloudtik-virtual-scheduler.state"


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
            detected_resources["accelerator_type:GPU"] = 1

        memory_total_in_bytes = get_memory_in_bytes(
                instance_type.get("memory"))
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


def post_prepare_virtual(config: Dict[str, Any]) -> Dict[str, Any]:
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


def _safe_remove_file(file_to_remove):
    if os.path.exists(file_to_remove):
        os.remove(file_to_remove)


def _get_network_name(workspace_name):
    return "cloudtik-{}".format(workspace_name)


def _get_bridge_interface_name(network_name):
    interface_suffix = str(uuid.uuid3(uuid.NAMESPACE_OID, network_name))[:8]
    return "tik-{}".format(interface_suffix)


def _get_sshd_config_file(workspace_name):
    return os.path.expanduser(
        "~/.ssh/cloudtik-{}-sshd_config".format(workspace_name))


def _get_ssh_control_key_file(workspace_name):
    return os.path.expanduser(
        "~/.ssh/cloudtik-{}-control_key".format(workspace_name))


def _get_authorized_keys_file(workspace_name):
    return os.path.expanduser(
        "~/.ssh/cloudtik-{}-authorized_keys".format(workspace_name))


def _get_host_key_file(workspace_name):
    return os.path.expanduser(
        "~/.ssh/cloudtik-{}-host_key".format(workspace_name))


def _get_data_path():
    return os.path.expanduser("~/.cloudtik/virtual")


def _make_sure_data_path():
    data_path = _get_data_path()
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)


def get_ssh_server_process_file(workspace_name: str):
    ssh_server_process_file = os.path.join(
        get_cloudtik_temp_dir(), "cloudtik-{}-sshd".format(workspace_name))
    return ssh_server_process_file


def _get_ssh_server_process(ssh_server_process_file: str):
    ssh_server_process = utils.get_server_process(ssh_server_process_file)
    if ssh_server_process is None:
        return None, None, None
    pid = ssh_server_process.get("pid")
    if not pid:
        return None, None, None
    return (ssh_server_process["pid"],
            ssh_server_process.get("bind_address"),
            ssh_server_process.get("port"))


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
