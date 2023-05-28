import logging
import os
import socket
from typing import Any, Optional
from typing import Dict

import cloudtik.core._private.utils as utils
from cloudtik.core._private.core_utils import get_memory_in_bytes, get_cloudtik_temp_dir
from cloudtik.core._private.resource_spec import ResourceSpec
from cloudtik.core._private.utils import exec_with_output
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME

logger = logging.getLogger(__name__)

LOCAL_WORKSPACE_NAME = "default"
LOCAL_INSTANCE_TYPE = "default"
CONFIG_KEY_NODES = "nodes"


def bootstrap_local(config):
    config = _configure_workspace_name(config)
    config = _configure_local_node_id(config)
    config = _configure_node_instance_types(config)
    config = _configure_docker(config)

    return config


def _configure_local_node_id(config):
    # if the config has a local ip specified
    # use this one, else use a default ip
    provider = config["provider"]

    # in case that user don't specify any other nodes
    if CONFIG_KEY_NODES not in provider:
        provider[CONFIG_KEY_NODES] = {}

    local_ip = provider.get("local_ip")
    listed_local_ip = None
    if not local_ip:
        # check whether there is local ip already list in the nodes list
        listed_local_ip = _get_listed_local_ip(provider)
        local_ip = listed_local_ip
        if not local_ip:
            local_ip = socket.gethostbyname(socket.gethostname())
        if not local_ip:
            raise RuntimeError("Failed to get local ip.")
    provider["local_node_id"] = local_ip

    # we also need to add the local ip to the nodes list of provider
    if not listed_local_ip:
        _add_local_node_to_provider(provider, local_ip)

    return config


def _get_listed_local_ip(provider):
    nodes = provider[CONFIG_KEY_NODES]
    host_ips = utils.get_host_address(address_type="all")
    for node in nodes:
        node_ip = node["ip"]
        if node_ip in host_ips:
            return node_ip
    return None


def _add_local_node_to_provider(provider, local_node_ip):
    nodes = provider[CONFIG_KEY_NODES]
    node_ips = {node["ip"] for node in nodes}
    if local_node_ip in node_ips:
        return

    local_node = {"ip": local_node_ip}
    nodes.append(local_node)


def _configure_instance_types(config):
    for node_type, node_type_config in config["available_node_types"].items():
        node_config = node_type_config["node_config"]
        if "instance_type" not in node_config:
            node_config["instance_type"] = LOCAL_INSTANCE_TYPE
        instance_type = node_config["instance_type"]
        if instance_type == LOCAL_INSTANCE_TYPE:
            _make_default_instance_type(config)

    return config


def _make_default_instance_type(config):
    provider = config["provider"]
    if "instance_types" not in provider:
        provider["instance_types"] = {}
    instance_types = provider["instance_types"]
    if LOCAL_INSTANCE_TYPE in instance_types:
        return
    resource_spec = ResourceSpec().resolve(available_memory=False)
    local_instance_type = {
        "CPU": resource_spec.num_cpus,
        "memory": resource_spec.memory,
    }
    if resource_spec.num_gpus:
        local_instance_type["GPU"] = resource_spec.num_gpus
    instance_types[LOCAL_INSTANCE_TYPE] = local_instance_type


def _configure_node_instance_types(config):
    provider = config["provider"]
    nodes = provider[CONFIG_KEY_NODES]
    for node in nodes:
        if "instance_type" not in node:
            node["instance_type"] = LOCAL_INSTANCE_TYPE
    return config


def _configure_docker(config):
    if not utils.is_docker_enabled(config):
        return config
    provider = config["provider"]
    rootless = is_rootless_docker()
    if not rootless:
        provider["docker_with_sudo"] = True

    state_path = get_state_path()
    exec_with_output(
        "mkdir -p '{path}' && chmod -R 777 '{path}'".format(path=state_path))
    return config


def bootstrap_local_for_api(config):
    config = _configure_workspace_name(config)
    return config


def _configure_workspace_name(config):
    workspace_name = LOCAL_WORKSPACE_NAME
    config["workspace_name"] = workspace_name
    config["provider"]["workspace_name"] = workspace_name
    return config


def prepare_local(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare local cluster config for ingestion by cluster launcher and scaler.
    """
    return config


def get_local_scheduler_lock_path() -> str:
    return os.path.join(
        get_cloudtik_temp_dir(), "cloudtik-local-scheduler.lock")


def get_state_path():
    return _get_data_path()


def get_local_scheduler_state_path() -> str:
    return os.path.join(
        _get_data_path(), get_local_scheduler_state_file_name())


def get_local_scheduler_state_file_name() -> str:
    return "cloudtik-local-scheduler.state"


def get_cluster_name_from_node(node_info) -> Optional[str]:
    for key, value in node_info.items():
        if key == CLOUDTIK_TAG_CLUSTER_NAME:
            return value
    return None


def get_available_nodes(provider_config: Dict[str, Any]):
    if CONFIG_KEY_NODES not in provider_config:
        raise RuntimeError("No 'nodes' defined in provider configuration.")

    return provider_config[CONFIG_KEY_NODES]


def _get_request_instance_type(node_config):
    if "instance_type" not in node_config:
        raise ValueError("Invalid node request. 'instance_type' is required.")
    return node_config["instance_type"]


def _get_node_id_mapping(provider_config: Dict[str, Any]):
    nodes = get_available_nodes(provider_config)
    return {node["ip"]: node for node in nodes}


def _get_node_instance_type(node_id_mapping, node_id):
    node = node_id_mapping.get(node_id)
    if node is None:
        raise RuntimeError(f"Node with ip {node_id} is not found in the original node list.")
    return node["instance_type"]


def get_all_node_ids(provider_config: Dict[str, Any]):
    nodes = get_available_nodes(provider_config)
    # ip here may the host ip or host name
    return [node["ip"] for node in nodes]


def set_node_types_resources(
            config: Dict[str, Any]):
    # Update the instance information to node type
    provider = config["provider"]
    instance_types = provider.get("instance_types", {})

    available_node_types = config["available_node_types"]
    for node_type in available_node_types:
        instance_type_name = available_node_types[node_type]["node_config"].get(
            "instance_type", LOCAL_INSTANCE_TYPE)
        instance_type = instance_types.get(instance_type_name)
        if not instance_type:
            raise RuntimeError("Instance type: {} is not defined.".format(instance_type_name))
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


def post_prepare_local(config: Dict[str, Any]) -> Dict[str, Any]:
    config = _configure_instance_types(config)
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
    except:  # noqa: E722
        return False


def with_sudo(docker_cmd):
    # check whether we need to run as sudo based whether we run rootless docker
    if is_rootless_docker():
        return docker_cmd
    return "sudo " + docker_cmd


def _safe_remove_file(file_to_remove):
    if os.path.exists(file_to_remove):
        os.remove(file_to_remove)


def _get_data_path():
    return os.path.expanduser("~/.cloudtik/local")


def _make_sure_data_path():
    data_path = _get_data_path()
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
