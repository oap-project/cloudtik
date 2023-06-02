import json
import socket
from http.client import RemoteDisconnected
import os
from typing import Any, Optional
from typing import Dict
import logging

from cloudtik.core._private.cli_logger import cli_logger
import cloudtik.core._private.utils as utils
from cloudtik.core._private.core_utils import get_memory_in_bytes, get_cloudtik_temp_dir
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME

logger = logging.getLogger(__name__)


DEFAULT_CLOUD_SIMULATOR_PORT = 8282


def get_cloud_simulator_process_file():
    return os.path.join(get_cloudtik_temp_dir(), "cloudtik-cloud-simulator")


def _discover_cloud_simulator():
    cloud_simulator_process_file = get_cloud_simulator_process_file()
    server_process = utils.get_server_process(cloud_simulator_process_file)
    if server_process is None:
        return None
    bind_address = server_process.get("bind_address")
    port = server_process.get("port")
    if bind_address and port:
        if "0.0.0.0" == bind_address:
            bind_address = socket.gethostbyname(socket.gethostname())
        return "{}:{}".format(bind_address, port)
    return None


def _get_cloud_simulator_address(provider_config):
    cloud_simulator_address = provider_config.get("cloud_simulator_address")
    if not cloud_simulator_address:
        cloud_simulator_address = _discover_cloud_simulator()
        if not cloud_simulator_address:
            raise RuntimeError(
                "Failed to discover the cloud simulator address."
                "Please configure cloud_simulator_address in provider configuration.")
    # Add the default port if not specified
    if ":" not in cloud_simulator_address:
        cloud_simulator_address += (":{}".format(DEFAULT_CLOUD_SIMULATOR_PORT))
    return cloud_simulator_address


def _get_http_response_from_simulator(cloud_simulator_address, request):
    headers = {
        "Content-Type": "application/json",
    }
    request_message = json.dumps(request).encode()
    cloud_simulator_endpoint = "http://" + cloud_simulator_address

    try:
        import requests  # `requests` is not part of stdlib.
        from requests.exceptions import ConnectionError

        r = requests.get(
            cloud_simulator_endpoint,
            data=request_message,
            headers=headers,
            timeout=None,
        )
    except (RemoteDisconnected, ConnectionError):
        logger.exception("Could not connect to: " +
                         cloud_simulator_endpoint +
                         ". Did you launched the Cloud Simulator by running cloudtik-simulator " +
                         " --config nodes-config-file --port <PORT>?")
        raise
    except ImportError:
        logger.exception(
            "Not all dependencies were found. Please "
            "update your install command.")
        raise

    response = r.json()
    return response


def bootstrap_onpremise(config):
    workspace_name = config.get("workspace_name")
    if not workspace_name:
        raise RuntimeError("Workspace name is not specified in cluster configuration.")

    config["provider"]["workspace_name"] = workspace_name
    config = _configure_cloud_simulator(config)
    return config


def _configure_cloud_simulator(config):
    provider = config["provider"]
    if "cloud_simulator_address" not in provider:
        provider["cloud_simulator_address"] = _get_cloud_simulator_address(provider)
    return config


def prepare_onpremise(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare onpremise cluster config for ingestion by cluster launcher and scaler.
    """
    return config


def get_cloud_simulator_lock_path() -> str:
    return os.path.join(get_cloudtik_temp_dir(), "cloudtik-cloud-simulator.lock")


def get_cloud_simulator_state_path() -> str:
    return os.path.join(_get_data_path(), "cloudtik-cloud-simulator.state")


def _get_data_path():
    return os.path.expanduser("~/.cloudtik/onpremise")


def _make_sure_data_path():
    data_path = _get_data_path()
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)


def get_available_nodes(provider_config: Dict[str, Any]):
    if "nodes" not in provider_config:
        raise RuntimeError("No 'nodes' defined in cloud simulator configuration.")

    return provider_config["nodes"]


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
    node_ids = [node["ip"] for node in nodes]
    return node_ids


def _get_node_tags(node):
    if node is None:
        return {}
    return node.get("tags", {})


def _get_node_info(node):
    node_info = {"node_id": node["name"],
                 "instance_type": node.get("instance_type"),
                 "private_ip": node["ip"],
                 "public_ip": node.get("external_ip"),
                 "instance_status": node["state"]}
    node_info.update(node["tags"])
    return node_info


def set_node_types_resources(
            config: Dict[str, Any], instance_types):
    # Update the instance information to node type
    available_node_types = config["available_node_types"]
    for node_type in available_node_types:
        instance_type = available_node_types[node_type]["node_config"][
            "instance_type"]
        if instance_type in instance_types:
            resources = instance_types[instance_type]
            detected_resources = {"CPU": resources["CPU"]}

            num_gpus = resources.get("GPU", 0)
            if num_gpus > 0:
                detected_resources["GPU"] = num_gpus
                detected_resources["accelerator_type:GPU"] = 1

            memory_total_in_bytes = get_memory_in_bytes(
                resources.get("memory"))
            detected_resources["memory"] = memory_total_in_bytes

            detected_resources.update(
                available_node_types[node_type].get("resources", {}))
            if detected_resources != \
                    available_node_types[node_type].get("resources", {}):
                available_node_types[node_type][
                    "resources"] = detected_resources
                logger.debug("Updating the resources of {} to {}.".format(
                    node_type, detected_resources))
        else:
            raise ValueError("Instance type " + instance_type +
                             " is not available in onpremise configuration.")


def _get_instance_types(provider_config: Dict[str, Any]) -> Dict[str, Any]:
    if "instance_types" not in provider_config:
        cli_logger.warning("No instance types definition found. No node can be created!"
                           "Please supply the instance types definition in the config.")
    instance_types = provider_config.get("instance_types", {})
    return instance_types


def get_cluster_name_from_node(node_info) -> Optional[str]:
    for key, value in node_info.items():
        if key == CLOUDTIK_TAG_CLUSTER_NAME:
            return value
    return None


def post_prepare_onpremise(config: Dict[str, Any]) -> Dict[str, Any]:
    config = fill_available_node_types_resources(config)
    return config


def fill_available_node_types_resources(
        cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fills out missing "resources" field for available_node_types."""
    request = {"type": "get_instance_types", "args": ()}
    cloud_simulator_address = _get_cloud_simulator_address(cluster_config["provider"])
    instance_types = _get_http_response_from_simulator(cloud_simulator_address, request)
    set_node_types_resources(cluster_config, instance_types)
    return cluster_config
