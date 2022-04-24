import os
from typing import Any
from typing import Dict
import logging

from cloudtik.core._private.cli_logger import cli_logger
import cloudtik.core._private.utils as utils

logger = logging.getLogger(__name__)


def prepare_local(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare local cluster config for ingestion by cluster launcher and scaler.
    """
    if "cloud_simulator_address" not in config["provider"]:
        cli_logger.abort("No Cloud Simulator address specified. "
                         "You must specify use cloud_simulator_address.")

    return config


def get_cloud_simulator_lock_path() -> str:
    return os.path.join(utils.get_user_temp_dir(), "cloudtik-cloud-simulator.lock")


def get_cloud_simulator_state_path() -> str:
    return os.path.join(utils.get_user_temp_dir(), "cloudtik-cloud-simulator.state")


def get_local_nodes(provider_config: Dict[str, Any]):
    if "nodes" not in provider_config:
        raise RuntimeError("No 'nodes' defined in local provider configuration.")

    return provider_config["nodes"]


def _get_request_instance_type(node_config):
    if "instance_type" not in node_config:
        raise ValueError("Invalid node request. 'instance_type' is required.")


def _get_node_id_mapping(provider_config: Dict[str, Any]):
    nodes = get_local_nodes(provider_config)
    node_id_mapping = {}
    for node in nodes:
        node_id_mapping[node.ip] = node
    return node_id_mapping


def _get_node_instance_type(node_id_mapping, node_ip):
    node = node_id_mapping.get(node_ip)
    if node is None:
        raise RuntimeError(f"Node with node ip {node_ip} is not found in the original node list.")
    return node["instance_type"]


def get_list_of_node_ips(provider_config: Dict[str, Any]):
    nodes = get_local_nodes(provider_config)
    node_ips = [node.ip for node in nodes]
    return node_ips


def _get_num_node_of_instance_type(provider_config: Dict[str, Any], instance_type) -> int:
    nodes = get_local_nodes(provider_config)
    num_node_of_instance_type = 0
    for node in nodes:
        if instance_type == node[instance_type]:
            num_node_of_instance_type += 1
    return num_node_of_instance_type


def set_node_types_resources(
            config: Dict[str, Any], instance_types):
    # Update the instance information to node type
    available_node_types = config["available_node_types"]
    for node_type in available_node_types:
        instance_type = available_node_types[node_type]["node_config"][
            "instance_type"]
        if instance_type in instance_types:
            resources = instance_types[instance_type]["resources"]
            detected_resources = {"CPU": resources["CPU"]}

            memory_total = resources["memoryMb"]
            memory_total_in_bytes = int(memory_total) * 1024 * 1024
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
                             " is not available in local configuration.")


def _get_instance_types(provider_config: Dict[str, Any]) -> Dict[str, Any]:
    if "instance_types" not in provider_config:
        cli_logger.warning("No instance types definition found. No node can be created!"
                           "Please supply the instance types definition in the config.")
    instance_types = provider_config.get("instance_types", {})
    return instance_types
