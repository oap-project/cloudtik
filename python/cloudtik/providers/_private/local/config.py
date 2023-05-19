import json
import os
from typing import Any, Optional
from typing import Dict
import logging

from cloudtik.core._private.cli_logger import cli_logger
import cloudtik.core._private.utils as utils
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, CLOUDTIK_TAG_CLUSTER_NAME

logger = logging.getLogger(__name__)


DEFAULT_BRIDGE_SSH_PORT = 8282


def _get_bridge_address(provider_config):
    bridge_address = provider_config["bridge_address"]
    if bridge_address:
        # within the cluster
        # Add the default port if not specified
        if ":" not in bridge_address:
            bridge_address += (":{}".format(DEFAULT_BRIDGE_SSH_PORT))
    else:
        # on host
        # TODO: search the docker0 ip address
        bridge_address = ""

    return bridge_address


def prepare_local(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare local cluster config for ingestion by cluster launcher and scaler.
    """
    return config


def get_host_scheduler_lock_path() -> str:
    return os.path.join(utils.get_user_temp_dir(), "cloudtik-local-host-scheduler.lock")


def get_host_scheduler_state_path() -> str:
    return os.path.join(utils.get_user_temp_dir(), "cloudtik-local-host-scheduler.state")


def get_local_nodes(provider_config: Dict[str, Any]):
    if "nodes" not in provider_config:
        raise RuntimeError("No 'nodes' defined in local provider configuration.")

    return provider_config["nodes"]


def _get_request_instance_type(node_config):
    if "instance_type" not in node_config:
        raise ValueError("Invalid node request. 'instance_type' is required.")

    return node_config["instance_type"]


def _get_node_id_mapping(provider_config: Dict[str, Any]):
    nodes = get_local_nodes(provider_config)
    node_id_mapping = {}
    for node in nodes:
        node_id_mapping[node["ip"]] = node
    return node_id_mapping


def _get_node_instance_type(node_id_mapping, node_ip):
    node = node_id_mapping.get(node_ip)
    if node is None:
        raise RuntimeError(f"Node with node ip {node_ip} is not found in the original node list.")
    return node["instance_type"]


def get_list_of_node_ips(provider_config: Dict[str, Any]):
    nodes = get_local_nodes(provider_config)
    node_ips = [node["ip"] for node in nodes]
    return node_ips


def set_node_types_resources(
            config: Dict[str, Any]):
    # Update the instance information to node type
    available_node_types = config["available_node_types"]
    for node_type in available_node_types:
        instance_type = available_node_types[node_type]["node_config"][
            "instance_type"]
        if instance_type:
            detected_resources = {"CPU": instance_type["CPU"]}

            memory_total = instance_type["memory"]
            memory_total_in_bytes = int(memory_total) * 1024 * 1024 * 1024
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
            raise ValueError("Instance type is not specified in local configuration.")


def get_workspace_head_nodes(provider_config: Dict[str, Any]):
    tag_filters = {CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD}
    # TODO: handle getting nodes with tag filters
    all_heads = []
    return all_heads


def _get_node_info(provider_config: Dict[str, Any], node_id):
    # TODO: handle node info
    node_info = {}
    return node_info


def _get_node_tags(provider_config: Dict[str, Any], node_id):
    # TODO: handle node tags
    node_tags = {}
    return node_tags


def get_cluster_name_from_head(node_info) -> Optional[str]:
    for key, value in node_info.items():
        if key == CLOUDTIK_TAG_CLUSTER_NAME:
            return value
    return None


def list_local_clusters(provider_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    head_nodes = get_workspace_head_nodes(provider_config)
    clusters = {}
    for head_node in head_nodes:
        node_info = _get_node_info(provider_config, head_node)
        cluster_name = get_cluster_name_from_head(node_info)
        if cluster_name:
            clusters[cluster_name] = node_info
    return clusters


def post_prepare_local(config: Dict[str, Any]) -> Dict[str, Any]:
    config = fill_available_node_types_resources(config)
    return config


def fill_available_node_types_resources(
        cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fills out missing "resources" field for available_node_types."""
    set_node_types_resources(cluster_config)
    return cluster_config
