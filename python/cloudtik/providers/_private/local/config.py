import os
from typing import Any, Optional
from typing import Dict
import logging

import cloudtik.core._private.utils as utils
from cloudtik.core._private.resource_spec import ResourceSpec
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME

logger = logging.getLogger(__name__)


DEFAULT_BRIDGE_SSH_PORT = 8282


def _get_bridge_address(provider_config):
    bridge_address = provider_config.get("bridge_address")
    if bridge_address:
        # within container
        # Add the default port if not specified
        if ":" not in bridge_address:
            bridge_address += (":{}".format(DEFAULT_BRIDGE_SSH_PORT))

    return bridge_address


def prepare_local(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare local cluster config for ingestion by cluster launcher and scaler.
    """
    return config


def get_host_scheduler_lock_path() -> str:
    return os.path.join(
        utils.get_user_temp_dir(), "cloudtik-local-host-scheduler.lock")


def get_host_scheduler_state_path() -> str:
    return os.path.join(
        utils.get_user_temp_dir(), "cloudtik-local-host-scheduler.state")


def _get_request_instance_type(node_config):
    if "instance_type" not in node_config:
        raise ValueError("Invalid node request. 'instance_type' is required.")

    return node_config["instance_type"]


def get_instance_type_name(instance_type):
    instance_type_name = instance_type.get("name")
    if instance_type_name:
        return

    # combine a name from CPU, memory
    num_cpus = instance_type.get("CPU", 0)
    memory_gb = instance_type.get("memory", 0)
    if num_cpus and memory_gb:
        return "{}CPU:{}GB".format(num_cpus, memory_gb)
    elif num_cpus:
        return "{}CPU".format(num_cpus)
    elif memory_gb:
        return "{}GB".format(memory_gb)
    return "unknown_type"


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
