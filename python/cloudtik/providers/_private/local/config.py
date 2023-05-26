import logging
import os
from typing import Any, Optional
from typing import Dict

import cloudtik.core._private.utils as utils
from cloudtik.core._private.resource_spec import ResourceSpec
from cloudtik.core._private.utils import exec_with_output
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME

logger = logging.getLogger(__name__)


def bootstrap_local(config):
    workspace_name = config.get("workspace_name")
    if not workspace_name:
        raise RuntimeError("Workspace name is not specified in cluster configuration.")

    config["provider"]["workspace_name"] = workspace_name
    return config


def bootstrap_local_for_api(config):
    workspace_name = config.get("workspace_name")
    if not workspace_name:
        raise RuntimeError("Workspace name is not specified.")

    config["provider"]["workspace_name"] = workspace_name
    return config


def prepare_local(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare local cluster config for ingestion by cluster launcher and scaler.
    """
    return config


def get_local_scheduler_lock_path() -> str:
    return os.path.join(
        utils.get_user_temp_dir(), "cloudtik-local-scheduler.lock")


def get_local_scheduler_state_path() -> str:
    return os.path.join(
        _get_data_path(), "cloudtik-local-scheduler.state")


def get_cluster_name_from_node(node_info) -> Optional[str]:
    for key, value in node_info.items():
        if key == CLOUDTIK_TAG_CLUSTER_NAME:
            return value
    return None


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
