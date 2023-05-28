import copy
import logging
from typing import Any, Optional
from typing import Dict

from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, CLOUDTIK_TAG_WORKSPACE_NAME
from cloudtik.core.workspace_provider import Existence
from cloudtik.providers._private.onpremise.config import _get_cloud_simulator_address, \
    _get_http_response_from_simulator, \
    get_cluster_name_from_node, _get_node_info

logger = logging.getLogger(__name__)

ON_PREMISE_WORKSPACE_NUM_CREATION_STEPS = 1
ON_PREMISE_WORKSPACE_NUM_DELETION_STEPS = 1
ON_PREMISE_WORKSPACE_TARGET_RESOURCES = 1


def get_workspace_head_nodes(workspace_name, provider_config: Dict[str, Any]):
    tag_filters = {
        CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD}
    request = {"type": "list_nodes", "args": (workspace_name, tag_filters,)}
    cloud_simulator_address = _get_cloud_simulator_address(provider_config)
    all_heads = _get_http_response_from_simulator(cloud_simulator_address, request)
    return all_heads


def list_onpremise_clusters(
        workspace_name,
        provider_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    head_nodes = get_workspace_head_nodes(workspace_name, provider_config)
    clusters = {}
    for head_node in head_nodes:
        node_info = _get_node_info(head_node)
        cluster_name = get_cluster_name_from_node(node_info)
        if cluster_name:
            clusters[cluster_name] = node_info
    return clusters


def check_onpremise_workspace_integrity(config):
    existence = check_onpremise_workspace_existence(config)
    return True if existence == Existence.COMPLETED else False


def check_onpremise_workspace_existence(config):
    provider_config = config["provider"]
    workspace_name = config["workspace_name"]

    skipped_resources = 0
    target_resources = ON_PREMISE_WORKSPACE_TARGET_RESOURCES
    existing_resources = 0

    # check docker bridge network
    if _is_workspace_exists(provider_config, workspace_name):
        existing_resources += 1

    if existing_resources <= skipped_resources:
        return Existence.NOT_EXIST
    elif existing_resources == target_resources:
        return Existence.COMPLETED
    else:
        return Existence.IN_COMPLETED


def _is_workspace_exists(provider_config, workspace_name):
    request = {"type": "get_workspace", "args": (workspace_name,)}
    cloud_simulator_address = _get_cloud_simulator_address(provider_config)
    workspace = _get_http_response_from_simulator(cloud_simulator_address, request)
    return True if workspace else False


def _create_simulator_workspace(provider_config, workspace_name):
    if _is_workspace_exists(provider_config, workspace_name):
        cli_logger.print("On-premise workspace already exists. Skip creation.")
        return

    try:
        cli_logger.print("Creating on-premise workspace: {}.", workspace_name)

        request = {"type": "create_workspace", "args": (workspace_name,)}
        cloud_simulator_address = _get_cloud_simulator_address(provider_config)
        _get_http_response_from_simulator(cloud_simulator_address, request)

        cli_logger.print("Successfully created on-premise workspace.")
    except Exception as e:
        cli_logger.error("Failed to create on-premise workspace: {}", str(e))
        raise e


def _delete_simulator_workspace(provider_config, workspace_name):
    if not _is_workspace_exists(provider_config, workspace_name):
        cli_logger.print("On-premise workspace doesn't exist. Skip deletion.")
        return

    try:
        cli_logger.print("Deleting on-premise workspace: {}.", workspace_name)

        request = {"type": "delete_workspace", "args": (workspace_name,)}
        cloud_simulator_address = _get_cloud_simulator_address(provider_config)
        _get_http_response_from_simulator(cloud_simulator_address, request)

        cli_logger.print("Successfully deleted on-premise workspace.")
    except Exception as e:
        cli_logger.error("Failed to delete on-premise workspace: {}", str(e))
        raise e


def create_onpremise_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # create workspace
    config = _create_workspace(config)
    return config


def _create_workspace(config):
    provider_config = config["provider"]
    workspace_name = config["workspace_name"]

    current_step = 1
    total_steps = ON_PREMISE_WORKSPACE_NUM_CREATION_STEPS

    try:
        with cli_logger.group("Creating workspace: {}", workspace_name):
            with cli_logger.group(
                    "Creating on-promise workspace",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _create_simulator_workspace(provider_config, workspace_name)

    except Exception as e:
        cli_logger.error("Failed to update workspace with the name {}. "
                         "You need to delete and try create again. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
        "Successfully updated workspace: {}.",
        cf.bold(workspace_name))
    return config


def delete_onpremise_workspace(
        config):
    provider_config = config["provider"]
    workspace_name = config["workspace_name"]

    current_step = 1
    total_steps = ON_PREMISE_WORKSPACE_NUM_DELETION_STEPS
    try:
        with cli_logger.group("Deleting workspace: {}", workspace_name):
            with cli_logger.group(
                    "Deleting simulator workspace",
                    _numbered=("[]", current_step, total_steps)):
                current_step += 1
                _delete_simulator_workspace(provider_config, workspace_name)
    except Exception as e:
        cli_logger.error(
            "Failed to delete workspace {}. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
            "Successfully deleted workspace: {}.",
            cf.bold(workspace_name))
