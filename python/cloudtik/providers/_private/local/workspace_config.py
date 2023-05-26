import copy
import logging
from typing import Any, Optional
from typing import Dict

from cloudtik.core._private.cli_logger import cli_logger, cf
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD
from cloudtik.core.workspace_provider import Existence
from cloudtik.providers._private.local.config import get_cluster_name_from_node
from cloudtik.providers._private.local.local_scheduler import LocalScheduler
from cloudtik.providers._private.local.utils import _get_node_info

logger = logging.getLogger(__name__)

LOCAL_WORKSPACE_TARGET_RESOURCES = 1


def _create_local_scheduler(provider_config):
    return LocalScheduler(provider_config, None)


def get_workspace_head_nodes(workspace_name, provider_config: Dict[str, Any]):
    tag_filters = {CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD}
    # The provider config is workspace provider
    # while scheduler expect cluster provider with bootstrap
    # we need to make sure of that
    local_scheduler = _create_local_scheduler(provider_config)
    return local_scheduler.list_nodes(workspace_name, tag_filters)


def list_local_clusters(
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


def create_local_workspace(config):
    # create a copy of the input config to modify
    config = copy.deepcopy(config)

    # create workspace
    config = _create_workspace(config)
    return config


def _create_workspace(config):
    workspace_name = config["workspace_name"]

    try:
        with cli_logger.group("Creating workspace: {}", workspace_name):
            cli_logger.print(
                "No creation needed for local workspace. It's default.")

    except Exception as e:
        cli_logger.error("Failed to update workspace with the name {}. "
                         "You need to delete and try create again. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
        "Successfully updated workspace: {}.",
        cf.bold(workspace_name))
    return config


def delete_local_workspace(
        config):
    workspace_name = config["workspace_name"]

    try:
        with cli_logger.group("Deleting workspace: {}", workspace_name):
            raise RuntimeError(
                "Cannot delete local workspace. It's default.")
    except Exception as e:
        cli_logger.error(
            "Failed to delete workspace {}. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
            "Successfully deleted workspace: {}.",
            cf.bold(workspace_name))


def check_local_workspace_integrity(config):
    existence = check_local_workspace_existence(config)
    return True if existence == Existence.COMPLETED else False


def check_local_workspace_existence(config):
    workspace_name = config["workspace_name"]

    skipped_resources = 0
    target_resources = LOCAL_WORKSPACE_TARGET_RESOURCES
    existing_resources = 0

    if _is_local_workspace_exists(config, workspace_name):
        existing_resources += 1

    if existing_resources <= skipped_resources:
        return Existence.NOT_EXIST
    elif existing_resources == target_resources:
        return Existence.COMPLETED
    else:
        return Existence.IN_COMPLETED


def _is_local_workspace_exists(config, workspace_name):
    return True


def update_local_workspace(
        config):
    workspace_name = config["workspace_name"]
    try:
        with cli_logger.group("Updating workspace: {}", workspace_name):
            cli_logger.print(
                "No update operation needed for local workspace.")
    except Exception as e:
        cli_logger.error("Failed to update workspace with the name {}. "
                         "You need to delete and try create again. {}", workspace_name, str(e))
        raise e

    cli_logger.success(
        "Successfully updated workspace: {}.",
        cf.bold(workspace_name))


def bootstrap_local_workspace_config(config):
    config["provider"]["workspace_name"] = "default"
    return config
