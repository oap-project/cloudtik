import logging
from typing import Any, Optional
from typing import Dict

from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD
from cloudtik.providers._private.local.config import get_cluster_name_from_node
from cloudtik.providers._private.local.local_container_scheduler import LocalContainerScheduler
from cloudtik.providers._private.local.local_host_scheduler import LocalHostScheduler

logger = logging.getLogger(__name__)


def is_docker_workspace(provider_config: Dict[str, Any]) -> bool:
    return provider_config.get("docker", False)


def _create_local_scheduler(provider_config):
    if is_docker_workspace(provider_config):
        local_scheduler = LocalContainerScheduler(provider_config)
    else:
        local_scheduler = LocalHostScheduler(provider_config)
    return local_scheduler


def get_workspace_head_nodes(provider_config: Dict[str, Any]):
    tag_filters = {CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD}
    local_scheduler = _create_local_scheduler(provider_config)
    return local_scheduler.get_non_terminated_nodes(tag_filters)


def _get_node_info(provider_config: Dict[str, Any], node_id):
    local_scheduler = _create_local_scheduler(provider_config)
    return local_scheduler.get_node_info(node_id)


def _get_node_tags(provider_config: Dict[str, Any], node_id):
    local_scheduler = _create_local_scheduler(provider_config)
    return local_scheduler.get_node_tags(node_id)


def list_local_clusters(provider_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    head_nodes = get_workspace_head_nodes(provider_config)
    clusters = {}
    for head_node in head_nodes:
        node_info = _get_node_info(provider_config, head_node)
        cluster_name = get_cluster_name_from_node(node_info)
        if cluster_name:
            clusters[cluster_name] = node_info
    return clusters
