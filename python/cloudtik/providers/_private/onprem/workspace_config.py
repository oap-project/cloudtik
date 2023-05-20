import logging
from typing import Any, Optional
from typing import Dict

from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD
from cloudtik.providers._private.onprem.config import _get_cloud_simulator_address, _get_http_response_from_simulator, \
    get_cluster_name_from_node

logger = logging.getLogger(__name__)


def get_workspace_head_nodes(provider_config: Dict[str, Any]):
    tag_filters = {CLOUDTIK_TAG_NODE_KIND: NODE_KIND_HEAD}
    request = {"type": "non_terminated_nodes", "args": (tag_filters,)}
    cloud_simulator_address = _get_cloud_simulator_address(provider_config)
    all_heads = _get_http_response_from_simulator(cloud_simulator_address, request)
    return all_heads


def _get_node_info(provider_config: Dict[str, Any], node_id):
    request = {"type": "get_node_info", "args": (node_id,)}
    cloud_simulator_address = _get_cloud_simulator_address(provider_config)
    node_info = _get_http_response_from_simulator(cloud_simulator_address, request)
    return node_info


def _get_node_tags(provider_config: Dict[str, Any], node_id):
    request = {"type": "node_tags", "args": (node_id, )}
    cloud_simulator_address = _get_cloud_simulator_address(provider_config)
    node_tags = _get_http_response_from_simulator(cloud_simulator_address, request)
    return node_tags


def list_onprem_clusters(provider_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    head_nodes = get_workspace_head_nodes(provider_config)
    clusters = {}
    for head_node in head_nodes:
        node_info = _get_node_info(provider_config, head_node)
        cluster_name = get_cluster_name_from_node(node_info)
        if cluster_name:
            clusters[cluster_name] = node_info
    return clusters
