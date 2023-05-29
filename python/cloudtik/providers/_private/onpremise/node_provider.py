import logging
from threading import RLock
from typing import Any, Dict

from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME, CLOUDTIK_TAG_WORKSPACE_NAME
from cloudtik.providers._private.onpremise.config import prepare_onpremise, \
    _get_cloud_simulator_address, _get_http_response_from_simulator, post_prepare_onpremise, \
    bootstrap_onpremise, _get_node_info

logger = logging.getLogger(__name__)


class OnPremiseNodeProvider(NodeProvider):
    """NodeProvider for automatically managed private/on-premise clusters.

    The cluster management is handled by a remote Cloud Simulator.
    The server listens on <cloud_simulator_address>, therefore, the address
    should be provided in the provider section in the cluster config.
    The server receives HTTP requests from this class and uses
    OnPremiseNodeProvider to get their responses.
    """

    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        self.cloud_simulator_address = _get_cloud_simulator_address(provider_config)

        self.lock = RLock()
        # Cache of node objects from the last list nodes call. This avoids
        # excessive read from remote http requests
        self.cached_nodes: Dict[str, Any] = {}

    def _get_http_response(self, request):
        return _get_http_response_from_simulator(self.cloud_simulator_address, request)

    def non_terminated_nodes(self, tag_filters):
        # Only get the non terminated nodes associated with this cluster name.
        tag_filters = {} if tag_filters is None else tag_filters
        tag_filters[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        # list nodes is thread safe, put it out of the lock block
        matching_nodes = self._list_nodes(tag_filters)
        with self.lock:
            self.cached_nodes = {
                n["name"]: n for n in matching_nodes
            }
            return [node["name"] for node in matching_nodes]

    def is_running(self, node_id):
        # always get current status
        node = self._get_node(node_id=node_id)
        return node["state"] == "running" if node else False

    def is_terminated(self, node_id):
        # always get current status
        return not self.is_running(node_id)

    def node_tags(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            return node.get("tags", {}) if node else {}

    def external_ip(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            return node.get("external_ip")

    def internal_ip(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            return node.get("ip")

    def create_node(self, node_config, tags, count):
        # Tag the newly created node with this cluster name. Helps to get
        # the right nodes when calling non_terminated_nodes.
        tags[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        tags[CLOUDTIK_TAG_WORKSPACE_NAME] = self.provider_config["workspace_name"]
        request = {
            "type": "create_node",
            "args": (node_config, tags, count),
        }
        self._get_http_response(request)

    def set_node_tags(self, node_id, tags):
        # after set node tags to remote, we need to update local node cache
        with self.lock:
            node = self._get_cached_node(node_id)
            self._set_node_tags(node_id, tags)
            # update the cached node tags, although it will refresh at next non_terminated_nodes
            node["tags"].update(tags)

    def terminate_node(self, node_id):
        request = {"type": "terminate_node", "args": (node_id, )}
        self._get_http_response(request)

    def terminate_nodes(self, node_ids):
        request = {"type": "terminate_nodes", "args": (node_ids, )}
        self._get_http_response(request)

    def get_node_info(self, node_id):
        with self.lock:
            node = self._get_cached_node(node_id)
            return _get_node_info(node)

    def with_environment_variables(self, node_type_config: Dict[str, Any], node_id: str):
        return {}

    @staticmethod
    def bootstrap_config(cluster_config):
        return bootstrap_onpremise(cluster_config)

    @staticmethod
    def bootstrap_config_for_api(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        return bootstrap_onpremise(cluster_config)

    @staticmethod
    def prepare_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        return prepare_onpremise(cluster_config)

    @staticmethod
    def post_prepare(
            cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fills out missing fields after the user config is merged with defaults and before validate"""
        return post_prepare_onpremise(cluster_config)

    def _get_cached_node(self, node_id: str):
        if node_id in self.cached_nodes:
            return self.cached_nodes[node_id]

        return self._get_node(node_id)

    def _get_node(self, node_id: str):
        self.non_terminated_nodes({})  # Side effect: updates cache
        with self.lock:
            if node_id in self.cached_nodes:
                return self.cached_nodes[node_id]
            node = self._describe_node(node_id)
            if node is None:
                raise RuntimeError("No node found with id: {}.".format(node_id))
            return node

    def _describe_node(self, node_id: str):
        request = {"type": "describe_node", "args": (node_id,)}
        return self._get_http_response(request)

    def _list_nodes(self, tag_filters):
        workspace_name = self.provider_config["workspace_name"]
        request = {"type": "list_nodes", "args": (workspace_name, tag_filters, )}
        return self._get_http_response(request)

    def _set_node_tags(self, node_id, tags):
        request = {"type": "set_node_tags", "args": (node_id, tags)}
        self._get_http_response(request)
