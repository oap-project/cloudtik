import logging
from typing import Any, Dict

from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.providers._private.onprem.config import prepare_onprem, \
    _get_cloud_simulator_address, _get_http_response_from_simulator, post_prepare_onprem, \
    TAG_WORKSPACE_NAME, bootstrap_onprem

logger = logging.getLogger(__name__)


class OnPremNodeProvider(NodeProvider):
    """NodeProvider for automatically managed private/on-premise clusters.

    The cluster management is handled by a remote Cloud Simulator.
    The server listens on <cloud_simulator_address>, therefore, the address
    should be provided in the provider section in the cluster config.
    The server receives HTTP requests from this class and uses
    OnPremNodeProvider to get their responses.
    """

    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        self.cloud_simulator_address = _get_cloud_simulator_address(provider_config)

    def _get_http_response(self, request):
        return _get_http_response_from_simulator(self.cloud_simulator_address, request)

    def non_terminated_nodes(self, tag_filters):
        # Only get the non terminated nodes associated with this cluster name.
        tag_filters[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        tag_filters[TAG_WORKSPACE_NAME] = self.provider_config["workspace_name"]
        request = {"type": "non_terminated_nodes", "args": (tag_filters, )}
        return self._get_http_response(request)

    def is_running(self, node_id):
        request = {"type": "is_running", "args": (node_id, )}
        return self._get_http_response(request)

    def is_terminated(self, node_id):
        request = {"type": "is_terminated", "args": (node_id, )}
        return self._get_http_response(request)

    def node_tags(self, node_id):
        request = {"type": "node_tags", "args": (node_id, )}
        return self._get_http_response(request)

    def external_ip(self, node_id):
        request = {"type": "external_ip", "args": (node_id, )}
        response = self._get_http_response(request)
        return response

    def internal_ip(self, node_id):
        request = {"type": "internal_ip", "args": (node_id, )}
        response = self._get_http_response(request)
        return response

    def create_node(self, node_config, tags, count):
        # Tag the newly created node with this cluster name. Helps to get
        # the right nodes when calling non_terminated_nodes.
        tags[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        tags[TAG_WORKSPACE_NAME] = self.provider_config["workspace_name"]
        request = {
            "type": "create_node",
            "args": (node_config, tags, count),
        }
        self._get_http_response(request)

    def set_node_tags(self, node_id, tags):
        request = {"type": "set_node_tags", "args": (node_id, tags)}
        self._get_http_response(request)

    def terminate_node(self, node_id):
        request = {"type": "terminate_node", "args": (node_id, )}
        self._get_http_response(request)

    def terminate_nodes(self, node_ids):
        request = {"type": "terminate_nodes", "args": (node_ids, )}
        self._get_http_response(request)

    def get_node_info(self, node_id):
        request = {"type": "get_node_info", "args": (node_id,)}
        response = self._get_http_response(request)
        return response

    def with_environment_variables(self, node_type_config: Dict[str, Any], node_id: str):
        request = {"type": "with_environment_variables", "args": (node_type_config, node_id, )}
        response = self._get_http_response(request)
        return response

    @staticmethod
    def bootstrap_config(cluster_config):
        return bootstrap_onprem(cluster_config)

    @staticmethod
    def bootstrap_config_for_api(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        return bootstrap_onprem(cluster_config)

    @staticmethod
    def prepare_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        return prepare_onprem(cluster_config)

    @staticmethod
    def post_prepare(
            cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fills out missing fields after the user config is merged with defaults and before validate"""
        return post_prepare_onprem(cluster_config)
