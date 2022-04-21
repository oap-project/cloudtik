import json
import logging
from http.client import RemoteDisconnected

from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME

logger = logging.getLogger(__name__)


class CloudSimulatorNodeProvider(NodeProvider):
    """NodeProvider for automatically managed private/local clusters.

    The cluster management is handled by a remote Cloud Simulator.
    The server listens on <cloud_simulator_address>, therefore, the address
    should be provided in the provider section in the cluster config.
    The server receives HTTP requests from this class and uses
    LocalNodeProvider to get their responses.
    """

    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        self.cloud_simulator_address = provider_config["cloud_simulator_address"]

    def _get_http_response(self, request):
        headers = {
            "Content-Type": "application/json",
        }
        request_message = json.dumps(request).encode()
        cloud_simulator_endpoint = "http://" + self.cloud_simulator_address

        try:
            import requests  # `requests` is not part of stdlib.
            from requests.exceptions import ConnectionError

            r = requests.get(
                cloud_simulator_endpoint,
                data=request_message,
                headers=headers,
                timeout=None,
            )
        except (RemoteDisconnected, ConnectionError):
            logger.exception("Could not connect to: " +
                             cloud_simulator_endpoint +
                             ". Did you launched the Cloud Simulator by running python cloud_simulator.py" +
                             " --ips <list_of_node_ips> --port <PORT>?")
            raise
        except ImportError:
            logger.exception(
                "Not all dependencies were found. Please "
                "update your install command.")
            raise

        response = r.json()
        return response

    def non_terminated_nodes(self, tag_filters):
        # Only get the non terminated nodes associated with this cluster name.
        tag_filters[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
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
