import logging
from typing import Any, Dict

from cloudtik.core._private.utils import is_docker_enabled
from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.providers._private.local.config import prepare_local, \
    _get_bridge_address, post_prepare_local
from cloudtik.providers._private.local.local_container_scheduler import LocalContainerScheduler
from cloudtik.providers._private.local.local_host_scheduler import LocalHostScheduler
from cloudtik.providers._private.local.local_scheduler import LocalScheduler

logger = logging.getLogger(__name__)


class LocalNodeProvider(NodeProvider):
    """NodeProvider for automatically managed host or containers on local node.

    The host or container management is handled by ssh to docker0 bridge of
    local node and execute docker commands over it.
    On working node environment, it can simply to run docker commands.
    Within the cluster environment, for example, on head, you cannot run docker command
    because it is within the container. For this case, it ssh to docker0 and
    execute all the commands including run new docker containers, attach to a worker container.
    """

    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        if self._is_docker_enabled():
            self.local_scheduler = LocalContainerScheduler(provider_config)
        else:
            self.local_scheduler = LocalHostScheduler(provider_config)

    def non_terminated_nodes(self, tag_filters):
        # Only get the non terminated nodes associated with this cluster name.
        tag_filters[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        return self.local_scheduler.get_non_terminated_nodes(tag_filters)

    def is_running(self, node_id):
        return self.local_scheduler.is_running(node_id)

    def is_terminated(self, node_id):
        return self.local_scheduler.is_terminated(node_id)

    def node_tags(self, node_id):
        return self.local_scheduler.get_node_tags(node_id)

    def external_ip(self, node_id):
        return None

    def internal_ip(self, node_id):
        return self.local_scheduler.get_internal_ip(node_id)

    def create_node(self, node_config, tags, count):
        # Tag the newly created node with this cluster name. Helps to get
        # the right nodes when calling non_terminated_nodes.
        tags[CLOUDTIK_TAG_CLUSTER_NAME] = self.cluster_name
        self.local_scheduler.create_node(
            self.cluster_name, node_config, tags, count)

    def set_node_tags(self, node_id, tags):
        self.local_scheduler.set_node_tags(node_id, tags)

    def terminate_node(self, node_id):
        self.local_scheduler.terminate_node(node_id)

    def get_node_info(self, node_id):
        return self.local_scheduler.get_node_info(node_id)

    def with_environment_variables(
            self, node_type_config: Dict[str, Any], node_id: str):
        return {}

    @staticmethod
    def prepare_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        return prepare_local(cluster_config)

    @staticmethod
    def post_prepare(
            cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fills out missing fields after the user config is merged
        with defaults and before validate"""
        return post_prepare_local(cluster_config)

    def _is_docker_enabled(self):
        # We bootstrap global docker configuration to provider config
        return is_docker_enabled(self.provider_config)
