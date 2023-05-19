import logging

from cloudtik.providers._private.local.config import \
    _get_bridge_address
from cloudtik.providers._private.local.local_scheduler import LocalScheduler

logger = logging.getLogger(__name__)


class LocalContainerScheduler(LocalScheduler):
    def __init__(self, provider_config):
        LocalScheduler.__init__(self, provider_config)
        self.bridge_address = _get_bridge_address(provider_config)

    def create_node(self, cluster_name, node_config, tags, count):
        # TODO
        pass

    def get_non_terminated_nodes(self, tag_filters):
        # TODO
        pass

    def is_running(self, node_id):
        # TODO
        pass

    def is_terminated(self, node_id):
        # TODO
        pass

    def get_node_tags(self, node_id):
        # TODO
        pass

    def get_internal_ip(self, node_id):
        # TODO
        pass

    def set_node_tags(self, node_id, tags):
        # TODO
        pass

    def terminate_node(self, node_id):
        # TODO
        pass

    def get_node_info(self, node_id):
        # TODO
        pass
